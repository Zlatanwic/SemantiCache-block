"""Multi-tenant Poisson load generator against a vanilla vLLM server.

Produces the *real-data* version of the simulator's money-shot left panel: as
arrival rate (load) rises, measure real TTFT p50/p99, TPOT, throughput, the
vLLM preemption counter (scraped from /metrics), and a quality-aware goodput
(SLO-met AND needle-correct). Output JSON mirrors bench/run_moneyshot.py so the
two can be overlaid (real vs calibrated sim).

This is a pure HTTP client — run it on bc01 against a running `vllm serve`.
To actually trigger preemption with a 14B model on an 85GB card, serve with a
constrained KV budget and push long prompts, e.g.:

    vllm serve Qwen/Qwen2.5-14B-Instruct --max-model-len 32768 \
        --gpu-memory-utilization 0.55 --max-num-seqs 32 --port 8000
    uv run python bench/vllm_loadgen.py --rates 2 4 8 12 16 --duration 45 \
        --metrics-url http://localhost:8000/metrics --plot
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI


@dataclass(frozen=True)
class TenantLoad:
    """A tenant's request shape and SLO for the load generator."""

    name: str
    weight: float          # arrival share
    prompt_tokens: int
    max_tokens: int
    slo_ttft: float        # seconds
    needle: bool           # embed a known-answer needle for correctness


DEFAULT_TENANTS = [
    TenantLoad("chat", weight=0.6, prompt_tokens=400, max_tokens=128, slo_ttft=1.0, needle=False),
    TenantLoad("rag",  weight=0.4, prompt_tokens=8000, max_tokens=256, slo_ttft=6.0, needle=True),
]
_FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go again. "


def build_prompt(tenant: TenantLoad, rng: random.Random) -> tuple[str, str | None]:
    """Return (prompt, answer_keyword|None). RAG tenant hides a needle to score."""
    reps = max(1, tenant.prompt_tokens // 14)
    filler = _FILLER * reps
    if not tenant.needle:
        return filler + "\n\nSummarize the passage above in one sentence.", None
    code = str(rng.randint(10000, 99999))
    needle = f"\nIMPORTANT: the access code is {code}.\n"
    mid = len(filler) // 2
    prompt = (filler[:mid] + needle + filler[mid:]
              + "\n\nWhat is the access code mentioned above? Answer with the number only.")
    return prompt, code


def scrape_preemptions(metrics_url: str) -> float:
    """Sum the vLLM preemption counter from the Prometheus /metrics endpoint."""
    try:
        with urllib.request.urlopen(metrics_url, timeout=5) as resp:
            body = resp.read().decode("utf-8", "ignore")
    except Exception:
        return float("nan")
    total = 0.0
    for line in body.splitlines():
        if line.startswith("vllm:num_preemption") and not line.startswith("#"):
            try:
                total += float(line.rsplit(" ", 1)[1])
            except (ValueError, IndexError):
                continue
    return total


async def one_request(client: AsyncOpenAI, model: str, tenant: TenantLoad,
                      prompt: str, answer: str | None) -> dict:
    t0 = time.perf_counter()
    first_t, n_tok, text = None, 0, []
    try:
        stream = await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            max_tokens=tenant.max_tokens, stream=True, temperature=0.0,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                if first_t is None:
                    first_t = time.perf_counter()
                n_tok += 1
                text.append(chunk.choices[0].delta.content)
    except Exception:
        return {"tenant": tenant.name, "ttft": None, "ok": False}
    t1 = time.perf_counter()
    ttft = (first_t - t0) if first_t else None
    correct = True if answer is None else (answer in "".join(text))
    slo_met = ttft is not None and ttft <= tenant.slo_ttft
    return {
        "tenant": tenant.name, "ttft": ttft,
        "tpot": (t1 - first_t) / max(n_tok - 1, 1) if first_t else None,
        "correct": correct, "slo_met": slo_met, "ok": first_t is not None,
    }


async def run_phase(client: AsyncOpenAI, model: str, rate: float, duration: float,
                    tenants: list[TenantLoad], rng: random.Random) -> list[dict]:
    """Fire Poisson(rate) arrivals for `duration` seconds; gather all records."""
    weights = [t.weight for t in tenants]
    tasks: list[asyncio.Task] = []
    start = time.perf_counter()
    while time.perf_counter() - start < duration:
        await asyncio.sleep(rng.expovariate(rate) if rate > 0 else 1.0)
        tenant = rng.choices(tenants, weights=weights, k=1)[0]
        prompt, answer = build_prompt(tenant, rng)
        tasks.append(asyncio.create_task(one_request(client, model, tenant, prompt, answer)))
    if not tasks:
        return []
    return list(await asyncio.gather(*tasks))


def _percentile(vals: list[float], p: float) -> float:
    s = sorted(vals)
    return s[max(0, min(len(s) - 1, math.ceil(p * len(s)) - 1))] if s else float("nan")


def aggregate(records: list[dict], preemptions: float) -> dict:
    ok = [r for r in records if r.get("ok")]
    ttfts = [r["ttft"] for r in ok if r["ttft"] is not None]
    n = len(records)
    good = sum(1 for r in ok if r.get("slo_met") and r.get("correct"))
    return {
        "n": n, "completed": len(ok),
        "ttft_p50": _percentile(ttfts, 0.50), "ttft_p99": _percentile(ttfts, 0.99),
        "mean_tpot": (sum(r["tpot"] for r in ok if r["tpot"]) / max(1, len(ok))),
        "goodput": good / n if n else 0.0,
        "preemptions": preemptions,
    }


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--metrics-url", default="http://localhost:8000/metrics")
    p.add_argument("--rates", type=float, nargs="+", default=[2, 4, 8, 12, 16])
    p.add_argument("--duration", type=float, default=45.0, help="seconds per rate")
    p.add_argument("--out", type=Path, default=Path("results/v3/vllm_loadgen.json"))
    p.add_argument("--plot", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")
    rng = random.Random(args.seed)
    out: dict[str, list[dict]] = {"vanilla_vllm": []}
    for rate in args.rates:
        before = scrape_preemptions(args.metrics_url)
        records = await run_phase(client, args.model, rate, args.duration, DEFAULT_TENANTS, rng)
        after = scrape_preemptions(args.metrics_url)
        row = aggregate(records, after - before)
        row["rate"] = rate
        out["vanilla_vllm"].append(row)
        print(json.dumps({"rate": rate, **{k: row[k] for k in
              ("n", "completed", "ttft_p50", "ttft_p99", "goodput", "preemptions")}}, indent=2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  -> {args.out}")
    if args.plot:
        _plot(out, args.out.with_name("fig_vllm_loadgen.png"))


def _plot(data: dict, path: Path) -> None:
    import matplotlib.pyplot as plt
    rows = data["vanilla_vllm"]
    rates = [r["rate"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(rates, [r["ttft_p99"] for r in rows], "o-", color="#d62728", label="vanilla vLLM")
    ax1.set_xlabel("arrival rate (req/s)"); ax1.set_ylabel("TTFT p99 (s)")
    ax1.set_title("(a) Real tail latency under load"); ax1.legend()
    ax2.plot(rates, [r["preemptions"] for r in rows], "s-", color="#9467bd")
    ax2.set_xlabel("arrival rate (req/s)"); ax2.set_ylabel("preemptions (per phase)")
    ax2.set_title("(b) vLLM preemptions under load")
    fig.suptitle(f"Vanilla vLLM serving under multi-tenant load ({data.get('model','Qwen2.5-14B')})",
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"  -> {path}")


if __name__ == "__main__":
    asyncio.run(main())
