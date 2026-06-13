"""Smoke benchmark: N concurrent long-context requests against a vLLM server.

Phase 0 Task 0.3 — establish a vanilla-vLLM serving baseline and find the load
point that starts triggering preemption. Pure HTTP client; run from anywhere
against a running ``vllm serve`` instance.

Run (on bc01, vLLM already serving):
    uv run python bench/baseline_smoke.py --concurrency 16 --prompt-tokens 8000
    # then check the vLLM server log for "preempt", or scrape /metrics
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

from openai import AsyncOpenAI


async def one_request(client: AsyncOpenAI, model: str, prompt: str, max_tokens: int) -> dict:
    t0 = time.perf_counter()
    first_token_t = None
    n_tokens = 0
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True,
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if first_token_t is None:
                first_token_t = time.perf_counter()
            n_tokens += 1
    t1 = time.perf_counter()
    return {
        "ttft": first_token_t - t0 if first_token_t else None,
        "tpot": (t1 - first_token_t) / max(n_tokens - 1, 1) if first_token_t else None,
        "total": t1 - t0,
        "tokens": n_tokens,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--prompt-tokens", type=int, default=8000)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    args = parser.parse_args()

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")
    filler = "The quick brown fox jumps over the lazy dog. " * (args.prompt_tokens // 9)
    prompt = filler + "\n\nSummarize the above in one sentence."

    results = await asyncio.gather(
        *[one_request(client, args.model, prompt, args.max_tokens) for _ in range(args.concurrency)]
    )
    ttfts = sorted(r["ttft"] for r in results if r["ttft"])
    print(json.dumps({
        "model": args.model,
        "concurrency": args.concurrency,
        "prompt_tokens": args.prompt_tokens,
        "ttft_p50": ttfts[len(ttfts) // 2] if ttfts else None,
        "ttft_p99": ttfts[min(len(ttfts) - 1, int(len(ttfts) * 0.99))] if ttfts else None,
        "mean_tpot": sum(r["tpot"] for r in results if r["tpot"]) / max(1, sum(1 for r in results if r["tpot"])),
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
