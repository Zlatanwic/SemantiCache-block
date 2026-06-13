"""Run the SemServe multi-tenant 'money shot' experiment and plot it.

Sweeps arrival rate (load) for three policies and reports tail latency + the
quality-aware goodput, averaged over seeds. CPU-only and fast — no GPU.

  vanilla  : whole-request preemption + recompute (vLLM behavior)
  uniform  : compress everyone to 0.5 budget (no semantics)
  semserve : tenant cgroup budgets + cross-request semantic water-filling

Quality is grounded in the *measured* block-gate curve (semantic retention holds
to ~25% budget); latency is an explicit prefill+recompute model. See semserve.sim.

Run:
    uv run python bench/run_moneyshot.py --out-dir results/v3
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams

from semserve.sim import SimConfig, TenantSpec, aggregate_metrics, simulate

rcParams.update({
    "font.family": "DejaVu Serif", "font.size": 11,
    "axes.spines.right": False, "axes.spines.top": False,
    "axes.grid": True, "grid.alpha": 0.3, "lines.linewidth": 2.2, "lines.markersize": 7,
})
POLICY_STYLE = {
    "vanilla":  {"color": "#d62728", "marker": "v", "label": "vanilla vLLM (whole-request preempt)"},
    "uniform":  {"color": "#ff7f0e", "marker": "s", "label": "uniform compress (no semantics)"},
    "semserve": {"color": "#2ca02c", "marker": "^", "label": "SemServe (semantic, ours)"},
}

# Block-gate-calibrated quality: semantic retention holds quality down to ~25% budget.
QUALITY_CURVE = [(0.1, 0.32), (0.25, 0.95), (0.5, 0.95), (1.0, 0.95)]
TENANTS = [
    TenantSpec("rag",  slo_class=1, slo_ttft=40, prompt_blocks=8, decode_tokens=10, mean_score=0.20),
    TenantSpec("chat", slo_class=2, slo_ttft=40, prompt_blocks=2, decode_tokens=5,  mean_score=0.85),
]


def sweep(rates: list[float], seeds: list[int], horizon: int) -> dict:
    """Return {policy: [{rate, ttft_p99, ttft_p50, goodput, fairness, preemptions}, ...]}."""
    out: dict[str, list[dict]] = {p: [] for p in POLICY_STYLE}
    for rate in rates:
        cfg = SimConfig(total_blocks=20, arrival_rate=rate, horizon=horizon,
                        block_quality_curve=QUALITY_CURVE, uniform_quality=0.45, full_quality=0.95)
        for policy in POLICY_STYLE:
            runs = [simulate(policy, TENANTS, cfg, seed=s) for s in seeds]
            metrics = [aggregate_metrics(r.records) for r in runs]
            out[policy].append({
                "rate": rate,
                "ttft_p50": statistics.mean(m["ttft_p50"] for m in metrics),
                "ttft_p99": statistics.mean(m["ttft_p99"] for m in metrics),
                "goodput": statistics.mean(m["goodput"] for m in metrics),
                "fairness": statistics.mean(m["fairness"] for m in metrics),
                "preemptions": statistics.mean(r.preemptions for r in runs),
            })
    return out


def plot(sweep_data: dict, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for policy, rows in sweep_data.items():
        s = POLICY_STYLE[policy]
        rates = [r["rate"] for r in rows]
        ax1.plot(rates, [r["ttft_p99"] for r in rows], color=s["color"], marker=s["marker"], label=s["label"])
        ax2.plot(rates, [r["goodput"] * 100 for r in rows], color=s["color"], marker=s["marker"], label=s["label"])
    ax1.set_xlabel("arrival rate (requests / step → load)")
    ax1.set_ylabel("TTFT p99 (sim steps)")
    ax1.set_title("(a) Tail latency under load")
    ax1.legend(fontsize=8, loc="upper left")
    ax2.set_xlabel("arrival rate (requests / step → load)")
    ax2.set_ylabel("quality-aware goodput (%)")
    ax2.set_ylim(0, 100)
    ax2.set_title("(b) Quality-aware goodput under load")
    ax2.legend(fontsize=8, loc="lower left")
    fig.suptitle("SemServe multi-tenant simulation (quality calibrated by measured block-gate curve)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("results/v3"))
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--rates", type=float, nargs="+",
                   default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()

    data = sweep(args.rates, args.seeds, args.horizon)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sim_moneyshot.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8")
    print(f"  -> {args.out_dir / 'sim_moneyshot.json'}")
    plot(data, args.out_dir / "fig_moneyshot.png")


if __name__ == "__main__":
    main()
