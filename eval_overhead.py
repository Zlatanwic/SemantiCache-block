"""
Systems overhead profiling for SYSTOR 2026 full paper.

Measures per-policy: prefill time, decode time, eviction overhead,
snapshot overhead, forward time, throughput, and KV memory savings.

Usage:
    python eval_overhead.py                       # all policies, default settings
    python eval_overhead.py --policies full semantic kvzip --runs 3
    python eval_overhead.py --output results/overhead_table.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

from config import CacheConfig, ExperimentConfig, ModelConfig
from run_generation import generate_with_eviction, load_model


# Policies to profile and their cache budget (0 means full / no eviction)
DEFAULT_POLICIES = [
    ("full", 0.0),
    ("streaming", 0.2),
    ("h2o", 0.2),
    ("snapkv", 0.2),
    ("kvzip", 0.2),
    ("semantic", 0.2),
    ("tiered_semantic", 0.2),
]

# A fixed NIAH-style prompt so all policies see the same workload
HAYSTACK_FILLER = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again. "
) * 80  # ~800 tokens of filler

NEEDLE = "The special key is: diamond-falcon-7."

SYSTEM_MSG = "You are a helpful assistant."
USER_MSG = (
    f"Read the following passage carefully.\n\n"
    f"{HAYSTACK_FILLER}\n"
    f"Remember this fact: {NEEDLE}\n"
    f"{HAYSTACK_FILLER}\n\n"
    f"What is the special key mentioned in the passage above? "
    f"Answer with just the key value."
)


def build_messages():
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": USER_MSG},
    ]


def profile_policy(
    model, tokenizer, policy_name: str, budget_ratio: float,
    max_new_tokens: int, run_idx: int,
) -> dict:
    """Run one generation and collect timing metrics."""
    cache_cfg = CacheConfig(
        policy=policy_name,
        cache_budget=budget_ratio if budget_ratio > 0 else 1.0,
    )
    model_cfg = ModelConfig(max_new_tokens=max_new_tokens)
    config = ExperimentConfig(model=model_cfg, cache=cache_cfg)
    messages = build_messages()

    # Warm up GPU on first run
    if run_idx == 0:
        torch.cuda.synchronize() if torch.cuda.is_available() else None

    result = generate_with_eviction(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        config=config,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    stats = result.get("stats", {})

    return {
        "policy": policy_name,
        "budget_ratio": budget_ratio,
        "run": run_idx,
        "prefill_time": result.get("prefill_time", 0),
        "decode_time": result.get("decode_time", 0),
        "snapshot_time": result.get("snapshot_time", 0),
        "eviction_time_total": result.get("eviction_time_total", 0),
        "eviction_time_per_step": result.get("eviction_time_per_step", 0),
        "forward_time_total": result.get("forward_time_total", 0),
        "forward_time_per_step": result.get("forward_time_per_step", 0),
        "elapsed_time": result.get("elapsed_time", 0),
        "num_decode_steps": result.get("num_decode_steps", 0),
        "tokens_per_second": result.get("tokens_per_second", 0),
        "output_text": result.get("output_text", "")[:200],
        # KV memory stats from cache manager
        "current_cache_len": stats.get("current_cache_len", 0),
        "total_evicted": stats.get("total_evicted", 0),
        "hot_cache_len": stats.get("hot_cache_len", 0),
        "warm_cache_len": stats.get("warm_cache_len", 0),
        "warm_quantize_time_s": stats.get("warm_quantize_time_s", 0),
        "warm_dequantize_time_s": stats.get("warm_dequantize_time_s", 0),
    }


def print_summary_table(results: list[dict]):
    """Print a compact summary table for the paper."""
    # Aggregate by policy
    from collections import defaultdict
    agg = defaultdict(list)
    for r in results:
        agg[r["policy"]].append(r)

    header = (
        f"{'Policy':<18} {'Prefill':>8} {'Decode':>8} {'Snap':>7} "
        f"{'Evict/step':>11} {'Fwd/step':>9} {'Total':>7} {'tok/s':>7} "
        f"{'Evict%':>7}"
    )
    print("\n" + "=" * len(header))
    print("SYSTEMS OVERHEAD PROFILING SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for policy_name, _ in DEFAULT_POLICIES:
        if policy_name not in agg:
            continue
        runs = agg[policy_name]
        n = len(runs)
        avg = lambda key: sum(r[key] for r in runs) / n

        prefill = avg("prefill_time")
        decode = avg("decode_time")
        snap = avg("snapshot_time")
        evict_step = avg("eviction_time_per_step")
        fwd_step = avg("forward_time_per_step")
        total = avg("elapsed_time")
        tps = avg("tokens_per_second")
        evict_total = avg("eviction_time_total")
        evict_pct = (evict_total / max(1e-6, decode)) * 100

        print(
            f"{policy_name:<18} {prefill:>7.3f}s {decode:>7.3f}s {snap:>6.3f}s "
            f"{evict_step * 1000:>9.2f}ms {fwd_step * 1000:>7.2f}ms {total:>6.2f}s {tps:>6.1f} "
            f"{evict_pct:>6.1f}%"
        )

    print("=" * len(header))


def export_latex_table(results: list[dict]) -> str:
    """Generate a LaTeX table suitable for the paper."""
    from collections import defaultdict
    agg = defaultdict(list)
    for r in results:
        agg[r["policy"]].append(r)

    lines = [
        r"\begin{table}[t]",
        r"\caption{Systems overhead per policy at $b{=}20\%$ cache budget.",
        r"Averaged over multiple runs on \texttt{Qwen2.5-3B-Instruct} (4-bit NF4).}",
        r"\label{tab:overhead}",
        r"\centering",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Policy & Prefill (s) & Snap (s) & Evict/step (ms) & Decode (s) & tok/s \\",
        r"\midrule",
    ]

    policy_display = {
        "full": "Full KV",
        "streaming": "StreamingLLM",
        "h2o": "H2O",
        "snapkv": "SnapKV",
        "kvzip": "KVzip",
        "semantic": "SemantiCache",
        "tiered_semantic": "Tiered SemantiCache",
    }

    for policy_name, _ in DEFAULT_POLICIES:
        if policy_name not in agg:
            continue
        runs = agg[policy_name]
        n = len(runs)
        avg = lambda key: sum(r[key] for r in runs) / n

        name = policy_display.get(policy_name, policy_name)
        prefill = avg("prefill_time")
        snap = avg("snapshot_time")
        evict_step = avg("eviction_time_per_step") * 1000  # ms
        decode = avg("decode_time")
        tps = avg("tokens_per_second")

        if policy_name in ("semantic", "tiered_semantic"):
            name = r"\textbf{" + name + "}"

        lines.append(
            f"{name} & {prefill:.2f} & {snap:.3f} & {evict_step:.2f} & {decode:.2f} & {tps:.1f} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Systems overhead profiling")
    parser.add_argument(
        "--policies", nargs="+", default=None,
        help="Policies to profile (default: all)",
    )
    parser.add_argument(
        "--budget", type=float, default=0.2,
        help="Cache budget ratio for non-full policies (default: 0.2)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=16,
        help="Max tokens to generate per run (default: 16)",
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of runs per policy for averaging (default: 3)",
    )
    parser.add_argument(
        "--output", type=str, default="results/overhead_profiling.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--latex", type=str, default="results/overhead_table.tex",
        help="Output LaTeX table path",
    )
    args = parser.parse_args()

    # Determine which policies to run
    if args.policies:
        policies = []
        for p in args.policies:
            budget = 0.0 if p == "full" else args.budget
            policies.append((p, budget))
    else:
        policies = [(name, 0.0 if name == "full" else args.budget) for name, _ in DEFAULT_POLICIES]

    print(f"Loading model...")
    model_cfg = ModelConfig()
    model, tokenizer = load_model(model_cfg)
    print(f"Model loaded on {model.device}")

    all_results = []
    total_runs = len(policies) * args.runs

    for pi, (policy_name, budget) in enumerate(policies):
        print(f"\n{'='*60}")
        print(f"Profiling: {policy_name} (budget={budget:.0%}) — {args.runs} runs")
        print(f"{'='*60}")

        for run_idx in range(args.runs):
            print(f"\n  Run {run_idx + 1}/{args.runs}...")
            try:
                result = profile_policy(
                    model=model,
                    tokenizer=tokenizer,
                    policy_name=policy_name,
                    budget_ratio=budget,
                    max_new_tokens=args.max_new_tokens,
                    run_idx=run_idx,
                )
                all_results.append(result)
                print(
                    f"  -> {result['elapsed_time']:.2f}s total, "
                    f"{result['tokens_per_second']:.1f} tok/s, "
                    f"evict/step={result['eviction_time_per_step']*1000:.2f}ms"
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Summary table
    print_summary_table(all_results)

    # Save JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nJSON results saved to {args.output}")

    # Save LaTeX
    latex = export_latex_table(all_results)
    os.makedirs(os.path.dirname(args.latex), exist_ok=True)
    with open(args.latex, "w") as f:
        f.write(latex)
    print(f"LaTeX table saved to {args.latex}")
    print(f"\nLaTeX output:\n{latex}")


if __name__ == "__main__":
    main()
