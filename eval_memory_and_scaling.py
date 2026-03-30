"""Experiment script for three supplementary evaluations:

1. Memory footprint profiling — actual KV cache size (MB) and peak GPU memory
2. Longer context scaling — accuracy and overhead at 1800/4000/8000 tokens
3. Multi-needle on Llama — cross-architecture multi-needle validation

Usage examples:
    # Experiment 1: Memory footprint (quick, ~10 min)
    python eval_memory_and_scaling.py memory \
        --policies full streaming h2o snapkv semantic \
        --budgets 0.5 0.3 0.2 0.1

    # Experiment 2: Context scaling (longer, ~30 min)
    python eval_memory_and_scaling.py scaling \
        --policies full streaming h2o snapkv semantic \
        --target-tokens 1800 4000 8000

    # Experiment 3: Multi-needle on Llama
    python eval_memory_and_scaling.py multi-needle-llama \
        --model LLM-Research/Llama-3.2-3B-Instruct \
        --policies full snapkv semantic \
        --needle-counts 1 2 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import numpy as np

from config import ExperimentConfig
from eval_ruler_niah import build_pg_haystack, RULER_KEYS, string_match_all
from eval_multi_needle import run_multi_needle_suite
from run_generation import generate_with_eviction, load_model

import random


# ---------------------------------------------------------------------------
# Experiment 1: Memory footprint profiling
# ---------------------------------------------------------------------------

def measure_memory_footprint(
    model,
    tokenizer,
    policies: list[str],
    budgets: list[float],
    target_tokens: int = 1800,
    num_trials: int = 3,
    seed: int = 42,
    output_path: str | None = None,
    model_cfg=None,
) -> list[dict]:
    """Measure KV cache memory and peak GPU memory per policy × budget."""
    rng = random.Random(seed)
    results = []

    total = len(policies) * len(budgets) * num_trials
    print(f"Memory footprint profiling: {total} runs")
    run_idx = 0
    t0 = time.time()

    for policy in policies:
        for budget in budgets:
            for trial in range(num_trials):
                run_idx += 1
                elapsed = time.time() - t0
                eta = (elapsed / run_idx) * (total - run_idx) if run_idx > 1 else 0
                print(
                    f"[{run_idx}/{total}] {policy} b={budget:.0%} trial={trial+1} "
                    f"(elapsed={elapsed:.0f}s eta={eta:.0f}s)"
                )

                # Build a simple NIAH prompt
                key = rng.choice(RULER_KEYS)
                value = str(rng.randint(10000, 99999))
                needle = f"One of the special magic numbers for {key} is: {value}."
                haystack = build_pg_haystack(tokenizer, target_tokens)
                sentences = haystack.split("\n")
                mid = len(sentences) // 2
                sentences.insert(mid, needle)
                text = "\n".join(sentences)
                question = f"What is one of the special magic numbers for {key} mentioned in the provided text?"

                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Answer based only on the provided text."},
                    {"role": "user", "content": f"{text}\n\n{question}"},
                ]

                cfg = ExperimentConfig()
                if model_cfg is not None:
                    cfg.model = model_cfg
                cfg.cache.policy = policy
                cfg.cache.cache_budget = budget
                cfg.model.do_sample = False
                cfg.model.max_new_tokens = 32
                cfg.model.stop_when_output_contains = [value]

                # Reset peak memory counter
                torch.cuda.reset_peak_memory_stats()
                mem_before = torch.cuda.memory_allocated()

                run_result = generate_with_eviction(model, tokenizer, messages, cfg)

                mem_after = torch.cuda.memory_allocated()
                peak_mem = torch.cuda.max_memory_allocated()

                stats = run_result.get("stats", {})
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt_tokens = len(tokenizer.encode(prompt_text))

                # Estimate KV cache size: retained_tokens × num_layers × 2(K+V) × num_kv_heads × head_dim × dtype_bytes
                retained = stats.get("current_cache_len", prompt_tokens)
                num_layers = cfg.model.num_layers
                num_kv_heads = cfg.model.num_kv_heads
                head_dim = cfg.model.head_dim
                # 4-bit quantized model but KV cache is in float16
                kv_bytes = retained * num_layers * 2 * num_kv_heads * head_dim * 2  # fp16 = 2 bytes
                kv_mb = kv_bytes / (1024 * 1024)

                # Full KV cache size for comparison
                full_kv_bytes = prompt_tokens * num_layers * 2 * num_kv_heads * head_dim * 2
                full_kv_mb = full_kv_bytes / (1024 * 1024)

                result = {
                    "policy": policy,
                    "budget": budget,
                    "trial": trial + 1,
                    "prompt_tokens": prompt_tokens,
                    "retained_tokens": retained,
                    "retention_ratio": retained / prompt_tokens if prompt_tokens > 0 else 1.0,
                    "kv_cache_mb": round(kv_mb, 2),
                    "full_kv_cache_mb": round(full_kv_mb, 2),
                    "kv_savings_pct": round((1 - kv_mb / full_kv_mb) * 100, 1) if full_kv_mb > 0 else 0,
                    "peak_gpu_mb": round(peak_mem / (1024 * 1024), 1),
                    "score": 100 if value in run_result["output_text"] else 0,
                }
                results.append(result)
                print(
                    f"  retained={retained}/{prompt_tokens} "
                    f"kv={kv_mb:.1f}MB (save {result['kv_savings_pct']}%) "
                    f"peak_gpu={result['peak_gpu_mb']:.0f}MB"
                )

    # Summary table
    print("\n" + "=" * 70)
    print("MEMORY FOOTPRINT SUMMARY")
    print("=" * 70)
    print(f"{'Policy':<16} {'Budget':>6} {'Retained':>8} {'KV (MB)':>8} {'Savings':>8} {'Peak GPU':>10}")
    print("-" * 60)
    for policy in policies:
        for budget in budgets:
            subset = [r for r in results if r["policy"] == policy and r["budget"] == budget]
            if not subset:
                continue
            avg_retained = int(np.mean([r["retained_tokens"] for r in subset]))
            avg_kv = np.mean([r["kv_cache_mb"] for r in subset])
            avg_savings = np.mean([r["kv_savings_pct"] for r in subset])
            avg_peak = np.mean([r["peak_gpu_mb"] for r in subset])
            print(
                f"{policy:<16} {budget:>5.0%} {avg_retained:>8} "
                f"{avg_kv:>7.1f} {avg_savings:>7.1f}% {avg_peak:>9.0f}MB"
            )

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved to: {out}")

    return results


# ---------------------------------------------------------------------------
# Experiment 2: Longer context scaling
# ---------------------------------------------------------------------------

def run_scaling_experiment(
    model,
    tokenizer,
    policies: list[str],
    target_tokens_list: list[int],
    budget: float = 0.2,
    num_trials: int = 10,
    seed: int = 42,
    max_new_tokens: int = 64,
    output_path: str | None = None,
    model_cfg=None,
) -> list[dict]:
    """Evaluate accuracy and overhead across different context lengths."""
    rng = random.Random(seed)
    results = []

    total = len(policies) * len(target_tokens_list) * num_trials
    print(f"Context scaling experiment: {total} runs")
    print(f"  policies: {policies}")
    print(f"  target_tokens: {target_tokens_list}")
    print(f"  budget: {budget:.0%}")
    print(f"  trials: {num_trials}")

    run_idx = 0
    t0 = time.time()

    for target_tokens in target_tokens_list:
        haystack = build_pg_haystack(tokenizer, target_tokens)

        for policy in policies:
            for trial in range(num_trials):
                run_idx += 1
                elapsed = time.time() - t0
                eta = (elapsed / run_idx) * (total - run_idx) if run_idx > 1 else 0

                # Random needle
                key = rng.choice(RULER_KEYS)
                value = str(rng.randint(10000, 99999))
                needle = f"One of the special magic numbers for {key} is: {value}."
                depth = rng.random()

                sentences = haystack.split("\n")
                idx = max(1, min(int(len(sentences) * depth), len(sentences) - 1))
                sentences_copy = sentences.copy()
                sentences_copy.insert(idx, needle)
                text = "\n".join(sentences_copy)
                question = f"What is one of the special magic numbers for {key} mentioned in the provided text?"

                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Answer based only on the provided text."},
                    {"role": "user", "content": f"{text}\n\n{question}"},
                ]

                cfg = ExperimentConfig()
                if model_cfg is not None:
                    cfg.model = model_cfg
                cfg.cache.policy = policy
                cfg.cache.cache_budget = budget
                cfg.model.do_sample = False
                cfg.model.max_new_tokens = max_new_tokens
                cfg.model.stop_when_output_contains = [value]

                print(
                    f"[{run_idx}/{total}] tokens={target_tokens} {policy} "
                    f"trial={trial+1} depth={depth:.2f} "
                    f"(elapsed={elapsed:.0f}s eta={eta:.0f}s)"
                )

                torch.cuda.reset_peak_memory_stats()
                run_result = generate_with_eviction(model, tokenizer, messages, cfg)

                output_text = run_result["output_text"]
                score = 100 if value.lower() in output_text.lower() else 0
                stats = run_result.get("stats", {})

                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt_tokens = len(tokenizer.encode(prompt_text))

                result = {
                    "target_tokens": target_tokens,
                    "actual_prompt_tokens": prompt_tokens,
                    "policy": policy,
                    "budget": budget,
                    "trial": trial + 1,
                    "depth": round(depth, 3),
                    "score": score,
                    "eviction_time_per_step_ms": round(run_result.get("eviction_time_per_step", 0) * 1000, 2),
                    "prefill_time": round(run_result.get("prefill_time", 0), 3),
                    "decode_time": round(run_result.get("decode_time", 0), 3),
                    "tokens_per_second": round(run_result.get("tokens_per_second", 0), 2),
                    "retained_tokens": stats.get("current_cache_len", 0),
                    "total_evicted": stats.get("total_evicted", 0),
                    "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1),
                }
                results.append(result)
                print(
                    f"  score={score}% retained={result['retained_tokens']}/{prompt_tokens} "
                    f"evict/step={result['eviction_time_per_step_ms']:.1f}ms "
                    f"peak={result['peak_gpu_mb']:.0f}MB"
                )

        # Per-length summary
        print(f"\n--- target_tokens={target_tokens} summary ---")
        for policy in policies:
            subset = [r for r in results if r["target_tokens"] == target_tokens and r["policy"] == policy]
            if not subset:
                continue
            avg_score = np.mean([r["score"] for r in subset])
            avg_evict = np.mean([r["eviction_time_per_step_ms"] for r in subset])
            avg_tps = np.mean([r["tokens_per_second"] for r in subset])
            print(f"  {policy:<16} score={avg_score:.1f}% evict/step={avg_evict:.1f}ms tok/s={avg_tps:.1f}")
        print()

    # Final summary
    print("\n" + "=" * 70)
    print("CONTEXT SCALING SUMMARY")
    print("=" * 70)
    print(f"{'Policy':<16} ", end="")
    for t in target_tokens_list:
        print(f"{'%dt' % t:>12}", end="")
    print()
    print("-" * (16 + 12 * len(target_tokens_list)))

    for policy in policies:
        print(f"{policy:<16} ", end="")
        for t in target_tokens_list:
            subset = [r for r in results if r["target_tokens"] == t and r["policy"] == policy]
            avg = np.mean([r["score"] for r in subset]) if subset else 0
            print(f"{avg:>11.1f}%", end="")
        print()

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved to: {out}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supplementary experiments: memory, scaling, multi-needle-llama")
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    # --- Experiment 1: Memory ---
    mem_parser = subparsers.add_parser("memory", help="Memory footprint profiling")
    mem_parser.add_argument("--model", type=str, default=None)
    mem_parser.add_argument("--policies", nargs="+", default=["full", "streaming", "h2o", "snapkv", "semantic"])
    mem_parser.add_argument("--budgets", nargs="+", type=float, default=[0.5, 0.3, 0.2, 0.1])
    mem_parser.add_argument("--target-tokens", type=int, default=1800)
    mem_parser.add_argument("--num-trials", type=int, default=3)
    mem_parser.add_argument("--seed", type=int, default=42)
    mem_parser.add_argument("--output", default="results/memory/memory_footprint.json")

    # --- Experiment 2: Scaling ---
    scale_parser = subparsers.add_parser("scaling", help="Context length scaling")
    scale_parser.add_argument("--model", type=str, default=None)
    scale_parser.add_argument("--policies", nargs="+", default=["full", "streaming", "h2o", "snapkv", "semantic"])
    scale_parser.add_argument("--budgets", nargs="+", type=float, default=[0.2])
    scale_parser.add_argument("--target-tokens", nargs="+", type=int, default=[1800, 4000, 8000])
    scale_parser.add_argument("--num-trials", type=int, default=10)
    scale_parser.add_argument("--max-new-tokens", type=int, default=64)
    scale_parser.add_argument("--seed", type=int, default=42)
    scale_parser.add_argument("--output", default="results/scaling/context_scaling.json")

    # --- Experiment 3: Multi-needle Llama ---
    mn_parser = subparsers.add_parser("multi-needle-llama", help="Multi-needle on Llama")
    mn_parser.add_argument("--model", type=str, default="LLM-Research/Llama-3.2-3B-Instruct")
    mn_parser.add_argument("--policies", nargs="+", default=["full", "snapkv", "semantic"])
    mn_parser.add_argument("--budgets", nargs="+", type=float, default=[0.2])
    mn_parser.add_argument("--needle-counts", nargs="+", type=int, default=[1, 2, 4])
    mn_parser.add_argument("--target-tokens", type=int, default=1800)
    mn_parser.add_argument("--num-trials", type=int, default=10)
    mn_parser.add_argument("--max-new-tokens", type=int, default=128)
    mn_parser.add_argument("--seed", type=int, default=42)
    mn_parser.add_argument("--output", default="results/multi_needle/multi_needle_llama.json")

    args = parser.parse_args()

    cfg = ExperimentConfig()
    model_name = getattr(args, "model", None)
    if model_name:
        cfg.model.model_name = model_name
    model, tokenizer = load_model(cfg.model)

    if args.experiment == "memory":
        measure_memory_footprint(
            model, tokenizer,
            policies=args.policies,
            budgets=args.budgets,
            target_tokens=args.target_tokens,
            num_trials=args.num_trials,
            seed=args.seed,
            output_path=args.output,
            model_cfg=cfg.model,
        )

    elif args.experiment == "scaling":
        for budget in args.budgets:
            run_scaling_experiment(
                model, tokenizer,
                policies=args.policies,
                target_tokens_list=args.target_tokens,
                budget=budget,
                num_trials=args.num_trials,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
                output_path=args.output,
                model_cfg=cfg.model,
            )

    elif args.experiment == "multi-needle-llama":
        run_multi_needle_suite(
            model=model,
            tokenizer=tokenizer,
            policies=args.policies,
            budgets=args.budgets,
            needle_counts=args.needle_counts,
            target_tokens=args.target_tokens,
            num_trials=args.num_trials,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            output_path=args.output,
            model_cfg=cfg.model,
        )
