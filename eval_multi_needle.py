"""RULER Multi-Needle-in-a-Haystack evaluation.

Tests whether eviction policies can simultaneously retain multiple factual
spans scattered across different depths in the context.

Reference: RULER benchmark (NVIDIA, NAACL 2024) — multi-key NIAH variant.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np

from config import ExperimentConfig
from eval_ruler_niah import (
    RULER_KEYS,
    build_pg_haystack,
    string_match_all,
)
from run_generation import generate_with_eviction, load_model


# ---------------------------------------------------------------------------
# Multi-needle generation and insertion
# ---------------------------------------------------------------------------

NEEDLE_TEMPLATE = "One of the special magic numbers for {key} is: {value}."
QUERY_TEMPLATE = "What are all the special magic numbers for {key} mentioned in the provided text?"


def make_multi_needles(
    num_needles: int,
    rng: random.Random,
) -> tuple[str, str, list[str]]:
    """Generate num_needles facts for the same key, each with a unique value.

    Returns (key, question, list_of_facts, list_of_values).
    """
    key = rng.choice(RULER_KEYS)
    values = []
    facts = []
    for _ in range(num_needles):
        val = str(rng.randint(10000, 99999))
        while val in values:  # ensure unique
            val = str(rng.randint(10000, 99999))
        values.append(val)
        facts.append(NEEDLE_TEMPLATE.format(key=key, value=val))
    question = QUERY_TEMPLATE.format(key=key)
    return key, question, facts, values


def insert_needles_at_depths(
    haystack: str,
    facts: list[str],
    depths: list[float],
) -> str:
    """Insert multiple needles at specified depths (sorted to preserve positions)."""
    sentences = haystack.split("\n")
    total = len(sentences)

    # Build (position, fact) pairs, sorted by position descending so inserts
    # don't shift later indices
    inserts = []
    for fact, depth in zip(facts, depths):
        idx = max(1, min(int(total * depth), total - 1))
        inserts.append((idx, fact))
    inserts.sort(key=lambda x: x[0], reverse=True)

    for idx, fact in inserts:
        sentences.insert(idx, fact)

    return "\n".join(sentences)


# ---------------------------------------------------------------------------
# Single evaluation run
# ---------------------------------------------------------------------------

def run_multi_needle_eval(
    model,
    tokenizer,
    config: ExperimentConfig,
    num_needles: int,
    target_tokens: int,
    rng: random.Random,
    max_new_tokens: int = 128,
) -> dict:
    """Run a single multi-needle NIAH evaluation."""
    key, question, facts, values = make_multi_needles(num_needles, rng)

    # Spread needles evenly across depths
    depths = list(np.linspace(0.0, 1.0, num=num_needles, endpoint=True))

    haystack = build_pg_haystack(tokenizer, target_tokens)
    text_with_needles = insert_needles_at_depths(haystack, facts, depths)

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer questions based only on the provided text."},
        {"role": "user", "content": f"{text_with_needles}\n\n{question}"},
    ]

    config.model.do_sample = False
    config.model.max_new_tokens = max_new_tokens
    config.model.stop_when_output_contains = []

    t0 = time.time()
    run_result = generate_with_eviction(model, tokenizer, messages, config)
    elapsed = time.time() - t0

    output_text = run_result["output_text"]
    score = string_match_all(output_text, values)
    num_found = sum(1 for v in values if v.lower() in output_text.lower())

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = len(tokenizer.encode(prompt_text))

    stats = run_result.get("stats", {})
    return {
        "benchmark": "ruler_multi_niah",
        "policy": config.cache.policy,
        "budget": config.cache.cache_budget,
        "num_needles": num_needles,
        "target_tokens": target_tokens,
        "actual_prompt_tokens": prompt_tokens,
        "needle_key": key,
        "needle_values": values,
        "needle_depths": depths,
        "output_text": output_text,
        "score": score,
        "num_found": num_found,
        "elapsed_time": elapsed,
        "tokens_per_second": len(run_result.get("output_ids", [])) / elapsed if elapsed > 0 else 0,
        "retained_tokens": stats.get("current_cache_len", 0),
        "initial_seq_len": stats.get("initial_seq_len", 0),
        "total_evicted": stats.get("total_evicted", 0),
    }


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_multi_needle_suite(
    model,
    tokenizer,
    policies: list[str],
    budgets: list[float],
    needle_counts: list[int],
    target_tokens: int,
    num_trials: int = 10,
    seed: int = 42,
    max_new_tokens: int = 128,
    output_path: Optional[str] = None,
    model_cfg=None,
) -> list[dict]:
    """Run multi-needle evaluation grid."""
    rng = random.Random(seed)

    total = len(policies) * len(budgets) * len(needle_counts) * num_trials
    print(f"Multi-Needle NIAH suite: {total} total runs")
    print(f"  policies: {policies}")
    print(f"  budgets: {budgets}")
    print(f"  needle_counts: {needle_counts}")
    print(f"  trials per cell: {num_trials}")
    print(f"  target_tokens: {target_tokens}")

    results: list[dict] = []
    run_idx = 0
    t0 = time.time()

    for policy in policies:
        for budget in budgets:
            for n_needles in needle_counts:
                for trial in range(num_trials):
                    run_idx += 1
                    elapsed = time.time() - t0
                    eta = (elapsed / run_idx) * (total - run_idx) if run_idx > 1 else 0
                    print(
                        f"[{run_idx}/{total}] {policy} b={budget:.0%} "
                        f"needles={n_needles} trial={trial+1} "
                        f"(elapsed={elapsed:.0f}s eta={eta:.0f}s)"
                    )

                    cfg = ExperimentConfig()
                    if model_cfg is not None:
                        cfg.model = model_cfg
                    cfg.cache.policy = policy
                    cfg.cache.cache_budget = budget

                    result = run_multi_needle_eval(
                        model, tokenizer, cfg,
                        num_needles=n_needles,
                        target_tokens=target_tokens,
                        rng=rng,
                        max_new_tokens=max_new_tokens,
                    )
                    result["trial"] = trial + 1
                    results.append(result)
                    print(
                        f"  found={result['num_found']}/{n_needles} "
                        f"score={result['score']:.0f}% "
                        f"-> {result['output_text'][:60]}"
                    )

        # Per-policy checkpoint
        policy_results = [r for r in results if r["policy"] == policy]
        avg_score = sum(r["score"] for r in policy_results) / len(policy_results) if policy_results else 0
        print(f"\n=== {policy} done: avg_score={avg_score:.1f}% ===\n")

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved to: {out}")

    # Summary
    print("\n" + "=" * 70)
    print("MULTI-NEEDLE NIAH SUMMARY")
    print("=" * 70)
    for policy in policies:
        for budget in budgets:
            for n_needles in needle_counts:
                subset = [
                    r for r in results
                    if r["policy"] == policy and r["budget"] == budget and r["num_needles"] == n_needles
                ]
                if not subset:
                    continue
                avg_score = sum(r["score"] for r in subset) / len(subset)
                avg_found = sum(r["num_found"] for r in subset) / len(subset)
                print(
                    f"  {policy:<16} b={budget:.0%}  k={n_needles}  "
                    f"score={avg_score:.1f}%  avg_found={avg_found:.1f}/{n_needles}"
                )
    print(f"\nTotal time: {time.time()-t0:.0f}s")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RULER Multi-Needle NIAH evaluation")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--policies", nargs="+", default=["full", "semantic", "snapkv"])
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.2])
    parser.add_argument("--needle-counts", nargs="+", type=int, default=[1, 2, 4],
                        help="Number of needles to insert simultaneously")
    parser.add_argument("--target-tokens", type=int, default=1800)
    parser.add_argument("--num-trials", type=int, default=10,
                        help="Trials per (policy, budget, needle_count) cell")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/multi_needle/multi_needle_results.json")
    args = parser.parse_args()

    cfg = ExperimentConfig()
    if args.model:
        cfg.model.model_name = args.model
    model, tokenizer = load_model(cfg.model)

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
