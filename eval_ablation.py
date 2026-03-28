"""Ablation study for SemantiCache on RULER NIAH.

Tests the contribution of each semantic signal by disabling them one at a time.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from config import ExperimentConfig
from eval_ruler_niah import (
    build_pg_haystack,
    insert_needle_at_depth,
    make_ruler_needle,
    string_match_all,
)
from run_generation import generate_with_eviction, load_model

import random


# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------

@dataclass
class AblationConfig:
    """Defines which semantic signals to disable for an ablation variant."""
    name: str
    description: str
    # Set to 0.0 to disable, None to keep default
    alpha: Optional[float] = None        # attention weight
    beta: Optional[float] = None         # info density weight
    gamma: Optional[float] = None        # head entropy weight
    query_weight: Optional[float] = None # query relevance weight
    factual_weight: Optional[float] = None  # factual bonus weight
    pin_system: Optional[bool] = None    # role pinning for system tokens
    pin_latest_user: Optional[bool] = None  # role pinning for user tokens


ABLATION_VARIANTS = [
    AblationConfig(
        name="full_model",
        description="Full SemantiCache (all components)",
    ),
    AblationConfig(
        name="no_attention",
        description="Remove attention signal",
        alpha=0.0,
        gamma=0.0,  # head entropy also depends on attention
    ),
    AblationConfig(
        name="no_info_density",
        description="Remove information density signal",
        beta=0.0,
    ),
    AblationConfig(
        name="no_query_relevance",
        description="Remove query relevance signal",
        query_weight=0.0,
    ),
    AblationConfig(
        name="no_factual_bonus",
        description="Remove factual importance signal",
        factual_weight=0.0,
    ),
    AblationConfig(
        name="no_role_pinning",
        description="Remove role-based token protection",
        pin_system=False,
        pin_latest_user=False,
    ),
    AblationConfig(
        name="attention_only",
        description="Only attention (no semantic signals, no role pinning)",
        beta=0.0,
        query_weight=0.0,
        factual_weight=0.0,
        pin_system=False,
        pin_latest_user=False,
    ),
]


def build_ablation_config(ablation: AblationConfig, budget: float) -> ExperimentConfig:
    """Build an ExperimentConfig with ablation overrides applied."""
    cfg = ExperimentConfig()
    cfg.cache.policy = "semantic"
    cfg.cache.cache_budget = budget

    if ablation.alpha is not None:
        cfg.cache.alpha = ablation.alpha
    if ablation.beta is not None:
        cfg.cache.beta = ablation.beta
    if ablation.gamma is not None:
        cfg.cache.gamma = ablation.gamma
    if ablation.query_weight is not None:
        cfg.cache.query_weight = ablation.query_weight
    if ablation.factual_weight is not None:
        cfg.cache.factual_weight = ablation.factual_weight
    if ablation.pin_system is not None:
        cfg.cache.pin_system = ablation.pin_system
    if ablation.pin_latest_user is not None:
        cfg.cache.pin_latest_user = ablation.pin_latest_user

    return cfg


# ---------------------------------------------------------------------------
# Single ablation run
# ---------------------------------------------------------------------------

def run_ablation_eval(
    model,
    tokenizer,
    ablation: AblationConfig,
    needle,
    budget: float = 0.2,
    target_tokens: int = 1800,
    depth: float = 0.5,
    max_new_tokens: int = 64,
) -> dict:
    """Run a single RULER NIAH eval with ablation config."""
    haystack = build_pg_haystack(tokenizer, target_tokens)
    text_with_needle = insert_needle_at_depth(haystack, needle.fact, depth)

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer questions based only on the provided text."},
        {"role": "user", "content": f"{text_with_needle}\n\n{needle.question}"},
    ]

    config = build_ablation_config(ablation, budget)
    config.model.do_sample = False
    config.model.max_new_tokens = max_new_tokens
    config.model.stop_when_output_contains = []

    t0 = time.time()
    run_result = generate_with_eviction(model, tokenizer, messages, config)
    elapsed = time.time() - t0

    output_text = run_result["output_text"]
    score = string_match_all(output_text, needle.answer_keywords)
    correct = score >= 100.0

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = len(tokenizer.encode(prompt_text))

    stats = run_result.get("stats", {})
    return {
        "benchmark": "ablation_ruler_niah",
        "ablation": ablation.name,
        "ablation_desc": ablation.description,
        "budget": budget,
        "target_tokens": target_tokens,
        "actual_prompt_tokens": prompt_tokens,
        "depth": depth,
        "needle_key": needle.key,
        "needle_value": needle.value,
        "output_text": output_text,
        "score": score,
        "correct": correct,
        "elapsed_time": elapsed,
        "tokens_per_second": len(run_result.get("output_ids", [])) / elapsed if elapsed > 0 else 0,
        "retained_tokens": stats.get("current_cache_len", 0),
        "total_evicted": stats.get("total_evicted", 0),
    }


# ---------------------------------------------------------------------------
# Ablation suite runner
# ---------------------------------------------------------------------------

def run_ablation_suite(
    model,
    tokenizer,
    variants: list[AblationConfig],
    budget: float = 0.2,
    target_tokens: int = 1800,
    depths: list[float] | None = None,
    num_needles: int = 5,
    seed: int = 42,
    max_new_tokens: int = 64,
    output_path: str | None = None,
) -> list[dict]:
    """Run the full ablation study."""
    if depths is None:
        depths = list(np.round(np.linspace(0, 1, num=10, endpoint=True), 2))

    rng = random.Random(seed)
    needles = [make_ruler_needle("numbers", rng) for _ in range(num_needles)]

    total = len(variants) * len(depths) * num_needles
    print(f"Ablation study: {total} total runs")
    print(f"  variants: {[v.name for v in variants]}")
    print(f"  budget: {budget:.0%}")
    print(f"  target_tokens: {target_tokens}")
    print(f"  depths: {len(depths)} points")
    print(f"  needles: {num_needles}")
    print()

    results: list[dict] = []
    run_idx = 0
    t0 = time.time()

    for variant in variants:
        variant_results = []
        for depth in depths:
            for ni, needle in enumerate(needles):
                run_idx += 1
                elapsed = time.time() - t0
                eta = (elapsed / run_idx) * (total - run_idx) if run_idx > 1 else 0
                print(
                    f"[{run_idx}/{total}] {variant.name} "
                    f"d={depth:.2f} n={ni+1} "
                    f"(elapsed={elapsed:.0f}s eta={eta:.0f}s)"
                )

                result = run_ablation_eval(
                    model, tokenizer, variant, needle,
                    budget=budget,
                    target_tokens=target_tokens,
                    depth=depth,
                    max_new_tokens=max_new_tokens,
                )
                result["needle_index"] = ni + 1
                variant_results.append(result)
                results.append(result)
                tag = "O" if result["correct"] else "X"
                print(f"  [{tag}] val={needle.value} -> {result['output_text'][:60]}")

        correct = sum(1 for r in variant_results if r["correct"])
        print(f"\n=== {variant.name}: {correct}/{len(variant_results)} correct ===\n")

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved to: {out}")

    # Summary table
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)
    print(f"{'Variant':<22} {'Correct':>8} {'Total':>6} {'Acc':>8}")
    print("-" * 48)
    for variant in variants:
        subset = [r for r in results if r["ablation"] == variant.name]
        c = sum(1 for r in subset if r["correct"])
        t = len(subset)
        delta = ""
        if variant.name != "full_model":
            full_subset = [r for r in results if r["ablation"] == "full_model"]
            full_c = sum(1 for r in full_subset if r["correct"])
            full_t = len(full_subset)
            if full_t > 0 and t > 0:
                diff = c / t - full_c / full_t
                delta = f"  ({diff:+.0%})"
        print(f"{variant.name:<22} {c:>8} {t:>6} {c/t:>7.0%}{delta}")

    print(f"\nTotal time: {time.time()-t0:.0f}s")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SemantiCache ablation study on RULER NIAH")
    parser.add_argument("--budget", type=float, default=0.2, help="Cache budget for all variants")
    parser.add_argument("--target-tokens", type=int, default=1800, help="Haystack target tokens")
    parser.add_argument("--num-depths", type=int, default=10, help="Number of depth points")
    parser.add_argument("--num-needles", type=int, default=5, help="Number of needles")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variants", nargs="+", default=None,
        help="Subset of variants to run (default: all). "
             "Options: full_model, no_attention, no_info_density, no_query_relevance, "
             "no_factual_bonus, no_role_pinning, attention_only"
    )
    parser.add_argument("--output", default="results/ablation/ablation_results.json")
    args = parser.parse_args()

    depths = list(np.round(np.linspace(0, 1, num=args.num_depths, endpoint=True), 2))

    variants = ABLATION_VARIANTS
    if args.variants:
        name_set = set(args.variants)
        variants = [v for v in ABLATION_VARIANTS if v.name in name_set]
        if not variants:
            parser.error(f"No matching variants. Available: {[v.name for v in ABLATION_VARIANTS]}")

    cfg = ExperimentConfig()
    model, tokenizer = load_model(cfg.model)

    run_ablation_suite(
        model=model,
        tokenizer=tokenizer,
        variants=variants,
        budget=args.budget,
        target_tokens=args.target_tokens,
        depths=depths,
        num_needles=args.num_needles,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        output_path=args.output,
    )
