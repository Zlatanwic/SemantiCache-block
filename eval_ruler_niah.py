"""RULER-compatible Needle-in-a-Haystack evaluation using Paul Graham essays.

This module generates test cases following the RULER benchmark format
(NVIDIA, NAACL 2024) with Paul Graham essays as haystack text, randomized
key/value needles, and string_match_all scoring.

Reference: https://github.com/NVIDIA/RULER (arXiv 2404.06654)
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import ExperimentConfig
from run_generation import generate_with_eviction


# ---------------------------------------------------------------------------
# Haystack builder from Paul Graham essays
# ---------------------------------------------------------------------------

_ESSAYS_CACHE: list[str] | None = None


def _load_essays(path: str = "data/paul_graham_essays.json") -> list[str]:
    global _ESSAYS_CACHE
    if _ESSAYS_CACHE is None:
        _ESSAYS_CACHE = json.loads(Path(path).read_text(encoding="utf-8"))
    return _ESSAYS_CACHE


def build_pg_haystack(tokenizer, target_tokens: int) -> str:
    """Build a haystack from Paul Graham essays to approximately target_tokens."""
    essays = _load_essays()
    # Concatenate essays, cycling as needed
    haystack_parts: list[str] = []
    total_tokens = 0
    idx = 0
    while total_tokens < target_tokens:
        essay = essays[idx % len(essays)]
        essay_tokens = len(tokenizer.encode(essay, add_special_tokens=False))
        haystack_parts.append(essay)
        total_tokens += essay_tokens
        idx += 1
    return "\n\n".join(haystack_parts)


# ---------------------------------------------------------------------------
# RULER-style needle generation
# ---------------------------------------------------------------------------

RULER_KEYS = [
    "Alice", "Bob", "Charlie", "David", "Eve", "Frank",
    "Grace", "Henry", "Iris", "Jack", "Kate", "Leo",
]

RULER_NEEDLE_TEMPLATE = "One of the special magic {type_v} for {key} is: {value}."
RULER_QUERY_TEMPLATE = "What are all the special magic {type_v} for {key} mentioned in the provided text?"


@dataclass
class RulerNeedle:
    key: str
    value: str
    type_v: str
    fact: str
    question: str
    answer_keywords: list[str]


def make_ruler_needle(
    type_v: str = "numbers",
    rng: random.Random | None = None,
) -> RulerNeedle:
    """Generate a single RULER-style needle with random key/value."""
    rng = rng or random.Random()
    key = rng.choice(RULER_KEYS)
    if type_v == "numbers":
        value = str(rng.randint(10000, 99999))
    elif type_v == "words":
        words = ["cherry", "sunset", "harbor", "crystal", "thunder",
                 "meadow", "falcon", "silver", "lantern", "compass"]
        value = rng.choice(words)
    elif type_v == "uuids":
        import uuid
        value = str(uuid.UUID(int=rng.getrandbits(128)))[:8]
    else:
        value = str(rng.randint(10000, 99999))

    fact = RULER_NEEDLE_TEMPLATE.format(type_v=type_v, key=key, value=value)
    question = RULER_QUERY_TEMPLATE.format(type_v=type_v, key=key)
    return RulerNeedle(
        key=key,
        value=value,
        type_v=type_v,
        fact=fact,
        question=question,
        answer_keywords=[value],
    )


# ---------------------------------------------------------------------------
# Scoring (RULER string_match_all)
# ---------------------------------------------------------------------------

def string_match_all(pred: str, refs: list[str]) -> float:
    """RULER-compatible scoring: fraction of refs found as substrings in pred."""
    if not refs:
        return 100.0
    pred_lower = pred.lower()
    matched = sum(1.0 if r.lower() in pred_lower else 0.0 for r in refs)
    return matched / len(refs) * 100


# ---------------------------------------------------------------------------
# Single evaluation run
# ---------------------------------------------------------------------------

def insert_needle_at_depth(haystack: str, needle: str, depth: float) -> str:
    """Insert needle into haystack at the given depth (0.0 = start, 1.0 = end)."""
    sentences = haystack.split("\n")
    insert_idx = max(1, min(int(len(sentences) * depth), len(sentences) - 1))
    sentences.insert(insert_idx, needle)
    return "\n".join(sentences)


def run_ruler_eval(
    model,
    tokenizer,
    config: ExperimentConfig,
    needle: RulerNeedle,
    target_tokens: int = 4096,
    depth: float = 0.5,
    max_new_tokens: int = 64,
) -> dict:
    """Run a single RULER-style NIAH evaluation."""
    haystack = build_pg_haystack(tokenizer, target_tokens)
    text_with_needle = insert_needle_at_depth(haystack, needle.fact, depth)

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer questions based only on the provided text."},
        {"role": "user", "content": f"{text_with_needle}\n\n{needle.question}"},
    ]

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
        "benchmark": "ruler_niah",
        "policy": config.cache.policy,
        "budget": config.cache.cache_budget,
        "target_tokens": target_tokens,
        "actual_prompt_tokens": prompt_tokens,
        "depth": depth,
        "needle_key": needle.key,
        "needle_value": needle.value,
        "needle_type": needle.type_v,
        "output_text": output_text,
        "score": score,
        "correct": correct,
        "elapsed_time": elapsed,
        "tokens_per_second": len(run_result.get("output_ids", [])) / elapsed if elapsed > 0 else 0,
        "retained_tokens": stats.get("current_cache_len", 0),
        "initial_seq_len": stats.get("initial_seq_len", 0),
        "total_evicted": stats.get("total_evicted", 0),
    }


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_ruler_suite(
    model,
    tokenizer,
    policies: list[str],
    budgets: list[float],
    target_tokens_list: list[int],
    depths: list[float],
    num_needles: int = 5,
    seed: int = 42,
    max_new_tokens: int = 64,
    output_path: Optional[str] = None,
) -> list[dict]:
    """Run a full RULER NIAH evaluation grid."""
    rng = random.Random(seed)
    needles = [make_ruler_needle("numbers", rng) for _ in range(num_needles)]

    total = len(policies) * len(budgets) * len(target_tokens_list) * len(depths) * num_needles
    print(f"RULER NIAH suite: {total} total runs")
    print(f"  policies: {policies}")
    print(f"  budgets: {budgets}")
    print(f"  target_tokens: {target_tokens_list}")
    print(f"  depths: {len(depths)} points")
    print(f"  needles: {num_needles}")

    results: list[dict] = []
    run_idx = 0
    t0 = time.time()

    for policy in policies:
        for budget in budgets:
            for target_tok in target_tokens_list:
                for depth in depths:
                    for ni, needle in enumerate(needles):
                        run_idx += 1
                        elapsed = time.time() - t0
                        eta = (elapsed / run_idx) * (total - run_idx) if run_idx > 1 else 0
                        print(
                            f"[{run_idx}/{total}] {policy} b={budget:.0%} "
                            f"tok={target_tok} d={depth:.2f} n={ni+1} "
                            f"(elapsed={elapsed:.0f}s eta={eta:.0f}s)"
                        )

                        cfg = ExperimentConfig()
                        cfg.cache.policy = policy
                        cfg.cache.cache_budget = budget

                        result = run_ruler_eval(
                            model, tokenizer, cfg, needle,
                            target_tokens=target_tok,
                            depth=depth,
                            max_new_tokens=max_new_tokens,
                        )
                        result["needle_index"] = ni + 1
                        results.append(result)
                        tag = "O" if result["correct"] else "X"
                        print(f"  [{tag}] val={needle.value} -> {result['output_text'][:60]}")

        # Save per-policy checkpoint
        policy_results = [r for r in results if r["policy"] == policy]
        correct = sum(1 for r in policy_results if r["correct"])
        print(f"\n=== {policy} done: {correct}/{len(policy_results)} correct ===\n")

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Saved to: {out}")

    # Summary
    print("\n" + "=" * 70)
    print("RULER NIAH SUMMARY")
    print("=" * 70)
    for policy in policies:
        for budget in budgets:
            subset = [r for r in results if r["policy"] == policy and r["budget"] == budget]
            c = sum(1 for r in subset if r["correct"])
            t = len(subset)
            avg_tps = sum(r["tokens_per_second"] for r in subset) / t if t else 0
            print(f"  {policy:<16} budget={budget:.0%}  acc={c}/{t} ({c/t:.0%})  avg_tok/s={avg_tps:.1f}")
    print(f"\nTotal time: {time.time()-t0:.0f}s")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RULER-compatible NIAH evaluation")
    parser.add_argument("--policies", nargs="+", default=["full", "h2o", "snapkv", "kvzip", "streaming", "semantic"])
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.5, 0.3, 0.2])
    parser.add_argument("--target-tokens", nargs="+", type=int, default=[4096])
    parser.add_argument("--num-depths", type=int, default=10, help="Number of evenly-spaced depth points")
    parser.add_argument("--num-needles", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/ruler_niah/ruler_niah_results.json")
    args = parser.parse_args()

    import numpy as np
    depths = list(np.round(np.linspace(0, 1, num=args.num_depths, endpoint=True), 2))

    from run_generation import load_model
    cfg = ExperimentConfig()
    model, tokenizer = load_model(cfg.model)

    run_ruler_suite(
        model=model,
        tokenizer=tokenizer,
        policies=args.policies,
        budgets=args.budgets,
        target_tokens_list=args.target_tokens,
        depths=depths,
        num_needles=args.num_needles,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        output_path=args.output,
    )
