"""BABILong benchmark for SemantiCache evaluation.

Tests whether eviction policies can retain scattered reasoning facts
embedded in long PG19 book text.

Reference: Kuratov et al., "BABILong: Testing the Limits of LLMs in
Long Context Reasoning", NeurIPS 2024 Datasets Track.

Dataset: RMT-team/babilong on HuggingFace.
Tasks:
  - qa1: Single supporting fact retrieval
  - qa2: Two supporting facts (conjunction)
  - qa3: Three supporting facts (chaining)
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional

from config import ExperimentConfig, ModelConfig
from run_generation import generate_with_eviction, load_model


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def score_babilong(pred: str, target: str) -> float:
    """Score a BABILong prediction.

    Returns 1.0 if the target answer appears as a substring in the
    model output (case-insensitive), else 0.0.
    This follows the BABILong evaluation protocol.
    """
    pred_norm = normalize_answer(pred)
    target_norm = normalize_answer(target)
    if not target_norm:
        return 0.0
    # Exact substring match
    if target_norm in pred_norm:
        return 1.0
    # Also check individual words for single-word answers
    target_words = target_norm.split()
    if len(target_words) == 1:
        pred_words = pred_norm.split()
        if target_words[0] in pred_words:
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_babilong_tasks(
    task_names: list[str],
    context_lengths: list[str],
    max_examples: int = 50,
    seed: int = 42,
) -> list[dict]:
    """Load BABILong examples from HuggingFace.

    Dataset API: config = task name (e.g. "qa1"), split = context length
    (e.g. "2k").  Each example has columns: input, target, question.

    Args:
        task_names: e.g. ["qa1", "qa2", "qa3"]
        context_lengths: e.g. ["1k", "2k", "4k"]
        max_examples: max examples per (task, length) pair
        seed: for reproducible sampling

    Returns:
        List of dicts with keys: task, context_length, input, target, idx
    """
    from datasets import load_dataset
    import random

    rng = random.Random(seed)
    tasks = []

    for task_name in task_names:
        for ctx_len in context_lengths:
            print(f"Loading BABILong: config={task_name}, split={ctx_len} ...")
            try:
                ds = load_dataset(
                    "RMT-team/babilong",
                    task_name,
                    split=ctx_len,
                    trust_remote_code=True,
                )
            except Exception as e:
                print(f"  WARNING: could not load {task_name}/{ctx_len}: {e}")
                # Try reversed: config=context_length, split=task
                try:
                    ds = load_dataset(
                        "RMT-team/babilong",
                        ctx_len,
                        split=task_name,
                        trust_remote_code=True,
                    )
                except Exception as e2:
                    print(f"  SKIPPING {task_name}/{ctx_len}: {e2}")
                    continue

            # Sample up to max_examples
            indices = list(range(len(ds)))
            if len(indices) > max_examples:
                rng.shuffle(indices)
                indices = sorted(indices[:max_examples])

            for idx in indices:
                example = ds[idx]
                # BABILong has separate 'input' (context) and 'question' fields
                context = example["input"]
                question = example.get("question", "")
                if question:
                    full_input = f"{context}\n\n{question}"
                else:
                    full_input = context
                tasks.append({
                    "task": task_name,
                    "context_length": ctx_len,
                    "input": full_input,
                    "target": example["target"],
                    "idx": idx,
                })

            print(f"  Loaded {len(indices)} examples for {task_name}/{ctx_len}")

    print(f"Total BABILong examples: {len(tasks)}")
    return tasks


# ---------------------------------------------------------------------------
# Single evaluation
# ---------------------------------------------------------------------------

def run_babilong_eval(
    model,
    tokenizer,
    config: ExperimentConfig,
    task_input: str,
    target: str,
    max_new_tokens: int = 64,
) -> dict:
    """Run a single BABILong evaluation."""

    # BABILong input already contains context + question.
    # We wrap it in a chat template.
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Read the context carefully "
                "and answer the question with just the answer, "
                "no explanation needed."
            ),
        },
        {"role": "user", "content": task_input},
    ]

    config.model.max_new_tokens = max_new_tokens
    config.model.do_sample = False
    config.model.temperature = 0.0

    t0 = time.time()
    try:
        result = generate_with_eviction(model, tokenizer, messages, config)
    except torch.cuda.OutOfMemoryError:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "output_text": "[OOM]",
            "score": 0.0,
            "elapsed_time": time.time() - t0,
            "tokens_per_second": 0.0,
            "error": "CUDA OOM",
        }
    elapsed = time.time() - t0

    output_text = result["output_text"]
    score = score_babilong(output_text, target)

    # Prompt token count
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = len(tokenizer.encode(prompt_text))

    stats = result.get("stats", {})
    return {
        "output_text": output_text,
        "target": target,
        "score": score,
        "actual_prompt_tokens": prompt_tokens,
        "elapsed_time": elapsed,
        "tokens_per_second": result.get("tokens_per_second", 0),
        "retained_tokens": stats.get("current_cache_len", 0),
        "initial_seq_len": stats.get("initial_seq_len", 0),
        "total_evicted": stats.get("total_evicted", 0),
    }


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------

def run_babilong_suite(
    model,
    tokenizer,
    policies: list[str],
    budgets: list[float],
    tasks: list[dict],
    max_new_tokens: int = 64,
    output_path: Optional[str] = None,
    model_cfg=None,
) -> list[dict]:
    """Run BABILong evaluation grid."""

    results: list[dict] = []
    total = len(policies) * len(budgets) * len(tasks)
    print(f"\nBABILong suite: {len(policies)} policies x "
          f"{len(budgets)} budgets x {len(tasks)} examples = {total} runs")

    run_idx = 0
    t0 = time.time()

    for policy in policies:
        for budget in budgets:
            for task in tasks:
                run_idx += 1
                elapsed_total = time.time() - t0
                eta = (elapsed_total / run_idx) * (total - run_idx) if run_idx > 0 else 0

                cfg = ExperimentConfig()
                if model_cfg is not None:
                    cfg.model = model_cfg
                cfg.cache.policy = policy
                cfg.cache.cache_budget = budget

                try:
                    result = run_babilong_eval(
                        model, tokenizer, cfg,
                        task_input=task["input"],
                        target=task["target"],
                        max_new_tokens=max_new_tokens,
                    )
                except Exception as e:
                    result = {
                        "output_text": f"[ERROR: {e}]",
                        "score": 0.0,
                        "elapsed_time": 0,
                        "tokens_per_second": 0,
                        "error": str(e),
                    }

                # Add metadata
                result["benchmark"] = "babilong"
                result["policy"] = policy
                result["budget"] = budget
                result["task"] = task["task"]
                result["context_length"] = task["context_length"]
                result["example_idx"] = task["idx"]

                results.append(result)

                status = "OK" if result["score"] > 0 else "MISS"
                print(
                    f"  [{run_idx}/{total}] {policy:<12} b={budget:.0%}  "
                    f"{task['task']}_{task['context_length']}  "
                    f"{status}  (eta {eta:.0f}s)"
                )

    # Save
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(results, indent=2))
        print(f"\nSaved to: {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("BABILONG RESULTS SUMMARY")
    print("=" * 70)
    for task_name in sorted(set(t["task"] for t in tasks)):
        for ctx_len in sorted(set(t["context_length"] for t in tasks)):
            print(f"\n--- {task_name}_{ctx_len} ---")
            print(f"{'Policy':<16} {'Budget':<8} {'Acc':>6}  {'n':>4}")
            print("-" * 40)
            for policy in policies:
                for budget in budgets:
                    subset = [
                        r for r in results
                        if r["policy"] == policy
                        and r["budget"] == budget
                        and r["task"] == task_name
                        and r["context_length"] == ctx_len
                    ]
                    if not subset:
                        continue
                    acc = sum(r["score"] for r in subset) / len(subset) * 100
                    print(f"  {policy:<14} {budget:>5.0%}   {acc:>5.1f}%  {len(subset):>4}")

    # Overall
    print(f"\n{'=' * 70}")
    print("OVERALL AVERAGE")
    print(f"{'=' * 70}")
    print(f"{'Policy':<16} ", end="")
    for budget in budgets:
        print(f"  {budget:.0%}", end="    ")
    print()
    print("-" * (16 + len(budgets) * 10))
    for policy in policies:
        print(f"  {policy:<14}", end="")
        for budget in budgets:
            subset = [
                r for r in results
                if r["policy"] == policy and r["budget"] == budget
            ]
            if subset:
                acc = sum(r["score"] for r in subset) / len(subset) * 100
                print(f"  {acc:>5.1f}%", end="  ")
            else:
                print(f"  {'N/A':>6}", end="  ")
        print()

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

try:
    import torch
except ImportError:
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BABILong benchmark for KV cache eviction policies"
    )
    parser.add_argument(
        "--policies", nargs="+",
        default=["full", "streaming", "h2o", "snapkv", "semantic"],
        help="Eviction policies to evaluate",
    )
    parser.add_argument(
        "--budgets", nargs="+", type=float,
        default=[0.2],
        help="Cache budget fractions",
    )
    parser.add_argument(
        "--tasks", nargs="+",
        default=["qa1", "qa2", "qa3"],
        help="BABILong task names (qa1-qa5)",
    )
    parser.add_argument(
        "--context-lengths", nargs="+",
        default=["1k", "2k", "4k"],
        help="Context length splits to evaluate",
    )
    parser.add_argument(
        "--max-examples", type=int, default=50,
        help="Max examples per (task, context_length) pair",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=64,
        help="Max tokens to generate per example",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Model name (default: from config)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--output", type=str,
        default="results/babilong/babilong_results.json",
    )
    args = parser.parse_args()

    # Load model
    model_cfg = ModelConfig()
    if args.model:
        model_cfg.model_name = args.model
    model, tokenizer = load_model(model_cfg)

    # Load dataset
    tasks = load_babilong_tasks(
        task_names=args.tasks,
        context_lengths=args.context_lengths,
        max_examples=args.max_examples,
        seed=args.seed,
    )

    # Run
    run_babilong_suite(
        model=model,
        tokenizer=tokenizer,
        policies=args.policies,
        budgets=args.budgets,
        tasks=tasks,
        max_new_tokens=args.max_new_tokens,
        output_path=args.output,
        model_cfg=model_cfg,
    )
