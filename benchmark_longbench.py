"""LongBench evaluation for SemantiCache.

Runs a representative subset of LongBench tasks using our existing
generation harness with eviction policies.

Task subset (representative, diverse, feasible):
  - narrativeqa: single-doc QA (English, ~9k chars)
  - hotpotqa: multi-doc QA requiring multi-hop reasoning
  - gov_report: summarization (English, ~10k chars)
  - passage_count: synthetic count-the-passages task

Each task evaluates at a fixed cache budget to stress-test retention.

Usage on cloud server:
    python benchmark_longbench.py \
        --policies full semantic snapkv streaming h2o \
        --budgets 0.2 0.3 0.5 \
        --tasks narrativeqa hotpotqa gov_report passage_count \
        --model Qwen/Qwen2.5-3B-Instruct \
        --max-examples 50 \
        --output results/longbench/longbench_results.json

    # Then compute summary:
    python -c "
    import json
    import numpy as np
    from pathlib import Path
    data = json.load(open('results/longbench/longbench_results.json'))
    # Group by task, policy, budget
    # Compute mean F1 / Rouge-L / Accuracy per group
    # Print table
    "
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Optional

from collections import Counter

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

# LongBench uses 'THUDM/LongBench' dataset
# Each example: {context, input, answers, length, dataset, language, ...}
# Metrics: F1 for QA, Rouge-L for summarization, Accuracy for classification/synthetic


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def normalize_answer(pred: str) -> str:
    """Lowercase and strip whitespace for comparison."""
    return pred.lower().strip()


def f1_score(pred: str, answers: list[str]) -> float:
    """Compute max F1 between prediction and any answer (LongBench style).

    Uses Counter-based multiset intersection to handle token frequency correctly.
    """
    pred_tokens = normalize_answer(pred).split()
    if not pred_tokens:
        return 0.0
    pred_counter = Counter(pred_tokens)
    best_f1 = 0.0
    for ans in answers:
        ans_tokens = normalize_answer(ans).split()
        if not ans_tokens:
            continue
        ans_counter = Counter(ans_tokens)
        common = sum((pred_counter & ans_counter).values())
        if common == 0:
            continue
        prec = common / len(pred_tokens)
        rec = common / len(ans_tokens)
        f1 = 2 * prec * rec / (prec + rec)
        best_f1 = max(best_f1, f1)
    return best_f1


def rouge_l_score(pred: str, answers: list[str], beta: float = 1.2) -> float:
    """Compute word-level Rouge-L F-measure (LongBench style).

    Uses word-level LCS and computes F-measure with beta weighting,
    matching the official LongBench evaluation.
    """
    pred_words = normalize_answer(pred).split()
    if not pred_words:
        return 0.0
    m = len(pred_words)
    best_f = 0.0
    for ans in answers:
        ans_words = normalize_answer(ans).split()
        n = len(ans_words)
        if n == 0:
            continue
        # Word-level LCS via DP (O(m*n) space-optimized)
        dp = [0] * (n + 1)
        for i in range(1, m + 1):
            prev = 0
            for j in range(1, n + 1):
                temp = dp[j]
                if pred_words[i - 1] == ans_words[j - 1]:
                    dp[j] = prev + 1
                else:
                    dp[j] = max(dp[j], dp[j - 1])
                prev = temp
        lcs_len = dp[n]
        if lcs_len == 0:
            continue
        prec = lcs_len / m
        rec = lcs_len / n
        f = ((1 + beta ** 2) * prec * rec) / (rec + beta ** 2 * prec)
        best_f = max(best_f, f)
    return best_f


def accuracy_score(pred: str, answers: list[str]) -> float:
    """Exact match accuracy - prediction must contain answer substring."""
    pred_lower = normalize_answer(pred)
    for ans in answers:
        if normalize_answer(ans) in pred_lower:
            return 1.0
    return 0.0


def count_score(pred: str, answers: list[str]) -> float:
    """For passage_count: extract integer from prediction."""
    import re
    pred_lower = pred.lower()
    # Find all numbers in prediction
    numbers = re.findall(r'\d+', pred_lower)
    if not numbers:
        return 0.0
    # For passage_count, the answer is an integer
    # We use exact match with the first integer found
    try:
        pred_int = int(numbers[0])
        true_int = int(answers[0])
        return 1.0 if pred_int == true_int else 0.0
    except (ValueError, IndexError):
        return 0.0


def compute_score(task: str, pred: str, answers: list[str]) -> float:
    """Compute task-appropriate metric."""
    if task in ("narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh",
                "hotpotqa", "2wikimqa", "musique", "dureader",
                "triviaqa", "lsht", "trec"):
        return f1_score(pred, answers)
    elif task in ("gov_report", "qmsum", "multi_news", "vcsum", "samsum"):
        return rouge_l_score(pred, answers)
    elif task in ("passage_count",):
        return count_score(pred, answers)
    elif task in ("passage_retrieval_en", "passage_retrieval_zh"):
        return accuracy_score(pred, answers)
    else:
        # Default: F1
        return f1_score(pred, answers)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def make_prompt_longbench(task: str, context: str, question: str, model_name: str) -> list[dict]:
    """Format prompt for LongBench task.

    For Qwen models use ChatML format.
    For Llama models use standard chat format.
    """
    if "qwen" in model_name.lower():
        # ChatML format
        if task in ("gov_report", "qmsum", "multi_news", "vcsum", "samsum"):
            # Summarization tasks
            user_content = f"{context}\n\nPlease provide a summary of the above content."
        elif task in ("passage_count",):
            # Count task
            user_content = f"{context}\n\nHow many passages are mentioned in the text above? Only output the number."
        else:
            # QA tasks
            user_content = f"{context}\n\n{question}"
    else:
        # Generic format for other models
        if task in ("gov_report", "qmsum", "multi_news", "vcsum", "samsum"):
            user_content = f"[Context]\n{context}\n\n[Task]\nSummarize the above content."
        elif task in ("passage_count",):
            user_content = f"[Context]\n{context}\n\n[Task]\nHow many passages are mentioned? Only output the number."
        else:
            user_content = f"[Context]\n{context}\n\n[Question]\n{question}"

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer based only on the provided context."},
        {"role": "user", "content": user_content},
    ]
    return messages


# ---------------------------------------------------------------------------
# Single evaluation run
# ---------------------------------------------------------------------------

def run_single_longbench_eval(
    model,
    tokenizer,
    config,
    task: str,
    example: dict,
    model_name: str,
    max_new_tokens: int = 256,
) -> dict:
    """Run a single LongBench evaluation."""
    from run_generation import generate_with_eviction

    context = example.get("context", "")
    question = example.get("input", "")
    answers = example.get("answers", [])
    if isinstance(answers, str):
        answers = [answers]

    messages = make_prompt_longbench(task, context, question, model_name)

    t0 = time.time()
    result = generate_with_eviction(model, tokenizer, messages, config)
    elapsed = time.time() - t0

    output_text = result.get("output_text", "")
    score = compute_score(task, output_text, answers)

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = len(tokenizer.encode(prompt_text))

    stats = result.get("stats", {})
    return {
        "task": task,
        "policy": config.cache.policy,
        "budget": config.cache.cache_budget,
        "context_len_chars": len(context),
        "context_len_tokens": prompt_tokens,
        "question": question[:100],
        "output_text": output_text[:200],
        "answers": answers,
        "score": score,
        "elapsed_time": elapsed,
        "tokens_per_second": result.get("tokens_per_second", 0),
        "retained_tokens": stats.get("current_cache_len", 0),
        "total_evicted": stats.get("total_evicted", 0),
    }


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_longbench_suite(
    model,
    tokenizer,
    policies: list[str],
    budgets: list[float],
    tasks: list[str],
    max_examples: int = 50,
    max_new_tokens: int = 256,
    max_context_tokens: int = 8192,
    seed: int = 42,
    output_path: Optional[str] = None,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    model_cfg=None,
):
    """Run LongBench evaluation grid.

    Args:
        model: loaded model
        tokenizer: loaded tokenizer
        policies: list of eviction policies to test
        budgets: list of cache budgets (fraction of prompt length)
        tasks: list of LongBench task names
        max_examples: max examples per task (LongBench test set is large)
        max_new_tokens: max tokens for generation
        seed: random seed
        output_path: where to save results JSON
        model_name: model name for prompt formatting
        model_cfg: ExperimentConfig model settings
    """
    rng = random.Random(seed)
    results: list[dict] = []

    total = len(policies) * len(budgets) * len(tasks) * max_examples
    print(f"LongBench evaluation: {total} total runs")
    print(f"  policies: {policies}")
    print(f"  budgets: {budgets}")
    print(f"  tasks: {tasks}")
    print(f"  max_examples per task: {max_examples}")
    print(f"  max_new_tokens: {max_new_tokens}")

    run_idx = 0
    t0 = time.time()

    for task in tasks:
        # Load LongBench data
        print(f"\nLoading task: {task}")
        try:
            dataset = load_dataset("THUDM/LongBench", task, split="test", trust_remote_code=True)
        except Exception as e:
            print(f"  ERROR loading {task}: {e}")
            continue

        # Sample examples, filtering out those exceeding max_context_tokens
        all_examples = list(dataset)
        rng.shuffle(all_examples)
        sampled = []
        skipped = 0
        for ex in all_examples:
            if len(sampled) >= max_examples:
                break
            ctx = ex.get("context", "")
            n_tok = len(tokenizer.encode(ctx, add_special_tokens=False))
            if n_tok > max_context_tokens:
                skipped += 1
                continue
            sampled.append(ex)
        print(f"  Loaded {len(all_examples)}, using {len(sampled)}, skipped {skipped} (>{max_context_tokens} tokens)")

        for policy in policies:
            for budget in budgets:
                for ex_idx, example in enumerate(sampled):
                    run_idx += 1
                    elapsed_total = time.time() - t0
                    eta = (elapsed_total / run_idx) * (total - run_idx) if run_idx > 1 else 0

                    cfg = ExperimentConfig()
                    if model_cfg is not None:
                        cfg.model = model_cfg
                    cfg.cache.policy = policy
                    cfg.cache.cache_budget = budget
                    cfg.model.do_sample = False
                    cfg.model.max_new_tokens = max_new_tokens

                    print(
                        f"[{run_idx}/{total}] {task} {policy} b={budget:.0%} "
                        f"ex={ex_idx+1}/{len(sampled)} "
                        f"(eta={eta:.0f}s)"
                    )

                    try:
                        result = run_single_longbench_eval(
                            model, tokenizer, cfg, task, example,
                            model_name, max_new_tokens
                        )
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        print(f"  OOM — skipped, freed GPU cache")
                        continue
                    except Exception as e:
                        print(f"  ERROR: {e}")
                        result = {
                            "task": task,
                            "policy": policy,
                            "budget": budget,
                            "context_len_chars": len(example.get("context", "")),
                            "score": 0.0,
                            "error": str(e),
                        }

                    results.append(result)
                    print(f"  score={result.get('score', 0):.3f} "
                          f"retained={result.get('retained_tokens', 0)} "
                          f"evicted={result.get('total_evicted', 0)}")

    # Save results
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved {len(results)} results to {out}")

    # Print summary
    print("\n" + "=" * 70)
    print("LONGBENCH SUMMARY")
    print("=" * 70)
    print(f"\n{'Task':<20} {'Policy':<16} {'Budget':>8} {'Score':>8} {'n':>5}")
    print("-" * 65)

    for task in tasks:
        for policy in policies:
            for budget in budgets:
                subset = [
                    r for r in results
                    if r.get("task") == task
                    and r.get("policy") == policy
                    and abs(r.get("budget", 0) - budget) < 0.001
                ]
                if not subset:
                    continue
                scores = [r.get("score", 0) for r in subset]
                mean = float(np.mean(scores))
                n = len(scores)
                print(f"{task:<20} {policy:<16} {budget:>7.0%} {mean:>8.3f} {n:>5d}")

    print()
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LongBench evaluation for SemantiCache")
    parser.add_argument("--policies", nargs="+", default=["full", "semantic", "snapkv", "streaming", "h2o"])
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.2, 0.3, 0.5])
    parser.add_argument("--tasks", nargs="+",
                        default=["narrativeqa", "hotpotqa", "gov_report", "passage_count"],
                        help="LongBench task names")
    parser.add_argument("--max-examples", type=int, default=50,
                        help="Max examples per task (default 50)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-context-tokens", type=int, default=8192,
                        help="Skip examples with context longer than this (default 8192)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/longbench/longbench_results.json")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")

    args = parser.parse_args()

    from config import ExperimentConfig, ModelConfig
    from run_generation import load_model

    model_cfg = ModelConfig()
    model_cfg.model_name = args.model

    print(f"Loading model: {model_cfg.model_name}")
    model, tokenizer = load_model(model_cfg)
    print(f"Model loaded. vocab_size={tokenizer.vocab_size}")

    run_longbench_suite(
        model=model,
        tokenizer=tokenizer,
        policies=args.policies,
        budgets=args.budgets,
        tasks=args.tasks,
        max_examples=args.max_examples,
        max_new_tokens=args.max_new_tokens,
        max_context_tokens=args.max_context_tokens,
        seed=args.seed,
        output_path=args.output,
        model_name=args.model,
        model_cfg=model_cfg,
    )
