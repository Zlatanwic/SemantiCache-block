"""RULER Variable Tracking (VT) and Common Words Extraction (CWE) evaluation.

Complements RULER NIAH with two additional subtasks:
  - VT: Multi-hop variable binding chains (X=Y, Y=Z -> find value of X)
  - CWE: Extract words appearing >= threshold times in context

Reference: Hsieh et al., "RULER: What's the Real Context Size of Your
Long-Context Language Models?" (COLM 2024), arXiv 2404.06654

Usage:
    python eval_ruler_vt_cwe.py --task vt \
        --policies full semantic snapkv streaming h2o \
        --budgets 0.2 --num-trials 30 --target-tokens 1800 \
        --output results/ruler_vt/ruler_vt_results.json

    python eval_ruler_vt_cwe.py --task cwe \
        --policies full semantic snapkv streaming h2o \
        --budgets 0.2 --num-trials 30 --target-tokens 1800 \
        --output results/ruler_cwe/ruler_cwe_results.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from config import ExperimentConfig
from eval_ruler_niah import build_pg_haystack
from run_generation import generate_with_eviction


# ---------------------------------------------------------------------------
# Variable Tracking (VT) — multi-hop variable binding
# ---------------------------------------------------------------------------

VT_NAMES = [
    "Alice", "Bob", "Charlie", "David", "Eve", "Frank",
    "Grace", "Henry", "Iris", "Jack", "Kate", "Leo",
    "Mia", "Noah", "Olive", "Paul", "Quinn", "Rose",
]

VT_BINDING_TEMPLATE = "VAR {src} = {dst}"
VT_QUERY_TEMPLATE = "What is the final assigned value of VAR {var}? Only output the value, nothing else."


def make_vt_case(
    num_hops: int = 2,
    num_chains: int = 1,
    rng: random.Random | None = None,
) -> dict:
    """Generate a Variable Tracking test case.

    Creates binding chains like:
        VAR X1 = X2
        VAR X2 = 57392
    Query: What is the final assigned value of VAR X1?
    Answer: 57392
    """
    rng = rng or random.Random()
    all_facts = []
    queries = []
    answers = []

    for _ in range(num_chains):
        # Pick unique names for this chain
        chain_names = rng.sample(VT_NAMES, num_hops + 1)
        final_value = str(rng.randint(10000, 99999))

        # Build binding chain
        chain_facts = []
        for i in range(num_hops):
            if i < num_hops - 1:
                fact = VT_BINDING_TEMPLATE.format(src=chain_names[i], dst=chain_names[i + 1])
            else:
                fact = VT_BINDING_TEMPLATE.format(src=chain_names[i], dst=final_value)
            chain_facts.append(fact)

        all_facts.extend(chain_facts)
        queries.append(VT_QUERY_TEMPLATE.format(var=chain_names[0]))
        answers.append(final_value)

    return {
        "facts": all_facts,
        "queries": queries,
        "answers": answers,
    }


def insert_facts_scattered(haystack: str, facts: list[str], rng: random.Random) -> str:
    """Insert facts at random positions throughout the haystack."""
    sentences = haystack.split("\n")
    # Pick random insertion points, sorted to maintain relative order
    if len(sentences) < len(facts) + 2:
        # Very short haystack — just interleave
        result = []
        for i, fact in enumerate(facts):
            if i < len(sentences):
                result.append(sentences[i])
            result.append(fact)
        result.extend(sentences[len(facts):])
        return "\n".join(result)

    positions = sorted(rng.sample(range(1, len(sentences)), min(len(facts), len(sentences) - 1)))
    for i, pos in enumerate(positions):
        sentences.insert(pos + i, facts[i])  # +i to account for previous insertions
    return "\n".join(sentences)


# ---------------------------------------------------------------------------
# Common Words Extraction (CWE) — global frequency aggregation
# ---------------------------------------------------------------------------

# Words to inject at controlled frequencies
CWE_INJECT_WORDS = [
    "umbrella", "telescope", "carousel", "labyrinth", "dandelion",
    "saxophone", "cathedral", "avalanche", "chrysalis", "pendulum",
    "nightingale", "silhouette", "cinnamon", "trapezoid", "fibonacci",
]

CWE_INJECT_TEMPLATE = "The word {word} appeared here."
CWE_QUERY_TEMPLATE = (
    "List all special words that appeared {threshold} or more times in the text above. "
    "Output only the words separated by commas, nothing else."
)


def make_cwe_case(
    num_target_words: int = 3,
    num_distractor_words: int = 3,
    target_freq: int = 5,
    distractor_freq: int = 2,
    threshold: int = 4,
    rng: random.Random | None = None,
) -> dict:
    """Generate a Common Words Extraction test case.

    Injects target words (freq >= threshold) and distractor words (freq < threshold).
    Model must list only the target words.
    """
    rng = rng or random.Random()
    available = rng.sample(CWE_INJECT_WORDS, num_target_words + num_distractor_words)
    target_words = available[:num_target_words]
    distractor_words = available[num_target_words:]

    facts = []
    for w in target_words:
        for _ in range(target_freq):
            facts.append(CWE_INJECT_TEMPLATE.format(word=w))
    for w in distractor_words:
        for _ in range(distractor_freq):
            facts.append(CWE_INJECT_TEMPLATE.format(word=w))

    rng.shuffle(facts)

    query = CWE_QUERY_TEMPLATE.format(threshold=threshold)
    return {
        "facts": facts,
        "queries": [query],
        "answers": sorted(target_words),  # sorted for consistent matching
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_vt(pred: str, answers: list[str]) -> float:
    """Score VT: fraction of answer values found in prediction."""
    pred_lower = pred.lower().strip()
    matched = sum(1.0 for a in answers if a.lower() in pred_lower)
    return matched / len(answers) * 100 if answers else 100.0


def score_cwe(pred: str, answers: list[str]) -> float:
    """Score CWE: F1 between predicted words and target words."""
    pred_words = set(w.strip().lower() for w in pred.replace(",", " ").split() if w.strip())
    answer_set = set(a.lower() for a in answers)

    if not pred_words and not answer_set:
        return 100.0
    if not pred_words or not answer_set:
        return 0.0

    tp = len(pred_words & answer_set)
    prec = tp / len(pred_words) if pred_words else 0
    rec = tp / len(answer_set) if answer_set else 0
    if prec + rec == 0:
        return 0.0
    f1 = 2 * prec * rec / (prec + rec)
    return f1 * 100


# ---------------------------------------------------------------------------
# Single evaluation run
# ---------------------------------------------------------------------------

def run_single_eval(
    model,
    tokenizer,
    config: ExperimentConfig,
    task: str,
    case: dict,
    target_tokens: int = 1800,
    max_new_tokens: int = 64,
) -> dict:
    """Run a single VT or CWE evaluation."""
    rng = random.Random(42)  # deterministic haystack
    haystack = build_pg_haystack(tokenizer, target_tokens)
    text = insert_facts_scattered(haystack, case["facts"], rng)

    # Combine all queries into one prompt
    query_text = "\n".join(case["queries"])

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer based only on the provided text."},
        {"role": "user", "content": f"{text}\n\n{query_text}"},
    ]

    config.model.do_sample = False
    config.model.max_new_tokens = max_new_tokens

    t0 = time.time()
    result = generate_with_eviction(model, tokenizer, messages, config)
    elapsed = time.time() - t0

    output_text = result["output_text"]

    if task == "vt":
        score = score_vt(output_text, case["answers"])
    else:
        score = score_cwe(output_text, case["answers"])

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = len(tokenizer.encode(prompt_text))

    stats = result.get("stats", {})
    return {
        "benchmark": f"ruler_{task}",
        "task": task,
        "policy": config.cache.policy,
        "budget": config.cache.cache_budget,
        "target_tokens": target_tokens,
        "actual_prompt_tokens": prompt_tokens,
        "output_text": output_text[:200],
        "answers": case["answers"],
        "score": score,
        "correct": score >= 100.0,
        "elapsed_time": elapsed,
        "tokens_per_second": len(result.get("output_ids", [])) / elapsed if elapsed > 0 else 0,
        "retained_tokens": stats.get("current_cache_len", 0),
        "total_evicted": stats.get("total_evicted", 0),
    }


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_ruler_subtask_suite(
    model,
    tokenizer,
    task: str,
    policies: list[str],
    budgets: list[float],
    target_tokens: int = 1800,
    num_trials: int = 30,
    num_hops: int = 3,
    seed: int = 42,
    max_new_tokens: int = 64,
    output_path: Optional[str] = None,
    model_cfg=None,
) -> list[dict]:
    """Run VT or CWE evaluation grid."""
    rng = random.Random(seed)

    # Pre-generate all test cases
    cases = []
    for _ in range(num_trials):
        if task == "vt":
            cases.append(make_vt_case(num_hops=num_hops, rng=rng))
        else:
            cases.append(make_cwe_case(rng=rng))

    total = len(policies) * len(budgets) * num_trials
    task_name = "Variable Tracking" if task == "vt" else "Common Words Extraction"
    print(f"RULER {task_name} suite: {total} total runs")
    print(f"  policies: {policies}")
    print(f"  budgets: {budgets}")
    print(f"  trials: {num_trials}")
    print(f"  target_tokens: {target_tokens}")
    if task == "vt":
        print(f"  num_hops: {num_hops}")

    results: list[dict] = []
    run_idx = 0
    t0 = time.time()

    for policy in policies:
        for budget in budgets:
            for trial_idx, case in enumerate(cases):
                run_idx += 1
                elapsed = time.time() - t0
                eta = (elapsed / run_idx) * (total - run_idx) if run_idx > 1 else 0
                print(
                    f"[{run_idx}/{total}] {task} {policy} b={budget:.0%} "
                    f"trial={trial_idx+1}/{num_trials} "
                    f"(eta={eta:.0f}s)"
                )

                cfg = ExperimentConfig()
                if model_cfg is not None:
                    cfg.model = model_cfg
                cfg.cache.policy = policy
                cfg.cache.cache_budget = budget

                try:
                    result = run_single_eval(
                        model, tokenizer, cfg, task, case,
                        target_tokens=target_tokens,
                        max_new_tokens=max_new_tokens,
                    )
                except Exception as e:
                    import torch
                    if isinstance(e, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                        print(f"  OOM — skipped")
                        continue
                    print(f"  ERROR: {e}")
                    result = {
                        "benchmark": f"ruler_{task}",
                        "task": task,
                        "policy": policy,
                        "budget": budget,
                        "score": 0.0,
                        "correct": False,
                        "error": str(e),
                    }

                result["trial"] = trial_idx + 1
                results.append(result)
                tag = "O" if result.get("correct") else "X"
                print(f"  [{tag}] score={result['score']:.1f} -> {result.get('output_text', '')[:60]}")

    # Save results
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved {len(results)} results to {out}")

    # Summary
    print("\n" + "=" * 70)
    print(f"RULER {task_name.upper()} SUMMARY")
    print("=" * 70)
    for policy in policies:
        for budget in budgets:
            subset = [r for r in results if r["policy"] == policy and r["budget"] == budget]
            if not subset:
                continue
            scores = [r["score"] for r in subset]
            n = len(scores)
            mean = sum(scores) / n
            correct = sum(1 for r in subset if r.get("correct"))
            print(f"  {policy:<16} b={budget:.0%}  mean_score={mean:.1f}%  exact={correct}/{n}  n={n}")
    print(f"\nTotal time: {time.time()-t0:.0f}s")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RULER VT/CWE evaluation for SemantiCache")
    parser.add_argument("--task", choices=["vt", "cwe"], required=True,
                        help="Task: vt (Variable Tracking) or cwe (Common Words Extraction)")
    parser.add_argument("--policies", nargs="+",
                        default=["full", "semantic", "snapkv", "streaming", "h2o"])
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.2])
    parser.add_argument("--target-tokens", type=int, default=1800)
    parser.add_argument("--num-trials", type=int, default=30)
    parser.add_argument("--num-hops", type=int, default=2,
                        help="Number of hops in VT chain (default 2)")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None,
                        help="Output path (default: results/ruler_{task}/ruler_{task}_results.json)")
    parser.add_argument("--model", type=str, default=None)

    args = parser.parse_args()

    if args.output is None:
        args.output = f"results/ruler_{args.task}/ruler_{args.task}_results.json"

    from config import ModelConfig
    from run_generation import load_model

    model_cfg = ModelConfig()
    if args.model:
        model_cfg.model_name = args.model

    print(f"Loading model: {model_cfg.model_name}")
    model, tokenizer = load_model(model_cfg)
    print(f"Model loaded. vocab_size={tokenizer.vocab_size}")

    run_ruler_subtask_suite(
        model=model,
        tokenizer=tokenizer,
        task=args.task,
        policies=args.policies,
        budgets=args.budgets,
        target_tokens=args.target_tokens,
        num_trials=args.num_trials,
        num_hops=args.num_hops,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        output_path=args.output,
        model_cfg=model_cfg,
    )
