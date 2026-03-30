"""Dump retained/evicted tokens for a single NIAH run.

Outputs a JSON with per-token retention info for case study analysis.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch

from config import ExperimentConfig
from eval_ruler_niah import build_pg_haystack, RULER_KEYS
from run_generation import load_model, generate_with_eviction
from semantic_analyzer import SemanticAnalyzer


def dump_retention(
    model, tokenizer, model_cfg,
    policy: str = "semantic",
    budget: float = 0.2,
    target_tokens: int = 1800,
    depth: float = 0.5,
    seed: int = 42,
    output_path: str = "results/case_study/token_retention.json",
):
    rng = random.Random(seed)
    key = rng.choice(RULER_KEYS)
    value = str(rng.randint(10000, 99999))
    needle = f"One of the special magic numbers for {key} is: {value}."

    haystack = build_pg_haystack(tokenizer, target_tokens)
    sentences = haystack.split("\n")
    idx = max(1, min(int(len(sentences) * depth), len(sentences) - 1))
    sentences.insert(idx, needle)
    text = "\n".join(sentences)
    question = f"What is one of the special magic numbers for {key} mentioned in the provided text?"

    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer based only on the provided text."},
        {"role": "user", "content": f"{text}\n\n{question}"},
    ]

    cfg = ExperimentConfig()
    cfg.model = model_cfg
    cfg.cache.policy = policy
    cfg.cache.cache_budget = budget
    cfg.model.do_sample = False
    cfg.model.max_new_tokens = 32
    cfg.model.stop_when_output_contains = [value]

    # Get full token list before generation
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(prompt_text)
    prompt_len = len(input_ids)

    # Decode each token for annotation
    tokens = []
    for i, tid in enumerate(input_ids):
        token_str = tokenizer.decode([tid], skip_special_tokens=False)
        tokens.append({"pos": i, "token_id": tid, "text": token_str})

    # Find needle token positions by decoding windows and checking for substring
    def find_substring_positions(token_ids, substring, tokenizer, window=30):
        """Find token positions that decode to contain the substring."""
        positions = []
        for start in range(len(token_ids) - window + 1):
            window_text = tokenizer.decode(token_ids[start:start + window], skip_special_tokens=False)
            if substring in window_text:
                # Narrow down to the exact tokens
                for i in range(start, min(start + window, len(token_ids))):
                    tok_text = tokenizer.decode([token_ids[i]], skip_special_tokens=False)
                    # Check if this token is part of the needle by checking prefix/suffix
                    positions.append(i)
                # Deduplicate and trim: find minimal span
                break
        if not positions:
            return positions
        # Refine: find the tightest span that contains the full substring
        for trim_start in range(positions[0], positions[-1] + 1):
            for trim_end in range(positions[-1], trim_start, -1):
                span_text = tokenizer.decode(token_ids[trim_start:trim_end + 1], skip_special_tokens=False)
                if substring in span_text:
                    return list(range(trim_start, trim_end + 1))
        return positions

    needle_positions = find_substring_positions(input_ids, needle, tokenizer)
    query_positions = find_substring_positions(input_ids, question, tokenizer)

    print(f"Prompt: {prompt_len} tokens")
    if needle_positions:
        print(f"Needle: '{needle}' at positions {needle_positions[0]}--{needle_positions[-1]} ({len(needle_positions)} tokens)")
    else:
        print(f"WARNING: needle not found in token stream")
    if query_positions:
        print(f"Query: positions {query_positions[0]}--{query_positions[-1]} ({len(query_positions)} tokens)")
    else:
        print(f"WARNING: query not found in token stream")
    print(f"Policy: {policy}, Budget: {budget:.0%}")
    print(f"Running generation...")

    run_result = generate_with_eviction(model, tokenizer, messages, cfg)

    stats = run_result.get("stats", {})
    retained_count = stats.get("current_cache_len", prompt_len)
    total_evicted = stats.get("total_evicted", 0)
    output_text = run_result["output_text"]
    score = 100 if value.lower() in output_text.lower() else 0

    print(f"\nOutput: {output_text}")
    print(f"Score: {score}%")
    print(f"Retained: {retained_count}/{prompt_len} tokens")

    # Compute semantic signals using the analyzer's actual API
    analyzer = SemanticAnalyzer(tokenizer)
    input_tensor = torch.tensor(input_ids)

    info_density = analyzer.compute_info_density(input_tensor)
    query_relevance = analyzer.compute_query_relevance(input_tensor, question)
    factual_bonus = analyzer.compute_factual_bonus(input_tensor)

    # Build per-token annotation
    annotated = []
    needle_set = set(needle_positions)
    query_set = set(query_positions)
    for i, tid in enumerate(input_ids):
        token_str = tokenizer.decode([tid], skip_special_tokens=False)
        entry = {
            "pos": i,
            "token_id": tid,
            "text": token_str,
            "is_needle": i in needle_set,
            "is_query": i in query_set,
            "density": round(float(info_density[i]), 4),
            "query_relevance": round(float(query_relevance[i]), 4),
            "factual": round(float(factual_bonus[i]), 4),
        }
        annotated.append(entry)

    # Compute summary stats
    needle_density = [a["density"] for a in annotated if a["is_needle"]]
    haystack_density = [a["density"] for a in annotated if not a["is_needle"] and not a["is_query"]]
    needle_query_rel = [a["query_relevance"] for a in annotated if a["is_needle"]]
    haystack_query_rel = [a["query_relevance"] for a in annotated if not a["is_needle"] and not a["is_query"]]
    needle_factual = [a["factual"] for a in annotated if a["is_needle"]]
    haystack_factual = [a["factual"] for a in annotated if not a["is_needle"] and not a["is_query"]]

    import numpy as np
    if not needle_density:
        print("WARNING: no needle tokens annotated, signal comparison will be empty")
        needle_density = [0.0]
        needle_query_rel = [0.0]
        needle_factual = [0.0]
    summary = {
        "prompt_tokens": prompt_len,
        "needle": needle,
        "needle_key": key,
        "needle_value": value,
        "needle_positions": needle_positions,
        "query_positions": query_positions,
        "depth": depth,
        "policy": policy,
        "budget": budget,
        "retained_tokens": retained_count,
        "total_evicted": total_evicted,
        "output_text": output_text,
        "score": score,
        "signal_comparison": {
            "density": {
                "needle_mean": round(float(np.mean(needle_density)), 4),
                "haystack_mean": round(float(np.mean(haystack_density)), 4),
                "needle_max": round(float(np.max(needle_density)), 4),
                "haystack_p95": round(float(np.percentile(haystack_density, 95)), 4),
            },
            "query_relevance": {
                "needle_mean": round(float(np.mean(needle_query_rel)), 4),
                "haystack_mean": round(float(np.mean(haystack_query_rel)), 4),
                "needle_max": round(float(np.max(needle_query_rel)), 4),
                "haystack_p95": round(float(np.percentile(haystack_query_rel, 95)), 4),
            },
            "factual": {
                "needle_mean": round(float(np.mean(needle_factual)), 4),
                "haystack_mean": round(float(np.mean(haystack_factual)), 4),
                "needle_max": round(float(np.max(needle_factual)), 4),
                "haystack_p95": round(float(np.percentile(haystack_factual, 95)), 4),
            },
        },
    }

    print("\n=== SIGNAL COMPARISON ===")
    for signal, vals in summary["signal_comparison"].items():
        print(f"  {signal}: needle_mean={vals['needle_mean']:.4f} vs haystack_mean={vals['haystack_mean']:.4f} (ratio={vals['needle_mean']/max(vals['haystack_mean'],1e-6):.1f}x)")

    result = {"summary": summary, "tokens": annotated}

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved to: {out}")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--policy", default="semantic")
    parser.add_argument("--budget", type=float, default=0.2)
    parser.add_argument("--target-tokens", type=int, default=1800)
    parser.add_argument("--depth", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/case_study/token_retention.json")
    args = parser.parse_args()

    cfg = ExperimentConfig()
    if args.model:
        cfg.model.model_name = args.model
    model, tokenizer = load_model(cfg.model)

    dump_retention(
        model, tokenizer, cfg.model,
        policy=args.policy,
        budget=args.budget,
        target_tokens=args.target_tokens,
        depth=args.depth,
        seed=args.seed,
        output_path=args.output,
    )
