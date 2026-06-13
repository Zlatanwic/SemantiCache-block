"""
Needle-in-a-Haystack evaluation for KV-cache eviction policies.

This script inserts a small factual "needle" into a long irrelevant "haystack"
and evaluates whether the model can still recover that fact under different
KV-cache eviction policies and cache budgets.
"""

import argparse
import json
import random
from pathlib import Path

from config import ExperimentConfig
from run_generation import generate_with_eviction, load_model


NEEDLES = [
    {
        "fact": "The secret project code name is Operation Starlight and it was initiated on March 15th.",
        "question": "What is the secret project code name and when was it initiated?",
        "answer_keywords": ["Operation Starlight", "March 15"],
    },
    {
        "fact": "The server room temperature must be maintained at exactly 18.5 degrees Celsius for optimal performance.",
        "question": "What temperature must the server room be maintained at?",
        "answer_keywords": ["18.5"],
    },
    {
        "fact": "The database migration is scheduled for November 27th at 3:00 AM UTC and requires exactly 4 hours of downtime.",
        "question": "When is the database migration scheduled and how long is the downtime?",
        "answer_keywords": ["November 27", "3:00 AM", "4 hours"],
    },
]

HAYSTACK_PARAGRAPHS = [
    "The development team held their weekly standup meeting to discuss progress on various fronts. Several engineers reported completing their assigned tasks ahead of schedule, while others noted some unexpected challenges with third-party API integrations.",
    "Cloud computing has transformed the way organizations deploy and manage their infrastructure. With the advent of containerization and microservices architecture, teams can now scale their applications more efficiently than ever before.",
    "Machine learning models require careful tuning of hyperparameters to achieve optimal performance. The process of model selection involves evaluating multiple architectures and training configurations on validation datasets.",
    "The cybersecurity landscape continues to evolve with new threat vectors emerging regularly. Organizations must adopt a defense-in-depth strategy that includes network segmentation, endpoint protection, and regular security audits.",
    "Agile software development methodologies emphasize iterative development, collaboration, and adaptability. Scrum and Kanban are two popular frameworks that help teams manage their workflow effectively.",
    "Database optimization is crucial for maintaining application performance at scale. Techniques such as indexing, query optimization, and connection pooling can significantly reduce response times.",
    "The adoption of DevOps practices has bridged the gap between development and operations teams. Continuous integration and continuous deployment pipelines automate the software delivery process.",
    "Open source software has become a cornerstone of modern technology stacks. Communities around projects like Linux, Kubernetes, and TensorFlow drive innovation through collaborative development.",
    "Network latency remains a critical factor in distributed systems design. Techniques like content delivery networks, edge computing, and protocol optimization help minimize the impact of network delays.",
    "Software testing strategies range from unit tests to integration tests to end-to-end tests. A comprehensive testing pyramid ensures that bugs are caught early in the development lifecycle.",
    "Version control systems like Git enable teams to collaborate on codebases efficiently. Branching strategies such as GitFlow and trunk-based development provide structured approaches to managing code changes.",
    "API design principles emphasize consistency, discoverability, and backward compatibility. RESTful APIs and GraphQL offer different approaches to building interfaces between services.",
    "Monitoring and observability are essential for maintaining reliable distributed systems. Tools for logging, metrics collection, and distributed tracing help teams identify and resolve issues quickly.",
    "The evolution of programming languages reflects changing priorities in software development. Languages like Rust prioritize memory safety, while Go emphasizes simplicity and concurrency.",
    "Data engineering pipelines transform raw data into actionable insights. ETL processes, stream processing, and data warehousing form the backbone of modern data infrastructure.",
]

SWEEP_POLICIES = ["full", "window", "streaming", "h2o", "semantic", "block_semantic", "tiered_semantic", "op_sievekv_lite"]
SWEEP_BUDGETS = [1.0, 0.5, 0.3, 0.1]
SWEEP_POSITIONS = [0.0, 0.25, 0.5, 0.75, 1.0]


def build_haystack_prompt(
    needle: dict,
    haystack_length: int = 2000,
    needle_position: float = 0.5,
    paragraph_offset: int = 0,
) -> list[dict]:
    """Build a long-context prompt with one inserted target fact."""
    paragraphs = []
    total_chars = 0
    target_chars = haystack_length * 4
    offset = int(paragraph_offset) % len(HAYSTACK_PARAGRAPHS)
    haystack_paragraphs = HAYSTACK_PARAGRAPHS[offset:] + HAYSTACK_PARAGRAPHS[:offset]

    while total_chars < target_chars:
        for paragraph in haystack_paragraphs:
            paragraphs.append(paragraph)
            total_chars += len(paragraph)
            if total_chars >= target_chars:
                break

    insert_idx = int(len(paragraphs) * needle_position)
    insert_idx = max(1, min(insert_idx, len(paragraphs) - 1))
    paragraphs.insert(insert_idx, needle["fact"])

    haystack_text = "\n\n".join(paragraphs)
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer questions based on the provided text.",
        },
        {
            "role": "user",
            "content": (
                f"Please read the following text carefully:\n\n{haystack_text}\n\n"
                f"Now answer this question: {needle['question']}"
            ),
        },
    ]


def check_answer(output_text: str, keywords: list[str]) -> bool:
    """Return True if all answer keywords appear in the output."""
    output_lower = output_text.lower()
    return all(keyword.lower() in output_lower for keyword in keywords)


def _dtype_bytes(torch_dtype_name: str) -> int:
    """Return the byte width for the configured floating-point dtype."""
    mapping = {
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
    }
    return mapping.get(torch_dtype_name, 2)


def _estimate_fp16_kv_token_bytes(config: ExperimentConfig) -> int:
    """Estimate full-precision KV bytes retained per logical token."""
    dtype_bytes = _dtype_bytes(config.model.torch_dtype)
    return (
        config.model.num_layers
        * config.model.num_kv_heads
        * config.model.head_dim
        * 2  # keys + values
        * dtype_bytes
    )


def _estimate_warm_quantized_kv_token_bytes(config: ExperimentConfig) -> int:
    """Estimate quantized warm-tier bytes retained per logical token."""
    quant_bytes = max(1, int(config.cache.semantic_warm_bits) // 8)
    scale_bytes = 4  # per-vector fp32 scale
    return (
        config.model.num_layers
        * config.model.num_kv_heads
        * 2  # keys + values
        * ((config.model.head_dim * quant_bytes) + scale_bytes)
    )


def _build_system_metrics(run_result: dict, config: ExperimentConfig) -> dict:
    """Derive systems-facing cache and overhead metrics from one generation run."""
    stats = run_result["stats"]
    generated_tokens = len(run_result["output_ids"])
    logical_total_tokens = int((stats.get("initial_seq_len") or 0) + generated_tokens)

    fp_token_bytes = _estimate_fp16_kv_token_bytes(config)
    warm_token_bytes = _estimate_warm_quantized_kv_token_bytes(config)

    hot_cache_len = int(stats.get("hot_cache_len") or 0)
    warm_cache_len = int(stats.get("warm_cache_len") or 0)
    tiered_retained_tokens = hot_cache_len + warm_cache_len

    if config.cache.policy == "tiered_semantic":
        retained_tokens = tiered_retained_tokens
        retained_kv_bytes = (hot_cache_len * fp_token_bytes) + (warm_cache_len * warm_token_bytes)
        full_precision_retained_kv_bytes = tiered_retained_tokens * fp_token_bytes
        active_fp_kv_bytes = hot_cache_len * fp_token_bytes
        warm_quantized_kv_bytes = warm_cache_len * warm_token_bytes
    else:
        retained_tokens = int(stats.get("current_cache_len") or 0)
        retained_kv_bytes = retained_tokens * fp_token_bytes
        full_precision_retained_kv_bytes = retained_kv_bytes
        active_fp_kv_bytes = retained_kv_bytes
        warm_quantized_kv_bytes = 0

    logical_full_kv_bytes = logical_total_tokens * fp_token_bytes
    kv_bytes_saved = max(0, logical_full_kv_bytes - retained_kv_bytes)
    kv_bytes_saved_vs_uncompressed_retained = max(0, full_precision_retained_kv_bytes - retained_kv_bytes)

    quantize_time_s = float(stats.get("warm_quantize_time_s") or 0.0)
    dequantize_time_s = float(stats.get("warm_dequantize_time_s") or 0.0)
    retention_overhead_s = quantize_time_s + dequantize_time_s
    elapsed_time = float(run_result["elapsed_time"])

    return {
        "logical_total_tokens": logical_total_tokens,
        "retained_tokens": retained_tokens,
        "retained_token_ratio_vs_logical": retained_tokens / logical_total_tokens if logical_total_tokens else 0.0,
        "fp_kv_token_bytes": fp_token_bytes,
        "warm_kv_token_bytes": warm_token_bytes,
        "logical_full_kv_bytes": logical_full_kv_bytes,
        "retained_kv_bytes": retained_kv_bytes,
        "active_fp_kv_bytes": active_fp_kv_bytes,
        "warm_quantized_kv_bytes": warm_quantized_kv_bytes,
        "full_precision_retained_kv_bytes": full_precision_retained_kv_bytes,
        "kv_bytes_saved": kv_bytes_saved,
        "kv_savings_ratio_vs_logical_full": kv_bytes_saved / logical_full_kv_bytes if logical_full_kv_bytes else 0.0,
        "warm_quantization_savings_ratio": (
            kv_bytes_saved_vs_uncompressed_retained / full_precision_retained_kv_bytes
            if full_precision_retained_kv_bytes
            else 0.0
        ),
        "retention_overhead_s": retention_overhead_s,
        "retention_overhead_ratio": retention_overhead_s / elapsed_time if elapsed_time else 0.0,
        "tokens_per_second": generated_tokens / elapsed_time if elapsed_time else 0.0,
    }


def build_result_record(
    run_result: dict,
    config: ExperimentConfig,
    needle: dict,
    policy: str,
    budget: float,
    haystack_length: int,
    needle_position: float,
    paragraph_offset: int = 0,
) -> dict:
    """Normalize one evaluation run into a stable JSON-friendly schema."""
    stats = run_result["stats"]
    output_text = run_result["output_text"]
    correct = check_answer(output_text, needle["answer_keywords"])
    system_metrics = _build_system_metrics(run_result, config)

    return {
        "task": "niah",
        "policy": policy,
        "budget": budget,
        "haystack_length": haystack_length,
        "needle_position": needle_position,
        "paragraph_offset": paragraph_offset,
        "needle_fact": needle["fact"],
        "question": needle["question"],
        "answer_keywords": needle["answer_keywords"],
        "correct": correct,
        "output_text": output_text,
        "output_preview": output_text[:200],
        "elapsed_time": run_result["elapsed_time"],
        "generated_tokens": len(run_result["output_ids"]),
        "tokens_per_second": system_metrics["tokens_per_second"],
        "initial_seq_len": stats["initial_seq_len"],
        "cache_budget_tokens": stats["cache_budget_tokens"],
        "cache_budget_ratio": stats["cache_budget_ratio"],
        "current_cache_len": stats["current_cache_len"],
        "hot_cache_len": stats.get("hot_cache_len"),
        "warm_cache_len": stats.get("warm_cache_len"),
        "cold_cache_len": stats.get("cold_cache_len"),
        "peak_hot_cache_len": stats.get("peak_hot_cache_len"),
        "peak_warm_cache_len": stats.get("peak_warm_cache_len"),
        "peak_cold_cache_len": stats.get("peak_cold_cache_len"),
        "hot_budget_tokens": stats.get("hot_budget_tokens"),
        "hot_ratio": stats.get("hot_ratio"),
        "warm_top_k": stats.get("warm_top_k"),
        "warm_quantize_time_s": stats.get("warm_quantize_time_s"),
        "warm_dequantize_time_s": stats.get("warm_dequantize_time_s"),
        "warm_quantize_ops": stats.get("warm_quantize_ops"),
        "warm_dequantize_ops": stats.get("warm_dequantize_ops"),
        "materialization_steps": stats.get("materialization_steps"),
        "last_promoted_warm_count": stats.get("last_promoted_warm_count"),
        "peak_promoted_warm_count": stats.get("peak_promoted_warm_count"),
        "promotion_steps": stats.get("promotion_steps"),
        "total_evicted": stats["total_evicted"],
        "eviction_steps": stats["eviction_steps"],
        "logical_total_tokens": system_metrics["logical_total_tokens"],
        "retained_tokens": system_metrics["retained_tokens"],
        "retained_token_ratio_vs_logical": system_metrics["retained_token_ratio_vs_logical"],
        "fp_kv_token_bytes": system_metrics["fp_kv_token_bytes"],
        "warm_kv_token_bytes": system_metrics["warm_kv_token_bytes"],
        "logical_full_kv_bytes": system_metrics["logical_full_kv_bytes"],
        "retained_kv_bytes": system_metrics["retained_kv_bytes"],
        "active_fp_kv_bytes": system_metrics["active_fp_kv_bytes"],
        "warm_quantized_kv_bytes": system_metrics["warm_quantized_kv_bytes"],
        "full_precision_retained_kv_bytes": system_metrics["full_precision_retained_kv_bytes"],
        "kv_bytes_saved": system_metrics["kv_bytes_saved"],
        "kv_savings_ratio_vs_logical_full": system_metrics["kv_savings_ratio_vs_logical_full"],
        "warm_quantization_savings_ratio": system_metrics["warm_quantization_savings_ratio"],
        "retention_overhead_s": system_metrics["retention_overhead_s"],
        "retention_overhead_ratio": system_metrics["retention_overhead_ratio"],
    }


def run_single_eval(
    model,
    tokenizer,
    config: ExperimentConfig,
    needle: dict,
    haystack_length: int,
    needle_position: float,
    paragraph_offset: int = 0,
    max_new_tokens: int | None = None,
    stop_when_answer_found: bool = False,
) -> dict:
    """Run one NIAH evaluation item."""
    messages = build_haystack_prompt(needle, haystack_length, needle_position, paragraph_offset)
    config.model.do_sample = False
    config.model.max_new_tokens = max_new_tokens if max_new_tokens is not None else 128
    config.model.stop_when_output_contains = list(needle["answer_keywords"]) if stop_when_answer_found else []

    run_result = generate_with_eviction(model, tokenizer, messages, config)
    return build_result_record(
        run_result=run_result,
        config=config,
        needle=needle,
        policy=config.cache.policy,
        budget=config.cache.cache_budget,
        haystack_length=haystack_length,
        needle_position=needle_position,
        paragraph_offset=paragraph_offset,
    )


def make_niah_manifest(
    *,
    num_samples: int,
    haystack_length: int,
    seed: int,
    positions: list[float] | None = None,
) -> list[dict]:
    """Create fixed NIAH samples for paired policy comparisons."""
    rng = random.Random(seed)
    position_choices = positions or SWEEP_POSITIONS
    manifest: list[dict] = []
    for sample_id in range(num_samples):
        needle_index = sample_id % len(NEEDLES)
        if sample_id < len(NEEDLES) * len(position_choices):
            position = position_choices[(sample_id // len(NEEDLES)) % len(position_choices)]
        else:
            position = round(rng.uniform(0.0, 1.0), 4)
        manifest.append(
            {
                "task": "niah",
                "sample_id": sample_id,
                "needle_index": needle_index,
                "haystack_length": haystack_length,
                "needle_position": position,
                "paragraph_offset": rng.randrange(len(HAYSTACK_PARAGRAPHS)),
            }
        )
    return manifest


def run_manifest(
    model,
    tokenizer,
    manifest_path: Path,
    output_path: Path,
    policies: list[str],
    budgets: list[float],
    hot_ratio: float = 0.5,
    warm_top_k: int = 16,
    op_policy_ckpt: str | None = None,
) -> list[dict]:
    """Run a fixed NIAH manifest across policies and budgets."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    valid_policy_budget_pairs = [
        (policy, budget)
        for policy in policies
        for budget in budgets
        if not (policy == "full" and budget != 1.0)
    ]
    total = len(valid_policy_budget_pairs) * len(manifest)
    results: list[dict] = []
    run_index = 0

    for policy, budget in valid_policy_budget_pairs:
        for item in manifest:
            run_index += 1
            needle = NEEDLES[int(item["needle_index"])]
            print(
                f"\n[{run_index}/{total}] policy={policy}, budget={budget:.0%}, "
                f"sample={item.get('sample_id')}, pos={float(item['needle_position']):.2f}"
            )
            config = ExperimentConfig()
            config.cache.policy = policy
            config.cache.cache_budget = budget
            config.cache.semantic_hot_ratio = hot_ratio
            config.cache.semantic_warm_top_k = warm_top_k
            config.cache.op_policy_ckpt = op_policy_ckpt
            result = run_single_eval(
                model=model,
                tokenizer=tokenizer,
                config=config,
                needle=needle,
                haystack_length=int(item["haystack_length"]),
                needle_position=float(item["needle_position"]),
                paragraph_offset=int(item.get("paragraph_offset", 0)),
            )
            result["sample_id"] = item.get("sample_id")
            results.append(result)
            print(f"  -> {'Correct' if result['correct'] else 'Wrong'}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {output_path}")
    summarize_results(results, policies=policies, budgets=budgets)
    return results


def summarize_results(results: list[dict], policies: list[str], budgets: list[float]) -> None:
    """Print a compact summary grouped by policy and budget."""
    print("\n" + "=" * 60)
    print("NIAH Results Summary")
    print("=" * 60)

    for policy in policies:
        for budget in budgets:
            if policy == "full" and budget != 1.0:
                continue

            subset = [
                result
                for result in results
                if result["policy"] == policy and result["budget"] == budget
            ]
            if not subset:
                continue

            accuracy = sum(result["correct"] for result in subset) / len(subset)
            avg_time = sum(result["elapsed_time"] for result in subset) / len(subset)
            print(
                f"  {policy:12s} budget={budget:.0%}: "
                f"accuracy={accuracy:.1%} ({sum(result['correct'] for result in subset)}/{len(subset)}), "
                f"avg_time={avg_time:.2f}s"
            )


def run_sweep(
    model,
    tokenizer,
    output_path: Path,
    haystack_length: int = 2000,
    policies: list[str] | None = None,
    budgets: list[float] | None = None,
    positions: list[float] | None = None,
    hot_ratio: float = 0.5,
    warm_top_k: int = 16,
    op_policy_ckpt: str | None = None,
) -> list[dict]:
    """Run the full sweep across policies, budgets and needle positions."""
    policies = policies or SWEEP_POLICIES
    budgets = budgets or SWEEP_BUDGETS
    positions = positions or SWEEP_POSITIONS

    experiment_grid = []
    for policy in policies:
        for budget in budgets:
            if policy == "full" and budget != 1.0:
                continue
            for position in positions:
                for needle in NEEDLES:
                    experiment_grid.append((policy, budget, position, needle))

    results = []
    total = len(experiment_grid)

    for index, (policy, budget, position, needle) in enumerate(experiment_grid, start=1):
        print(f"\n[{index}/{total}] policy={policy}, budget={budget:.0%}, pos={position}")

        config = ExperimentConfig()
        config.cache.policy = policy
        config.cache.cache_budget = budget
        config.cache.semantic_hot_ratio = hot_ratio
        config.cache.semantic_warm_top_k = warm_top_k
        config.cache.op_policy_ckpt = op_policy_ckpt

        result = run_single_eval(
            model=model,
            tokenizer=tokenizer,
            config=config,
            needle=needle,
            haystack_length=haystack_length,
            needle_position=position,
        )
        results.append(result)
        print(f"  -> {'Correct' if result['correct'] else 'Wrong'}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")
    summarize_results(results, policies=policies, budgets=budgets)
    return results


def main():
    parser = argparse.ArgumentParser(description="Needle-in-a-Haystack Evaluation")
    parser.add_argument("--sweep", action="store_true", help="Run the full experiment sweep")
    parser.add_argument(
        "--policy",
        type=str,
        default="semantic",
        choices=SWEEP_POLICIES,
        help="Policy for single-run evaluation",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=None,
        choices=SWEEP_POLICIES,
        help="Policies for manifest mode. Defaults to --policy.",
    )
    parser.add_argument("--budget", type=float, default=0.5, help="Cache budget ratio")
    parser.add_argument(
        "--no-bnb-4bit",
        action="store_true",
        help="Disable BitsAndBytes 4-bit quantization (load full dtype; for large-VRAM GPUs)",
    )
    parser.add_argument(
        "--eviction-block-size",
        type=int,
        default=16,
        help="Whole-block eviction granularity for block_semantic policy (16/32/64)",
    )
    parser.add_argument(
        "--sweep-budgets",
        nargs="+",
        type=float,
        default=None,
        help="Budget ratios for sweep mode",
    )
    parser.add_argument("--haystack-length", type=int, default=2000, help="Approximate haystack token budget")
    parser.add_argument("--needle-pos", type=float, default=0.5, help="Needle insertion position in [0, 1]")
    parser.add_argument(
        "--hot-ratio",
        type=float,
        default=0.5,
        help="Hot-tier ratio within the total retained budget for tiered_semantic",
    )
    parser.add_argument(
        "--warm-top-k",
        type=int,
        default=16,
        help="How many warm-tier tokens to promote per decode step for tiered_semantic",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/niah_results.json",
        help="Output path for JSON results",
    )
    parser.add_argument("--op-policy-ckpt", type=str, default=None, help="Learned OP-SieveKV policy checkpoint")
    parser.add_argument("--manifest", type=str, default=None, help="Fixed NIAH manifest to evaluate")
    parser.add_argument("--create-manifest", type=str, default=None, help="Write a fixed NIAH manifest and exit")
    parser.add_argument("--manifest-samples", type=int, default=90, help="Number of samples for --create-manifest")
    parser.add_argument("--manifest-seed", type=int, default=42, help="Seed for --create-manifest")
    args = parser.parse_args()

    if args.create_manifest:
        manifest = make_niah_manifest(
            num_samples=args.manifest_samples,
            haystack_length=args.haystack_length,
            seed=args.manifest_seed,
        )
        manifest_path = Path(args.create_manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Wrote {len(manifest)} NIAH manifest samples to {manifest_path}")
        return

    config = ExperimentConfig()
    config.cache.op_policy_ckpt = args.op_policy_ckpt
    if args.no_bnb_4bit:
        config.model.use_bnb_4bit = False
    model, tokenizer = load_model(config.model)
    output_path = Path(args.output)

    if args.manifest:
        manifest_budgets = args.sweep_budgets or [args.budget]
        manifest_policies = args.policies or [args.policy]
        run_manifest(
            model=model,
            tokenizer=tokenizer,
            manifest_path=Path(args.manifest),
            output_path=output_path,
            policies=manifest_policies,
            budgets=manifest_budgets,
            hot_ratio=args.hot_ratio,
            warm_top_k=args.warm_top_k,
            op_policy_ckpt=args.op_policy_ckpt,
        )
        return

    if args.sweep:
        run_sweep(
            model=model,
            tokenizer=tokenizer,
            output_path=output_path,
            haystack_length=args.haystack_length,
            policies=args.policies,
            budgets=args.sweep_budgets,
            hot_ratio=args.hot_ratio,
            warm_top_k=args.warm_top_k,
            op_policy_ckpt=args.op_policy_ckpt,
        )
        return

    config.cache.policy = args.policy
    config.cache.cache_budget = args.budget
    config.cache.eviction_block_size = args.eviction_block_size
    config.cache.semantic_hot_ratio = args.hot_ratio
    config.cache.semantic_warm_top_k = args.warm_top_k
    config.cache.op_policy_ckpt = args.op_policy_ckpt

    needle = NEEDLES[0]
    result = run_single_eval(
        model=model,
        tokenizer=tokenizer,
        config=config,
        needle=needle,
        haystack_length=args.haystack_length,
        needle_position=args.needle_pos,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump([result], file, indent=2, ensure_ascii=False)

    print(f"\nResult: {'Correct' if result['correct'] else 'Wrong'}")
    print(f"Output preview: {result['output_preview']}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
