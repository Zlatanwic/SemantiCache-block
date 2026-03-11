"""
Needle-in-a-Haystack evaluation for KV-cache eviction policies.

This script inserts a small factual "needle" into a long irrelevant "haystack"
and evaluates whether the model can still recover that fact under different
KV-cache eviction policies and cache budgets.
"""

import argparse
import json
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

SWEEP_POLICIES = ["full", "window", "streaming", "h2o", "semantic"]
SWEEP_BUDGETS = [1.0, 0.5, 0.3, 0.1]
SWEEP_POSITIONS = [0.0, 0.25, 0.5, 0.75, 1.0]


def build_haystack_prompt(
    needle: dict,
    haystack_length: int = 2000,
    needle_position: float = 0.5,
) -> list[dict]:
    """Build a long-context prompt with one inserted target fact."""
    paragraphs = []
    total_chars = 0
    target_chars = haystack_length * 4

    while total_chars < target_chars:
        for paragraph in HAYSTACK_PARAGRAPHS:
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


def build_result_record(
    run_result: dict,
    needle: dict,
    policy: str,
    budget: float,
    haystack_length: int,
    needle_position: float,
) -> dict:
    """Normalize one evaluation run into a stable JSON-friendly schema."""
    stats = run_result["stats"]
    output_text = run_result["output_text"]
    correct = check_answer(output_text, needle["answer_keywords"])

    return {
        "task": "niah",
        "policy": policy,
        "budget": budget,
        "haystack_length": haystack_length,
        "needle_position": needle_position,
        "needle_fact": needle["fact"],
        "question": needle["question"],
        "answer_keywords": needle["answer_keywords"],
        "correct": correct,
        "output_text": output_text,
        "output_preview": output_text[:200],
        "elapsed_time": run_result["elapsed_time"],
        "generated_tokens": len(run_result["output_ids"]),
        "initial_seq_len": stats["initial_seq_len"],
        "cache_budget_tokens": stats["cache_budget_tokens"],
        "cache_budget_ratio": stats["cache_budget_ratio"],
        "current_cache_len": stats["current_cache_len"],
        "total_evicted": stats["total_evicted"],
        "eviction_steps": stats["eviction_steps"],
    }


def run_single_eval(
    model,
    tokenizer,
    config: ExperimentConfig,
    needle: dict,
    haystack_length: int,
    needle_position: float,
) -> dict:
    """Run one NIAH evaluation item."""
    messages = build_haystack_prompt(needle, haystack_length, needle_position)
    config.model.do_sample = False
    config.model.max_new_tokens = 128

    run_result = generate_with_eviction(model, tokenizer, messages, config)
    return build_result_record(
        run_result=run_result,
        needle=needle,
        policy=config.cache.policy,
        budget=config.cache.cache_budget,
        haystack_length=haystack_length,
        needle_position=needle_position,
    )


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
    parser.add_argument("--budget", type=float, default=0.5, help="Cache budget ratio")
    parser.add_argument("--haystack-length", type=int, default=2000, help="Approximate haystack token budget")
    parser.add_argument("--needle-pos", type=float, default=0.5, help="Needle insertion position in [0, 1]")
    parser.add_argument(
        "--output",
        type=str,
        default="results/niah_results.json",
        help="Output path for JSON results",
    )
    args = parser.parse_args()

    config = ExperimentConfig()
    model, tokenizer = load_model(config.model)
    output_path = Path(args.output)

    if args.sweep:
        run_sweep(
            model=model,
            tokenizer=tokenizer,
            output_path=output_path,
            haystack_length=args.haystack_length,
        )
        return

    config.cache.policy = args.policy
    config.cache.cache_budget = args.budget

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
