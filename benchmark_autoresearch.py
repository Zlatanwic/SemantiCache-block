"""Autoresearch benchmark harness for SemantiCache policy experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import ExperimentConfig
from eval_niah import HAYSTACK_PARAGRAPHS, NEEDLES, build_result_record, run_single_eval
from run_generation import generate_with_eviction, load_model
from tqdm.auto import tqdm


GAUNTLET_CASES = [
    {
        "name": "project_codename_current",
        "fact": "The current source-of-truth memo states that the secret project code name is Operation Starlight and it was initiated on March 15th.",
        "question": "According to the current source-of-truth memo, what is the secret project code name and when was it initiated?",
        "answer_keywords": ["Operation Starlight", "March 15"],
        "distractors": [
            "An outdated planning note incorrectly says the project code name was Operation Daybreak and that it began on March 1st.",
            "A rehearsal document once referred to Operation Starlight but paired it with March 1st instead of the real initiation date.",
        ],
    },
    {
        "name": "server_temperature_authoritative",
        "fact": "The authoritative facilities note says the server room temperature must be maintained at exactly 18.5 degrees Celsius for optimal performance.",
        "question": "What exact temperature does the authoritative facilities note require for the server room?",
        "answer_keywords": ["18.5"],
        "distractors": [
            "A stale maintenance checklist still mentions 19.5 degrees Celsius, but it is marked superseded.",
            "An onboarding slide once rounded the target to 18 degrees Celsius, which is no longer the approved value.",
        ],
    },
    {
        "name": "migration_schedule_authoritative",
        "fact": "The approved migration plan says the database migration is scheduled for November 27th at 3:00 AM UTC and requires exactly 4 hours of downtime.",
        "question": "In the approved migration plan, when is the database migration scheduled and how long is the downtime?",
        "answer_keywords": ["November 27", "3:00 AM", "4 hours"],
        "distractors": [
            "A canceled rehearsal plan mentioned November 27th at 3:00 AM UTC but only 1 hour of downtime.",
            "An obsolete draft referred to 4 hours of downtime but listed the start time as 2:00 AM UTC.",
        ],
    },
    {
        "name": "incident_bridge_authoritative",
        "fact": "The current incident bridge opens on channel Delta-42 at 09:30 AM UTC and remains active for exactly 90 minutes.",
        "question": "Which channel does the current incident bridge use, when does it open, and how long does it remain active?",
        "answer_keywords": ["Delta-42", "09:30", "90 minutes"],
        "distractors": [
            "An old runbook still lists channel Delta-24 at 09:00 AM UTC for 60 minutes.",
            "A rehearsal script used Delta-42 at 09:00 AM UTC for only 45 minutes.",
        ],
    },
    {
        "name": "backup_window_authoritative",
        "fact": "The authoritative retention policy says encrypted backups are retained for 37 days and the export window opens at 02:30 AM UTC.",
        "question": "According to the authoritative retention policy, how long are encrypted backups retained and when does the export window open?",
        "answer_keywords": ["37 days", "02:30"],
        "distractors": [
            "A superseded checklist still mentions 30 days and an export window at 02:00 AM UTC.",
            "A temporary incident waiver once used 14 days of retention while keeping the same export window.",
        ],
    },
]


SUITES = {
    "smoke": {
        "case_builder": "standard",
        "cases": NEEDLES,
        "haystack_lengths": [1000],
        "budgets": [0.5, 0.3],
        "positions": [0.5],
        "hard_filter": {"min_haystack_length": 1000, "max_budget": 0.3},
    },
    "frontier": {
        "case_builder": "standard",
        "cases": NEEDLES,
        "haystack_lengths": [1000, 2000],
        "budgets": [0.5, 0.3, 0.2],
        "positions": [0.0, 0.25, 0.5, 0.75, 1.0],
        "hard_filter": {"min_haystack_length": 2000, "max_budget": 0.3},
    },
    "gauntlet": {
        "case_builder": "adversarial",
        "cases": GAUNTLET_CASES,
        "haystack_lengths": [2000, 4000],
        "budgets": [0.2, 0.15, 0.1],
        "positions": [0.1, 0.5, 0.9],
        "hard_filter": {"min_haystack_length": 4000, "max_budget": 0.15},
    },
}


def parse_csv_filter(raw_value: str | None) -> list[str]:
    """Parse a comma-separated CLI filter into normalized item names."""
    if raw_value is None:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def get_case_name(case: dict, index: int) -> str:
    """Return a stable display/filter name for a benchmark case."""
    if "name" in case and case["name"]:
        return str(case["name"])
    return f"needle_{index + 1}"


def select_cases(cases: list[dict], case_filters: list[str]) -> list[tuple[str, dict]]:
    """Return named cases filtered by optional user-provided case names."""
    named_cases = [(get_case_name(case, index), case) for index, case in enumerate(cases)]
    if not case_filters:
        return named_cases

    selected = [(case_name, case) for case_name, case in named_cases if case_name in case_filters]
    available = {case_name for case_name, _ in named_cases}
    missing = [case_name for case_name in case_filters if case_name not in available]
    if missing:
        raise ValueError(f"Unknown case name(s): {', '.join(missing)}")
    return selected


def build_experiment_grid(
    suite: dict,
    policy: str,
    case_filters: list[str] | None = None,
    mini: bool = False,
    limit: int | None = None,
) -> list[tuple[int, float, float, str, dict]]:
    """Build a filtered experiment grid for one suite/policy combination."""
    haystack_lengths = list(suite["haystack_lengths"])
    budgets = list(suite["budgets"])
    positions = list(suite["positions"])
    named_cases = select_cases(suite["cases"], case_filters or [])

    if mini:
        haystack_lengths = haystack_lengths[:1]
        budgets = budgets[:1]
        positions = positions[:1]
        named_cases = named_cases[:1]

    experiment_grid: list[tuple[int, float, float, str, dict]] = []
    for haystack_length in haystack_lengths:
        for budget in budgets:
            for needle_position in positions:
                for case_name, case in named_cases:
                    experiment_grid.append((haystack_length, budget, needle_position, case_name, case))

    if limit is not None:
        experiment_grid = experiment_grid[: max(0, limit)]

    return experiment_grid


def print_grid_preview(
    suite_name: str,
    policy: str,
    experiment_grid: list[tuple[int, float, float, str, dict]],
    preview_count: int = 10,
) -> None:
    """Print a compact preview of the planned benchmark runs."""
    print(f"suite_name: {suite_name}")
    print(f"policy: {policy}")
    print(f"planned_runs: {len(experiment_grid)}")
    if not experiment_grid:
        print("planned_preview: <empty>")
        return

    print("planned_preview:")
    for index, (haystack_length, budget, needle_position, case_name, _) in enumerate(
        experiment_grid[:preview_count],
        start=1,
    ):
        print(
            f"  [{index}] case={case_name}, len={haystack_length}, "
            f"budget={budget:.0%}, pos={needle_position:.2f}"
        )
    if len(experiment_grid) > preview_count:
        print(f"  ... {len(experiment_grid) - preview_count} more")


def _build_haystack_paragraphs(haystack_length: int) -> list[str]:
    """Create a deterministic haystack body with approximately stable size."""
    paragraphs = []
    total_chars = 0
    target_chars = haystack_length * 4

    while total_chars < target_chars:
        for paragraph in HAYSTACK_PARAGRAPHS:
            paragraphs.append(paragraph)
            total_chars += len(paragraph)
            if total_chars >= target_chars:
                break

    return paragraphs


def build_adversarial_prompt(
    case: dict,
    haystack_length: int,
    needle_position: float,
) -> list[dict]:
    """Build a harder prompt with nearby confusable distractors."""
    paragraphs = _build_haystack_paragraphs(haystack_length)
    anchor = int(len(paragraphs) * needle_position)
    anchor = max(2, min(anchor, len(paragraphs) - 3))

    insertions = [
        (anchor - 2, case["distractors"][0]),
        (anchor, case["fact"]),
        (anchor + 2, case["distractors"][1]),
    ]
    insertions.sort(key=lambda item: item[0])

    offset = 0
    for index, text in insertions:
        paragraphs.insert(index + offset, text)
        offset += 1

    haystack_text = "\n\n".join(paragraphs)
    return [
        {
            "role": "system",
            "content": (
                "You are a careful assistant. The text may contain outdated or contradictory notes. "
                "Answer only using the current or authoritative statement in the text."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Please read the following text carefully:\n\n{haystack_text}\n\n"
                f"Now answer this question: {case['question']}"
            ),
        },
    ]


def run_adversarial_eval(
    model,
    tokenizer,
    config: ExperimentConfig,
    case: dict,
    haystack_length: int,
    needle_position: float,
    max_new_tokens: int | None = None,
    stop_when_answer_found: bool = False,
) -> dict:
    """Run one adversarial NIAH item with near-neighbor distractors."""
    messages = build_adversarial_prompt(case, haystack_length, needle_position)
    config.model.do_sample = False
    config.model.max_new_tokens = max_new_tokens if max_new_tokens is not None else 128
    config.model.stop_when_output_contains = list(case["answer_keywords"]) if stop_when_answer_found else []

    run_result = generate_with_eviction(model, tokenizer, messages, config)
    record = build_result_record(
        run_result=run_result,
        config=config,
        needle=case,
        policy=config.cache.policy,
        budget=config.cache.cache_budget,
        haystack_length=haystack_length,
        needle_position=needle_position,
    )
    record["task"] = "niah_adversarial"
    record["case_name"] = case["name"]
    record["distractors"] = case["distractors"]
    return record


def summarize_cases(results: list[dict], suite: dict) -> dict:
    """Compute stable summary metrics for one benchmark run."""
    case_count = len(results)
    correct_count = sum(1 for result in results if result["correct"])
    accuracy = correct_count / case_count if case_count else 0.0
    avg_time_s = sum(result["elapsed_time"] for result in results) / case_count if case_count else 0.0
    avg_tokens_per_second = (
        sum(result.get("tokens_per_second", 0.0) for result in results) / case_count if case_count else 0.0
    )
    avg_retained_tokens = (
        sum(result.get("retained_tokens", 0) for result in results) / case_count if case_count else 0.0
    )
    avg_retained_token_ratio = (
        sum(result.get("retained_token_ratio_vs_logical", 0.0) for result in results) / case_count
        if case_count
        else 0.0
    )
    avg_retained_kv_bytes = (
        sum(result.get("retained_kv_bytes", 0) for result in results) / case_count if case_count else 0.0
    )
    avg_logical_full_kv_bytes = (
        sum(result.get("logical_full_kv_bytes", 0) for result in results) / case_count if case_count else 0.0
    )
    avg_kv_bytes_saved = (
        sum(result.get("kv_bytes_saved", 0) for result in results) / case_count if case_count else 0.0
    )
    avg_kv_savings_ratio = (
        sum(result.get("kv_savings_ratio_vs_logical_full", 0.0) for result in results) / case_count
        if case_count
        else 0.0
    )
    avg_retention_overhead_s = (
        sum(result.get("retention_overhead_s", 0.0) for result in results) / case_count if case_count else 0.0
    )
    avg_retention_overhead_ratio = (
        sum(result.get("retention_overhead_ratio", 0.0) for result in results) / case_count
        if case_count
        else 0.0
    )

    hard_filter = suite["hard_filter"]
    hard_cases = [
        result
        for result in results
        if result["haystack_length"] >= hard_filter["min_haystack_length"]
        and result["budget"] <= hard_filter["max_budget"]
    ]
    hard_correct = sum(1 for result in hard_cases if result["correct"])
    hard_accuracy = hard_correct / len(hard_cases) if hard_cases else accuracy

    return {
        "case_count": case_count,
        "correct_count": correct_count,
        "accuracy": accuracy,
        "hard_case_count": len(hard_cases),
        "hard_correct_count": hard_correct,
        "hard_accuracy": hard_accuracy,
        "avg_time_s": avg_time_s,
        "avg_tokens_per_second": avg_tokens_per_second,
        "avg_retained_tokens": avg_retained_tokens,
        "avg_retained_token_ratio": avg_retained_token_ratio,
        "avg_retained_kv_bytes": avg_retained_kv_bytes,
        "avg_logical_full_kv_bytes": avg_logical_full_kv_bytes,
        "avg_kv_bytes_saved": avg_kv_bytes_saved,
        "avg_kv_savings_ratio": avg_kv_savings_ratio,
        "avg_retention_overhead_s": avg_retention_overhead_s,
        "avg_retention_overhead_ratio": avg_retention_overhead_ratio,
    }


def print_summary(summary: dict, suite_name: str) -> None:
    """Emit machine-readable summary lines for the experiment loop."""
    print(f"suite_name: {suite_name}")
    print(f"suite_cases: {summary['case_count']}")
    print(f"suite_accuracy: {summary['accuracy']:.6f}")
    print(f"suite_hard_accuracy: {summary['hard_accuracy']:.6f}")
    print(f"suite_avg_time_s: {summary['avg_time_s']:.6f}")
    print(f"suite_avg_tokens_per_second: {summary['avg_tokens_per_second']:.6f}")
    print(f"suite_avg_retained_tokens: {summary['avg_retained_tokens']:.6f}")
    print(f"suite_avg_retained_token_ratio: {summary['avg_retained_token_ratio']:.6f}")
    print(f"suite_avg_retained_kv_bytes: {summary['avg_retained_kv_bytes']:.2f}")
    print(f"suite_avg_logical_full_kv_bytes: {summary['avg_logical_full_kv_bytes']:.2f}")
    print(f"suite_avg_kv_bytes_saved: {summary['avg_kv_bytes_saved']:.2f}")
    print(f"suite_avg_kv_savings_ratio: {summary['avg_kv_savings_ratio']:.6f}")
    print(f"suite_avg_retention_overhead_s: {summary['avg_retention_overhead_s']:.6f}")
    print(f"suite_avg_retention_overhead_ratio: {summary['avg_retention_overhead_ratio']:.6f}")
    print(f"correct_count: {summary['correct_count']}/{summary['case_count']}")
    if summary["hard_case_count"] > 0:
        print(
            "hard_correct_count: "
            f"{summary['hard_correct_count']}/{summary['hard_case_count']}"
        )


def run_suite(
    suite_name: str,
    policy: str,
    output_path: Path,
    hot_ratio: float = 0.5,
    warm_top_k: int = 16,
    case_filters: list[str] | None = None,
    limit: int | None = None,
    mini: bool = False,
    dry_run: bool = False,
    max_new_tokens: int | None = None,
    haystack_length_override: int | None = None,
    stop_when_answer_found: bool = False,
) -> dict:
    """Run one named benchmark suite and persist raw results plus a summary."""
    suite = SUITES[suite_name]
    experiment_grid = build_experiment_grid(
        suite=suite,
        policy=policy,
        case_filters=case_filters,
        mini=mini,
        limit=limit,
    )
    if haystack_length_override is not None:
        experiment_grid = [
            (haystack_length_override, budget, needle_position, case_name, case)
            for _, budget, needle_position, case_name, case in experiment_grid
        ]
    print_grid_preview(suite_name=suite_name, policy=policy, experiment_grid=experiment_grid)
    if dry_run:
        return {
            "suite": suite_name,
            "policy": policy,
            "hot_ratio": hot_ratio,
            "warm_top_k": warm_top_k,
            "planned_runs": len(experiment_grid),
            "max_new_tokens": max_new_tokens,
            "haystack_length_override": haystack_length_override,
            "stop_when_answer_found": stop_when_answer_found,
            "results": [],
        }

    if not experiment_grid:
        payload = {
            "suite": suite_name,
            "policy": policy,
            "hot_ratio": hot_ratio,
            "warm_top_k": warm_top_k,
            "summary": summarize_cases(results=[], suite=suite),
            "results": [],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"saved_to: {output_path}")
        return payload

    model_cfg = ExperimentConfig().model
    model, tokenizer = load_model(model_cfg)

    results = []
    total = len(experiment_grid)
    run_progress = tqdm(experiment_grid, desc="Benchmark runs", dynamic_ncols=True)
    try:
        for index, (haystack_length, budget, needle_position, case_name, case) in enumerate(
            run_progress,
            start=1,
        ):
            run_progress.set_postfix_str(
                f"{index}/{total} case={case_name} len={haystack_length} budget={budget:.0%} pos={needle_position:.2f}"
            )
            print(
                f"[{index}/{total}] policy={policy}, case={case_name}, len={haystack_length}, "
                f"budget={budget:.0%}, pos={needle_position:.2f}"
            )

            cfg = ExperimentConfig()
            cfg.cache.policy = policy
            cfg.cache.cache_budget = budget
            cfg.cache.semantic_hot_ratio = hot_ratio
            cfg.cache.semantic_warm_top_k = warm_top_k
            cfg.model.do_sample = False
            cfg.model.show_progress_bar = True

            if suite["case_builder"] == "standard":
                result = run_single_eval(
                    model=model,
                    tokenizer=tokenizer,
                    config=cfg,
                    needle=case,
                    haystack_length=haystack_length,
                    needle_position=needle_position,
                    max_new_tokens=max_new_tokens,
                    stop_when_answer_found=stop_when_answer_found,
                )
            else:
                cfg.model.max_new_tokens = max_new_tokens if max_new_tokens is not None else 128
                cfg.model.stop_when_output_contains = list(case["answer_keywords"]) if stop_when_answer_found else []
                result = run_adversarial_eval(
                    model=model,
                    tokenizer=tokenizer,
                    config=cfg,
                    case=case,
                    haystack_length=haystack_length,
                    needle_position=needle_position,
                    max_new_tokens=max_new_tokens,
                    stop_when_answer_found=stop_when_answer_found,
                )

            result.setdefault("case_name", case_name)
            results.append(result)
            print(f"  -> {'Correct' if result['correct'] else 'Wrong'}")
    finally:
        run_progress.close()

    summary = summarize_cases(results, suite=suite)
    payload = {
        "suite": suite_name,
        "policy": policy,
        "hot_ratio": hot_ratio,
        "warm_top_k": warm_top_k,
        "summary": summary,
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print_summary(summary, suite_name)
    print(f"saved_to: {output_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Expanded benchmark harness for SemantiCache autoresearch.")
    parser.add_argument(
        "--suite",
        choices=sorted(SUITES),
        default="frontier",
        help="Benchmark suite to execute.",
    )
    parser.add_argument(
        "--policy",
        default="semantic",
        choices=["full", "window", "streaming", "h2o", "semantic", "tiered_semantic"],
        help="Policy to evaluate.",
    )
    parser.add_argument(
        "--output",
        default="results/autoresearch/candidate_latest.json",
        help="Where to save raw results plus summary.",
    )
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
        "--case",
        type=str,
        default=None,
        help="Comma-separated case names to run (use --dry-run or --list-cases to inspect names)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of planned runs after filtering",
    )
    parser.add_argument(
        "--mini",
        action="store_true",
        help="Run a minimal subset: first haystack, first budget, first position, first case",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned run grid and exit without loading the model",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override generation length for quick benchmark validation",
    )
    parser.add_argument(
        "--haystack-length-override",
        type=int,
        default=None,
        help="Override the suite haystack length for all planned runs",
    )
    parser.add_argument(
        "--stop-when-answer-found",
        action="store_true",
        help="Stop generation early once all expected answer keywords appear in the output",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List available case names for the selected suite and exit",
    )
    args = parser.parse_args()

    if args.list_cases:
        named_cases = select_cases(SUITES[args.suite]["cases"], [])
        print(f"suite_name: {args.suite}")
        print("available_cases:")
        for case_name, _ in named_cases:
            print(f"  {case_name}")
        return

    case_filters = parse_csv_filter(args.case)

    run_suite(
        suite_name=args.suite,
        policy=args.policy,
        output_path=Path(args.output),
        hot_ratio=args.hot_ratio,
        warm_top_k=args.warm_top_k,
        case_filters=case_filters,
        limit=args.limit,
        mini=args.mini,
        dry_run=args.dry_run,
        max_new_tokens=args.max_new_tokens,
        haystack_length_override=args.haystack_length_override,
        stop_when_answer_found=args.stop_when_answer_found,
    )


if __name__ == "__main__":
    main()
