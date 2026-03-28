"""
Small memory-recall benchmark for comparing full cache vs tiered_semantic.

This benchmark keeps prompts short and explicit so we can measure whether the
cache policy preserves user-provided facts across a few conversational turns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from datetime import datetime

from config import ExperimentConfig
from run_generation import generate_with_eviction, load_model


AVAILABLE_POLICIES = ["full", "tiered_semantic"]


MEMORY_RECALL_CASES = [
    {
        "name": "project_name",
        "fact": "My operating system project is called NovaOS.",
        "bridge": "I'm implementing a batch syscall mechanism this week.",
        "question": "What is the name of my project? Answer with only the project name.",
        "answer_keywords": ["NovaOS"],
        "max_new_tokens": 4,
    },
    {
        "name": "codename",
        "fact": "The internal codename for this release is Starlight.",
        "bridge": "I'm currently rewriting the deployment checklist and incident notes.",
        "question": "What is the codename of the release? Answer with only the codename.",
        "answer_keywords": ["Starlight"],
        "max_new_tokens": 4,
    },
    {
        "name": "language",
        "fact": "The kernel module I am writing is implemented in Rust.",
        "bridge": "Right now I am debugging DMA buffer ownership and interrupt ordering.",
        "question": "Which programming language is the kernel module implemented in? Answer with only the language.",
        "answer_keywords": ["Rust"],
        "max_new_tokens": 4,
    },
    {
        "name": "launch_date",
        "fact": "The planned launch date for the prototype is March 15.",
        "bridge": "I am also preparing a small demo and writing release notes for the team.",
        "question": "What is the planned launch date of the prototype? Answer with only the date.",
        "answer_keywords": ["March 15"],
        "max_new_tokens": 6,
    },
    {
        "name": "incident_channel",
        "fact": "The incident bridge channel for this service is Delta-42.",
        "bridge": "I am tuning alert routing and cleaning up the on-call handoff template.",
        "question": "What is the incident bridge channel for the service? Answer with only the channel name.",
        "answer_keywords": ["Delta-42"],
        "max_new_tokens": 6,
    },
]


def parse_csv_filter(raw_value: str | None) -> list[str]:
    """Parse a comma-separated CLI filter into normalized item names."""
    if raw_value is None:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def select_cases(case_filters: list[str]) -> list[dict]:
    """Return benchmark cases matching the requested case names."""
    if not case_filters:
        return MEMORY_RECALL_CASES

    selected = [case for case in MEMORY_RECALL_CASES if case["name"] in case_filters]
    missing = [name for name in case_filters if name not in {case["name"] for case in MEMORY_RECALL_CASES}]
    if missing:
        raise ValueError(f"Unknown case name(s): {', '.join(missing)}")
    return selected


def select_policies(policy_filters: list[str]) -> list[str]:
    """Return benchmark policies matching the requested policy names."""
    if not policy_filters:
        return AVAILABLE_POLICIES

    missing = [policy for policy in policy_filters if policy not in AVAILABLE_POLICIES]
    if missing:
        raise ValueError(f"Unknown policy name(s): {', '.join(missing)}")
    return [policy for policy in AVAILABLE_POLICIES if policy in policy_filters]


def format_duration(seconds: float) -> str:
    """Format a duration in a compact human-readable form."""
    rounded = max(0, int(round(seconds)))
    minutes, secs = divmod(rounded, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def log_progress(message: str) -> None:
    """Print a timestamped progress line immediately."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def expected_budget(policy: str, requested_budget: float) -> float:
    """Return the effective budget for the given policy."""
    return 1.0 if policy == "full" else requested_budget


def load_resumable_records(
    output_path: Path,
    selected_case_names: set[str],
    selected_policies: set[str],
    budget: float,
    hot_ratio: float,
    warm_top_k: int,
    follow_token_bias: float,
    max_new_tokens_override: int | None,
) -> list[dict]:
    """Load matching records for the current benchmark selection and settings."""
    if not output_path.exists():
        return []

    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    if not isinstance(payload, list):
        return []

    records: list[dict] = []
    for record in payload:
        if not isinstance(record, dict):
            continue
        if record.get("case_name") not in selected_case_names:
            continue
        if record.get("policy") not in selected_policies:
            continue
        if record.get("budget") != expected_budget(record["policy"], budget):
            continue
        if record.get("hot_ratio") != hot_ratio:
            continue
        if record.get("warm_top_k") != warm_top_k:
            continue
        if record.get("follow_token_bias") != follow_token_bias:
            continue
        expected_max_new_tokens = (
            max_new_tokens_override
            if max_new_tokens_override is not None
            else next(
                int(case.get("max_new_tokens", 8))
                for case in MEMORY_RECALL_CASES
                if case["name"] == record["case_name"]
            )
        )
        if record.get("max_new_tokens") != expected_max_new_tokens:
            continue
        records.append(record)
    return records


def write_records(output_path: Path, records: list[dict]) -> None:
    """Persist records to disk in a stable JSON format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def record_key(record: dict) -> tuple[str, str]:
    """Return the unique record key for resume and sorting."""
    return record["case_name"], record["policy"]


def build_messages(case: dict) -> list[dict]:
    """Create a short multi-turn prompt with one user fact to recall later."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": case["fact"]},
        {
            "role": "assistant",
            "content": (
                "Understood. I'll remember that detail and keep my answers concise when you ask for it later."
            ),
        },
        {"role": "user", "content": case["bridge"]},
        {
            "role": "assistant",
            "content": "Thanks for the context. Ask me for the specific detail whenever you need it.",
        },
        {"role": "user", "content": case["question"]},
    ]


def check_answer(output_text: str, answer_keywords: list[str]) -> bool:
    """Return whether all expected answer keywords appear in the output."""
    normalized = output_text.strip().lower()
    return all(keyword.lower() in normalized for keyword in answer_keywords)


def run_case(
    model,
    tokenizer,
    policy: str,
    budget: float,
    hot_ratio: float,
    warm_top_k: int,
    follow_token_bias: float,
    max_new_tokens_override: int | None,
    case: dict,
) -> dict:
    """Run one benchmark case under one cache policy."""
    config = ExperimentConfig()
    config.cache.policy = policy
    config.cache.cache_budget = budget
    config.cache.semantic_hot_ratio = hot_ratio
    config.cache.semantic_warm_top_k = warm_top_k
    config.cache.semantic_follow_token_bias = follow_token_bias
    config.model.do_sample = False
    config.model.max_new_tokens = (
        max_new_tokens_override
        if max_new_tokens_override is not None
        else int(case.get("max_new_tokens", 8))
    )
    config.model.stop_when_output_contains = list(case["answer_keywords"])

    if policy == "full":
        config.cache.cache_budget = 1.0

    result = generate_with_eviction(model, tokenizer, build_messages(case), config)
    output_text = result["output_text"].strip()
    stats = result["stats"]
    return {
        "case_name": case["name"],
        "policy": policy,
        "budget": config.cache.cache_budget,
        "hot_ratio": hot_ratio,
        "warm_top_k": warm_top_k,
        "follow_token_bias": follow_token_bias,
        "max_new_tokens": config.model.max_new_tokens,
        "stop_when_output_contains": config.model.stop_when_output_contains,
        "question": case["question"],
        "answer_keywords": case["answer_keywords"],
        "output_text": output_text,
        "correct": check_answer(output_text, case["answer_keywords"]),
        "generated_tokens": len(result["output_ids"]),
        "elapsed_time": result["elapsed_time"],
        "initial_seq_len": stats["initial_seq_len"],
        "current_cache_len": stats["current_cache_len"],
        "hot_cache_len": stats.get("hot_cache_len"),
        "warm_cache_len": stats.get("warm_cache_len"),
        "promotion_steps": stats.get("promotion_steps"),
        "peak_promoted_warm_count": stats.get("peak_promoted_warm_count"),
    }


def print_summary(records: list[dict], policies: list[str]) -> None:
    """Print a compact by-policy summary plus per-case outputs."""
    if not records:
        print("\nNo benchmark records to summarize.")
        return

    print("\n" + "=" * 72)
    print("Memory Recall Summary")
    print("=" * 72)
    for policy in policies:
        subset = [record for record in records if record["policy"] == policy]
        correct = sum(1 for record in subset if record["correct"])
        print(f"{policy:16s} {correct}/{len(subset)} correct")

    print("\nPer-case outputs")
    print("-" * 72)
    case_names = []
    for record in records:
        if record["case_name"] not in case_names:
            case_names.append(record["case_name"])

    for case_name in case_names:
        print(case_name)
        for policy in policies:
            record = next(
                (record for record in records if record["case_name"] == case_name and record["policy"] == policy),
                None,
            )
            if record is None:
                continue
            verdict = "OK" if record["correct"] else "MISS"
            print(f"  {policy:16s} [{verdict}] {record['output_text']!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory-recall benchmark for SemantiCache")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/memory_recall_benchmark.json"),
        help="Path to write raw JSON results",
    )
    parser.add_argument("--budget", type=float, default=0.3, help="Cache budget for tiered_semantic")
    parser.add_argument("--hot-ratio", type=float, default=0.7, help="Hot-tier ratio for tiered_semantic")
    parser.add_argument("--warm-top-k", type=int, default=8, help="Warm promotion top-k for tiered_semantic")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override max_new_tokens for all selected cases (default: per-case optimized limits)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume matching runs already present in the output file and save progress after each case",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List available case and policy names, then exit without loading the model",
    )
    parser.add_argument(
        "--case",
        type=str,
        default=None,
        help="Comma-separated benchmark case names to run (default: all)",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Comma-separated policy names to run from: full,tiered_semantic (default: all)",
    )
    parser.add_argument(
        "--follow-token-bias",
        type=float,
        default=ExperimentConfig().cache.semantic_follow_token_bias,
        help="Bias next-token logits toward retained continuation tokens in tiered_semantic",
    )
    args = parser.parse_args()

    if args.list_cases:
        print("Cases:")
        for case in MEMORY_RECALL_CASES:
            print(f"  {case['name']}")
        print("Policies:")
        for policy in AVAILABLE_POLICIES:
            print(f"  {policy}")
        return

    try:
        selected_cases = select_cases(parse_csv_filter(args.case))
        policies = select_policies(parse_csv_filter(args.policy))
    except ValueError as exc:
        parser.error(str(exc))

    selected_case_names = {case["name"] for case in selected_cases}
    policy_names = set(policies)
    records: list[dict] = []
    if args.resume:
        records = load_resumable_records(
            output_path=args.output,
            selected_case_names=selected_case_names,
            selected_policies=policy_names,
            budget=args.budget,
            hot_ratio=args.hot_ratio,
            warm_top_k=args.warm_top_k,
            follow_token_bias=args.follow_token_bias,
            max_new_tokens_override=args.max_new_tokens,
        )
        records.sort(key=record_key)

    completed_keys = {record_key(record) for record in records}
    planned_pairs = [(case, policy) for case in selected_cases for policy in policies]
    pending_pairs = [(case, policy) for case, policy in planned_pairs if (case["name"], policy) not in completed_keys]
    total = len(planned_pairs)

    log_progress(
        "Benchmark plan: "
        f"{len(selected_cases)} case(s), {len(policies)} policy(s), "
        f"{len(pending_pairs)}/{total} run(s) pending"
    )
    log_progress(f"Cases: {', '.join(case['name'] for case in selected_cases)}")
    log_progress(f"Policies: {', '.join(policies)}")
    if args.resume and records:
        log_progress(f"Resuming from {len(records)} saved record(s) in {args.output}")

    if not pending_pairs:
        log_progress("Nothing left to run. Writing summary from resumed records.")
        write_records(args.output, records)
        print_summary(records, policies)
        return

    load_started = time.perf_counter()
    log_progress("Loading model and tokenizer...")
    model, tokenizer = load_model(ExperimentConfig().model)
    log_progress(f"Model ready in {format_duration(time.perf_counter() - load_started)}")

    benchmark_started = time.perf_counter()
    for completed, (case, policy) in enumerate(pending_pairs, start=len(records) + 1):
        run_started = time.perf_counter()
        log_progress(f"[{completed}/{total}] Starting case={case['name']} policy={policy}")
        record = run_case(
            model=model,
            tokenizer=tokenizer,
            policy=policy,
            budget=args.budget,
            hot_ratio=args.hot_ratio,
            warm_top_k=args.warm_top_k,
            follow_token_bias=args.follow_token_bias,
            max_new_tokens_override=args.max_new_tokens,
            case=case,
        )
        records.append(record)
        records.sort(key=record_key)
        write_records(args.output, records)

        run_elapsed = time.perf_counter() - run_started
        total_elapsed = time.perf_counter() - benchmark_started
        avg_run = total_elapsed / max(1, completed - len(completed_keys))
        remaining_runs = total - completed
        eta = avg_run * remaining_runs
        verdict = "OK" if record["correct"] else "MISS"
        log_progress(
            f"[{completed}/{total}] Finished case={case['name']} policy={policy} "
            f"in {format_duration(run_elapsed)} -> {verdict} {record['output_text']!r}"
        )
        if remaining_runs:
            log_progress(
                f"Progress saved to {args.output}. "
                f"Elapsed {format_duration(total_elapsed)}, ETA {format_duration(eta)}"
            )

    log_progress(f"Saved raw results to {args.output}")
    print_summary(records, policies)


if __name__ == "__main__":
    main()
