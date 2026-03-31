"""Continuous autoresearch runner for overnight gauntlet experiments."""

from __future__ import annotations

import csv
import json
import random
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS_TSV = ROOT / "results.tsv"
ARTIFACT_DIR = ROOT / "results" / "autoresearch"
RUN_LOG = ROOT / "run.log"
STOP_FILE = ROOT / "autoresearch.stop"
STATE_FILE = ARTIFACT_DIR / "runner_state.json"
CANDIDATE_JSON = ARTIFACT_DIR / "candidate_latest.json"
ACTIVE_SUITE = "gauntlet"


@dataclass
class Frontier:
    commit: str
    accuracy: float
    hard_accuracy: float
    avg_time_s: float


@dataclass
class Candidate:
    slug: str
    description: str
    files: tuple[str, ...]
    commit_message: str
    apply: callable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def run_command(args: list[str], stdout=None, stderr=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=stdout,
        stderr=stderr,
        check=False,
    )


def git_command(args: list[str]) -> subprocess.CompletedProcess:
    result = run_command(["git", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        return result

    combined = (result.stdout or "") + (result.stderr or "")
    if "index.lock" in combined:
        lock_file = ROOT / ".git" / "index.lock"
        lock_file.unlink(missing_ok=True)
        result = run_command(["git", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return result


def git_output(args: list[str]) -> str:
    result = git_command(args)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "").strip())
    return (result.stdout or "").strip()


def load_results() -> list[dict[str, str]]:
    if not RESULTS_TSV.exists():
        return []

    with RESULTS_TSV.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def append_result(row: dict[str, str]) -> None:
    file_exists = RESULTS_TSV.exists()
    with RESULTS_TSV.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment_id",
                "frontier_commit",
                "suite",
                "accuracy",
                "hard_accuracy",
                "avg_time_s",
                "status",
                "description",
                "output_json",
            ],
            delimiter="\t",
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def next_experiment_id(rows: list[dict[str, str]]) -> str:
    max_seen = 0
    for row in rows:
        match = re.search(r"(\d+)", row.get("experiment_id", ""))
        if match:
            max_seen = max(max_seen, int(match.group(1)))
    return f"exp-{max_seen + 1:03d}"


def current_frontier(rows: list[dict[str, str]]) -> Frontier:
    keeps = [
        row
        for row in rows
        if row.get("suite") == ACTIVE_SUITE and row.get("status") == "keep"
    ]
    if keeps:
        latest = keeps[-1]
        return Frontier(
            commit=git_output(["rev-parse", "--short", "HEAD"]),
            accuracy=float(latest["accuracy"]),
            hard_accuracy=float(latest["hard_accuracy"]),
            avg_time_s=float(latest["avg_time_s"]),
        )

    return Frontier(
        commit=git_output(["rev-parse", "--short", "HEAD"]),
        accuracy=0.0,
        hard_accuracy=0.0,
        avg_time_s=float("inf"),
    )


def experiment_seen(rows: list[dict[str, str]], frontier_commit: str, description: str) -> bool:
    for row in rows:
        if row.get("frontier_commit") == frontier_commit and row.get("description") == description:
            return True
    return False


def update_state(**state: object) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    previous: dict[str, object] = {}
    if STATE_FILE.exists():
        try:
            previous = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            previous = {}
    payload = {**previous, "updated_at": utc_now(), **state}
    STATE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_state() -> dict[str, object]:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def skipped_noop_descriptions(frontier_commit: str) -> set[str]:
    state = load_state()
    skip_map = state.get("noop_skip_by_frontier", {})
    if not isinstance(skip_map, dict):
        return set()
    values = skip_map.get(frontier_commit, [])
    if not isinstance(values, list):
        return set()
    return set(str(item) for item in values)


def remember_noop_candidate(frontier_commit: str, description: str) -> None:
    state = load_state()
    skip_map = state.get("noop_skip_by_frontier", {})
    if not isinstance(skip_map, dict):
        skip_map = {}
    existing = skip_map.get(frontier_commit, [])
    if not isinstance(existing, list):
        existing = []
    if description not in existing:
        existing.append(description)
    skip_map[frontier_commit] = existing
    update_state(noop_skip_by_frontier=skip_map)


def replace_line(text: str, pattern: str, replacement: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Failed to replace pattern: {pattern}")
    return updated


def update_file(path: str, updater) -> None:
    file_path = ROOT / path
    original = file_path.read_text(encoding="utf-8")
    updated = updater(original)
    if updated == original:
        raise ValueError(f"No effective change in {path}")
    file_path.write_text(updated, encoding="utf-8")


def restore_files(paths: tuple[str, ...]) -> None:
    result = git_command(["restore", "--source", "HEAD", "--", *paths])
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "").strip())


def benchmark() -> subprocess.CompletedProcess:
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("w", encoding="utf-8") as handle:
        return run_command(
            [
                "uv",
                "run",
                "python",
                "benchmark_autoresearch.py",
                "--suite",
                ACTIVE_SUITE,
                "--policy",
                "semantic",
                "--output",
                str(CANDIDATE_JSON.relative_to(ROOT)),
            ],
            stdout=handle,
            stderr=subprocess.STDOUT,
        )


def load_candidate_summary() -> dict[str, float]:
    payload = json.loads(CANDIDATE_JSON.read_text(encoding="utf-8"))
    return payload["summary"]


def create_crash_artifact(exp_id: str, candidate: Candidate, frontier_commit: str) -> Path:
    artifact_path = ARTIFACT_DIR / f"{exp_id}.json"
    tail = ""
    if RUN_LOG.exists():
        lines = RUN_LOG.read_text(encoding="utf-8", errors="ignore").splitlines()
        tail = "\n".join(lines[-80:])

    payload = {
        "suite": ACTIVE_SUITE,
        "frontier_commit": frontier_commit,
        "description": candidate.description,
        "status": "crash",
        "run_log_tail": tail,
    }
    artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return artifact_path


def archive_candidate(exp_id: str) -> Path:
    artifact_path = ARTIFACT_DIR / f"{exp_id}.json"
    shutil.move(str(CANDIDATE_JSON), artifact_path)
    return artifact_path


def should_keep(frontier: Frontier, accuracy: float, hard_accuracy: float, avg_time_s: float) -> bool:
    eps = 1e-9
    if accuracy > frontier.accuracy + eps:
        return True
    if abs(accuracy - frontier.accuracy) <= eps and hard_accuracy > frontier.hard_accuracy + eps:
        return True
    if (
        abs(accuracy - frontier.accuracy) <= eps
        and abs(hard_accuracy - frontier.hard_accuracy) <= eps
        and avg_time_s <= frontier.avg_time_s * 0.95
    ):
        return True
    return False


def set_factual_floor(value: float) -> Candidate:
    description = f"eviction_policies: raise factual-weight floor to {value:.2f}"
    return Candidate(
        slug=f"factual-floor-{value:.2f}",
        description=description,
        files=("eviction_policies.py",),
        commit_message=f"autoresearch: raise factual weight floor to {value:.2f}",
        apply=lambda: update_file(
            "eviction_policies.py",
            lambda text: replace_line(
                text,
                r"self\.factual_weight = .+",
                f"self.factual_weight = max(factual_weight, {value:.2f})",
            ),
        ),
    )


def set_query_floor(value: float) -> Candidate:
    description = f"eviction_policies: raise query-weight floor to {value:.2f}"
    return Candidate(
        slug=f"query-floor-{value:.2f}",
        description=description,
        files=("eviction_policies.py",),
        commit_message=f"autoresearch: raise query weight floor to {value:.2f}",
        apply=lambda: update_file(
            "eviction_policies.py",
            lambda text: replace_line(
                text,
                r"self\.query_weight = .+",
                f"self.query_weight = max(query_weight, {value:.2f})",
            ),
        ),
    )


def set_recent_window_floor(value: int) -> Candidate:
    description = f"eviction_policies: enforce recent decode protection floor of {value} tokens"
    return Candidate(
        slug=f"recent-window-{value}",
        description=description,
        files=("eviction_policies.py",),
        commit_message=f"autoresearch: enforce recent window floor of {value}",
        apply=lambda: update_file(
            "eviction_policies.py",
            lambda text: replace_line(
                text,
                r"self\.recent_window_size = .+",
                f"self.recent_window_size = max(recent_window_size, {value})",
            ),
        ),
    )


def set_block_size_cap(value: int) -> Candidate:
    description = f"eviction_policies: cap semantic block size at {value}"
    return Candidate(
        slug=f"block-size-{value}",
        description=description,
        files=("eviction_policies.py",),
        commit_message=f"autoresearch: cap block size at {value}",
        apply=lambda: update_file(
            "eviction_policies.py",
            lambda text: replace_line(
                text,
                r"self\.block_size = .+",
                f"self.block_size = min(block_size, {value})",
            ),
        ),
    )


def set_latest_user_tail(value: int) -> Candidate:
    description = f"eviction_policies: cap latest-user tail pinning at {value}"
    return Candidate(
        slug=f"latest-user-tail-{value}",
        description=description,
        files=("eviction_policies.py",),
        commit_message=f"autoresearch: cap latest-user tail at {value}",
        apply=lambda: update_file(
            "eviction_policies.py",
            lambda text: replace_line(
                text,
                r"self\.latest_user_tail_tokens = .+",
                f"self.latest_user_tail_tokens = min(latest_user_tail_tokens, {value})",
            ),
        ),
    )


def set_block_metric_smallest_mean(k: int) -> Candidate:
    description = f"eviction_policies: rank blocks by the mean of their {k} safest tokens"
    replacement = (
        "block_score = "
        f"scores[candidate_positions].topk(min({k}, candidate_positions.numel()), "
        "largest=False).values.mean().item()"
    )
    return Candidate(
        slug=f"block-metric-smallest-{k}",
        description=description,
        files=("eviction_policies.py",),
        commit_message=f"autoresearch: rank blocks by the {k} safest tokens",
        apply=lambda: update_file(
            "eviction_policies.py",
            lambda text: replace_line(text, r"block_score = .+", replacement),
        ),
    )


def add_day_units() -> Candidate:
    description = "semantic_analyzer: add day and days to factual unit detection"

    def updater(text: str) -> str:
        if '"day",' in text or '"days",' in text:
            raise ValueError("day/day units already present")
        marker = '            "fahrenheit",\n'
        if marker not in text:
            raise ValueError("Could not find fact_units marker")
        return text.replace(marker, marker + '            "day",\n            "days",\n', 1)

    return Candidate(
        slug="semantic-days-units",
        description=description,
        files=("semantic_analyzer.py",),
        commit_message="autoresearch: add day units to factual detection",
        apply=lambda: update_file("semantic_analyzer.py", updater),
    )


def set_fact_unit_weight(value: float) -> Candidate:
    description = f"semantic_analyzer: raise factual unit bonus weight to {value:.2f}"
    return Candidate(
        slug=f"fact-unit-weight-{value:.2f}",
        description=description,
        files=("semantic_analyzer.py",),
        commit_message=f"autoresearch: raise factual unit weight to {value:.2f}",
        apply=lambda: update_file(
            "semantic_analyzer.py",
            lambda text: replace_line(
                text,
                r"\+\s*[0-9.]+\s*\* fact_unit_ratio,",
                f"+ {value:.2f} * fact_unit_ratio,",
            ),
        ),
    )


def combine(candidate_a: Candidate, candidate_b: Candidate) -> Candidate:
    files = tuple(sorted(set(candidate_a.files + candidate_b.files)))
    return Candidate(
        slug=f"{candidate_a.slug}__{candidate_b.slug}",
        description=f"{candidate_a.description} + {candidate_b.description}",
        files=files,
        commit_message=f"autoresearch: {candidate_a.slug} + {candidate_b.slug}",
        apply=lambda: (candidate_a.apply(), candidate_b.apply()),
    )


def structured_candidates() -> list[Candidate]:
    base = [
        set_factual_floor(0.40),
        set_factual_floor(0.45),
        set_query_floor(0.30),
        set_query_floor(0.35),
        set_recent_window_floor(72),
        set_recent_window_floor(80),
        set_block_size_cap(3),
        set_block_size_cap(2),
        set_latest_user_tail(48),
        set_latest_user_tail(64),
        set_block_metric_smallest_mean(2),
        set_block_metric_smallest_mean(3),
        add_day_units(),
        set_fact_unit_weight(0.45),
        set_fact_unit_weight(0.50),
    ]

    combos = [
        combine(set_recent_window_floor(72), set_block_metric_smallest_mean(2)),
        combine(set_factual_floor(0.40), set_fact_unit_weight(0.45)),
        combine(set_query_floor(0.30), add_day_units()),
        combine(set_block_size_cap(3), set_recent_window_floor(72)),
        combine(set_latest_user_tail(64), set_block_metric_smallest_mean(2)),
    ]

    return base + combos


def random_candidate(seed: int) -> Candidate:
    rng = random.Random(seed)
    pools = [
        set_factual_floor(rng.choice([0.36, 0.38, 0.40, 0.42, 0.45])),
        set_query_floor(rng.choice([0.28, 0.30, 0.32, 0.35])),
        set_recent_window_floor(rng.choice([68, 72, 76, 80, 88])),
        set_block_size_cap(rng.choice([2, 3])),
        set_latest_user_tail(rng.choice([48, 52, 60, 64])),
        set_block_metric_smallest_mean(rng.choice([2, 3])),
        set_fact_unit_weight(rng.choice([0.40, 0.45, 0.50])),
    ]
    first = rng.choice(pools)
    second = rng.choice(pools)
    if first.slug == second.slug:
        return first
    return combine(first, second)


def choose_candidate(
    rows: list[dict[str, str]],
    frontier_commit: str,
    skipped_descriptions: set[str],
) -> Candidate:
    for candidate in structured_candidates():
        if (
            not experiment_seen(rows, frontier_commit, candidate.description)
            and candidate.description not in skipped_descriptions
        ):
            return candidate

    seed = len(rows) + 1
    while True:
        candidate = random_candidate(seed)
        seed += 1
        if (
            not experiment_seen(rows, frontier_commit, candidate.description)
            and candidate.description not in skipped_descriptions
        ):
            return candidate


def commit_files(candidate: Candidate) -> None:
    add_result = git_command(["add", *candidate.files])
    if add_result.returncode != 0:
        raise RuntimeError((add_result.stderr or add_result.stdout or "").strip())

    commit_result = git_command(["commit", "-m", candidate.commit_message, "--", *candidate.files])
    if commit_result.returncode != 0:
        raise RuntimeError((commit_result.stderr or commit_result.stdout or "").strip())


def run_one_iteration() -> None:
    rows = load_results()
    frontier = current_frontier(rows)
    exp_id = next_experiment_id(rows)
    candidate = choose_candidate(rows, frontier.commit, skipped_noop_descriptions(frontier.commit))

    update_state(
        mode="running",
        frontier_commit=frontier.commit,
        experiment_id=exp_id,
        candidate=candidate.description,
    )
    log(f"{exp_id}: frontier={frontier.commit} candidate={candidate.description}")

    try:
        candidate.apply()
    except Exception as exc:
        try:
            restore_files(candidate.files)
        except Exception:
            pass
        if "No effective change" in str(exc):
            remember_noop_candidate(frontier.commit, candidate.description)
            log(f"{exp_id}: skipped no-op candidate: {candidate.description}")
            update_state(
                mode="noop_skip",
                experiment_id=exp_id,
                frontier_commit=frontier.commit,
                candidate=candidate.description,
                error=str(exc),
            )
            time.sleep(1)
            return
        log(f"{exp_id}: candidate application failed: {exc}")
        update_state(mode="error", experiment_id=exp_id, error=str(exc))
        time.sleep(5)
        return

    result = benchmark()

    if result.returncode != 0 or not CANDIDATE_JSON.exists():
        artifact_path = create_crash_artifact(exp_id, candidate, frontier.commit)
        append_result(
            {
                "experiment_id": exp_id,
                "frontier_commit": frontier.commit,
                "suite": ACTIVE_SUITE,
                "accuracy": "0.000000",
                "hard_accuracy": "0.000000",
                "avg_time_s": "0.0",
                "status": "crash",
                "description": candidate.description,
                "output_json": str(artifact_path.relative_to(ROOT)),
            }
        )
        try:
            restore_files(candidate.files)
        except Exception as exc:
            log(f"{exp_id}: restore after crash failed: {exc}")
        update_state(mode="crash", experiment_id=exp_id, candidate=candidate.description)
        log(f"{exp_id}: crash logged")
        return

    summary = load_candidate_summary()
    artifact_path = archive_candidate(exp_id)
    accuracy = float(summary["accuracy"])
    hard_accuracy = float(summary["hard_accuracy"])
    avg_time_s = float(summary["avg_time_s"])
    keep = should_keep(frontier, accuracy, hard_accuracy, avg_time_s)
    status = "keep" if keep else "discard"

    append_result(
        {
            "experiment_id": exp_id,
            "frontier_commit": frontier.commit,
            "suite": ACTIVE_SUITE,
            "accuracy": f"{accuracy:.6f}",
            "hard_accuracy": f"{hard_accuracy:.6f}",
            "avg_time_s": f"{avg_time_s:.6f}",
            "status": status,
            "description": candidate.description,
            "output_json": str(artifact_path.relative_to(ROOT)),
        }
    )

    if keep:
        commit_files(candidate)
        new_commit = git_output(["rev-parse", "--short", "HEAD"])
        update_state(
            mode="keep",
            experiment_id=exp_id,
            candidate=candidate.description,
            new_frontier_commit=new_commit,
            accuracy=f"{accuracy:.6f}",
            hard_accuracy=f"{hard_accuracy:.6f}",
            avg_time_s=f"{avg_time_s:.6f}",
        )
        log(
            f"{exp_id}: KEEP -> {new_commit} "
            f"(acc={accuracy:.6f}, hard={hard_accuracy:.6f}, time={avg_time_s:.6f})"
        )
    else:
        restore_files(candidate.files)
        update_state(
            mode="discard",
            experiment_id=exp_id,
            candidate=candidate.description,
            accuracy=f"{accuracy:.6f}",
            hard_accuracy=f"{hard_accuracy:.6f}",
            avg_time_s=f"{avg_time_s:.6f}",
        )
        log(
            f"{exp_id}: discard "
            f"(acc={accuracy:.6f}, hard={hard_accuracy:.6f}, time={avg_time_s:.6f})"
        )


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    log("continuous autoresearch loop started")
    update_state(mode="starting", frontier_commit=git_output(["rev-parse", "--short", "HEAD"]))

    while not STOP_FILE.exists():
        try:
            run_one_iteration()
        except Exception as exc:
            update_state(mode="fatal_error", error=str(exc))
            log(f"fatal error: {exc}")
            time.sleep(15)

    update_state(mode="stopped")
    log("stop file detected; exiting")


if __name__ == "__main__":
    main()
