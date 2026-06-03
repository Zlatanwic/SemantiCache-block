"""Run the OP-SieveKV v2 fixed-manifest main table.

This is a thin orchestrator around the existing evaluators. It keeps the v2
paper workflow reproducible:

1. evaluate fixed NIAH manifest across baselines and OP-SieveKV checkpoints
2. evaluate fixed multi-needle manifest across the same policies
3. write JSON, CI summaries, and optionally append the summaries to the
   experiment log

The runner loads the model once and calls the Python evaluation functions
directly, so it is more convenient on AutoDL than a long chain of shell commands.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from config import ExperimentConfig
from eval_multi_needle import run_multi_needle_suite
from eval_niah import run_manifest as run_niah_manifest
from run_generation import load_model
from summarize_manifest_results import (
    group_key,
    group_label,
    paired_delta,
    summarize_group,
)


DEFAULT_POLICIES = ["full", "streaming", "h2o", "snapkv", "kvzip", "semantic", "op_sievekv_lite"]
DEFAULT_BUDGETS = [0.3, 0.2, 0.1, 0.05]


def parse_labeled_ckpt(values: list[str]) -> list[tuple[str, str]]:
    """Parse LABEL=PATH checkpoint arguments."""
    parsed: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected checkpoint argument as LABEL=PATH, got {value!r}")
        label, path = value.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"Invalid checkpoint argument: {value!r}")
        parsed.append((label, path))
    return parsed


def compressed_policies(policies: list[str]) -> list[str]:
    """Policies that should be evaluated at compressed cache budgets."""
    return [policy for policy in policies if policy not in {"full", "op_sievekv_lite"}]


def relabel_op_policy(rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    """Rename op_sievekv_lite rows so multiple checkpoints can share one JSON table."""
    relabeled: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if item.get("policy") == "op_sievekv_lite":
            item["base_policy"] = "op_sievekv_lite"
            item["policy"] = label
        relabeled.append(item)
    return relabeled


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def build_summary(
    *,
    rows: list[dict[str, Any]],
    input_path: Path,
    baseline_policy: str | None,
    bootstrap_samples: int,
    seed: int,
    title: str,
) -> tuple[str, list[dict[str, Any]]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)

    report: list[dict[str, Any]] = []
    for key in sorted(groups):
        item = {
            "group": group_label(key),
            "policy": key[0],
            "budget": key[1],
            **({"num_needles": key[2]} if len(key) == 3 else {}),
            **summarize_group(groups[key], bootstrap_samples, seed),
        }
        if baseline_policy:
            delta = paired_delta(rows, key, baseline_policy, bootstrap_samples, seed)
            if delta:
                item.update(delta)
        report.append(item)

    lines = [
        f"# {title}",
        "",
        f"- Input: `{input_path}`",
        f"- Rows: {len(rows)}",
        f"- Baseline policy: `{baseline_policy}`" if baseline_policy else "- Baseline policy: none",
        "",
        "| Group | n | Mean | 95% CI | Paired delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in report:
        mean_pct = item["mean"] * 100.0
        ci = f"[{item['ci_low'] * 100.0:.1f}, {item['ci_high'] * 100.0:.1f}]"
        if "delta" in item:
            delta = (
                f"{item['delta'] * 100.0:+.1f} "
                f"[{item['delta_ci_low'] * 100.0:+.1f}, {item['delta_ci_high'] * 100.0:+.1f}] "
                f"(n={item['paired_n']})"
            )
        else:
            delta = ""
        lines.append(f"| {item['group']} | {item['n']} | {mean_pct:.1f} | {ci} | {delta} |")
    return "\n".join(lines) + "\n", report


def write_summary_bundle(
    *,
    rows: list[dict[str, Any]],
    input_path: Path,
    output_prefix: Path,
    baseline_policy: str | None,
    bootstrap_samples: int,
    seed: int,
    title: str,
    append_to: Path | None,
) -> None:
    markdown, report = build_summary(
        rows=rows,
        input_path=input_path,
        baseline_policy=baseline_policy,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        title=title,
    )
    md_path = output_prefix.with_suffix(".md")
    json_path = output_prefix.with_suffix(".summary.json")
    md_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(markdown)
    print(f"Saved summary: {md_path}")
    print(f"Saved summary JSON: {json_path}")

    if append_to:
        append_to.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        with append_to.open("a", encoding="utf-8") as file:
            file.write(f"\n\n## {title}\n\nRecorded: {timestamp}\n\n{markdown}")
        print(f"Appended summary to: {append_to}")


def run_baseline_niah(
    *,
    model,
    tokenizer,
    manifest: Path,
    output: Path,
    policies: list[str],
    budgets: list[float],
    hot_ratio: float,
    warm_top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if "full" in policies:
        rows.extend(
            run_niah_manifest(
                model=model,
                tokenizer=tokenizer,
                manifest_path=manifest,
                output_path=output.with_name(f"{output.stem}_full.json"),
                policies=["full"],
                budgets=[1.0],
                hot_ratio=hot_ratio,
                warm_top_k=warm_top_k,
            )
        )
    compressed = compressed_policies(policies)
    if compressed:
        rows.extend(
            run_niah_manifest(
                model=model,
                tokenizer=tokenizer,
                manifest_path=manifest,
                output_path=output.with_name(f"{output.stem}_compressed.json"),
                policies=compressed,
                budgets=budgets,
                hot_ratio=hot_ratio,
                warm_top_k=warm_top_k,
            )
        )
    write_rows(output, rows)
    return rows


def run_op_niah(
    *,
    model,
    tokenizer,
    manifest: Path,
    output: Path,
    budgets: list[float],
    ckpt_label: str,
    ckpt_path: str,
    hot_ratio: float,
    warm_top_k: int,
) -> list[dict[str, Any]]:
    rows = run_niah_manifest(
        model=model,
        tokenizer=tokenizer,
        manifest_path=manifest,
        output_path=output,
        policies=["op_sievekv_lite"],
        budgets=budgets,
        hot_ratio=hot_ratio,
        warm_top_k=warm_top_k,
        op_policy_ckpt=ckpt_path,
    )
    rows = relabel_op_policy(rows, ckpt_label)
    write_rows(output, rows)
    return rows


def run_baseline_multi(
    *,
    model,
    tokenizer,
    manifest: Path,
    output: Path,
    policies: list[str],
    budgets: list[float],
    target_tokens: int,
    seed: int,
    max_new_tokens: int,
    model_cfg,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if "full" in policies:
        rows.extend(
            run_multi_needle_suite(
                model=model,
                tokenizer=tokenizer,
                policies=["full"],
                budgets=[1.0],
                needle_counts=[2, 4, 8],
                target_tokens=target_tokens,
                seed=seed,
                max_new_tokens=max_new_tokens,
                output_path=str(output.with_name(f"{output.stem}_full.json")),
                model_cfg=model_cfg,
                manifest_path=str(manifest),
            )
        )
    compressed = compressed_policies(policies)
    if compressed:
        rows.extend(
            run_multi_needle_suite(
                model=model,
                tokenizer=tokenizer,
                policies=compressed,
                budgets=budgets,
                needle_counts=[2, 4, 8],
                target_tokens=target_tokens,
                seed=seed,
                max_new_tokens=max_new_tokens,
                output_path=str(output.with_name(f"{output.stem}_compressed.json")),
                model_cfg=model_cfg,
                manifest_path=str(manifest),
            )
        )
    write_rows(output, rows)
    return rows


def run_op_multi(
    *,
    model,
    tokenizer,
    manifest: Path,
    output: Path,
    budgets: list[float],
    ckpt_label: str,
    ckpt_path: str,
    target_tokens: int,
    seed: int,
    max_new_tokens: int,
    model_cfg,
) -> list[dict[str, Any]]:
    rows = run_multi_needle_suite(
        model=model,
        tokenizer=tokenizer,
        policies=["op_sievekv_lite"],
        budgets=budgets,
        needle_counts=[2, 4, 8],
        target_tokens=target_tokens,
        seed=seed,
        max_new_tokens=max_new_tokens,
        output_path=str(output),
        model_cfg=model_cfg,
        op_policy_ckpt=ckpt_path,
        manifest_path=str(manifest),
    )
    rows = relabel_op_policy(rows, ckpt_label)
    write_rows(output, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v2 fixed-manifest main-table evaluations")
    parser.add_argument("--benchmarks", nargs="+", choices=["niah", "multi"], default=["niah", "multi"])
    parser.add_argument("--niah-manifest", default="results/v2/manifests/niah_120.json")
    parser.add_argument("--multi-manifest", default="results/v2/manifests/multi_90.json")
    parser.add_argument("--output-dir", default="results/v2/main_table")
    parser.add_argument("--policies", nargs="+", default=DEFAULT_POLICIES)
    parser.add_argument("--budgets", nargs="+", type=float, default=DEFAULT_BUDGETS)
    parser.add_argument("--include-full-baseline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--op-ckpt",
        action="append",
        default=[],
        help="OP checkpoint as LABEL=PATH. Can be supplied multiple times.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi-target-tokens", type=int, default=1800)
    parser.add_argument("--multi-max-new-tokens", type=int, default=128)
    parser.add_argument("--hot-ratio", type=float, default=0.5)
    parser.add_argument("--warm-top-k", type=int, default=16)
    parser.add_argument("--baseline-policy", default="semantic", help="Policy used for paired deltas in summaries")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--append-to", default=None, help="Append benchmark summaries to this experiment log")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    niah_manifest = Path(args.niah_manifest)
    multi_manifest = Path(args.multi_manifest)
    append_to = Path(args.append_to) if args.append_to else None
    op_ckpts = parse_labeled_ckpt(args.op_ckpt)
    policies = list(args.policies)
    budgets = list(args.budgets)
    if not args.include_full_baseline:
        policies = [policy for policy in policies if policy != "full"]

    planned = {
        "benchmarks": args.benchmarks,
        "policies": policies,
        "budgets": budgets,
        "full_baseline_budget": 1.0 if "full" in policies else None,
        "op_ckpts": [{"label": label, "path": path} for label, path in op_ckpts],
        "niah_manifest": str(niah_manifest),
        "multi_manifest": str(multi_manifest),
        "output_dir": str(output_dir),
    }
    plan_path = output_dir / "v2_main_table_plan.json"
    plan_path.write_text(json.dumps(planned, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(planned, indent=2, ensure_ascii=False))
    print(f"Saved plan: {plan_path}")
    if args.dry_run:
        return

    cfg = ExperimentConfig()
    if args.model:
        cfg.model.model_name = args.model
    cfg.model.do_sample = False
    model, tokenizer = load_model(cfg.model)

    if "niah" in args.benchmarks:
        combined_niah_path = output_dir / "niah_main_table.json"
        combined_rows: list[dict[str, Any]] = load_rows(combined_niah_path) if args.resume else []

        baseline_path = output_dir / "niah_baselines.json"
        if args.resume and baseline_path.exists():
            baseline_rows = load_rows(baseline_path)
            print(f"Resuming NIAH baselines from {baseline_path}")
        else:
            baseline_rows = run_baseline_niah(
                model=model,
                tokenizer=tokenizer,
                manifest=niah_manifest,
                output=baseline_path,
                policies=policies,
                budgets=budgets,
                hot_ratio=args.hot_ratio,
                warm_top_k=args.warm_top_k,
            )
        combined_rows.extend(row for row in baseline_rows if row not in combined_rows)

        for label, ckpt_path in op_ckpts:
            op_path = output_dir / f"niah_{label}.json"
            if args.resume and op_path.exists():
                rows = load_rows(op_path)
                print(f"Resuming NIAH {label} from {op_path}")
            else:
                rows = run_op_niah(
                    model=model,
                    tokenizer=tokenizer,
                    manifest=niah_manifest,
                    output=op_path,
                    budgets=budgets,
                    ckpt_label=label,
                    ckpt_path=ckpt_path,
                    hot_ratio=args.hot_ratio,
                    warm_top_k=args.warm_top_k,
                )
            combined_rows.extend(row for row in rows if row not in combined_rows)

        write_rows(combined_niah_path, combined_rows)
        write_summary_bundle(
            rows=combined_rows,
            input_path=combined_niah_path,
            output_prefix=output_dir / "niah_main_table_summary",
            baseline_policy=args.baseline_policy,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            title="V2 NIAH Fixed-Manifest Main Table",
            append_to=append_to,
        )

    if "multi" in args.benchmarks:
        combined_multi_path = output_dir / "multi_main_table.json"
        combined_rows = load_rows(combined_multi_path) if args.resume else []

        baseline_path = output_dir / "multi_baselines.json"
        if args.resume and baseline_path.exists():
            baseline_rows = load_rows(baseline_path)
            print(f"Resuming multi baselines from {baseline_path}")
        else:
            baseline_rows = run_baseline_multi(
                model=model,
                tokenizer=tokenizer,
                manifest=multi_manifest,
                output=baseline_path,
                policies=policies,
                budgets=budgets,
                target_tokens=args.multi_target_tokens,
                seed=args.seed,
                max_new_tokens=args.multi_max_new_tokens,
                model_cfg=cfg.model,
            )
        combined_rows.extend(row for row in baseline_rows if row not in combined_rows)

        for label, ckpt_path in op_ckpts:
            op_path = output_dir / f"multi_{label}.json"
            if args.resume and op_path.exists():
                rows = load_rows(op_path)
                print(f"Resuming multi {label} from {op_path}")
            else:
                rows = run_op_multi(
                    model=model,
                    tokenizer=tokenizer,
                    manifest=multi_manifest,
                    output=op_path,
                    budgets=budgets,
                    ckpt_label=label,
                    ckpt_path=ckpt_path,
                    target_tokens=args.multi_target_tokens,
                    seed=args.seed,
                    max_new_tokens=args.multi_max_new_tokens,
                    model_cfg=cfg.model,
                )
            combined_rows.extend(row for row in rows if row not in combined_rows)

        write_rows(combined_multi_path, combined_rows)
        write_summary_bundle(
            rows=combined_rows,
            input_path=combined_multi_path,
            output_prefix=output_dir / "multi_main_table_summary",
            baseline_policy=args.baseline_policy,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            title="V2 Multi-Needle Fixed-Manifest Main Table",
            append_to=append_to,
        )


if __name__ == "__main__":
    main()
