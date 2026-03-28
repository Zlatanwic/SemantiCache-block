"""Run the full main-table benchmark for the SYSTOR paper."""

import json
import time
from pathlib import Path

from config import ExperimentConfig
from eval_niah import NEEDLES, run_single_eval
from run_generation import load_model

POLICIES = ["full", "h2o", "semantic"]
BUDGETS = [0.5, 0.3, 0.2]
HAYSTACK_LENGTHS = [1000, 2000]
POSITIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
MAX_NEW_TOKENS = 64

output_dir = Path("results/main_table")
output_dir.mkdir(parents=True, exist_ok=True)

cfg = ExperimentConfig()
model, tokenizer = load_model(cfg.model)

total_runs = len(POLICIES) * len(BUDGETS) * len(HAYSTACK_LENGTHS) * len(POSITIONS) * len(NEEDLES)
print(f"Total planned runs: {total_runs}")

all_results = []
run_idx = 0
t0 = time.time()

for policy in POLICIES:
    for budget in BUDGETS:
        for hl in HAYSTACK_LENGTHS:
            for pos in POSITIONS:
                for ni, needle in enumerate(NEEDLES):
                    run_idx += 1
                    elapsed = time.time() - t0
                    eta = (elapsed / run_idx) * (total_runs - run_idx) if run_idx > 1 else 0
                    tag_prefix = f"[{run_idx}/{total_runs}]"
                    print(
                        f"{tag_prefix} policy={policy} budget={budget:.0%} "
                        f"len={hl} pos={pos:.2f} needle_{ni+1}  "
                        f"(elapsed={elapsed:.0f}s eta={eta:.0f}s)"
                    )
                    c = ExperimentConfig()
                    c.cache.policy = policy
                    c.cache.cache_budget = budget
                    c.model.do_sample = False
                    c.model.max_new_tokens = MAX_NEW_TOKENS
                    c.model.show_progress_bar = False
                    result = run_single_eval(
                        model, tokenizer, c, needle,
                        haystack_length=hl,
                        needle_position=pos,
                    )
                    result["policy"] = policy
                    result["budget"] = budget
                    result["haystack_length"] = hl
                    result["needle_position"] = pos
                    result["needle_index"] = ni + 1
                    all_results.append(result)
                    ok = "O" if result["correct"] else "X"
                    print(f"  [{ok}] {result['output_text'][:60]}")

    # Save after each policy completes
    partial_path = output_dir / f"main_table_partial_{policy}.json"
    policy_results = [r for r in all_results if r["policy"] == policy]
    partial_path.write_text(json.dumps(policy_results, indent=2), encoding="utf-8")
    correct = sum(1 for r in policy_results if r["correct"])
    print(f"\n=== {policy} done: {correct}/{len(policy_results)} correct ===\n")

# Final save
final_path = output_dir / "main_table_all.json"
final_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

# Print summary table
print("\n" + "=" * 80)
print("MAIN TABLE SUMMARY")
print("=" * 80)
header = f"{'Policy':<12} {'Budget':>6} {'HL':>5}  | {'pos=0':>5} {'0.25':>5} {'0.5':>5} {'0.75':>5} {'1.0':>5} | {'Total':>6}"
print(header)
print("-" * len(header))

for policy in POLICIES:
    for budget in BUDGETS:
        for hl in HAYSTACK_LENGTHS:
            pos_acc = []
            for pos in POSITIONS:
                subset = [
                    r for r in all_results
                    if r["policy"] == policy
                    and r["budget"] == budget
                    and r["haystack_length"] == hl
                    and r["needle_position"] == pos
                ]
                c = sum(1 for r in subset if r["correct"])
                pos_acc.append(f"{c}/{len(subset)}")
            total_subset = [
                r for r in all_results
                if r["policy"] == policy
                and r["budget"] == budget
                and r["haystack_length"] == hl
            ]
            tc = sum(1 for r in total_subset if r["correct"])
            tt = len(total_subset)
            print(
                f"{policy:<12} {budget:>5.0%} {hl:>5}  | "
                + " ".join(f"{a:>5}" for a in pos_acc)
                + f" | {tc:>2}/{tt:<2} {tc/tt:.0%}"
            )
        print()

total_elapsed = time.time() - t0
print(f"\nTotal time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
print(f"Saved to: {final_path}")
