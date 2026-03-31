# semanticache autoresearch

This repository adapts the Autoresearch pattern to one narrow goal:

Improve `SemantiCache` on a frozen Needle-in-a-Haystack benchmark without changing the benchmark itself.

The point is not open-ended "AI research". The point is disciplined search inside a tight harness.

## Objective

Turn KV-cache policy tuning into a bounded optimization problem:

1. modify only the semantic eviction strategy
2. run the same fixed benchmark slice
3. measure the same fixed metrics
4. keep the change only if the benchmark improves
5. otherwise restore the search-surface files and move on

The harness matters more than the cleverness of any single idea.

## Setup

For a new run, work with the human to:

1. **Agree on a run tag**: use something like `2026-03-14-a` and create branch `autoresearch/<tag>`.
2. **Create the branch**: branch from the current main working branch.
3. **Read the repo context**:
   - `README.md`
   - `program.md`
   - `config.py`
   - `eval_niah.py`
   - `run_generation.py`
   - `kv_cache_manager.py`
   - `eviction_policies.py`
   - `semantic_analyzer.py`
4. **Verify the environment**:
   - Run `uv run python run_generation.py --test`
   - If this fails, fix environment issues before any experiment loop begins.
5. **Create experiment artifacts**:
   - Create `results/autoresearch/` if missing.
   - Create an untracked `results.tsv` in the repo root if missing.
6. **Initialize `results.tsv`** with this header:

```tsv
experiment_id	frontier_commit	suite	accuracy	hard_accuracy	avg_time_s	status	description	output_json
```

7. **Run the semantic baseline once** before any edits. This establishes the starting frontier.

Do not start searching until setup is complete.

## Search Surface

Keep the editable surface as small as possible.

### Preferred editable file

- `eviction_policies.py`

The default target is `SemantiCachePolicy` only. You may add small local helpers in the same file if needed.

### Secondary editable file

- `semantic_analyzer.py`

Touch this file only when the experiment is explicitly about semantic signal definition, such as:

- role-tag behavior
- information-density scoring
- query relevance scoring
- factual bonus scoring

If you edit `semantic_analyzer.py`, say so explicitly in the experiment description.

### Read-only files

Do not modify these during autoresearch runs:

- `eval_niah.py`
- `run_generation.py`
- `kv_cache_manager.py`
- `config.py`
- `README.md`
- existing files under `results/` except for new run outputs in `results/autoresearch/`
- dependency files such as `pyproject.toml`, `uv.lock`, `requirements.txt`

No new packages. No benchmark edits. No silent harness drift.

## Frozen Evaluation Harness

Use the dedicated harness script:

- `benchmark_autoresearch.py`

It defines two fixed suites:

### Smoke suite

Use only for quick debugging and local sanity checks.

- haystack lengths: `1000`
- budgets: `0.5`, `0.3`
- positions: `0.5`
- needles: all entries in `NEEDLES`
- total cases: `6`

### Frontier suite

Use this for keep/discard decisions.

- haystack lengths: `1000`, `2000`
- budgets: `0.5`, `0.3`, `0.2`
- positions: `0.0`, `0.25`, `0.5`, `0.75`, `1.0`
- needles: all entries in `NEEDLES`
- total cases: `90`

### Gauntlet suite

Use this once `frontier` saturates. It is the current promotion gate.

- adversarial cases with nearby confusable distractors
- haystack lengths: `2000`, `4000`
- budgets: `0.2`, `0.15`, `0.1`
- positions: `0.1`, `0.5`, `0.9`
- total cases: `90`

The policy under optimization remains `semantic`.
Generation remains deterministic (`do_sample = False`).

This is the ground-truth benchmark. Do not widen, shrink, or tweak it mid-run.

## Benchmark Command

Use the `gauntlet` benchmark command for every real experiment now that `frontier` is saturated. Redirect all output to `run.log`.

```powershell
$ErrorActionPreference = 'SilentlyContinue'
uv run python benchmark_autoresearch.py --suite gauntlet --policy semantic --output results/autoresearch/candidate_latest.json *> run.log
exit $LASTEXITCODE
```

After the run, extract the summary with:

```powershell
Select-String -Path run.log -Pattern "^suite_name:", "^suite_cases:", "^suite_accuracy:", "^suite_hard_accuracy:", "^suite_avg_time_s:", "^correct_count:"
```

If the summary lines are missing, treat the run as a crash and inspect the tail:

```powershell
Get-Content run.log -Tail 80
```

## Optimization Target

The acceptance rule is lexicographic:

1. **Primary metric**: `suite_accuracy` on the active promotion suite. Higher is better.
2. **Secondary metric**: `suite_hard_accuracy` on that suite's hardest slice. Higher is better.
3. **Tertiary metric**: `suite_avg_time_s`. Lower is better, but only used if the accuracy metrics are tied.
4. **Simplicity criterion**: if metrics are effectively tied, prefer the simpler change.

This means:

- never keep a change that lowers `suite_accuracy`
- never keep a change that preserves average accuracy but weakens `suite_hard_accuracy`
- only keep a same-accuracy change if it is meaningfully faster or materially simpler
- do not chase tiny speed wins by harming retrieval quality

In practice, use this keep rule:

- keep if `suite_accuracy` is strictly higher than the frontier
- if `suite_accuracy` ties, keep only if `suite_hard_accuracy` is higher
- if both accuracy metrics tie, keep only if `suite_avg_time_s` improves by at least 5 percent or the code is materially simpler
- otherwise discard

## Logging Results

`results.tsv` is untracked. It stores the full experiment history, including failed and discarded attempts.

Columns:

1. `experiment_id`: short local id such as `exp-001`
2. `frontier_commit`: current kept commit before the experiment started
3. `suite`: `smoke` or `frontier`
4. `accuracy`: use `0.000000` for crashes
5. `hard_accuracy`: use `0.000000` for crashes
6. `avg_time_s`: use `0.0` for crashes
7. `status`: `keep`, `discard`, `crash`, or `env_failure`
8. `description`: short text for what changed
9. `output_json`: path to the run artifact in `results/autoresearch/`

Example:

```tsv
experiment_id	frontier_commit	suite	accuracy	hard_accuracy	avg_time_s	status	description	output_json
exp-001	a1b2c3d	smoke	1.000000	1.000000	2.140000	keep	baseline semantic policy	results/autoresearch/exp-001.json
exp-002	a1b2c3d	frontier	0.977778	0.933333	2.560000	keep	broader suite baseline	results/autoresearch/exp-002.json
exp-003	b2c3d4e	frontier	0.966667	0.900000	2.420000	discard	shrink recent window too aggressively	results/autoresearch/exp-003.json
exp-004	b2c3d4e	frontier	0.000000	0.000000	0.0	crash	change signal shape caused tensor mismatch	results/autoresearch/exp-004.json
```

Rename `candidate_latest.json` to the experiment-specific output path after each run.

## Experiment Loop

Loop until the human interrupts you.

For each experiment:

1. Record the current frontier commit with `git rev-parse --short HEAD`.
2. Pick one clear idea.
3. Edit only the allowed search-surface files.
4. Do **not** commit yet.
5. Run the fixed benchmark command into `run.log`.
6. If the run crashes:
   - inspect `run.log`
   - if the issue is a simple bug you just introduced, fix it and rerun
   - if the idea is fundamentally bad, log `crash` and restore the edited files
7. If the run succeeds:
   - parse `suite_accuracy`, `suite_hard_accuracy`, and `suite_avg_time_s`
   - copy `results/autoresearch/candidate_latest.json` to an experiment-specific filename
   - append one row to `results.tsv`
8. Decide:
   - if the result beats the frontier, commit the change and the new output artifact may stay
   - if it does not beat the frontier, restore the edited files and keep the branch on the frontier commit

## Restore Rule

Do not rewrite git history just to discard a candidate.

Because experiments are run before committing, discard by restoring the edited files:

```powershell
git restore --source HEAD -- eviction_policies.py semantic_analyzer.py
```

If only one file was edited, restore only that file.

Commit only winning changes. This keeps the branch history aligned with the frontier.

## Crash Policy

Assume failures will happen.

Common failure classes:

- tensor shape mismatch
- stale semantic signal lengths after eviction
- bad keep-index logic
- attention-dependent policies failing because attentions are missing
- GPU OOM
- environment breakage unrelated to the policy idea

Rules:

- `crash`: the experiment idea broke code or exhausted resources
- `env_failure`: the environment broke for unrelated reasons; do not count this as a policy result

If a crash is clearly caused by the idea, log it and move on.
If the environment is unstable, stabilize the environment first before continuing the loop.

## What Good Experiments Look Like

Good experiments are narrow and test one hypothesis at a time, for example:

- increase factual preservation relative to generic info density
- narrow or widen the recent decode protection window
- change how contiguous blocks are ranked
- make pinned-token behavior less blunt for the latest user span
- reduce overprotection of low-value assistant tokens
- rebalance attention vs semantic signals under the 30 percent budget

Bad experiments:

- changing the benchmark
- changing other policies to make semantic look better
- adding knobs in multiple files with unclear effect
- mixing several unrelated ideas in one candidate

## Baselines

The first kept run under a new suite is always the current semantic implementation baseline.
When a suite reaches `1.000000` accuracy and no longer differentiates candidates, promote the loop to the next harder suite.

Optional reference runs:

- a one-time `full` baseline on the same benchmark slice
- a one-time `h2o` baseline on the same benchmark slice

These are for interpretation only. They are not part of the search loop unless the human explicitly asks for cross-policy optimization.

## Stop Condition

Do not stop just because a run was bad.

Stop only when:

- the human interrupts you
- the environment becomes unusable
- every remaining idea would require widening the search surface beyond this harness

If stuck, do not loosen the harness first. Re-read the fixed files and propose a sharper policy hypothesis.

## Core Principle

Do not ask how to make the agent freer.

Ask how to make the harness tighter:

- fixed benchmark
- narrow search surface
- deterministic evaluation
- cheap discard path
- clear keep rule
- full visibility into every attempt

That is the whole point of autoresearch for this project.
