# SYSTOR 2026 Short Paper Implementation Plan

This implementation plan translates the design in
`docs/plans/2026-03-28-systor-short-paper-design.md` into a concrete execution
sequence. It is intentionally scoped for a strong short-paper submission rather
than a full-paper campaign.

## Approach

The fastest path to a credible `SYSTOR 2026` short paper is to freeze the core
SemantiCache method early and spend the remaining time improving evidence
quality rather than reopening broad method search. Implementation therefore
follows four tracks in order: freeze, measure, validate, and write.

The central rule is simple: after the freeze point, only make changes that
improve instrumentation, reproducibility, or presentation. New retention logic
after that point should be treated as out of scope unless it fixes a critical
correctness issue.

## Scope

- In:
  - Freeze one submission configuration for SemantiCache
  - Instrument system-facing metrics in the main benchmark harness
  - Generate final comparison and ablation tables
  - Add one public benchmark slice and one second-model sanity check
  - Produce paper-ready figures, appendix notes, and reproducibility materials
- Out:
  - More open-ended heuristic frontier search on final paper suites
  - Large multi-model or multi-dataset campaigns
  - New benchmark families beyond one external slice
  - Expanding the short memory-recall benchmark into a main paper result

## Phase 1: Freeze and Instrumentation

### Objective

Create a stable paper version and make the main benchmark measure the quantities
that matter for a systems venue.

### Tasks

[ ] Create a single source of truth for the submission configuration in a new
paper-facing config note, referencing the exact fields in [config.py](/d:/semanticache/config.py) and the relevant behavior in [eviction_policies.py](/d:/semanticache/eviction_policies.py), [kv_cache_manager.py](/d:/semanticache/kv_cache_manager.py), and [semantic_analyzer.py](/d:/semanticache/semantic_analyzer.py).

[ ] Define and document the `development` versus `final paper` split for
`frontier`, `gauntlet`, and diagnostics in [benchmark_autoresearch.py](/d:/semanticache/benchmark_autoresearch.py) and a short companion note under `docs/plans/`.

[ ] Add systems-facing metrics to the main benchmark output in
[benchmark_autoresearch.py](/d:/semanticache/benchmark_autoresearch.py), using
the stats already exposed by [run_generation.py](/d:/semanticache/run_generation.py) and [kv_cache_manager.py](/d:/semanticache/kv_cache_manager.py). Minimum metrics:
- accuracy
- hard-slice accuracy
- end-to-end latency
- retained cache length
- retained cache ratio versus prompt length
- eviction/materialization overhead

[ ] Add a paper-summary exporter that reads raw JSON result files from
[results/autoresearch](/d:/semanticache/results/autoresearch) and emits one
machine-readable summary table for later plotting or LaTeX inclusion.

[ ] Record the freeze point in a persistent note that links the submission
configuration, benchmark protocol, and expected output files.

### Validation Gate

- The main benchmark can run with frozen settings and emits all required systems
  metrics to disk.
- The submission configuration is explicitly documented and no longer ambiguous.

## Phase 2: Main Results and Ablations

### Objective

Produce the core evidence that will appear in the main body of the short paper.

### Tasks

[ ] Run the final `gauntlet` comparison for the paper baselines:
- `full`
- `window`
- `streaming`
- `h2o`
- frozen SemantiCache

[ ] Save each final run to a clearly named output file in
[results/autoresearch](/d:/semanticache/results/autoresearch) and generate one
paper-summary table from them.

[ ] Implement and run five focused ablations on the frozen protocol:
- remove query relevance
- remove factual signal
- remove role-aware protection
- remove contiguous block retention
- remove partial-block refinement

[ ] Add one plotting or table-generation script that converts the frozen result
files into:
- a main results table
- an accuracy-vs-budget figure
- an overhead or memory-tradeoff table
- an ablation table

[ ] Write a short interpretation note for each final table so the eventual paper
text can be grounded in the observed result rather than in memory.

### Validation Gate

- One main benchmark table exists and is reproducible from raw outputs.
- One ablation table exists and isolates each major design component.
- The paper’s central claim can be expressed using these tables alone.

## Phase 3: External Validation

### Objective

Reduce the most predictable reviewer objections around benchmark specialization
and single-model dependence.

### Tasks

[ ] Select one public long-context benchmark slice aligned with
retrieval-oriented or fact-heavy workloads. Document the selection rationale in
a short note before implementation.

[ ] Add the minimum required runner or adapter for that benchmark, reusing the
current generation path in [run_generation.py](/d:/semanticache/run_generation.py) rather than creating a separate inference implementation.

[ ] Evaluate only the strongest baselines and frozen SemantiCache on this
external slice to keep runtime and scope under control.

[ ] Select one second model for a reduced sanity-check matrix. Prioritize a
model that differs in tokenizer behavior or family, but is still practical to
run on current hardware.

[ ] Run a reduced matrix on that second model using the same frozen method and
report relative trend consistency rather than attempting a full large-scale
study.

[ ] Build a short failure taxonomy note based on final errors from
`gauntlet`, the external slice, and the short memory-recall diagnostics. Group
failures into a small number of repeatable categories.

### Validation Gate

- One external-validation table exists.
- One second-model sanity-check result exists.
- One failure taxonomy exists with representative examples.

## Phase 4: Paper Compression and Submission Preparation

### Objective

Turn the experimental evidence into a compact, defensible short paper and a
clean submission package.

### Tasks

[ ] Rewrite [systor_short_paper_draft.md](/d:/semanticache/systor_short_paper_draft.md) to align with the frozen final evidence and the narrower systems claim.

[ ] Update [systor_short_paper_acm.tex](/d:/semanticache/systor_short_paper_acm.tex) so that every figure, table, and claim points to a specific final result file or summary artifact.

[ ] Keep `gauntlet` as the main benchmark in the paper and move
[benchmark_memory_recall.py](/d:/semanticache/benchmark_memory_recall.py) plus
its findings into appendix or supplemental material.

[ ] Reuse [results/research_changelog_appendix_ready.md](/d:/semanticache/results/research_changelog_appendix_ready.md) and [results/session_2026-03-28_memory_recall_and_benchmark_updates.md](/d:/semanticache/results/session_2026-03-28_memory_recall_and_benchmark_updates.md) to draft appendix prose on late-stage prototype refinement.

[ ] Prepare anonymous reproducibility instructions covering:
- environment setup
- exact benchmark commands
- result file mapping
- figure/table regeneration commands

[ ] Run a final submission rehearsal:
- build the paper cleanly
- verify page limit
- verify double-anonymous compliance
- verify every referenced artifact exists and is reproducible

### Validation Gate

- The paper compiles and stays within short-paper limits.
- All main figures and tables are reproducible from named result files.
- Supplemental notes are clean enough to upload or convert into an anonymous artifact.

## Concrete Deliverables

- `docs/plans/<date>-submission-freeze-note.md`
- one updated main benchmark harness with systems metrics
- one paper-summary export script or notebook
- final baseline result JSON files
- final ablation result JSON files
- one external-validation result file
- one second-model sanity-check result file
- one failure-taxonomy note
- one revised short paper draft
- one reproducibility note for supplemental material

## Execution Order

1. Freeze configuration and split evaluation.
2. Instrument systems metrics in the main harness.
3. Run final main benchmark.
4. Run ablations.
5. Add one external benchmark slice.
6. Run one second-model sanity check.
7. Generate final figures and tables.
8. Rewrite the paper.
9. Prepare anonymous reproducibility material.
10. Rehearse submission end to end.

## Working Rules

- Do not continue method search on the final paper split after the freeze note.
- Treat `benchmark_memory_recall.py` as a diagnostic appendix tool only.
- Prefer one clean, reproducible table over three exploratory half-finished ones.
- If a new experiment does not clearly strengthen the short-paper argument, drop it.

## Immediate Next Move

The next concrete implementation step is:

- add systems metrics and freeze-note scaffolding around
  [benchmark_autoresearch.py](/d:/semanticache/benchmark_autoresearch.py)

This is the highest-leverage next move because it creates the measurement
backbone for the main table, the ablations, and the eventual paper claims.

