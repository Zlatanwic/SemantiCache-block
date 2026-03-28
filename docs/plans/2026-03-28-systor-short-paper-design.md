# SYSTOR 2026 Short Paper Design

## Goal

Submit a stronger `SYSTOR 2026` research-track short paper based on
SemantiCache. The target is not a broad full-paper claim, but a sharp,
well-supported systems result:

> Query-aware semantic KV retention improves retrieval-oriented long-context
> serving under tight memory budgets.

This design assumes a roughly two-month runway and chooses a
`strong-short-paper` strategy: keep the current core method and story, but
significantly strengthen the evidence with systems metrics, ablations, one
public-benchmark slice, one additional model, and cleaner reproducibility.

## Scope

- In:
  - Freeze one submission-grade SemantiCache configuration
  - Add systems-facing metrics beyond task accuracy
  - Add a formal ablation suite
  - Add one public benchmark slice
  - Add one second-model sanity check
  - Rewrite the short paper around a tighter systems claim
  - Prepare anonymous and reproducible submission materials
- Out:
  - Expanding into a full-paper-scale benchmark campaign
  - Continuing open-ended heuristic search on the final benchmark split
  - Treating `benchmark_memory_recall.py` as a main paper result
  - Building a large multi-model serving study

## Core Submission Strategy

The paper should keep `gauntlet` as the main benchmark because it best reflects
the motivating failure mode: preserving authoritative factual spans in long
prompts under strong distractors and tight budgets. However, `gauntlet` alone is
not enough for publication-grade evidence. The correct strategy is therefore:

1. keep `gauntlet` as the primary benchmark,
2. add systems metrics to make the work read like a storage/systems paper,
3. add a limited but credible external validation layer,
4. clearly separate development and final evaluation.

This is the most defensible path because the current repository already has a
strong prototype and a clear failure-driven optimization history, but still
lacks breadth and measurement discipline. The method should be mostly frozen
early, and most remaining effort should go into evidence quality rather than new
algorithmic complexity.

## Design Decisions

### 1. Freeze the method early

The current `tiered_semantic` / SemantiCache frontier is already coherent enough
for a short paper. Continued tuning on the same benchmark family is now more
likely to hurt credibility than help the final submission. We should therefore
choose one frozen submission configuration and define an explicit line between:

- `development suites`
- `final paper suites`

The paper should state that the method was finalized before running the final
tables. A small amount of post-freeze engineering is still acceptable if it
improves observability or reproducibility without changing the core ranking
logic, but any semantic-selection change after the freeze point should be
treated as a new method revision and avoided unless a severe flaw is found.

### 2. Reframe evaluation around system tradeoffs

SYSTOR reviewers will likely care less about raw benchmark cleverness and more
about whether the method exposes a convincing tradeoff between memory, latency,
and answer quality. The next experimental layer should therefore measure:

- task accuracy
- hard-slice accuracy
- end-to-end latency
- estimated retained KV size
- percentage KV reduction relative to full cache
- retention/eviction overhead

If feasible, a simple throughput-oriented proxy can be added later, but it is
not mandatory for the short-paper target. The important shift is that the paper
must no longer read as “we built a smarter heuristic,” but as “we characterized
a serving tradeoff and showed that query-aware retention improves the frontier.”

### 3. Add breadth, but only where it matters

The goal is not broad benchmark saturation. Instead, the paper needs just enough
external validation to neutralize the most predictable reviewer concerns:

- “this only works on the authors’ custom benchmark”
- “this may only work on one tokenizer or one model family”

The solution is a narrow external-validation layer:

- one public long-context benchmark slice that matches the retrieval-heavy
  problem setting,
- one additional model sanity check,
- one compact failure taxonomy with representative examples.

This is enough to strengthen the external-validity argument while keeping the
project within a short-paper scope.

## Experimental Architecture

The evaluation should be organized into four tiers.

### Tier A: Main benchmark

Primary paper benchmark:

- `gauntlet`

Purpose:

- show the main win under adversarial distractors and tight budgets

Outputs:

- main accuracy table
- hard-slice table
- accuracy-vs-budget figure
- latency/memory/overhead tradeoff table

### Tier B: Ablation benchmark

Purpose:

- isolate which components matter

Ablations to include:

- remove query relevance
- remove factual signal
- remove role-aware protection
- remove contiguous block retention
- remove partial-block refinement

Outputs:

- one compact ablation table

### Tier C: External validation

Purpose:

- show the behavior is not confined to `gauntlet`

Design:

- select one public benchmark slice aligned with retrieval-heavy or fact-heavy
  long-context tasks
- run only the strongest baselines plus frozen SemantiCache

Outputs:

- one external-validation table

### Tier D: Diagnostic appendix

Purpose:

- explain failure modes and debugging insights

Design:

- keep `benchmark_memory_recall.py` and short-answer traces in appendix or
  supplemental material only

Outputs:

- one appendix subsection on short structured-answer failures
- one appendix subsection on tokenizer/continuation issues

## Writing Structure

The paper should be written as a short systems result, not as a sprawling
benchmark compendium.

Recommended structure:

1. Problem and motivation
2. Why generic recency/salience fails for retrieval-oriented serving
3. SemantiCache design
4. Experimental setup
5. Main results
6. Ablation and limited generalization
7. Limitations and discussion

Claims should be tightly bounded. Avoid claiming universal KV-cache superiority.
Instead claim that, for retrieval-oriented long-context workloads under fixed
budgets, query-aware semantic retention improves the quality-memory tradeoff and
that retention shape is crucial for decode stability.

## Timeline

### Phase 1: Freeze and instrumentation

Target window:

- Week 1

Deliverables:

- frozen method configuration
- dev/final evaluation split
- systems-metric instrumentation
- finalized main experiment matrix

### Phase 2: Main results and ablations

Target window:

- Weeks 2-3

Deliverables:

- final `gauntlet` runs
- ablation runs
- main figures and tables

### Phase 3: External validation

Target window:

- Weeks 4-5

Deliverables:

- one public benchmark slice
- one second-model sanity check
- failure taxonomy

### Phase 4: Paper compression and submission prep

Target window:

- Weeks 6-8

Deliverables:

- rewritten short paper
- appendix-ready changelog and diagnostics
- anonymous reproducibility bundle
- final formatting and submission rehearsal

## Concrete Work Plan

- Freeze the SemantiCache submission configuration and record it in a single
  source of truth.
- Add memory-retention and overhead metrics to the current benchmark outputs.
- Build one script or notebook that produces paper-ready summary tables from raw
  result files.
- Run the main `gauntlet` comparison using the frozen method and common
  baselines.
- Run the five core ablations against the same final protocol.
- Choose and implement one public benchmark slice aligned with the paper’s
  retrieval-oriented workload.
- Run one second-model sanity check on a reduced matrix.
- Rewrite the short paper around the quality-memory tradeoff rather than around
  heuristic evolution.
- Prepare figures and tables early so the paper text conforms to what the data
  actually says.
- Build anonymous reproduction instructions and verify them before the final
  writing pass.

## Risks and Controls

### Risk: overfitting to `gauntlet`

Control:

- freeze the method early
- separate dev and final suites
- add one public benchmark slice

### Risk: too many experiments, not enough paper

Control:

- restrict external validation to one benchmark slice and one second model
- prioritize paper figures over exploratory runs

### Risk: systems story remains too weak

Control:

- measure memory and overhead explicitly
- report tradeoffs, not just accuracy

### Risk: short paper becomes overclaimed

Control:

- keep claims retrieval-oriented and budget-oriented
- put exploratory diagnostics in appendix, not in the main results core

## Success Criteria

This design is successful if, before submission, the project has:

- one frozen SemantiCache submission version
- one main result table on `gauntlet`
- one systems tradeoff table or figure
- one formal ablation table
- one external-validation table
- one second-model sanity-check result
- one concise, defensible short paper that stays within scope

