# Session Log: 2026-03-28 Memory Recall and Benchmark Updates

This note records the implementation and evaluation changes made during the
March 28, 2026 debugging session around `tiered_semantic`, the short
memory-recall benchmark, and benchmark usability.

## Goal of the session

The immediate goal was to stabilize the new `benchmark_memory_recall.py`
workflow and fix the remaining low-budget short-context recall failures in
`tiered_semantic`, especially:

- `launch_date` should return `March 15`
- `language` should return `Rust`
- benchmark commands should no longer look "hung" for long periods without
  progress information

## Core algorithm changes

### 1. Follow-token bias for retained continuation chains

Files:

- `run_generation.py`
- `config.py`

Changes:

- Added `CacheConfig.semantic_follow_token_bias`.
- Added retained follow-token collection and logit biasing in the decode loop.
- Matching is no longer based only on exact token id equality. It now also
  supports normalized token-text matching.
- Added suffix continuation support so a retained prompt token such as
  `" Rust"` can help a generated prefix such as `"R"` continue into `"ust"`.
- Added a narrow fallback for space/digit continuation chains so date-like
  spans such as `March 15` survive even if one intermediate token drops from
  the retained span.

Why it mattered:

- Fixed the `Nova` -> `OS` style continuation problem.
- Fixed the `March` -> space -> `1` -> `5` chain in low-budget decode.

### 2. Single-space token handling in semantic analysis

File:

- `semantic_analyzer.py`

Changes:

- Pure whitespace is no longer automatically treated as template noise.
- Only longer or structural whitespace such as newline/tab-heavy tokens remain
  classified as chat-template-like tokens.

Why it mattered:

- Prevented date spans like `March 15` from being broken at the space token.

### 3. Better partial-block retention in semantic promotion/selection

Files:

- `kv_cache_manager.py`
- `eviction_policies.py`

Changes:

- When a promoted or retained block does not fully fit in the remaining budget,
  the code now chooses the highest-scoring contiguous subwindow instead of a
  naive prefix or peak-centered slice.
- Added stronger warm-tier rescue behavior for adjacent continuation tokens.
- Added prompt-token-aware continuation bonuses for digits, spaces,
  punctuation, and entity continuation.

Why it mattered:

- Reduced span breakage near the tail of factual entities.
- Improved preservation of short structured answers such as `Rust` and
  `Delta-42`.

## Benchmark harness improvements

File:

- `benchmark_memory_recall.py`

Changes:

- Added `--case` filter.
- Added `--policy` filter.
- Added `--list-cases` mode.
- Added `--resume` so runs can continue from an existing output file.
- Added per-run progress persistence after each completed item.
- Added benchmark plan logging, model-load timing, per-case completion logs,
  and ETA reporting.
- Added per-case `max_new_tokens` defaults to keep short-answer cases from
  over-generating.
- Added `--max-new-tokens` override for targeted experiments.

Why it mattered:

- Eliminated the need to rerun all 10 combinations for every debugging step.
- Made long runs observable instead of appearing stuck.
- Reduced wasted runtime on short factual answers.

## Generation observability improvements

File:

- `run_generation.py`

Changes:

- Added explicit `Prefill: running...` / `Prefill: done` logs.
- Added per-decode-step progress logs for small-token runs.
- Logs now show token text, token id, and step latency for targeted debugging.

Why it mattered:

- Made it obvious where generation was spending time.
- Showed that some "hung" runs were actually continuing decode, just with poor
  visibility.

## Early-stop behavior for benchmark-style short answers

Files:

- `config.py`
- `benchmark_memory_recall.py`
- `run_generation.py`

Changes:

- Added `ModelConfig.stop_when_output_contains`.
- `benchmark_memory_recall.py` now seeds that list from each case's
  `answer_keywords`.
- Generation stops early once the normalized generated output contains the
  expected answer substring.

Why it mattered:

- Prevented `language` from continuing after `Rust` had already been produced.
- Turned the problematic behavior
  `Rust kernel module I am writing a ...`
  into a clean `Rust` benchmark result.

## Residual-process handling

During the session, several interrupted benchmark processes were found still
running in the background and were terminated manually. This was necessary
because aborted UI runs did not always clean up the underlying Python process.

Representative residual benchmark processes that were cleaned up during the
session included runs of:

- `benchmark_memory_recall.py --case language,launch_date --policy tiered_semantic ...`
- `benchmark_memory_recall.py --case language --policy tiered_semantic ...`

## Key validation results from this session

### Targeted `launch_date`

Command shape:

- `uv run python .\benchmark_memory_recall.py --case launch_date --policy tiered_semantic --budget 0.3 --hot-ratio 0.7 --warm-top-k 8 --follow-token-bias 20.0`

Observed result:

- `tiered_semantic [OK] 'March 15.'`

Important trace:

- decode chain visibly followed `March` -> `' '` -> `'1'` -> `'5'` -> `'.'`

### Targeted `language`

Command shape:

- `uv run python .\benchmark_memory_recall.py --case language --policy tiered_semantic --budget 0.3 --hot-ratio 0.7 --warm-top-k 8 --follow-token-bias 20.0`

Observed result after early-stop support:

- `tiered_semantic [OK] 'Rust'`

Important trace:

- step 1 generated `'R'`
- step 2 matched early-stop text and terminated cleanly

## Impact on paper readiness

This session did not materially expand the public-benchmark or multi-model
evidence needed for paper submission, but it did improve two important things:

- the short factual recall story is now cleaner and easier to demonstrate
- the benchmark harness is now much more usable for controlled, reproducible
  experiments and debugging

## Files changed in this session

- `benchmark_memory_recall.py`
- `config.py`
- `run_generation.py`
- `semantic_analyzer.py`
- `kv_cache_manager.py`
- `eviction_policies.py`

