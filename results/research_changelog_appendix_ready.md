# Research Changelog for Paper Appendix

This document summarizes recent SemantiCache changes in an appendix-friendly
format. Each entry is organized by:

- `Method`: what changed in the algorithm or system
- `Benchmark`: what evaluation surface or protocol was affected
- `Evidence`: what concrete observation supported the change
- `Impact`: why the change matters for the paper narrative

The focus here is the 2026-03-28 update cycle around short-answer factual
recall, low-budget decode stability, and benchmark usability.

## Changelog Table

| ID | Method | Benchmark | Evidence | Impact |
| --- | --- | --- | --- | --- |
| RC-01 | Added retained follow-token biasing during decode in `tiered_semantic`. The next-token logits are now biased toward tokens that historically followed retained prompt tokens. | Targeted short factual recall cases; especially `project_name`, `launch_date`, and `language`. | Exact token-id matching alone was insufficient for cases where the prompt used a fused token and generation produced a split prefix. This showed up as failures such as `Nova` failing to continue into `OS`, and date chains breaking after `March`. | Establishes that decode-time continuation support is necessary in addition to prefill-time retention. This is evidence that retention quality alone does not guarantee short factual recall. |
| RC-02 | Extended follow-token matching from exact token ids to normalized token-text matching. | Short memory-recall benchmark under `tiered_semantic`. | Cases with equivalent text but different tokenizer boundaries were still failing under exact-id logic. Normalized matching removed this brittleness. | Supports the paper's claim that workload-aware retention must tolerate tokenizer-specific fragmentation. |
| RC-03 | Added suffix continuation support so a retained fused token such as `" Rust"` can guide generation from `"R"` to `"ust"`. | `language` case in the short memory-recall benchmark. | The failing trace showed generation beginning with `R`, while the retained factual token in the prompt was fused as `" Rust"`. | Provides a concrete systems fix for a tokenizer-induced decode failure. This strengthens the argument that cache management and decoding interact in nontrivial ways. |
| RC-04 | Added a narrow fallback for space/digit continuation chains when retained spans lose one intermediate link. | `launch_date` case under low budget. | The model could generate `March`, but the continuation into `15` was unstable if the retained span lost the space or digit bridge. After the fallback, the decode trace visibly followed `March` -> `' '` -> `'1'` -> `'5'`. | Shows that short structured facts such as dates require preserving local token adjacency, not just semantic relevance. |
| RC-05 | Changed semantic whitespace handling so a single space token is no longer automatically treated as template noise. | `launch_date` and other structured short-answer cases. | Date-like spans were being fragmented at the space token even when surrounding content was retained. After the change, `March 15.` was recovered cleanly. | Strengthens the paper's structural-retention story by showing that low-level token typing decisions affect factual recall. |
| RC-06 | Reworked partial-block truncation to choose the highest-scoring contiguous subwindow instead of a naive prefix or peak-centered slice. | Low-budget `tiered_semantic` retention and promotion. | Tail-heavy factual entities and short spans were being clipped when a block partially fit within the remaining budget. | Supports the claim that retention shape is a first-class systems issue. The result is more faithful preservation of compact factual spans. |
| RC-07 | Improved warm-tier rescue behavior and adjacency-aware continuation rescue for short structured spans. | Low-budget short factual recall; especially identifiers, dates, and mixed alphanumeric answers. | Some failures preserved the topic span but dropped one adjacent continuation token, producing incomplete or drifted answers. | Helps explain why SemantiCache moved beyond pure scoring into structure-aware retention and rescue logic. |
| RC-08 | Added a dedicated short memory-recall benchmark for conversational factual recall under constrained cache budgets. | New benchmark surface: `benchmark_memory_recall.py`. | The existing gauntlet and NIAH-style experiments were useful but too coarse for quickly isolating short factual decode failures. | Gives the paper an additional diagnostic tool for studying failure modes such as short answer continuation, over-generation, and tokenizer fragmentation. |
| RC-09 | Added `--case` and `--policy` filtering to the short memory-recall benchmark. | `benchmark_memory_recall.py` evaluation workflow. | Full benchmark runs were too slow for targeted iteration and often encouraged repeated reruns of unrelated cases. | Improves reproducibility and debugging discipline by allowing single-case, single-policy evaluation. This is useful for appendix and artifact documentation. |
| RC-10 | Added `--resume`, on-disk progress persistence, and partial-run recovery for the short memory-recall benchmark. | Long-running or interrupted benchmark sessions. | Interrupted UI runs could leave Python processes alive and produce no final summary, wasting time and hiding partial progress. | Makes the evaluation harness more robust and artifact-friendly. This is helpful for supplemental reproducibility material. |
| RC-11 | Added benchmark plan logging, model-load timing, per-case completion logs, and ETA reporting. | `benchmark_memory_recall.py` runtime observability. | Before this change, long runs appeared to hang because model loading and decode progress were opaque. | Improves observability and reduces ambiguity between "slow" and "stuck" runs, which is important for stable experimental practice. |
| RC-12 | Added prefill and per-decode-step progress logging in `run_generation.py` for small-token runs. | Targeted debugging runs of short-answer cases. | A key `language` trace showed `R`, then `ust`, then ` kernel`, making it clear that the answer had already been produced before the run appeared to stall. | Provides direct evidence for where time is spent and why some runs over-generate. This is useful appendix material for explaining debugging methodology. |
| RC-13 | Introduced per-case `max_new_tokens` defaults in the short memory-recall benchmark. | All short factual benchmark cases. | Uniform `max_new_tokens=8` was excessive for answers like `Rust` or `NovaOS`, increasing runtime and over-generation risk. | Aligns benchmark generation length with task design, reducing evaluation noise and making short-answer measurements more faithful. |
| RC-14 | Added benchmark-oriented early stopping once the generated output already contains the expected answer substring. | `language` and other short-answer memory-recall cases. | The `language` case produced `R` then `ust`, meaning the correct answer was already present by step 2, but generation continued into ` kernel ...` and sometimes appeared stuck. After early stop, the case terminated cleanly with `Rust`. | Tightens the link between benchmark objective and generation protocol. This is especially important for short-answer diagnostics where additional tokens are not meaningful signal. |
| RC-15 | Recorded and cleaned residual benchmark processes left alive after interrupted runs. | Practical benchmarking workflow on Windows/PowerShell. | Multiple interrupted `benchmark_memory_recall.py` commands remained active in the background and continued consuming CPU/GPU resources. | Improves experimental hygiene and explains prior wall-clock anomalies in benchmark timing. This is useful context for artifact and evaluation notes. |

## Key Validated Outcomes

### Outcome A: `launch_date` recovery

Configuration:

- policy: `tiered_semantic`
- budget: `0.3`
- hot ratio: `0.7`
- warm top-k: `8`
- follow-token bias: `20.0`

Observed output:

- `March 15.`

Interpretation:

- The combination of single-space preservation, follow-token biasing, and
  space/digit continuation fallback was sufficient to stabilize short date
  generation under low budget.

### Outcome B: `language` recovery

Configuration:

- policy: `tiered_semantic`
- budget: `0.3`
- hot ratio: `0.7`
- warm top-k: `8`
- follow-token bias: `20.0`

Observed output after early-stop support:

- `Rust`

Interpretation:

- The retained factual signal plus fused-token suffix continuation fixed the
  initial tokenizer mismatch.
- Early stopping then prevented the benchmark from drifting into unnecessary
  continuation text such as `kernel ...`.

## How to cite these changes in the paper

If needed, these entries can be summarized in appendix prose as follows:

> During late-stage prototype refinement, we observed that low-budget factual
> failures were often caused not by complete loss of the supporting span, but by
> decode-time fragmentation at short continuations such as suffix tokens, spaces,
> and digits. We therefore augmented SemantiCache with continuation-aware
> follow-token biasing, less aggressive whitespace filtering, and improved
> contiguous subwindow selection for partial-block retention. In parallel, we
> extended the evaluation harness with a targeted short memory-recall benchmark,
> resume/progress support, and short-answer early stopping, which made these
> failure modes directly observable and reproducible.

## Related internal notes

- `results/session_2026-03-28_memory_recall_and_benchmark_updates.md`
- `results/semantic_optimization_summary.md`
- `benchmark_memory_recall.py`
- `run_generation.py`
- `semantic_analyzer.py`
- `kv_cache_manager.py`
- `eviction_policies.py`
- `config.py`

