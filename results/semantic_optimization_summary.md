# SemantiCache Optimization Summary

## 1. Budget Sensitivity on NIAH

Common setup:
- Model: `Qwen/Qwen2.5-3B-Instruct`
- Haystack length: `1000`
- Needle position: `0.5`
- Task: recover `Operation Starlight` and `March 15`

| Policy | Budget | Correct | Generated Tokens | Total Evicted | Time (s) | Representative Behavior | Result File |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| `window` | 50% | `false` | 128 | 465 | 8.37 | Degenerates into punctuation / blanks | `results/diag_window_1000_b50.json` |
| `streaming` | 50% | `false` | 128 | 465 | 8.42 | Repetitive loop, fails to recover needle | `results/diag_streaming_1000_b50.json` |
| `h2o` | 50% | `true` | 21 | 357 | 2.03 | Recovers full factual answer | `results/diag_h2o_1000_b50.json` |
| `semantic` | 50% | `true` | 21 | 357 | 2.06 | Recovers full factual answer | `results/diag_semantic_1000_b50.json` |
| `window` | 30% | `false` | 128 | 601 | 11.22 | Severe collapse | `results/diag_window_1000_b30.json` |
| `streaming` | 30% | `false` | 70 | 542 | 6.55 | Repetitive and incomplete | `results/diag_streaming_1000_b30.json` |
| `h2o` | 30% | `false` | 20 | 492 | 2.00 | Nearly correct, but outputs `March 1st` | `results/diag_h2o_1000_b30.json` |
| `semantic` (before query-aware) | 30% | `false` | 105 | 577 | 9.93 | Stable but conservative retrieval failure | `results/diag_semantic_1000_b30.json` |
| `semantic` (query-aware) | 30% | `true` | 21 | 493 | 2.16 | Recovers full factual answer | `results/diag_semantic_1000_b30_v2.json` |

### Main takeaway

- At `50%` budget, recency-based methods (`window`, `streaming`) already fail, while importance-based methods (`h2o`, `semantic`) still succeed.
- At `30%` budget, naive `semantic` also fails, but after adding query-aware factual preservation it recovers the full answer again.
- This is the strongest evidence so far that **query-aware semantic retention** matters under tight KV budgets.

## 2. SemantiCache Version Evolution

Representative setup:
- Policy: `semantic`
- Budget: `100%` unless otherwise noted
- Haystack length: `1000`
- Needle position: `0.5`

| Version | Key Change | Outcome | Observed Behavior | Result File |
| --- | --- | --- | --- | --- |
| `v1` | Initial sparse token-level semantic pruning | `false` | Repetition collapse; generated 128 tokens | `results/diag_semantic_1000_v2.json` |
| `v2` | Added explicit `cache_position` / `position_ids` | `false` | Still unstable under sparse pruning | `results/diag_semantic_1000_v3.json` |
| `v3` | Switched to contiguous block retention + narrower latest-user pinning | `true` | Stable generation, correct answer at 100% budget | `results/diag_semantic_1000_v4.json` |
| `v4` | Added query relevance + factual bonus | `true` at 30% | Recovers full answer even at tight budget | `results/diag_semantic_1000_b30_v2.json` |

### Main takeaway

- Fixing **position consistency** was necessary but not sufficient.
- Fixing **retention shape** (contiguous blocks instead of sparse token holes) restored stability.
- Fixing **selection target** (query-aware factual spans instead of generic semantic importance) restored low-budget recall.

## 3. Suggested Defense Narrative

You can summarize the full optimization path like this:

1. The first implementation failed because naive token-level pruning broke generation stability.
2. We diagnosed that the problem was not only "which tokens to evict", but also "how positions and retained structure are preserved".
3. After adding explicit position handling and switching to contiguous semantic retention, the method became stable.
4. Under tighter budgets, generic semantic importance was still not enough, so we added query-aware and factual signals.
5. This final version preserved both stability and retrieval quality under low KV budgets.
