# Fixed-Pipeline NIAH Diagnostics

Configuration:
- Needle position: `0.5`
- Cache budget: `100%`
- Model: `Qwen/Qwen2.5-3B-Instruct`
- Pipeline fixes included:
  - explicit `cache_position` / `position_ids`
  - contiguous semantic eviction
  - latest-user tail pinning

| Policy | Haystack | Correct | Generated Tokens | Total Evicted | Time (s) | Result File |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| `full` | 1000 | `true` | 21 | 0 | 2.26 | `results/diag_full_1000.json` |
| `full` | 2000 | `true` | 21 | 0 | 2.55 | `results/diag_full_2000.json` |
| `window` | 1000 | `true` | 32 | 30 | 3.17 | `results/diag_window_1000_v2.json` |
| `window` | 2000 | `true` | 21 | 19 | 2.72 | `results/diag_window_2000_v2.json` |
| `streaming` | 1000 | `true` | 21 | 19 | 2.11 | `results/diag_streaming_1000_v2.json` |
| `streaming` | 2000 | `true` | 21 | 19 | 3.15 | `results/diag_streaming_2000_v2.json` |
| `h2o` | 1000 | `true` | 21 | 19 | 2.82 | `results/diag_h2o_1000_v2.json` |
| `h2o` | 2000 | `true` | 21 | 19 | 2.52 | `results/diag_h2o_2000_v2.json` |
| `semantic` | 1000 | `true` | 21 | 19 | 2.14 | `results/diag_semantic_1000_v4.json` |
| `semantic` | 2000 | `true` | 21 | 19 | 2.30 | `results/diag_semantic_2000_v4.json` |

Notes:
- Before the pipeline fixes, all pruning-based methods showed degeneration or repetition on this setup.
- After adding explicit position handling, contiguous semantic blocks, and narrower latest-user pinning, all five policies are stable on this diagnostic slice.
- `window@1000` still generated a longer answer tail, but it preserved the correct needle answer and was scored as correct.
