# OP-SieveKV v2 Experiment Summary

Updated: 2026-05-28

## Current State

This note records the current OP-SieveKV-Lite / distillation status and compares it with the original plan in `research_plan/research_plan.md`.

## Implemented

- Added `op_sievekv_lite` as an executable cache policy.
- Added semantic segment construction at sentence / role-aware level, with segment-level features.
- Added OP-Lite heuristic scoring with recent-window protection, role pinning, query/evidence-style signals, and learned policy checkpoint loading.
- Added basic KV-TIP-style diagnostic scaffolding.
- Added NIAH and multi-needle evaluation support for `op_sievekv_lite`.
- Added distillation data collection pipeline:
  - `collect_op_distill_data.py`
  - NIAH + multi-needle examples
  - offset-based evidence span matching
  - soft labels from evidence, support tokens, recent/query/system/boundary support, and heuristic teacher support
  - label reason diagnostics
- Added learned OP policy training and loading:
  - `op_policy_model.py`
  - `train_op_policy.py`
  - MLP segment retention policy checkpoint path through eval scripts.

## Current Results

### NIAH heuristic OP-Lite

- `semantic`: 30%, 20%, 10% budgets all reached 100%; 5% budget dropped to 0%.
- `op_sievekv_lite`: 30%, 20%, 10% budgets all reached 100%; 5% budget reached 60%.

Interpretation: the heuristic OP-Lite MVP gives a real low-budget retrieval gain on single-needle NIAH.

### Multi-needle heuristic OP-Lite

From the reported `results/v2/multi_needle/` summary:

- `semantic`:
  - 30% average: 87.5%
  - 20% average: 80.6%
  - 10% average: 63.9%
  - 5% average: 30.6%
- `op_sievekv_lite`:
  - 30% average: 84.7%
  - 20% average: 65.3%
  - 10% average: 76.4%
  - 5% average: 41.7%

Interpretation: OP-Lite improves aggressive low-budget settings, especially 10% and 5%, but is not uniformly better at 20% and 30%.

### Distillation v1

Evidence-only / narrow cheap oracle:

- Dataset: 7896 rows, 192 positives.
- Training converged cleanly.
- NIAH distilled result:
  - 30%: 100%
  - 20%: 93.3%
  - 10%: 86.7%
  - 5%: 40%

Interpretation: labels were too narrow. The student learned evidence markers but lost robustness compared with heuristic OP-Lite.

### Distillation v2d

Latest collected dataset:

- 7896 rows.
- hard positives: 1520.
- label mass: 3321.2.
- label reasons:
  - boundary_support: 412
  - system_support: 168
  - teacher: 6300
  - evidence: 192
  - pinned_support: 168
  - heuristic_support: 364
  - recent_support: 100
  - query_support: 192
- hard-positive reasons:
  - boundary_support: 412
  - system_support: 168
  - evidence: 192
  - pinned_support: 168
  - heuristic_support: 364
  - query_support: 192
  - recent_support: 24

Training behavior:

- BCE loss plateaued because labels are soft, not because training failed.
- `val_mae` fell from about 0.0678 to about 0.0058.
- `val_mean_prob` matched `val_mean_label` at about 0.419.

NIAH distilled v2d:

- 30%: 100%
- 20%: 100%
- 10%: 100%
- 5%: 40%

Multi-needle distilled v2d:

- 30%, k=2: 83.3%, k=4: 83.3%, k=8: 87.5%
- 20%, k=2: 66.7%, k=4: 91.7%, k=8: 70.8%
- 10%, k=2: 66.7%, k=4: 25.0%, k=8: 50.0%
- 5%, k=2: 50.0%, k=4: 25.0%, k=8: 37.5%

Interpretation: v2d fixes mid-budget single-needle NIAH, but still hurts aggressive-budget multi-needle k=4/k=8. The policy needs budget-aware labels/objective and true on-policy oracle signals.

### Distillation v3

Budget-aware label collection result:

- 7896 rows.
- hard positives: 1847.
- label mass: 2462.4.
- label reasons:
  - boundary_support: 365
  - system_support: 168
  - teacher: 5995
  - heuristic_support: 453
  - evidence: 192
  - pinned_support: 139
  - evidence_neighbor: 218
  - recent_support: 78
  - query_support: 288
- hard-positive reasons:
  - boundary_support: 365
  - system_support: 168
  - heuristic_support: 453
  - evidence: 192
  - pinned_support: 139
  - evidence_neighbor: 218
  - query_support: 288
  - recent_support: 24

Budget label stats:

- 5%: rows 1974, hard 415, teacher 31, label mass 409.8
- 10%: rows 1974, hard 434, teacher 59, label mass 484.2
- 20%: rows 1974, hard 477, teacher 147, label mass 671.7
- 30%: rows 1974, hard 521, teacher 216, label mass 896.6

Interpretation: v3 tightened labels in the intended direction. Total label mass dropped from v2d 3321.2 to 2462.4, and teacher-selected segments are now strongly budget-dependent. Hard positives increased because evidence-neighbor/support labels crossed the 0.5 threshold, but aggressive-budget soft mass is much smaller.

Training result:

- Dataset mean label: 0.312.
- Dataset mean weight: 2.144.
- Validation mean probability matched mean label closely, e.g. about 0.313 / 0.309 near epoch 300.
- Validation MAE improved from 0.0915 to about 0.011-0.012.
- Best validation loss was around 0.424.

Interpretation: convergence is less visually sharp than v2d because v3 has harder, more budget-dependent labels and stronger low-budget weights. The calibration is still acceptable; downstream NIAH / multi-needle evaluation is the deciding signal.

Evaluation result:

- NIAH:
  - 30%: 100% (15/15)
  - 20%: 100% (15/15)
  - 10%: 80% (12/15)
  - 5%: 60% (9/15)
- Multi-needle:
  - 30%, k=2: 83.3%, k=4: 100.0%, k=8: 83.3%
  - 20%, k=2: 50.0%, k=4: 100.0%, k=8: 87.5%
  - 10%, k=2: 83.3%, k=4: 66.7%, k=8: 58.3%
  - 5%, k=2: 66.7%, k=4: 50.0%, k=8: 33.3%

Interpretation: v3 improves the main v2d low-budget multi-needle failures, especially 10% k=4 and 5% k=4, and it restores NIAH 5% from 40% to 60%. However, it sacrifices NIAH 10% and is still not consistently better than the heuristic OP-Lite. This suggests that budget-aware cheap labels help, but the next bottleneck is real oracle supervision rather than more label-shaping.

## Gap Versus Original Plan

### Done

- OP-SieveKV-Lite MVP exists and runs.
- Semantic segment retention is partially implemented.
- A learned lightweight retention policy exists.
- Evaluation is available for NIAH and multi-needle.
- The system can run on the AutoDL 3B Qwen setup.

### Partially Done

- Semantic segmentation:
  - Current: sentence / role-aware segments.
  - Missing: hierarchical document -> paragraph -> sentence -> token fallback, RAG chunk, code block, table row/cell segmenters.
- Learned adaptive policy:
  - Current: global MLP over segment features.
  - Missing: richer query/task features, layer features used as real layer-wise policy inputs, gated scoring analysis.
- Oracle distillation:
  - Current: cheap label proxy from evidence/support/teacher.
  - Missing: counterfactual drop / restore oracle based on answer likelihood or generation correctness.
- KV-TIP:
  - Current: diagnostic idea and scaffolding.
  - Missing: actual retention entropy + oracle-policy divergence labels, Q1-Q4 taxonomy, Q3 case mining.
- Experiments:
  - Current: NIAH and multi-needle on Qwen2.5-3B.
  - Missing: RULER variants, LongBench / multi-hop / summarization / RAG / planning, more model sizes.

### Not Yet Done

- True on-policy compressed-cache rollout dataset generation.
- Counterfactual restore/drop oracle in compressed-cache states.
- Soft-OR decision selection and ablations against entropy-only, divergence-only, Q3-only, all-decisions.
- Layer-wise or layer-group budget allocation.
- Uncertainty-aware dynamic eviction.
- System-level memory/latency profiling for the learned policy.
- Fixed-block vs semantic-segment ablation.
- Layer/shared-index ablation.
- Feature ablation and policy-form ablation.
- Broad baseline table against full/window/streaming/H2O/SnapKV/KVzip/original SieveKV.

## Immediate Next Steps

1. Implement budget-aware distillation v3.
   - Add budget-dependent label top fractions and budget-conditioned loss weights.
   - Make 5% and 10% examples more selective than 20% and 30%.
   - Track per-budget positive rate and label mass.
   - Status: implemented in `collect_op_distill_data.py` after this note was first created. The collector now tightens teacher thresholds, teacher top-k/fraction, teacher token budget, teacher cap, background scale, and row weights as budget becomes more aggressive.

2. Add an oracle pilot instead of only cheap labels.
   - For sampled candidate segments, run drop/restore tests.
   - Start with NIAH and multi-needle only.
   - Use generation correctness or answer-token log probability as the oracle score.
   - Status: prompt-level counterfactual drop pilot implemented in `collect_op_oracle_data.py`. It removes sampled semantic segments from prompt tokens and measures gold-answer average log-probability delta. Candidate oracle labels override cheap labels for scored segments.

Oracle smoke result:

- Dataset: 780 rows.
- Hard positives: 209.
- Label mass: 238.6.
- Oracle-scored rows: 144.
- Oracle delta stats:
  - min: -0.0611
  - mean: 0.4328
  - max: 3.2894
- Label reasons:
  - boundary_support: 90
  - system_support: 36
  - teacher: 440
  - oracle_drop: 116
  - oracle_keep: 28
  - pinned_support: 30
  - heuristic_support: 21
  - recent_support: 15
  - evidence_neighbor: 4
- Budget label stats:
  - 5%: rows 390, oracle 72, label mass 114.2
  - 10%: rows 390, oracle 72, label mass 124.4

Interpretation: the smoke run validates that the prompt-level counterfactual oracle produces meaningful signal. Most sampled oracle candidates are drops, which is useful because it corrects cheap teacher over-retention, while high positive deltas identify answer-critical segments.

Oracle pilot collection result:

- Dataset: 7896 rows.
- Hard positives: 1681.
- Label mass: 2480.0.
- Oracle-scored rows: 1076.
- Oracle delta stats:
  - min: -0.2886
  - mean: 0.3235
  - max: 3.2507
- Label reasons:
  - boundary_support: 365
  - system_support: 168
  - teacher: 5462
  - oracle_keep: 352
  - oracle_drop: 724
  - pinned_support: 95
  - heuristic_support: 282
  - recent_support: 53
  - query_support: 288
  - evidence_neighbor: 107
- Budget label stats:
  - 5%: rows 1974, oracle 269, label mass 445.4
  - 10%: rows 1974, oracle 269, label mass 505.4
  - 20%: rows 1974, oracle 269, label mass 663.4
  - 30%: rows 1974, oracle 269, label mass 865.8

Interpretation: the full oracle pilot is healthy. Compared with v3, label mass is similar but hard positives drop from 1847 to 1681, meaning oracle labels mainly remove over-broad positives while preserving high-impact evidence. The next check is whether this improves downstream NIAH / multi-needle stability.

Oracle pilot training result:

- Dataset mean label: about 0.314.
- Dataset mean weight: 2.984.
- Validation loss improved from 0.5674 to best 0.4851.
- Validation MAE improved from 0.1136 to about 0.034-0.037.
- Validation mean probability stayed close to mean label, usually around 0.31 / 0.309.
- Early stopped at epoch 188 with best validation loss 0.4851.
- Per-budget validation calibration near the end:
  - 5%: mean_prob around 0.257 vs mean_label 0.242
  - 10%: mean_prob around 0.277 vs mean_label 0.262
  - 20%: mean_prob around 0.327 vs mean_label 0.304
  - 30%: mean_prob around 0.442 vs mean_label 0.426

Interpretation: training is normal. Oracle-weighted labels are harder than v3 labels, so BCE and MAE are higher, but calibration is acceptable and early stopping selected the best checkpoint. Downstream evaluation is needed to judge whether oracle supervision improves retention quality.

Oracle pilot NIAH evaluation:

- 30%: 93.3% (14/15), avg time 1.87s
- 20%: 100.0% (15/15), avg time 1.78s
- 10%: 93.3% (14/15), avg time 1.69s
- 5%: 46.7% (7/15), avg time 1.62s

Interpretation: oracle pilot improves v3 at 10% NIAH but hurts 5% and slightly hurts 30%. This suggests the prompt-level drop oracle is useful but currently too aggressive or not budget-calibrated enough at 5%. Multi-needle evaluation is still needed before deciding whether to keep this checkpoint or adjust oracle label mixing.

Oracle pilot multi-needle evaluation:

- 30%, k=2: 100.0%, avg_found 2.0/2
- 30%, k=4: 100.0%, avg_found 4.0/4
- 30%, k=8: 95.8%, avg_found 7.7/8
- 20%, k=2: 50.0%, avg_found 1.0/2
- 20%, k=4: 91.7%, avg_found 3.7/4
- 20%, k=8: 66.7%, avg_found 5.3/8
- 10%, k=2: 50.0%, avg_found 1.0/2
- 10%, k=4: 83.3%, avg_found 3.3/4
- 10%, k=8: 79.2%, avg_found 6.3/8
- 5%, k=2: 83.3%, avg_found 1.7/2
- 5%, k=4: 41.7%, avg_found 1.7/4
- 5%, k=8: 20.8%, avg_found 1.7/8

Interpretation: oracle pilot strongly improves high-evidence / high-k settings, especially 30% and 10% k=8, but hurts the extreme 5% k=8 case. The oracle signal is valuable, but direct oracle label override is too aggressive for the lowest budget. Next step: implement conservative oracle mixing where oracle drops only down-weight uncertain teacher labels and cannot override evidence/query/support labels at 5%.

Implementation update: `collect_op_oracle_data.py` now supports `--oracle-mix-mode conservative` by default. In conservative mode:

- Oracle keep labels can still raise segment labels.
- Oracle drop labels mainly correct teacher/background labels.
- Evidence, query, system, boundary, pinned, and recent support labels receive a protected floor.
- Low-budget oracle drops are scaled down to avoid erasing the only surviving evidence at 5%.

Conservative oracle collection result:

- Dataset: 7896 rows.
- Hard positives: 1859.
- Label mass: 2499.7.
- Oracle-scored rows: 1076.
- Oracle delta stats:
  - min: -0.2886
  - mean: 0.3235
  - max: 3.2507
- Label reasons:
  - boundary_support: 365
  - system_support: 168
  - teacher: 5462
  - oracle_keep: 352
  - oracle_soft_drop: 77
  - oracle_mixed_drop: 512
  - oracle_protected_drop: 135
  - pinned_support: 95
  - heuristic_support: 282
  - recent_support: 53
  - query_support: 288
  - evidence_neighbor: 107
- Hard-positive reasons:
  - boundary_support: 365
  - system_support: 168
  - oracle_keep: 318
  - oracle_soft_drop: 77
  - oracle_protected_drop: 135
  - pinned_support: 95
  - heuristic_support: 282
  - query_support: 288
  - recent_support: 24
  - evidence_neighbor: 107
- Budget label stats:
  - 5%: rows 1974, oracle 269, label mass 435.8
  - 10%: rows 1974, oracle 269, label mass 505.4
  - 20%: rows 1974, oracle 269, label mass 677.6
  - 30%: rows 1974, oracle 269, label mass 880.9

Interpretation: conservative mixing is active. It keeps oracle signal but separates mixed/protected drops from direct drops. Hard positives increase relative to override because protected/soft drops preserve some evidence and structural support. The 5% label mass is slightly lower than override, so the change is not merely broadening labels; it is changing which positives survive.

Conservative oracle evaluation:

- NIAH:
  - 30%: 100.0% (15/15), avg time 1.99s
  - 20%: 100.0% (15/15), avg time 1.78s
  - 10%: 73.3% (11/15), avg time 1.66s
  - 5%: 53.3% (8/15), avg time 1.59s
- Multi-needle:
  - 30%, k=2: 100.0%, avg_found 2.0/2
  - 30%, k=4: 100.0%, avg_found 4.0/4
  - 30%, k=8: 100.0%, avg_found 8.0/8
  - 20%, k=2: 50.0%, avg_found 1.0/2
  - 20%, k=4: 100.0%, avg_found 4.0/4
  - 20%, k=8: 95.8%, avg_found 7.7/8
  - 10%, k=2: 83.3%, avg_found 1.7/2
  - 10%, k=4: 75.0%, avg_found 3.0/4
  - 10%, k=8: 54.2%, avg_found 4.3/8
  - 5%, k=2: 66.7%, avg_found 1.3/2
  - 5%, k=4: 66.7%, avg_found 2.7/4
  - 5%, k=8: 33.3%, avg_found 2.7/8

Interpretation: conservative oracle mixing improves high-budget and 20% multi-needle substantially, reaching perfect 30% k=8 and 95.8% at 20% k=8. It also restores 5% k=4 relative to override. However, it hurts NIAH 10% and multi-needle 10% k=8 relative to override and v3. The next issue is budget-specific mixing: 10% likely needs stronger oracle-keep / weaker oracle-drop protection than the current interpolation gives, while 5% needs evidence protection.

Research-plan alignment update:

- Added `analyze_op_oracle_dataset.py` for KV-TIP-style analysis.
- It reads oracle/distillation `.pt` datasets and reports:
  - retention entropy
  - oracle-policy divergence
  - Q1/Q2/Q3/Q4 quadrants
  - over-keep vs missed-keep decision types
  - per-budget and per-task oracle summaries
  - top Q3 confident-wrong rows
- This addresses the plan's KV-TIP taxonomy and Q3 case-analysis direction without doing more local threshold tuning.

Uploaded KV-TIP report observations:

- Override and conservative have the same oracle-scored segment set: 1076 oracle rows.
- Quadrants based on heuristic probability:
  - Q1 confident aligned: 364
  - Q2 uncertain aligned: 440
  - Q3 confident wrong: 172
  - Q4 uncertain wrong: 100
- Decision types based on heuristic probability:
  - aligned_keep: 352
  - over_keep: 724
  - missed_keep: 0
- Interpretation: current oracle candidate set mainly exposes teacher/heuristic over-retention, not false negatives. This is useful for pruning distractors but insufficient for discovering segments that the policy would wrongly drop.
- Important caveat: the current report compares oracle labels against `heuristic_keep_prob`, not the trained MLP policy. `analyze_op_oracle_dataset.py` now supports `--policy-ckpt` to compute learned-policy Q3 analysis.

Student-policy KV-TIP observations:

- Conservative student analysis:
  - Q1 confident aligned: 473
  - Q2 uncertain aligned: 334
  - Q3 confident wrong: 65
  - Q4 uncertain wrong: 204
  - aligned_keep: 301
  - aligned_drop: 509
  - over_keep: 215
  - missed_keep: 51
- Override student analysis:
  - Q1 confident aligned: 425
  - Q2 uncertain aligned: 382
  - Q3 confident wrong: 113
  - Q4 uncertain wrong: 156
  - aligned_keep: 303
  - aligned_drop: 717
  - over_keep: 7
  - missed_keep: 49

Interpretation: after training, the student policy introduces a new failure type that heuristic analysis could not see: missed_keep. Conservative greatly reduces Q3 count compared with override and improves high-budget retention, but it creates many more over_keep decisions. This matches the evaluation pattern: conservative is strong at 20/30% multi-needle but can crowd out key spans at lower budgets. Override avoids over_keep but still has missed_keep, explaining its 5% weakness. Next research-aligned step: collect an on-policy/student-policy oracle dataset that samples candidate segments from learned-policy decisions, not only heuristic teacher decisions.

Implementation update: `collect_op_oracle_data.py` now supports `--candidate-policy-ckpt`. When supplied, oracle candidates include:

- high student-probability segments for over-keep probes
- low/uncertain student-probability segments
- high heuristic/content but low student-probability segments for missed-keep probes

This moves the oracle pilot closer to on-policy distillation because candidate segments are sampled from the learned policy's own decisions rather than only from heuristic teacher scores.

Student-candidate oracle collection result:

- Dataset: 7896 rows.
- Hard positives: 1901.
- Label mass: 2556.3.
- Oracle-scored rows: 1585.
- Oracle delta stats:
  - min: -0.7092
  - mean: 0.2330
  - max: 3.2507
- Label reasons:
  - boundary_support: 365
  - system_support: 168
  - teacher: 5061
  - oracle_mixed_drop: 858
  - oracle_keep: 439
  - oracle_soft_drop: 68
  - oracle_protected_drop: 220
  - pinned_support: 95
  - heuristic_support: 291
  - query_support: 227
  - recent_support: 40
  - evidence_neighbor: 64
- Hard-positive reasons:
  - boundary_support: 365
  - system_support: 168
  - oracle_keep: 379
  - oracle_soft_drop: 68
  - oracle_protected_drop: 220
  - pinned_support: 95
  - heuristic_support: 291
  - query_support: 227
  - recent_support: 24
  - evidence_neighbor: 64
- Budget label stats:
  - 5%: rows 1974, oracle 474, label mass 458.3
  - 10%: rows 1974, oracle 451, label mass 524.8
  - 20%: rows 1974, oracle 373, label mass 688.3
  - 30%: rows 1974, oracle 287, label mass 884.8

Interpretation: student-policy candidate sampling changes the oracle distribution in the intended way. Oracle coverage increases from 1076 to 1585 rows, with much more coverage at 5% and 10%. Mean oracle delta drops because the sampler now includes more borderline and student-policy-specific decisions, not only the strongest heuristic candidates. This is closer to on-policy oracle distillation.

Student-candidate oracle evaluation:

- NIAH:
  - 30%: 100.0% (15/15), avg time 1.93s
  - 20%: 100.0% (15/15), avg time 1.81s
  - 10%: 93.3% (14/15), avg time 1.83s
  - 5%: 60.0% (9/15), avg time 1.58s
- Multi-needle:
  - 30%, k=2: 83.3%, avg_found 1.7/2
  - 30%, k=4: 100.0%, avg_found 4.0/4
  - 30%, k=8: 100.0%, avg_found 8.0/8
  - 20%, k=2: 66.7%, avg_found 1.3/2
  - 20%, k=4: 100.0%, avg_found 4.0/4
  - 20%, k=8: 87.5%, avg_found 7.0/8
  - 10%, k=2: 66.7%, avg_found 1.3/2
  - 10%, k=4: 75.0%, avg_found 3.0/4
  - 10%, k=8: 79.2%, avg_found 6.3/8
  - 5%, k=2: 66.7%, avg_found 1.3/2
  - 5%, k=4: 58.3%, avg_found 2.3/4
  - 5%, k=8: 25.0%, avg_found 2.0/8

Interpretation: student-candidate oracle training is the best balanced learned-policy result so far. It restores NIAH 5% to 60%, reaches 93.3% at NIAH 10%, preserves 30% k=8 at 100%, and keeps 10% k=8 at 79.2%. The remaining weakness is extreme 5% k=8 and some k=2 variance. This supports the research-plan claim that student/on-policy candidate oracle is more useful than pure cheap-label distillation.

3. Add Q3 / KV-TIP diagnostics.
   - Compute policy probability, retention entropy, and oracle disagreement.
   - Save confident-wrong examples for analysis.

4. Add retained-segment visualization.
   - Dump retained text spans per policy/budget.
   - Compare semantic vs OP-Lite vs distilled policy failures.

5. Expand data size after label logic is stable.
   - More multi-needle trials.
   - More needle positions and lengths.
   - Harder 5%/10% examples.

## True On-Policy Compressed-Cache Oracle (core method)

New module `collect_op_onpolicy_oracle_data.py` implements the research plan's
core contribution (RQ3, Sections 5.4-5.5): the **true on-policy compressed-cache
oracle**, replacing the prompt-level pilot. This closes the first two items on
the "Not Yet Done" list above.

What changed versus `collect_op_oracle_data.py`:

- The policy first evicts under its own budget (using the learned checkpoint when
  `--policy-ckpt` is supplied, otherwise the SieveKV heuristic as warm-start
  pi_theta) to build the compressed cache state `C^{pi_theta}`. Construction uses
  the exact runtime convention from `kv_cache_manager.py`:
  `budget_tokens = max(1, int(seq_len * budget))`, then
  `keep_indices = policy.select_keep_indices(policy.compute_eviction_scores(seq_len), budget_tokens)`.
- Oracle importance is measured **relative to the compressed state**, not the full
  prompt:
  - retained segment -> drop:    `I = logP(a|C) - logP(a|C \ g)`
  - evicted segment  -> restore: `I = logP(a|C u g) - logP(a|C)`
- The **restore** branch is the key addition: it discovers Q3 confident-wrong
  false negatives (segments the policy confidently dropped that are answer-critical
  once other evidence is gone). The prompt-level drop oracle could only find
  over-retention (`missed_keep = 0` in earlier heuristic KV-TIP reports).
- New per-row metadata: `on_policy=True`, `oracle_mode` (drop/restore),
  `segment_retained`, `policy_keep_prob`, `compressed_answer_avg_logprob`. The
  collector also prints a Q3 confident-wrong restore-candidate count.
- Dataset schema is unchanged, so `train_op_policy.py` and
  `analyze_op_oracle_dataset.py` consume it directly.

Local validation: syntax (`py_compile`) and import/argparse (`--help`) only —
model runs happen on AutoDL.

### AutoDL on-policy oracle results

Smoke result:

- Dataset: 59 rows.
- Hard positives: 14.
- Label mass: 15.0.
- Oracle-scored rows: 2.
- Oracle modes:
  - restore: 2
- Oracle delta stats:
  - min: 0.0014
  - mean: 0.0443
  - max: 0.0872

Interpretation: the cache-level oracle path runs successfully on AutoDL and can score restore candidates without DynamicCache / position-id failures.

NIAH pilot result:

- Dataset: 534 rows.
- Hard positives: 143.
- Label mass: 175.2.
- Oracle-scored rows: 138.
- Oracle modes:
  - restore: 105
  - drop: 33
- Oracle delta stats:
  - min: -1.5302
  - mean: 0.0283
  - max: 3.1801
- Budget label stats:
  - 5%: rows 178, oracle 48, label mass 47.3
  - 10%: rows 178, oracle 43, label mass 51.8
  - 30%: rows 178, oracle 47, label mass 76.2

Interpretation: the pilot confirms that the true on-policy oracle produces both drop and restore signals. The large positive restore delta shows that compressed-cache states contain real missed-keep opportunities that prompt-level drop-only oracle could not expose.

Full student-policy on-policy oracle collection result:

- Dataset: 7896 rows.
- Hard positives: 1951.
- Label mass: 2554.4.
- Oracle-scored rows: 1887.
- Oracle modes:
  - drop: 618
  - restore: 1269
- Oracle delta stats:
  - min: -3.6132
  - mean: 0.0648
  - max: 3.8701
- Missed-keep restore candidates: 227 / 1269 restore probes.
- Q3 low-prob confident-wrong subset: 160 / 227 missed-keep probes.
- Label reasons:
  - boundary_support: 365
  - system_support: 168
  - oracle_mixed_drop: 1016
  - teacher: 4859
  - oracle_soft_drop: 175
  - oracle_keep: 383
  - oracle_protected_drop: 313
  - pinned_support: 104
  - heuristic_support: 216
  - evidence: 23
  - query_support: 193
  - recent_support: 37
  - evidence_neighbor: 44
- Budget label stats:
  - 5%: rows 1974, oracle 477, label mass 483.6
  - 10%: rows 1974, oracle 458, label mass 530.8
  - 20%: rows 1974, oracle 465, label mass 677.9
  - 30%: rows 1974, oracle 487, label mass 862.1

Interpretation: this is the first result that directly supports the research plan's on-policy claim. The restore branch dominates the oracle rows, and 227 restore probes are positive missed-keep cases. Among them, 160 are low-policy-probability Q3-style confident-wrong decisions. This is qualitatively different from the earlier prompt-level oracle, which mainly exposed over-retention.

On-policy student training result:

- Dataset mean label: about 0.323.
- Dataset mean weight: 3.617.
- Best validation loss: 0.5164.
- Early stopped at epoch 104.
- Validation MAE improved from 0.1139 to about 0.05-0.06.
- Final validation calibration was slightly over-retentive overall, roughly mean probability 0.34 vs mean label 0.321.
- Per-budget calibration near epoch 100:
  - 5%: mean_prob 0.300 vs mean_label 0.269
  - 10%: mean_prob 0.290 vs mean_label 0.268
  - 20%: mean_prob 0.336 vs mean_label 0.318
  - 30%: mean_prob 0.431 vs mean_label 0.429

Interpretation: training converged normally. The higher BCE than earlier cheap-label / prompt-oracle runs is expected because true on-policy restore/drop labels are harder and more heavily weighted. The checkpoint is usable for downstream evaluation.

On-policy student evaluation:

- NIAH:
  - 30%: 100.0% (15/15), avg time 1.55s
  - 20%: 100.0% (15/15), avg time 1.47s
  - 10%: 100.0% (15/15), avg time 1.45s
  - 5%: 53.3% (8/15), avg time 1.39s
- Multi-needle:
  - 30%, k=2: 75.0%, avg_found 1.5/2
  - 30%, k=4: 62.5%, avg_found 2.5/4
  - 30%, k=8: 87.5%, avg_found 7.0/8
  - 20%, k=2: 100.0%, avg_found 2.0/2
  - 20%, k=4: 100.0%, avg_found 4.0/4
  - 20%, k=8: 87.5%, avg_found 7.0/8
  - 10%, k=2: 75.0%, avg_found 1.5/2
  - 10%, k=4: 62.5%, avg_found 2.5/4
  - 10%, k=8: 62.5%, avg_found 5.0/8
  - 5%, k=2: 50.0%, avg_found 1.0/2
  - 5%, k=4: 25.0%, avg_found 1.0/4
  - 5%, k=8: 37.5%, avg_found 3.0/8

Interpretation: true on-policy distillation strongly improves single-needle NIAH at 10%, reaching 100%, and keeps 30% / 20% perfect. However, it is not the best balanced multi-needle policy: it hurts several multi-needle settings relative to the student-candidate prompt-oracle checkpoint, especially 30% k=4 and 10% k=8. This suggests the on-policy oracle signal is real and paper-relevant, but the direct mixed-label training objective overfits or over-corrects restore/drop decisions. The next method step should be selection or weighting, not more raw oracle collection: Soft-OR / KV-TIP selection should train on the informative on-policy Q3/missed-keep rows while avoiding broad degradation on multi-needle high-budget cases.

Fixed-manifest on-policy student evaluation:

- NIAH manifest:
  - Samples: 120 fixed samples reused across budgets.
  - Output: `results/v2/niah_onpolicy_student_manifest.json`
  - 30%: 100.0% (120/120), avg time 1.52s
  - 20%: 99.2% (119/120), avg time 1.46s
  - 10%: 100.0% (120/120), avg time 1.39s
  - 5%: 45.0% (54/120), avg time 1.32s
- Multi-needle fixed manifest:
  - Output: `results/v2/multi_onpolicy_student_manifest.json`
  - 30%, k=2: 76.7%, avg_found 1.5/2
  - 30%, k=4: 66.7%, avg_found 2.7/4
  - 30%, k=8: 74.6%, avg_found 6.0/8
  - 20%, k=2: 75.0%, avg_found 1.5/2
  - 20%, k=4: 67.5%, avg_found 2.7/4
  - 20%, k=8: 80.4%, avg_found 6.4/8
  - 10%, k=2: 71.7%, avg_found 1.4/2
  - 10%, k=4: 63.3%, avg_found 2.5/4
  - 10%, k=8: 64.2%, avg_found 5.1/8
  - 5%, k=2: 51.7%, avg_found 1.0/2
  - 5%, k=4: 25.0%, avg_found 1.0/4
  - 5%, k=8: 44.2%, avg_found 3.5/8
  - Total time: 1989s

Interpretation: the larger fixed-manifest run confirms the small-sample trend. Single-needle NIAH is effectively solved at 10% and above, while 5% remains below the reliable operating region. Multi-needle is the more informative benchmark: scores plateau around 63-80% for 10-30% and degrade sharply at 5%, especially k=4. This means the current on-policy student is good at preserving one critical fact but still weak at distributing retention across multiple independent evidence items. The next training change should target evidence coverage and missed-keep selection, not merely more epochs.

## Current Bottom Line

The original MVP goal has now moved past cheap-label distillation. OP-Lite proves that semantic/role-aware segment retention can beat the semantic baseline under aggressive budget, and `collect_op_onpolicy_oracle_data.py` provides direct evidence for the paper's core on-policy claim: in the student's own compressed-cache state, restore oracle probes uncover missed-keep and Q3 confident-wrong decisions.

The fixed-manifest results make the tradeoff clearer. True on-policy training gives a strong single-needle result at 10-30% budget, but it does not yet solve multi-evidence retention. The next bottleneck is no longer whether on-policy oracle signal exists; it does. The next bottleneck is how to select and weight those oracle decisions so that restore corrections improve evidence coverage instead of over-specializing to single-fact retrieval. The most research-aligned next step is Soft-OR / KV-TIP selection on the true on-policy dataset, followed by ablations against all-decisions, entropy-only, divergence-only, and Q3/missed-keep-only training.
