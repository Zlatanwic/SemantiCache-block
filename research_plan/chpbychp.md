  ---
  OP-SieveKV AAAI Chapter Plan

  Venue: AAAI 2026 (8+2 pages)
  Framing: ML methodology — "on-policy distillation for KV retention is a new paradigm; Q3 confident-wrong errors are the key failure mode we
   identify and correct"
  Core narrative: Off-policy mismatch → Q3 confident-wrong → on-policy correction + Soft-OR → semantic segments preserve boundaries → better
  outcomes

  Paper Structure & Page Budget

  ┌──────────────────────────────┬───────┬─────────────────────────────────────────────────────────────────────────────────┐
  │           Section            │ Pages │                                     Purpose                                     │
  ├──────────────────────────────┼───────┼─────────────────────────────────────────────────────────────────────────────────┤
  │ Abstract                     │ —     │ Hook + result highlight                                                         │
  ├──────────────────────────────┼───────┼─────────────────────────────────────────────────────────────────────────────────┤
  │ 1. Introduction              │ 1.5   │ Problem framing, three limitations, OP-SieveKV preview, contribution list       │
  ├──────────────────────────────┼───────┼─────────────────────────────────────────────────────────────────────────────────┤
  │ 2. Background & Related Work │ 1.0   │ KV cache eviction landscape, on-policy distillation lineage                     │
  ├──────────────────────────────┼───────┼─────────────────────────────────────────────────────────────────────────────────┤
  │ 3. Motivation                │ 1.0   │ Off-policy mismatch evidence, Q3 existence proof, fixed-block failure           │
  ├──────────────────────────────┼───────┼─────────────────────────────────────────────────────────────────────────────────┤
  │ 4. Method                    │ 2.5   │ On-policy distillation, KV-TIP + Soft-OR, semantic segments, training objective │
  ├──────────────────────────────┼───────┼─────────────────────────────────────────────────────────────────────────────────┤
  │ 5. Experiments               │ 2.0   │ 5 experiment blocks, 1 Q3 case figure                                           │
  ├──────────────────────────────┼───────┼─────────────────────────────────────────────────────────────────────────────────┤
  │ 6. Conclusion                │ 0.5   │ Summary + future work (layer-wise, uncertainty-aware)                           │
  ├──────────────────────────────┼───────┼─────────────────────────────────────────────────────────────────────────────────┤
  │ Appendix                     │ 2.0   │ Layer-wise ablation, system overhead, full hyperparameters                      │
  └──────────────────────────────┴───────┴─────────────────────────────────────────────────────────────────────────────────┘

  ---
  Chapter-by-Chapter Plan

  Abstract (250 words)

  Hook: KV cache memory bottleneck in long-context LLM serving forces aggressive eviction, but current heuristic methods suffer from a
  fundamental problem: they are trained on full-context oracle labels while deployed on compressed-cache states.

  Gap: This off-policy supervision mismatch causes "confident-wrong" eviction decisions — the policy confidently drops segments that are
  critical under compressed state.

  Solution: OP-SieveKV, an on-policy oracle distillation framework that (1) generates counterfactual oracle labels on the policy's own
  compressed-cache trajectory, (2) proposes KV-TIP taxonomy identifying Q3 confident-wrong decisions via retention entropy × oracle-policy
  divergence, and (3) uses Soft-OR selection to prioritize training on informative decisions including Q3 cases. We further upgrade fixed
  token blocks to semantic segments to preserve entity/relation boundaries.

  Result highlight: On RULER NIAH, multi-needle, HotpotQA-long, and LongBench, OP-SieveKV achieves X% higher accuracy than SieveKV at the
  same cache budget, with Q3 error rate reduced by Y%. Policy overhead remains <Zms per eviction step.

  INSIGHT-1: The abstract must establish the causal chain (mismatch → Q3 → correction) in one sentence. Avoid listing all features; the
  reader should immediately grasp WHY on-policy matters.

  ---
  Section 1: Introduction (1.5 pages)

  Opening paragraph: Long-context LLM serving — KV cache grows linearly, creating memory-quality-latency tension. Existing eviction methods
  (H2O, SnapKV, etc.) rely on attention signals or heuristic rules.

  Second paragraph: SieveKV demonstrates that multi-signal, query-aware, content-aware eviction improves retrieval accuracy. But SieveKV uses
   fixed weights, fixed blocks, and no training — it cannot adapt across tasks or budgets.

  Third paragraph (THE KEY MOTIVATION): We identify a deeper structural problem. When oracle importance labels are generated under full-KV
  states (off-policy), the learned policy faces compressed-cache states at deployment (on-policy). This distribution mismatch creates a
  specific and dangerous failure mode: confident-wrong eviction — the policy is certain a segment is unimportant, but under the actual
  compressed state, that segment is critical for multi-hop reasoning or retrieval chains.

  Fourth paragraph: We propose OP-SieveKV, an on-policy oracle distillation framework. Three contributions:
  1. On-policy counterfactual oracle distillation — generates correction labels on the policy's own compressed trajectory, directly fixing
  off-policy mismatch.
  2. KV-TIP taxonomy — retention entropy × oracle-policy divergence quadrant analysis, identifying Q3 confident-wrong decisions; Soft-OR
  selection prioritizes these for training.
  3. Semantic segment retention — replacing fixed token blocks with semantically coherent units, preserving entity/relation boundaries that
  fixed blocks split.

  INSIGHT-2: The introduction must make Q3 feel inevitable, not invented. Show that off-policy mismatch naturally produces confident-wrong
  errors — it's not an arbitrary taxonomy but a structural consequence of the mismatch. One short example (e.g., "a multi-hop reasoning chain
   where the intermediate evidence has low query overlap but high causal importance") will make Q3 concrete before the reader reaches Section
   3.

  ---
  Section 2: Background & Related Work (1 page)

  KV cache eviction (4 paragraphs, tight):
  - Recent-window and heavy-hitter methods (StreamingLLM, H2O)
  - Query-aware methods (SnapKV, KVzip, SieveKV)
  - Learned/adaptive methods (AdaKV, recent work)
  - Quantization/offloading as orthogonal compression

  On-policy distillation lineage (2 paragraphs):
  - Knowledge distillation basics
  - TIP (Token Importance Prediction) entropy-divergence framework — this is the direct intellectual ancestor of KV-TIP
  - On-policy RL and DAgger-style correction — conceptual basis for on-policy oracle distillation

  INSIGHT-3: Do NOT let Related Work become a survey. Each paragraph should position OP-SieveKV relative to prior work, ending with "unlike
  X, we ..." or "X addresses Y but misses Z, which we fix." Keep it under 1 page.

  ---
  Section 3: Motivation (1 page)

  This section is the paper's emotional center. It must make the reader feel the problem before seeing the solution.

  Subsection 3.1: Off-policy supervision mismatch
  - Define the mismatch: oracle labels from full-KV states, deployment on compressed states.
  - Show that importance shifts: a segment unimportant under full-KV becomes critical when other evidence is evicted. A real example
  (preferably from pilot experiments on SieveKV).

  Subsection 3.2: Q3 confident-wrong eviction exists
  - Pilot experiment: run SieveKV on multi-hop queries, compute oracle importance on compressed cache, measure how many segments SieveKV
  confidently drops that oracle marks as essential.
  - One figure: scatter plot of retention probability vs. oracle importance, with Q3 quadrant highlighted and annotated.
  - Key statistic: "X% of critical evidence is dropped with confidence > 0.9" — this number makes Q3 tangible.

  Subsection 3.3: Fixed blocks split semantic units
  - Example: a 16-token block cuts an entity name ("San Francisco") into "San Fran" and "cisco", or splits a multi-hop reasoning chain across
   two blocks. When one half is kept and the other dropped, the retained fragment becomes useless.
  - This directly feeds the Q3 story: fixed blocks create false boundaries → policy confidently drops "half-entities" → Q3 errors.

  INSIGHT-4: Motivation section needs empirical evidence, not just logical argument. If you haven't run the pilot experiment yet, this is the
   first thing to do. The Q3 scatter plot figure is likely the single most impactful visual in the paper.

  ---
  Section 4: Method (2.5 pages)

  This is the densest section. Allocate space carefully.

  Subsection 4.1: Overview (0.3 pages)
  - Two-phase framework: offline training (oracle + on-policy rollout) → online inference (lightweight policy only). One figure showing the
  pipeline.

  Subsection 4.2: Semantic segment construction (0.4 pages)
  - Sentence/paragraph/chunk-level segmentation with hierarchical fallback.
  - Role-based pinning (system, query, recent-decode pinned).
  - Brief: this replaces fixed blocks; detailed segmenter descriptions go to Appendix.

  Subsection 4.3: On-policy counterfactual oracle (0.5 pages)
  - Policy rollout → compressed cache state C^{πθ}
  - Drop importance (for retained segments) and restore importance (for evicted segments)
  - Oracle label: y* = σ(I/τ)
  - Key point: oracle operates on the policy's actual compressed state, not full-KV.

  Subsection 4.4: KV-TIP taxonomy and Soft-OR selection (0.6 pages)
  - Retention entropy h_{j,l} and oracle-policy divergence δ_{j,l}
  - Four-quadrant table (Q1-Q4)
  - Q3 definition and why it's the most dangerous
  - Soft-OR score: z = ĥ + δ̂ - ĥδ̂ (captures both uncertain AND confident-wrong)
  - Selection: top-ρ decisions for training

  Subsection 4.5: Policy architecture and training (0.5 pages)
  - Adaptive gated scoring: w_{j,l} = G_θ(φ_{j,l}), S = Σ w_k · Pool_k, p = σ(S)
  - Feature vector composition (brief table)
  - Loss: L_distill (on Soft-OR selected decisions) + λ L_rank + μ L_budget
  - Initialize from SieveKV heuristic weights

  Subsection 4.6: Online serving (0.2 pages)
  - Prefill → segment → compute features → predict p_{j,l} → budget allocation → mask → decode
  - No oracle in serving path. Policy overhead analysis deferred to Appendix.

  INSIGHT-5: The method section should flow causally: (1) we need oracle on compressed states → (2) we discover Q3 errors → (3) we
  selectively train on them → (4) we use semantic segments to prevent boundary-split Q3 errors. Each subsection should reference the
  motivation that drives it, not just describe a module.

  ---
  Section 5: Experiments (2 pages)

  5.1 Setup (0.3 pages)
  - Models: Qwen2.5-3B-Instruct, Llama-3.2-3B-Instruct, Qwen2.5-7B-Instruct
  - Benchmarks: RULER NIAH, multi-needle NIAH, HotpotQA-long, LongBench subsets
  - Baselines: Full KV, StreamingLLM, H2O, SnapKV, KVzip, SieveKV
  - Budgets: 50%, 30%, 20%, 10%
  - Metrics: retrieval accuracy, F1, EM; KV cache memory; decode latency

  5.2 Main results (0.4 pages)
  - One main table: accuracy at each budget level for all methods × 3 models.
  - Key finding: OP-SieveKV matches SieveKV at 50% budget and significantly exceeds it at 20% and 10% — the aggressive budget regime is where
   on-policy correction matters most.

  5.3 Offline vs on-policy distillation (0.3 pages)
  - Table: Fixed SieveKV / Offline-distilled / On-policy-distilled at same budgets.
  - On-policy advantage is largest at aggressive budgets and on multi-hop tasks.

  5.4 KV-TIP and Soft-OR effectiveness (0.3 pages)
  - Table: All-decisions / Entropy-only / Divergence-only / Q3-only / Soft-OR
  - Soft-OR outperforms all single-axis selections.
  - Q3-only captures the most dangerous errors but misses uncertain cases; Soft-OR combines both.

  5.5 Semantic segment vs fixed block (0.3 pages)
  - Table: 16-token block / 32-token block / sentence / paragraph / hierarchical
  - Segments reduce Q3 error rate by X% because entities and relations are preserved intact.

  5.6 Q3 analysis and case study (0.3 pages)
  - One figure: Q3 quadrant scatter plot (retention prob vs. oracle importance) with annotated examples.
  - Two case studies: (a) multi-hop intermediate evidence confidently dropped, (b) entity name split across blocks — one half confidently
  dropped.

  INSIGHT-6: Every experiment table should tell one piece of the causal story:
  - Main table → "OP-SieveKV works"
  - Offline vs on-policy → "on-policy correction is necessary"
  - Soft-OR → "Q3 targeting is effective"
  - Segments → "semantic boundaries reduce Q3 errors"
  - Case study → "here's what Q3 looks like in practice"

  Do NOT include experiments that don't directly advance the causal chain. Layer-wise and uncertainty-aware go to Appendix.

  ---
  Section 6: Conclusion (0.5 pages)

  Summary: OP-SieveKV addresses the fundamental off-policy mismatch in KV retention by generating counterfactual oracle labels on the
  policy's own compressed trajectory. KV-TIP taxonomy reveals Q3 confident-wrong errors as the key failure mode, and Soft-OR selection
  targets them for training. Semantic segment retention preserves coherent information units that fixed blocks split.

  Future work: Layer-wise budget allocation (preliminary results in Appendix), uncertainty-aware dynamic eviction, extension to 14B+ models,
  and production serving integration.

  INSIGHT-7: Conclusion should restate the causal chain in one sentence, then list concrete next steps. Do NOT add new claims or results
  here.

  ---
  Appendix (2 pages)

  A. Layer-wise budget allocation: Layer-group (lower/middle/upper) budget experiment. Brief table + one figure showing per-layer retention
  patterns. (~0.5 pages)

  B. System overhead analysis: Policy inference time, eviction overhead, amortized analysis. Show that learned policy adds <5ms per eviction
  step and memory savings enable 2-3× larger batch size. (~0.5 pages)

  C. Implementation details: Hyperparameters, training procedure, segmenter specifics, feature computation cost. (~0.5 pages)

  D. Additional results: Per-task breakdown, per-model curves, failure case examples. (~0.5 pages)

  ---
  INSIGHT Collection

  ┌───────────┬─────────────────────────────────────────────────────────────────────────────────────────────┬──────────────┬─────────────┐
  │    ID     │                                           Insight                                           │   Chapter    │    Type     │
  ├───────────┼─────────────────────────────────────────────────────────────────────────────────────────────┼──────────────┼─────────────┤
  │ INSIGHT-1 │ Abstract must establish causal chain (mismatch → Q3 → correction) in one sentence; avoid    │ Abstract     │ structural  │
  │           │ feature listing                                                                             │              │             │
  ├───────────┼─────────────────────────────────────────────────────────────────────────────────────────────┼──────────────┼─────────────┤
  │ INSIGHT-2 │ Introduction must make Q3 feel inevitable (structural consequence of mismatch), not         │ §1 Intro     │ argument    │
  │           │ invented; one concrete example before Section 3                                             │              │             │
  ├───────────┼─────────────────────────────────────────────────────────────────────────────────────────────┼──────────────┼─────────────┤
  │ INSIGHT-3 │ Related Work: every paragraph ends with "unlike X, we..." positioning; keep under 1 page    │ §2 Related   │ positioning │
  ├───────────┼─────────────────────────────────────────────────────────────────────────────────────────────┼──────────────┼─────────────┤
  │ INSIGHT-4 │ Motivation needs empirical Q3 evidence (pilot scatter plot), not just logical argument; Q3  │ §3           │ evidence    │
  │           │ figure is the paper's most impactful visual                                                 │ Motivation   │             │
  ├───────────┼─────────────────────────────────────────────────────────────────────────────────────────────┼──────────────┼─────────────┤
  │ INSIGHT-5 │ Method section flows causally: need compressed oracle → discover Q3 → target Q3 → segments  │ §4 Method    │ flow        │
  │           │ prevent boundary-split Q3; each subsection references its motivation driver                 │              │             │
  ├───────────┼─────────────────────────────────────────────────────────────────────────────────────────────┼──────────────┼─────────────┤
  │ INSIGHT-6 │ Every experiment table advances one link in the causal chain; experiments that don't        │ §5           │ selection   │
  │           │ directly serve the story go to Appendix                                                     │ Experiments  │             │
  ├───────────┼─────────────────────────────────────────────────────────────────────────────────────────────┼──────────────┼─────────────┤
  │ INSIGHT-7 │ Conclusion restates causal chain in one sentence, lists concrete next steps (layer-wise,    │ §6           │ discipline  │
  │           │ uncertainty-aware in Appendix); no new claims                                               │ Conclusion   │             │
  └───────────┴─────────────────────────────────────────────────────────────────────────────────────────────┴──────────────┴─────────────┘

  ---
  Priority Action Items

  1. Run Q3 pilot experiment FIRST — before writing anything, you need the scatter plot showing confident-wrong eviction exists under
  SieveKV. This is the paper's empirical foundation.
  2. Implement OP-SieveKV-Lite MVP — fixed blocks + on-policy distillation + Soft-OR + global policy. Get initial numbers on NIAH and
  multi-needle.
  3. Then add semantic segments — compare sentence-level vs 16-token blocks on multi-hop tasks. Document Q3 error reduction.
  4. Write from the causal chain — every section, every paragraph, every figure should advance: mismatch → Q3 → correction → segments →
  results.

  ---
  This is the chapter plan. The next step is yours — either start the Q3 pilot experiment, or tell me which section you want to draft first.