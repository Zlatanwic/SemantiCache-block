# Query-Aware Semantic KV Cache Retention for Long-Context LLM Serving Under Tight Memory Budgets

## Abstract

Long-context large language model (LLM) serving is increasingly constrained by key-value (KV) cache growth during prefill and decoding. Existing cache-retention policies typically prioritize either recency or generic token salience, but these strategies often degrade factual recall under tight budgets, especially when relevant spans are surrounded by confusable distractors. This paper presents SemantiCache, a query-aware semantic KV-cache retention mechanism for long-context LLM serving. SemantiCache combines attention-based importance, information-density estimates, query relevance, factual-token detection, and role-aware protection of system instructions and the latest user query. The design further uses contiguous block retention to preserve positional stability during decoding and selectively retains the strongest tokens when only part of a block fits in the budget. We implement SemantiCache in a reproducible evaluation harness and compare it with full-cache execution, local window retention, StreamingLLM-style recency retention, and H2O-style heavy-hitter retention. On an adversarial long-context retrieval benchmark with authoritative and distractor spans, SemantiCache improves accuracy from 0.922 to 0.933 at the best retained frontier while preserving perfect hard-slice accuracy and substantially outperforming recency-only baselines under tight budgets. These results suggest that query-aware semantic retention is a promising systems direction for memory-efficient LLM inference.

## 1. Introduction

Serving large language models over long contexts is increasingly limited by the memory cost of the KV cache. Even when model weights fit on commodity accelerators, long prompts and long generations can cause KV state to dominate memory consumption and reduce serving efficiency. This tension is especially acute for practical deployments that must trade context length against latency, throughput, and hardware cost.

Prior work has proposed several approaches to limit KV-cache growth. Recency-based methods such as StreamingLLM preserve recent tokens and attention sinks to support long-running decoding without model retraining [1]. Salience-based methods such as H2O retain heavy-hitter tokens identified by attention statistics [2]. These designs are effective in many settings, but they still face failure modes when the answer depends on a small factual span embedded in long, noisy context. In such cases, generic salience and recency may be insufficient: the serving system must retain the right spans for the current query while preserving enough local structure to keep decoding stable.

This paper explores that systems problem through SemantiCache, a query-aware semantic KV-cache retention mechanism for long-context LLM serving. SemantiCache is motivated by a simple observation: in retrieval-heavy long-context workloads, not all tokens contribute equally to downstream answer quality, and the most valuable tokens are often identifiable through a combination of semantic signals rather than through recency alone. At the same time, token retention cannot be arbitrarily sparse, because aggressive token-level pruning may destabilize decoding by disrupting positional continuity.

SemantiCache addresses these two requirements jointly. First, it scores cached tokens using a blend of attention pressure, information density, entropy, query relevance, and factual cues. Second, it retains tokens in contiguous semantic blocks, while protecting system instructions, the latest user-query tail, and a recent decode window. The mechanism therefore aims to preserve both retrieval-critical content and decoding stability under fixed cache budgets.

This work makes the following contributions:

- It formulates query-aware KV-cache retention as a serving-time systems problem for long-context LLM inference under fixed memory budgets.
- It presents SemantiCache, a practical retention policy that combines semantic scoring, role-aware pinning, and block-level retention with partial-block token selection.
- It implements a reproducible evaluation harness with fixed benchmark suites and adversarial distractor cases for analyzing retention behavior under tight budgets.
- It shows that the resulting design improves adversarial retrieval accuracy over strong lightweight baselines while avoiding the severe failure modes of recency-only retention.

The rest of the paper is organized as follows. Section 2 discusses related work. Section 3 describes the SemantiCache design. Section 4 summarizes the implementation. Section 5 presents the evaluation methodology and current results. Section 6 discusses limitations and threats to validity. Section 7 concludes.

## 2. Related Work

### 2.1 KV-cache-efficient LLM inference

A growing body of work studies how to reduce the memory overhead of autoregressive LLM inference. StreamingLLM shows that models trained with finite-length attention can support effectively unbounded streaming through the preservation of attention sinks and recent context [1]. H2O proposes a heavy-hitter oracle that retains a combination of recent tokens and attention-dominant tokens, reducing memory while preserving quality in many settings [2]. These approaches establish that substantial KV compression is possible without retraining.

However, these methods primarily optimize for generic retention criteria. For workloads in which a user query targets a small factual span inside a long prompt, the most useful cache entries may be those aligned with the query and the underlying fact structure rather than those favored by recency or aggregate attention alone.

### 2.2 Long-context evaluation

LongBench provides a multitask benchmark for long-context understanding across question answering, summarization, few-shot learning, and code-related tasks [3]. RULER further emphasizes controlled long-context evaluation by probing the effective usable context length of long-context models [4]. More recently, NoLiMa highlights a central weakness of long-context systems by reducing lexical overlap between the question and the supporting span, thereby moving beyond literal-matching evaluation [5].

These benchmarks motivate a stricter view of long-context quality: high nominal context length does not guarantee reliable retrieval of relevant evidence under distraction. Our current harness is aligned with that perspective and focuses specifically on retention under authoritative-versus-distractor settings, while future work will extend the evaluation to public benchmark suites such as RULER, LongBench, and NoLiMa.

### 2.3 Positioning of this work

SemantiCache differs from prior KV-cache policies in two ways. First, it explicitly biases retention toward the current query and likely factual spans. Second, it treats structural retention as part of the systems design: preserving the right tokens is insufficient unless the remaining cache layout still supports stable generation. In this sense, the work sits at the boundary of LLM serving, cache management, and workload-aware inference optimization.

## 3. SemantiCache Design

### 3.1 Problem setting

We consider autoregressive generation with a fixed KV-cache budget expressed as a fraction of the prefill length. Given a prompt and a user query, the system must decide which cached tokens to retain during decoding so as to maximize answer quality while respecting the budget. A desirable policy should satisfy three properties:

- It should preserve answer-critical spans for the current query.
- It should avoid destabilizing generation through overly fragmented retention.
- It should incur low decision overhead at inference time.

### 3.2 Token scoring

SemantiCache computes a keep-oriented score for each token using a weighted combination of five signals:

- attention-based importance
- information density
- head entropy
- query relevance
- factual-token likelihood

The first three signals capture generic importance from the model state and local token statistics. Query relevance biases retention toward spans that overlap with the latest user request. Factual-token likelihood boosts tokens that resemble structured answer-bearing content such as dates, times, numbers, units, and other high-information spans frequently needed for retrieval-style questions.

In the current implementation, attention, density, and entropy are combined through weights `alpha`, `beta`, and `gamma`, while query and factual signals are controlled by separate floors that prevent them from becoming negligible. The retained frontier that currently defines the checked-in method was obtained after iterative keep/discard experiments that raised the query-weight floor to `0.30` and the factual-weight floor to `0.40`, which improved the adversarial benchmark frontier without changing the evaluation harness. These settings are documented in the experiment log and strategy report [6], [7].

### 3.3 Role-aware protection

Not all tokens should compete equally for eviction. SemantiCache therefore protects three classes of tokens before generic scoring is applied:

- all system tokens
- the tail of the latest user message
- a recent decode window

This choice is motivated by an observed tradeoff between semantic selectivity and generation stability. Weakening the latest-user tail or the recent decode region can reduce coherence or cause brittle failures even when factual spans remain partially retained. The role-aware protection mechanism narrows the search space by reserving part of the budget for tokens that are disproportionately important for preserving task intent and local autoregressive consistency.

### 3.4 Block-level retention

Early sparse token-level pruning led to unstable generation despite preserving nominally important tokens. SemantiCache therefore retains tokens in contiguous semantic blocks rather than arbitrary sparse subsets. The block-level design serves two purposes:

- it reduces positional fragmentation in the remaining cache
- it allows the system to preserve small answer-bearing spans together with enough local context to remain decodable

Blocks are ranked using the safest token within the block rather than the block-average score. This change prevents a single critical token from being drowned out by surrounding low-value tokens. If only part of a block fits within the budget, SemantiCache selects the strongest tokens within that block and then restores their original order. Together, these two decisions improved the retained frontier on the adversarial benchmark from `83/90` to `84/90` while preserving perfect hard-slice accuracy [6].

## 4. Implementation

SemantiCache is implemented as a serving-time KV-cache policy in a lightweight research codebase. The implementation consists of:

- a cache manager for budget-aware retention decisions
- policy implementations for full-cache, local-window, streaming-style, H2O-style, and semantic retention
- a semantic analyzer that computes role tags and factual/query-related signals
- a generation harness with deterministic decoding
- a benchmark harness that logs results for fixed suites and retains complete experiment history

The current evaluation uses `Qwen/Qwen2.5-3B-Instruct` as the underlying model [8]. The environment targets a commodity `RTX 5060 Laptop GPU` with `8 GB` of VRAM, emphasizing the practical setting in which KV-cache growth becomes a real systems bottleneck [9]. The benchmark harness supports multiple suites, including a stricter adversarial suite in which authoritative facts are surrounded by confusable distractor spans [10].

An important implementation lesson is that cache-retention quality cannot be studied independently from generation stability. Explicit handling of position consistency and contiguous retention shape were necessary precursors to the current semantic policy; without them, sparse pruning caused repetitive or collapsed outputs even when the retained tokens were intuitively relevant [7].

## 5. Evaluation

### 5.1 Methodology

The current evaluation focuses on adversarial retrieval-oriented long-context workloads. Each case embeds an authoritative factual statement within a long haystack and places distractor statements nearby. The model is instructed to answer using the authoritative statement only. This design stresses the exact failure mode most relevant to query-aware retention: preserving the correct factual span despite strong lexical and structural distraction.

The benchmark harness defines three suites: a smoke suite for debugging, a frontier suite for stable retention comparisons, and a gauntlet suite for harder adversarial evaluation [10]. The present paper reports current results from the gauntlet suite, which contains `90` cases spanning multiple haystack lengths, budgets, and insertion positions. The primary metrics are:

- task accuracy
- hard-slice accuracy
- average response time

The hard slice contains the longest contexts and tightest budgets in the suite, providing a stress-test view of the policy frontier.

### 5.2 Baselines

We compare SemantiCache against four policies:

- `full`: no eviction
- `window`: local-window retention
- `streaming`: a StreamingLLM-style recency policy
- `h2o`: a heavy-hitter retention policy

These baselines are implemented in the same codebase and evaluated through the same deterministic harness, limiting confounding differences in generation settings or infrastructure [9], [10].

### 5.3 Current results

The current checked-in frontier achieves `0.933333` accuracy (`84/90`) on the gauntlet suite, with `1.000000` hard-slice accuracy (`30/30`) and an average response time of `7.262574 s` [6], [7]. This result improves over the earlier gauntlet baseline of `0.922222` (`83/90`) while maintaining perfect hard-slice accuracy and avoiding the severe degradation observed for recency-only methods under lower budgets.

Additional targeted results on a fixed Needle-in-a-Haystack configuration show the qualitative advantage of semantic retention under constrained budgets. At `50%` budget, both H2O and SemantiCache recover the full factual answer, while local-window and streaming baselines fail. At `30%` budget, the initial semantic policy still fails, but the query-aware and factual-aware version recovers the correct answer, whereas the H2O baseline remains near-correct but factually wrong in at least one tested case [11].

These results support three claims. First, preserving query-aligned factual spans matters under tight memory budgets. Second, generic semantic importance alone is not sufficient; the policy benefits from explicitly modeling query relevance and factual structure. Third, retention shape matters: block-level contiguous retention is necessary to translate token-level importance into stable generation.

### 5.4 What remains to be validated

The present evidence is promising but incomplete for a full systems evaluation. In particular, the current study is limited to one model family and a custom adversarial workload. Before submission, the evaluation should be extended in three directions:

- public benchmark suites such as RULER and LongBench
- additional model families and scales
- explicit overhead analysis of retention decisions versus saved KV memory

These extensions would more fully establish generality and systems relevance.

## 6. Discussion

### 6.1 Limitations

This work has several limitations. First, the current evaluation is centered on retrieval-style factual questions, not on the full space of long-context reasoning tasks. Second, the strongest current evidence comes from a custom benchmark harness rather than from public standardized suites. Third, the implementation has been evaluated primarily on a single small open model and a commodity GPU setting. As a result, the current conclusions should be interpreted as evidence of feasibility rather than as a final claim of broad superiority.

### 6.2 Threats to validity

The main internal threat is benchmark specialization: a retention policy can overfit to the structure of a fixed evaluation suite. The iterative keep/discard search loop used in this project improves the frontier on the same benchmark family and therefore risks narrowing the policy toward those cases [6]. The main external threat is model specificity: token salience signals and factual heuristics that work well on one tokenizer or one model family may not transfer unchanged to others.

### 6.3 Lessons learned

Two lessons stand out. First, decoding stability is a first-class systems concern in KV-cache compression; preserving important tokens is not enough if the retained cache is structurally brittle. Second, the most useful retention signals are workload dependent. For retrieval-heavy long-context serving, query alignment and factual cues appear more valuable than generic salience alone. This suggests that future cache managers should be workload aware rather than model-state aware only.

## 7. Conclusion

This paper presented SemantiCache, a query-aware semantic KV-cache retention mechanism for long-context LLM serving under tight memory budgets. SemantiCache combines role-aware protection, semantic scoring, and contiguous block retention to preserve answer-critical spans while maintaining generation stability. In a reproducible adversarial retrieval harness, the method improves the retained frontier over simpler baselines and demonstrates that query-aware factual retention can materially improve low-budget behavior. Future work will extend the evaluation to public long-context benchmarks, additional model families, and broader serving metrics such as throughput and cost efficiency.

## References

[1] G. Xiao, Y. Tian, B. Chen, S. Han, and M. Lewis, "Efficient Streaming Language Models with Attention Sinks," arXiv:2309.17453, 2023. Available: https://arxiv.org/abs/2309.17453

[2] Z. Zhang, Y. Sheng, T. Zhou, T. Zheng, L. V. B. Krishna, A. Torralba, M. M. Hwu, Z. Wang, and S. Han, "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models," arXiv:2306.14048, 2023. Available: https://arxiv.org/abs/2306.14048

[3] Y. Bai, X. Lv, J. Zhang, H. Liao, G. Chen, Z. Liu, Y. Cui, and N. Duan, "LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding," arXiv:2308.14508, 2023. Available: https://arxiv.org/abs/2308.14508

[4] N. Hsieh et al., "RULER: What's the Real Context Size of Your Long-Context Language Models?" arXiv:2404.06654, 2024. Available: https://arxiv.org/abs/2404.06654

[5] A. Modarressi et al., "NoLiMa: Long-Context Evaluation Beyond Literal Matching," arXiv:2502.05167, 2025. Available: https://arxiv.org/abs/2502.05167

[6] "Current strategy report," SemantiCache repository, current frontier summary and experiment analysis. Available: [CURRENT_STRATEGY_REPORT.md](d:\semanticache\CURRENT_STRATEGY_REPORT.md)

[7] "Autoresearch results log," SemantiCache repository, iterative keep/discard experiment history. Available: [results.tsv](d:\semanticache\results.tsv)

[8] "Global configuration for SemantiCache experiments," repository configuration. Available: [config.py](d:\semanticache\config.py)

[9] "SemantiCache: Semantics-Aware KV Cache Eviction for LLM Inference," repository README. Available: [README.md](d:\semanticache\README.md)

[10] "Autoresearch benchmark harness for SemantiCache policy experiments," repository benchmark harness. Available: [benchmark_autoresearch.py](d:\semanticache\benchmark_autoresearch.py)

[11] "SemantiCache optimization summary," repository evaluation summary. Available: [results/semantic_optimization_summary.md](d:\semanticache\results\semantic_optimization_summary.md)
