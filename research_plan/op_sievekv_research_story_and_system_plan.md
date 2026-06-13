# OP-SieveKV 研究故事线与系统化实现计划

本文档是 OP-SieveKV 后续论文撰写和代码实现的主蓝图。它整合了两条关键启发：

1. **TrOPD 的算法启发**：on-policy 监督并不天然可靠，需要区分 trust region 与 oracle outlier；对 outlier 不应简单过滤，而应在完整数据分布上进行保守 reweighting。
2. **SAS 的系统启发**：系统论文不应只提出一个 sparse/retention pattern，而应给出从 pattern/decision 描述、分析、cache sizing、cache management 到 runtime profiling 的完整流水线。

因此，OP-SieveKV 后续应从“一个 learned KV eviction policy”升级为一个完整的 **语义 KV 保留系统**：离线通过 on-policy oracle 学习哪些语义片段应该保留，在线把 learned retention decision 编译成可执行、可分析、可 profile 的 compact KV cache plan。

---

## 1. 一句话主张

**OP-SieveKV 是一个 on-policy oracle 蒸馏式语义 KV cache retention runtime。它将静态结构保护、动态 learned retention score 和 compact KV cache materialization 统一到一个 RetentionPlan 中，用于长上下文 LLM 推理中的 memory-quality-latency trade-off 优化。**

更短的答辩版本：

> OP-SieveKV 把 KV cache eviction 从一个 heuristic token scoring 问题，转化为一个 on-policy decision learning + runtime planning 问题。

---

## 2. 研究动机与问题定义

长上下文 LLM 推理中，KV cache 显存占用随上下文长度线性增长。即使 attention 计算可以通过 flash attention 或 sparse attention 优化，decode 阶段仍然需要维护越来越长的 K/V 历史。对于 batch serving、RAG、多文档问答和 agent planning，KV cache 往往成为吞吐、延迟和最大上下文长度的实际瓶颈。

现有 KV cache 压缩或 eviction 方法大致有几类：

- recent window / sink token：保留最近 token 和少量 attention sink；
- heavy hitter：根据历史 attention 选择重要 token；
- SnapKV / H2O 类方法：prefill 或 decode 阶段用注意力统计挑选 KV；
- KVzip 类方法：通过 reconstruction 或 query-agnostic 压缩估计 token 重要性；
- semantic heuristic：用 query relevance、factual likelihood、role pinning 等人工信号保留语义片段。

这些方法有效，但仍有三个核心缺口。

### 2.1 Proxy importance 缺口

Attention score、query overlap、recency 和 heuristic factual score 都是 proxy signal。它们不能直接回答最关键的问题：

> 如果删除这个 segment，最终答案概率是否会下降？

OP-SieveKV 的第一步是引入 counterfactual oracle，用 drop / restore probe 构造更接近因果贡献的 oracle label。

### 2.2 Off-policy oracle 缺口

如果 oracle 在 full-KV prompt 上运行，那么它学习的是完整上下文下的静态重要性。但真实 serving 中，policy 面对的是自己已经压缩后的 cache state。一个 segment 在 full-KV 下可能冗余，但在其他 evidence 被删掉后可能变得关键。

因此，oracle 必须在当前 policy 的 compressed-cache state 上运行。

### 2.3 系统视角缺口

许多 KV eviction 工作只定义 importance score，没有完整说明：

- score 如何变成 segment-level decision；
- segment decision 如何变成 token-level KV mask；
- token-level mask 如何 materialize 成 compact KV cache；
- cache slot 如何映射回原始 token index；
- 实际 cache size、eviction overhead、decode throughput 如何统计。

借鉴 SAS 的系统论文写法，OP-SieveKV 需要显式引入 **RetentionPlan** 作为中间表示，将算法输出和系统执行连接起来。

---

## 3. 总体故事线

论文可以按下面这条线讲：

```text
KV cache 是长上下文 serving 的核心瓶颈
-> 现有 eviction 多依赖 proxy signal
-> full-context oracle 又存在 off-policy mismatch
-> 我们让 policy 在自己的 compressed cache 上 rollout
-> 用 drop/restore oracle 找到真实错删和冗余保留
-> 用 KV-TIP 诊断 policy 的自信错误
-> 用 trust-region oracle reweighting 稳定训练轻量 policy
-> 在线阶段把 policy score 编译成 RetentionPlan
-> materialize compact KV cache，并报告 quality/memory/latency
```

对应论文贡献：

1. **On-policy compressed-cache oracle**：在 policy 自己的压缩状态上做 drop / restore counterfactual。
2. **KV-TIP taxonomy**：用 retention entropy 和 oracle-policy divergence 识别 uncertainty、Q3 confident-wrong 和 missed-keep。
3. **Trust-region oracle reweighting**：保留完整 on-policy dataset，仅对 oracle outliers 加权，避免 naive filtering 破坏分布。
4. **RetentionPlan runtime**：把静态保护和动态 learned score 统一成可执行 KV cache plan。
5. **系统化评估**：不仅报告 accuracy，还报告实际 cache tokens、prefill/decode latency、eviction/materialization overhead 和 throughput。

---

## 4. 相关工作定位

### 4.1 与 StreamingLLM / H2O / SnapKV / KVzip 的关系

这些 baseline 可以统一看作“如何在有限 KV budget 下选择历史 tokens”。它们通常依赖规则或 attention/reconstruction proxy，而 OP-SieveKV 的区别是：

- 用 oracle label 修正 proxy signal；
- oracle 在 compressed-cache state 上运行；
- retention unit 从 token/block 升级为 semantic segment；
- 最终输出不是单纯 score，而是可执行 RetentionPlan。

### 4.2 与 TrOPD 的关系

TrOPD 的关键启发不是公式本身，而是：

> on-policy supervision 需要可靠性建模。

迁移到 KV retention：

- policy-oracle aligned rows 是 trust-region support；
- high-divergence rows 是 oracle outliers；
- Q3 / missed-keep 是最值得纠正的 outlier；
- naive top-rho filtering 会破坏 calibration；
- full-dataset reweighting 更稳定。

这直接解释了当前实验现象：Soft-OR filtering 在 multi-needle 中容易崩，而 Soft-OR reweighting 更稳。

### 4.3 与 SAS 的关系

SAS 的启发是系统组织方式：

```text
sparse pattern 描述
-> pattern analysis
-> KV cache sizing
-> cache index / mask function generation
-> backend execution
-> latency/memory profiling
```

OP-SieveKV 的系统对应为：

```text
semantic segment 描述
-> feature / policy scoring
-> retention plan synthesis
-> compact KV cache materialization
-> decode runtime
-> quality/memory/latency profiling
```

这使 OP-SieveKV 不只是算法，也是一套 runtime 设计。

---

## 5. 核心研究问题

### RQ1：如何定义 segment 对最终答案的真实贡献？

使用 counterfactual oracle：

- retained segment：删除它，看答案 log-prob 是否下降；
- evicted segment：恢复它，看答案 log-prob 是否上升；
- delta 越大，说明该 segment 对当前 compressed-cache state 越关键。

### RQ2：为什么 oracle 必须 on-policy？

因为 segment 重要性依赖当前 cache state。policy 已经删除了哪些 evidence，会改变剩余 segment 的边际价值。

所以 oracle state 应该是：

```text
C^pi_B = 当前 policy pi 在 budget B 下产生的 compressed KV cache
```

而不是完整 prompt。

### RQ3：哪些 oracle decision 最应该影响训练？

不是所有 oracle rows 都同等重要。需要区分：

- aligned rows：维持 calibration；
- high entropy rows：学习决策边界；
- high divergence rows：纠正 policy-oracle mismatch；
- Q3 / missed-keep rows：修复自信错删 evidence。

### RQ4：如何把 learned policy 变成系统？

policy 输出 keep probability 还不够。系统必须继续回答：

- 哪些 semantic segments 被选中；
- 对应哪些 token indices；
- compact cache slot 如何分配；
- 哪些 token 被 pin、evict、refresh；
- 实际 cache size 是否等于 nominal budget；
- runtime overhead 来自哪里。

这就是 RetentionPlan 的作用。

---

## 6. 方法总览

OP-SieveKV 包含离线训练和在线 serving 两个阶段。

### 6.1 离线训练阶段

```text
训练样本 prompt/query/answer
-> semantic segmentation
-> 当前 policy 在 budget 下执行 eviction
-> 得到 compressed-cache state
-> 对 candidate segments 做 drop/restore oracle
-> 得到 oracle label 和 oracle metadata
-> 计算 KV-TIP 指标
-> trust-region oracle reweighting
-> 训练轻量 retention policy
```

### 6.2 在线 serving 阶段

```text
prompt/query
-> semantic segmentation
-> feature extraction
-> learned retention policy scoring
-> RetentionPlan synthesis
-> compact KV cache materialization
-> decode
-> runtime profiling
```

在线阶段不调用 oracle。

---

## 7. Semantic Segment 设计

OP-SieveKV 的基本决策单位不是固定 token block，而是 semantic segment。

推荐层级：

```text
document
-> paragraph / retrieved chunk / code block / table row
-> sentence group
-> token block fallback
```

当前代码可以先使用已有 OP-SieveKV-Lite 的 sentence/role-aware segmentation，后续扩展到 LongBench/RAG/code/table。

系统层需要一个稳定数据结构：

```python
@dataclass
class SegmentInfo:
    segment_id: int
    token_start: int
    token_end: int
    token_count: int
    text: str
    role: str
    source: str
    is_pinned: bool
    is_recent: bool
    metadata: dict
```

这已经在 `retention_plan.py` 中开始实现。

---

## 8. On-Policy Compressed-Cache Oracle

### 8.1 Drop oracle

对于当前被保留的 segment：

```text
delta_drop(g)
= log P(answer | C^pi_B)
- log P(answer | C^pi_B without g)
```

如果 delta 高，说明删除该 segment 会伤害答案，因此它应该保留。

### 8.2 Restore oracle

对于当前被删除的 segment：

```text
delta_restore(g)
= log P(answer | C^pi_B with g restored)
- log P(answer | C^pi_B)
```

如果 delta 高，说明 policy 错删了它，即 missed-keep。

### 8.3 Oracle label

将 delta 转为 soft label：

```text
y* = sigmoid(delta / tau)
```

或在高置信场景下使用 threshold label。

每条 oracle row 应记录：

- `oracle_mode`: drop / restore；
- `oracle_delta`；
- `oracle_label`；
- `on_policy = True`；
- `oracle_state_type = compressed_kv`；
- `policy_keep_prob`；
- `budget`。

---

## 9. KV-TIP Taxonomy

KV-TIP 用两轴分类 retention decisions。

### 9.1 Retention entropy

```text
h(p) = -p log p - (1-p) log(1-p)
```

高 entropy 表示 policy 不确定。

### 9.2 Oracle-policy divergence

```text
d = |y* - p|
```

高 divergence 表示 policy 与 oracle 分歧大。

### 9.3 四象限

| 区域 | Entropy | Divergence | 含义 | 训练作用 |
|---|---:|---:|---|---|
| Q1 | 低 | 低 | 自信且正确 | calibration support |
| Q2 | 高 | 低 | 不确定但基本正确 | boundary smoothing |
| Q3 | 低 | 高 | 自信但错误 | critical correction |
| Q4 | 高 | 高 | 不确定且错误 | informative correction |

最重要的 failure mode：

```text
missed_keep = p < 0.5, y* >= 0.5, oracle_mode = restore
```

这类错误只有 on-policy restore oracle 才容易发现。

---

## 10. Trust-Region Oracle Reweighting

### 10.1 为什么不能把 Soft-OR filtering 当最终方法？

Soft-OR filtering 会只保留高价值 rows：

```text
selected rows = top-rho by z
```

它适合做 ablation，但不适合作为完整方法。原因是它会删除大量普通 negative rows、aligned rows 和 heuristic support rows，导致 learned policy calibration 失衡，在 multi-needle 中容易只保护少数 evidence，丢失覆盖能力。

### 10.2 完整方法：保留全量数据，只加权 outlier

Soft-OR score：

```text
h_hat = normalize(h)
d_hat = normalize(d)
z = h_hat + d_hat - h_hat * d_hat
```

在每个 budget 内选择 top-rho oracle rows，然后更新权重：

```text
w'_i = w_i
     * m_selected^{I[selected]}
     * m_Q3^{I[Q3]}
     * m_missed^{I[missed_keep]}
```

最后 clamp：

```text
w'_i = min(w'_i, w_max)
```

推荐参数：

```text
rho = 0.30
m_selected = 2.0
m_Q3 = 1.5
m_missed = 1.5
w_max = 20
```

当前代码入口：

```bash
python select_op_oracle_dataset.py \
  --dataset results/op_oracle_dataset_onpolicy_student.pt \
  --output results/op_oracle_dataset_onpolicy_trust_region_rw.pt \
  --method-preset trust_region_reweight \
  --rho 0.30 \
  --selected-weight-multiplier 2.0 \
  --q3-weight-multiplier 1.5 \
  --missed-keep-weight-multiplier 1.5 \
  --max-weight 20 \
  --output-md results/v2/trust_region_reweight_selection.md \
  --output-json results/v2/trust_region_reweight_selection.json
```

训练：

```bash
python train_op_policy.py \
  --dataset results/op_oracle_dataset_onpolicy_trust_region_rw.pt \
  --output results/op_policy_onpolicy_trust_region_rw.pt \
  --epochs 300 \
  --batch-size 256 \
  --lr 0.001 \
  --early-stop-patience 40 \
  --print-budget-metrics-every 25
```

---

## 11. 系统视角：Retention Runtime

系统贡献应写成：

> OP-SieveKV separates retention specification, policy scoring, retention planning, and cache materialization.

中文解释：

> OP-SieveKV 将“保留规则描述”“policy 打分”“保留计划合成”和“KV cache 实际 materialization”拆成清晰的系统层，使 learned retention policy 可以被执行、被调试、被 profile。

### 11.1 Runtime pipeline

```text
Prompt + Query
-> Semantic Segmenter
-> Feature Extractor
-> Learned Retention Policy
-> RetentionPlan Synthesizer
-> Compact KV Cache Materializer
-> Decoder Runtime
-> Quality / Memory / Latency Profiler
```

### 11.2 RetentionPlan IR

`RetentionPlan` 是算法和系统之间的中间表示。

核心字段：

```python
@dataclass
class RetentionDecision:
    segment_id: int
    token_start: int
    token_end: int
    keep_prob: float | None
    score: float | None
    selected: bool
    reason: str
    cache_slot_start: int | None
    cache_slot_end: int | None

@dataclass
class CacheEvent:
    event_type: Literal["store", "pin", "promote", "demote", "evict", "refresh"]
    segment_id: int
    token_start: int
    token_end: int
    step: int
    reason: str

@dataclass
class RetentionPlan:
    policy: str
    prompt_tokens: int
    budget_ratio: float
    budget_tokens: int
    retained_tokens: int
    evicted_tokens: int
    segments: list[SegmentInfo]
    decisions: list[RetentionDecision]
    events: list[CacheEvent]
    token_to_cache_slot: dict[int, int]
    stats: dict
```

第一版已在 `retention_plan.py` 实现。

### 11.3 Static + Dynamic composition

借鉴 SAS 的 static/dynamic sparse pattern：

Static retention：

- system prompt pinning；
- latest query pinning；
- recent window；
- boundary support；
- role-aware protection。

Dynamic retention：

- learned keep probability；
- semantic relevance；
- trust-region reweighted policy；
- decode-time uncertainty refresh。

组合规则：

```text
final_keep = static_pin OR top_budget(dynamic_score)
```

---

## 12. Cache Materialization 与 Profiling

### 12.1 Cache materialization

Policy score 必须转化成实际 KV layout：

```text
segment decisions
-> retained token indices
-> compact token order
-> old_token_id -> cache_slot_id mapping
-> pruned KV tensors
-> decode attention mask / position handling
```

系统需要记录：

- nominal budget tokens；
- actual retained tokens；
- pinned tokens；
- dynamic selected tokens；
- evicted tokens；
- cache materialization calls；
- decode refresh calls。

### 12.2 Runtime stats

建议统一输出：

```json
{
  "prompt_tokens": 3832,
  "budget_ratio": 0.30,
  "budget_tokens": 1149,
  "actual_cache_tokens": 1149,
  "evicted_tokens": 2739,
  "prefill_time_s": 0.29,
  "segmentation_time_s": 0.0,
  "feature_time_s": 0.0,
  "policy_time_s": 0.0,
  "materialization_time_s": 0.0,
  "decode_time_s": 5.89,
  "generated_tokens": 58,
  "tokens_per_second": 9.8,
  "eviction_steps": 57
}
```

---

## 13. 论文算法伪代码

### Algorithm 1: On-policy oracle collection

```text
Input: prompts D, policy pi, budgets B, model M
Output: oracle dataset O

for each example (x, q, a) in D:
    build semantic segments G
    for each budget b in B:
        p = pi(features(G, q, b))
        C = materialize_compressed_cache(x, G, p, b)
        base_score = log P_M(a | C)
        candidates = select_oracle_candidates(G, p, C)

        for each candidate g:
            if g is retained:
                C_drop = remove_segment(C, g)
                delta = base_score - log P_M(a | C_drop)
                mode = drop
            else:
                C_restore = restore_segment(C, g)
                delta = log P_M(a | C_restore) - base_score
                mode = restore

            y = oracle_label(delta)
            store(features(g), y, delta, mode, p_g, b)
```

### Algorithm 2: Trust-region oracle reweighting

```text
Input: oracle dataset O, rho, multipliers
Output: reweighted dataset O'

for row in O:
    if row has oracle label:
        h = entropy(policy_keep_prob)
        d = abs(oracle_label - policy_keep_prob)
        decision_type = classify(row)

normalize h and d over oracle rows
z = h_hat + d_hat - h_hat * d_hat

for each budget:
    S_b = top-rho oracle rows by z

for row in O:
    weight = base_weight
    if row in S_b:
        weight *= selected_multiplier
    if row is Q3:
        weight *= q3_multiplier
    if row is missed_keep:
        weight *= missed_keep_multiplier
    weight = min(weight, max_weight)

return full dataset with updated weights
```

### Algorithm 3: Online OP-SieveKV runtime

```text
Input: prompt x, query q, policy pi, budget b
Output: answer y, runtime stats

segments = segment(x)
features = extract_features(segments, q, b)
keep_probs = pi(features)
plan = synthesize_retention_plan(segments, keep_probs, b)
compact_cache = materialize_kv_cache(x, plan)

while decoding:
    if refresh condition:
        update dynamic stats
        optionally update plan
    generate next token using compact_cache

return answer and profiling stats
```

---

## 14. 后续代码路线

### Phase 1：RetentionPlan IR 与导出工具

已开始实现：

- `retention_plan.py`
- `dump_retention_plan.py`

目标：

```bash
python dump_retention_plan.py \
  --policy op_sievekv_lite \
  --op-policy-ckpt results/op_policy_onpolicy_trust_region_rw.pt \
  --budget 0.1 \
  --task niah \
  --output results/v2/case_study/retention_plan.json
```

输出内容：

- segments；
- retained / evicted decisions；
- reason；
- cache slots；
- token_to_cache_slot；
- plan stats。

### Phase 2：Runtime profiler

新增 `profile_opsievekv_runtime.py`：

```bash
python profile_opsievekv_runtime.py \
  --policies full semantic op_sievekv_lite \
  --op-ckpt trust_rw=results/op_policy_onpolicy_trust_region_rw.pt \
  --budgets 0.3 0.2 0.1 0.05 \
  --manifest results/v2/manifests/niah_120.json \
  --output results/v2/runtime_profile.json
```

统计：

- prompt tokens；
- actual retained tokens；
- peak KV tokens；
- prefill time；
- decode time；
- policy overhead；
- materialization overhead；
- tok/s。

### Phase 3：Runtime summary table

新增 `summarize_runtime_profile.py`：

```bash
python summarize_runtime_profile.py \
  --input results/v2/runtime_profile.json \
  --output-md results/v2/runtime_profile_summary.md \
  --append-to results/v2/experiment_summary.md
```

### Phase 4：动态 runtime 扩展

作为后续增强：

- decode-time uncertainty score；
- refresh interval；
- high uncertainty 时临时扩大 budget；
- cache event logging；
- layer-wise budget allocation。

---

## 15. 实验计划

### 15.1 主质量实验

固定 manifest：

- NIAH single-needle，n=120；
- multi-needle，k=2/4/8，n=90；
- 至少一个非 needle benchmark：LongBench HotpotQA 或 gov_report。

Budgets：

```text
5%, 10%, 20%, 30%
```

Methods：

- full KV；
- StreamingLLM；
- H2O；
- SnapKV；
- KVzip；
- semantic heuristic；
- onpolicy student；
- trust-region reweight OP-SieveKV。

### 15.2 消融实验

- all decisions；
- all oracle；
- entropy-only；
- divergence-only；
- Q3-only；
- missed-keep-only；
- Soft-OR filter；
- trust-region reweight。

预期叙事：

- Soft-OR 信号有用；
- naive filtering 会破坏 multi-needle；
- reweighting 保留分布，因此更稳定。

### 15.3 系统实验

同一硬件上报告：

- context scaling：2K / 4K / 8K / 16K；
- budget scaling：5 / 10 / 20 / 30%；
- overhead breakdown；
- actual vs nominal cache size；
- throughput 和 latency。

### 15.4 诊断指标

报告：

- oracle rows by budget；
- drop vs restore count；
- missed-keep count；
- Q3 count；
- trust-region vs outlier count；
- selected outlier rows；
- output weight mean/max。

---

## 16. 论文结构建议

### Abstract

必须包含：

- KV cache memory bottleneck；
- existing heuristic/off-policy mismatch；
- on-policy compressed-cache oracle；
- trust-region reweighted learned retention policy；
- semantic KV runtime and profiling；
- 主要 quality 和 system 结果。

### 1. Introduction

建议顺序：

1. 长上下文推理受 KV cache memory 限制。
2. 现有方法依赖 proxy importance 或 static sparse pattern。
3. proxy signal 在 retrieval/multi-evidence 下不稳定。
4. full-context oracle 存在 off-policy mismatch。
5. 提出 OP-SieveKV。
6. 总结贡献：
   - on-policy compressed-cache oracle；
   - KV-TIP and trust-region oracle reweighting；
   - RetentionPlan runtime；
   - fixed-manifest quality and system evaluation。

### 2. Background and Motivation

包括：

- KV cache cost；
- eviction baselines；
- semantic segment retention；
- on-policy oracle 的必要性；
- missed-keep restore case。

### 3. OP-SieveKV Algorithm

包括：

- semantic segmentation；
- feature design；
- on-policy oracle；
- KV-TIP；
- trust-region reweighting；
- training objective。

### 4. OP-SieveKV Runtime

这是 SAS 启发的系统章节。

包括：

- architecture diagram；
- RetentionPlan IR；
- static/dynamic retention composition；
- cache materialization；
- profiling hooks；
- online complexity。

### 5. Experiments

包括：

- setup；
- datasets/manifests；
- baselines；
- quality tables；
- ablations；
- system profiling；
- case study。

### 6. Discussion

包括：

- 为什么 on-policy signal 有价值；
- 为什么 filtering 会失败；
- semantic baseline 仍强说明什么；
- limitations：
  - 当前 oracle 是 post-prefill compressed-cache oracle，还不是完整 decode trajectory oracle；
  - layer-wise budget 尚未完整实现；
  - dynamic eviction 仍是未来工作；
  - 非 needle benchmark 还需补强；
  - 没有做 kernel-level optimization。

### 7. Conclusion

核心句：

> KV cache retention should be treated as an on-policy decision learning and runtime planning problem, not merely as token scoring.

---

## 17. 当前进度与缺口

已具备：

- on-policy compressed-cache oracle；
- drop and restore oracle；
- missed-keep / Q3 信号；
- trust-region reweighting 代码路径；
- fixed-manifest NIAH/multi baseline 表；
- 多个外部 baseline。

需要补强：

- RetentionPlan IR 的主流程接入；
- runtime profiling table；
- 非 needle benchmark；
- case study visualization；
- system architecture section；
- dynamic/layer-wise 作为未来工作或继续实现。

---

## 18. 答辩故事线

完整版本：

> 我们从一个实际系统瓶颈出发：长上下文 serving 中 KV cache memory 迅速增长。已有 eviction 方法轻量，但大多依赖 proxy signal。我们引入 on-policy compressed-cache oracle，在 policy 自己的压缩状态下问一个因果问题：哪些 segment 真的影响答案？这个 oracle 发现了 prompt-level oracle 看不到的 missed-keep 错误。随后我们用 KV-TIP 诊断 policy 的自信错误，并发现 naive Soft-OR filtering 会破坏 multi-evidence retention。借鉴 trust-region on-policy distillation，我们改为保留完整数据分布，只对 oracle outliers 加权。最后，借鉴 SAS 的系统视角，我们把 OP-SieveKV 写成一个 semantic KV retention runtime：它将 policy score 转化为 RetentionPlan，并 materialize 成 compact KV cache layout，从而系统评估 quality、memory 和 latency 的 trade-off。

短版本：

> OP-SieveKV 把 KV cache eviction 从 heuristic scoring 升级为 on-policy oracle-guided decision learning，再进一步落到可执行的 RetentionPlan runtime。

