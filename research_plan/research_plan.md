# Research Plan: OP-SieveKV: On-Policy Oracle-Distilled Adaptive Semantic KV Retention for Memory-Efficient Long-Context LLM Serving

## 1. 研究题目

**OP-SieveKV: On-Policy Oracle-Distilled Adaptive Semantic KV Retention for Memory-Efficient Long-Context LLM Serving**

中文题目可以写作：

**OP-SieveKV：面向长上下文大模型推理的 On-Policy Oracle 蒸馏式自适应语义 KV Cache 保留方法**

---

## 2. 研究背景与动机

大语言模型在长上下文推理、RAG、多文档问答、代码理解和智能体规划等场景中，需要处理越来越长的输入序列。随着上下文长度增加，KV cache 成为推理阶段最主要的显存瓶颈之一。KV cache 的大小随 sequence length 线性增长，在 batch size、context length、throughput 和硬件成本之间形成明显冲突。因此，如何在有限 KV cache budget 下尽量保留对生成质量最关键的上下文信息，是长上下文 LLM serving 中的重要问题。

已有 KV cache 压缩方法大致可以分为几类：基于 recent window 的保留、基于 attention heavy hitter 的 eviction、prefill 阶段的一次性剪枝、query-agnostic 的 reconstruction-based compression，以及 KV quantization / offloading 等。它们虽然能够降低显存占用，但在 retrieval-heavy 场景中仍存在明显问题：如果答案依赖于长上下文中某个稀疏 factual span，纯 attention、recency 或 query-agnostic 信号往往难以稳定区分真正的 answer-critical tokens 和语义相似的 distractors。

当前已有的 SieveKV 方法通过五类轻量信号进行 KV eviction：attention mass、information density、head entropy、query relevance 和 factual-token likelihood，并结合 role-aware pinning 与 contiguous block retention，在 RULER Needle-in-a-Haystack 等检索任务上取得了较好的效果。这个思路说明：query-aware、content-aware 的多信号组合确实能增强长上下文检索型推理下的 KV 保留质量。

然而，现有 SieveKV 仍然具有明显的进一步提升空间：

1. **固定权重问题**：不同任务、不同 query、不同 budget 下，各类信号的重要性并不相同。检索问答可能更依赖 query relevance 和 factual likelihood，summarization 可能更依赖 attention coverage，代码任务可能更依赖 syntax / identifier density。固定权重难以适配复杂任务分布。

2. **固定 block 问题**：原方法使用固定长度 block，例如 16-token block。固定 token block 容易切断语义边界，导致实体、关系和答案值被拆开保留或拆开删除。对于 RAG、多文档 QA、代码、表格等结构化场景，语义片段级 retention 更自然。

3. **shared-index 问题**：现有方法通常让所有 layer / head 保留同一批 token index。但 Transformer 不同层和不同 head 的功能并不相同。低层可能更关注局部结构，高层可能更关注 query-aligned evidence，中层可能更关注实体和关系组合。因此每层使用相同保留集合可能不是最优。

4. **off-policy oracle 问题**：如果只用 full-KV masking 生成静态 oracle importance，那么训练得到的 policy 只是在 full-KV 状态下学习“哪些 token 重要”。但实际 serving 中，policy 面对的是已经被自己多次 eviction 后的 compressed cache state。这个状态分布与 full-KV 状态存在 mismatch。

5. **confidence-blind 问题**：固定 heuristic 或普通 supervised distillation 可能只关注明显不确定的 segment，但容易漏掉另一类更危险的错误：policy 非常自信地 drop 了某个看似不相关、但实际上对 multi-hop / retrieval / planning 至关重要的 segment。这类 “confident but wrong eviction” 需要专门建模。

因此，本研究计划提出一种新的 KV cache retention 学习框架：**OP-SieveKV**。它将原有 SieveKV 的多信号启发式方法升级为一种 **on-policy oracle-distilled adaptive semantic retention policy**。核心思想是：让当前 KV retention policy 在自己的 compressed-cache trajectory 上执行 eviction，然后用 full-KV / counterfactual oracle 纠正它在真实压缩状态下犯的错误；同时借鉴 on-policy distillation 中 entropy + divergence 的 token selection 思想，识别最有信息量的 retention decisions，尤其关注 policy 自信但错误的 Q3-type eviction cases。

### 2.1 2026-06 方法修订：从 Soft-OR filtering 到 Trust-Region Oracle Reweighting

阅读 TrOPD（Trust Region On-Policy Distillation）后，本计划对 Soft-OR/KV-TIP 的使用方式做一个重要修订：**Soft-OR 不再被视为最终训练集过滤器，而是用于识别 on-policy oracle outliers；最终方法保留完整 on-policy 数据分布，并对这些 outliers 做 reweighting。**

这个修订来自一个更一般的判断：on-policy label 并不天然可靠。当前 policy 在自己的 compressed-cache state 上产生 retention decisions，其中一部分 decision 与 oracle 接近，可以视为 trust-region support；另一部分 decision 与 oracle 分歧很大，尤其是 missed-keep / Q3 confident-wrong cases，可以视为 oracle-outlier。直接过滤出 top-rho outliers 训练，会删除大量普通负类、teacher support 和稳定 aligned rows，容易破坏 policy calibration；而保留全量数据、只提高 outlier 权重，可以在纠正错删 evidence 的同时维持原始 retention 分布。

因此，本计划中的完整训练路线更新为：

1. 收集 on-policy compressed-cache oracle dataset；
2. 对 oracle rows 计算 retention entropy、oracle-policy divergence 和 Soft-OR score；
3. 每个 budget 内选择 top-rho oracle outliers；
4. 不过滤训练集，而是在完整 dataset 上对 selected outliers 乘以额外权重；
5. 对 Q3 confident-wrong 和 missed-keep restore positives 进一步加权；
6. 使用 weighted BCE 训练轻量 retention policy；
7. 将 filtering-only 版本保留为 ablation，用于证明 naive selection 会造成分布破坏。

详细实现规范、命令和字段约定见 [`trust_region_reweighting_design.md`](trust_region_reweighting_design.md)。

---

## 3. 研究目标

本研究的总体目标是：

**设计一种面向长上下文 LLM serving 的自适应 KV cache retention 方法，在有限 KV budget 下实现更优的 accuracy-memory-latency trade-off，并提升方法在 retrieval、multi-hop reasoning、RAG、summarization 和 agentic planning 等任务上的泛化能力。**

具体目标包括：

1. **从固定 heuristic 升级为 learned adaptive policy**
   将原有 SieveKV 中人工设定的多信号权重，升级为由 oracle distillation 学到的轻量 policy，使其能够根据 query、role、position、budget、layer 和上下文统计特征动态预测 segment 的 keep probability。

2. **从 fixed block retention 升级为 semantic segment retention**
   将固定长度 token block 替换为句子、段落、RAG chunk、代码函数、表格行等语义片段，提高保留上下文的完整性和可解释性。

3. **从 off-policy oracle distillation 升级为 on-policy oracle distillation**
   不只在 full-KV 状态下生成标签，而是在当前 policy 自己的 compressed cache trajectory 上生成 counterfactual oracle label，缓解训练分布与推理分布不一致的问题。

4. **提出 KV-TIP taxonomy**
   借鉴 TIP 中 student entropy 与 teacher-student divergence 的两轴思想，提出 KV retention 中的 retention entropy 与 oracle-policy divergence 两轴分类，识别 policy 不确定样本和 policy 自信但错误的 eviction 决策。

5. **实现 layer-wise budget allocation**
   根据 layer-level attention statistics、query concentration 和 retention uncertainty，为不同层分配不同 KV budget，突破所有层共享同一 token index 的限制。

6. **实现 uncertainty-aware dynamic eviction**
   在 decode-time 根据生成熵、attention entropy、policy margin 和 oracle-inspired confidence，动态调整 eviction aggressiveness 与 eviction frequency，降低误删关键证据的风险。

---

## 4. 核心研究问题

本研究围绕以下几个核心问题展开：

### RQ1: 如何定义 KV cache 中 segment 的真实重要性？

传统 attention score 或 query overlap 只能提供 proxy signal，不能直接回答“删掉这个 segment 是否会影响最终答案”。本研究计划通过 counterfactual masking / restoration 构造 oracle importance：

* 如果一个 segment 被保留，删除它后答案概率明显下降，说明它重要；
* 如果一个 segment 被删除，恢复它后答案概率明显提升，说明 policy 错删了它；
* 如果删除或恢复都几乎不影响结果，说明该 segment 对当前任务贡献较低。

### RQ2: 如何将 full-KV oracle 蒸馏成轻量级 online retention policy？

Oracle 计算成本高，不可能进入实际 serving 路径。因此需要训练一个轻量 policy，在推理时仅根据可快速计算的特征预测 keep probability：

[
p_{j,l} = \pi_\theta(\text{keep} \mid \phi_{j,l})
]

其中 (g_j) 表示第 (j) 个语义 segment，(l) 表示 layer，(\phi_{j,l}) 包括多信号统计、role、position、layer、budget 和 query features。

### RQ3: 为什么 on-policy distillation 对 KV retention 重要？

如果训练 label 来自 full-KV 状态，policy 学到的是 full context 下的静态重要性。但在真实 serving 中，cache 会被当前 policy 连续压缩，重要性会随已保留/已删除内容而变化。因此，本研究将让 policy 在自己的 compressed-cache state 上产生 trajectory，再由 oracle 针对这些 on-policy states 生成 correction label。

### RQ4: 如何识别 policy “自信但错误”的 eviction 决策？

借鉴 TIP 的 Q3 概念，本研究定义 KV retention 中的 Q3 case：

[
\text{low retention entropy} + \text{high oracle-policy divergence}
]

也就是：policy 对 keep/drop 决策非常自信，但 oracle 发现该决策明显错误。例如 policy 自信地 drop 了一个 query overlap 很低但对 multi-hop reasoning 至关重要的 intermediate evidence。识别并纠正这类 case 将是提升鲁棒性的关键。

### RQ5: 如何保证 learned policy 不引入过高 latency？

KV cache 优化的目标不是简单增加一个复杂模型，而是在 memory、quality、latency 之间取得更优 trade-off。因此 policy 必须满足：

* 推理阶段轻量；
* 静态特征 prefill 后缓存；
* 动态特征低频更新；
* eviction amortized 执行；
* 不把 oracle 或 teacher 放进 online serving 路径。

---

## 5. 方法设计

### 5.1 总体框架

OP-SieveKV 包含两个阶段：离线训练阶段和在线推理阶段。

#### 离线训练阶段

离线阶段允许使用较高成本的 oracle，用于训练轻量 retention policy。流程如下：

1. 输入 prompt、query 和 gold answer；
2. 运行当前 retention policy 进行 on-policy compressed-cache rollout；
3. 对 policy 保留和删除的 candidate segments 进行 counterfactual drop / restore；
4. 计算 oracle segment importance；
5. 构造 retention entropy 和 oracle-policy divergence；
6. 用 Soft-OR / KV-TIP 识别最有训练价值的 oracle-outlier decisions；
7. 保留完整 on-policy dataset，对 selected outliers 做 trust-region reweighting；
8. 使用 weighted BCE loss、ranking loss 和 budget-aware regularization 更新轻量 policy。

#### 在线推理阶段

在线阶段不再使用 oracle，只使用训练好的轻量 policy。流程如下：

1. 对 prompt 进行 prefill；
2. 构造 semantic segments；
3. 计算并缓存静态 semantic features；
4. 根据 query、budget、role、position 和 layer features 预测 keep probability；
5. 分层分配 budget；
6. 按 segment score 构造 layer-wise KV mask；
7. decode 过程中根据 uncertainty 低频更新 eviction mask；
8. 输出最终答案。

---

## 5.2 Semantic Segment Construction

原始 SieveKV 使用固定长度 token block，例如 16-token block。OP-SieveKV 将其升级为 semantic segment。

不同输入类型采用不同 segmenter：

1. **普通文本**
   使用 sentence-level 或 paragraph-level segmentation。对于长段落，可以进一步切分为 sentence groups。

2. **RAG / 多文档问答**
   使用 retrieval chunk、document title + passage、paragraph group 作为 segment。这样可以保留完整 evidence chunk。

3. **代码任务**
   使用函数、类、代码块、statement group 作为 segment。必要时可借助 AST 或简单正则规则切分。

4. **表格任务**
   使用 row、column group、cell neighborhood 作为 segment。

5. **Chat prompt**
   按 system message、user query、retrieved context、assistant history、latest user query 等 role 进行结构化划分。

为了防止 segment 太长，本研究采用 hierarchical segmentation：

[
\text{document} \rightarrow \text{paragraph} \rightarrow \text{sentence} \rightarrow \text{token block}
]

当某个高分 segment 超过 budget 时，在其内部继续细粒度选择。

---

## 5.3 Segment Feature Design

对于每个 segment (g_j) 和 layer (l)，构造特征向量：

[
\phi_{j,l} = [\phi_{signal}, \phi_{role}, \phi_{position}, \phi_{layer}, \phi_{budget}, \phi_{query}]
]

### 5.3.1 Multi-signal features

继承原始 SieveKV 的五类信号：

1. attention mass (s_\alpha)
2. information density (s_\beta)
3. head entropy (s_\gamma)
4. query relevance (s_q)
5. factual-token likelihood (s_f)

对每个 segment 做 pooling：

[
\text{Pool}*k(g_j) =
[
\max*{i \in g_j} s_k(i),
\text{mean}*{i \in g_j} s_k(i),
\text{std}*{i \in g_j} s_k(i)
]
]

其中 (k) 表示不同 signal type。

### 5.3.2 Role features

使用 one-hot 或 embedding 表示 segment role：

[
r_j \in {
\text{system},
\text{query},
\text{context},
\text{retrieved-doc},
\text{assistant-history},
\text{recent-decode}
}
]

system prompt、latest query 和 recent decode tokens 可以继续采用 hard pinning 或 high-priority soft pinning。

### 5.3.3 Position features

加入归一化位置和距离特征：

[
pos_j = \frac{start(g_j)}{L}
]

[
dist_j = \frac{|pos(g_j) - pos(q)|}{L}
]

同时加入 segment length、是否位于 beginning / middle / end 等特征，以处理 lost-in-the-middle 问题。

### 5.3.4 Layer features

加入 layer index：

[
\ell = \frac{l}{L_{model}}
]

也可以使用 layer group embedding：

[
l \in {\text{lower}, \text{middle}, \text{upper}}
]

先从 layer group 版本做起，降低工程复杂度。

### 5.3.5 Budget features

加入当前 cache budget：

[
b = \frac{B}{N}
]

以及 remaining budget、target compression ratio、current cache size 等。

### 5.3.6 Query features

加入 query length、query 中实体数量、数字比例、疑问词数量、query-context overlap peak、query overlap entropy 等特征，用于判断当前任务是否更偏检索、多跳、总结或开放生成。

---

## 5.4 Offline Oracle Importance via Counterfactual Masking

给定 gold answer (a)、prompt (x)、query (q)，定义当前 policy 下的 compressed cache：

[
C_t^{\pi_\theta}
]

### 5.4.1 Drop importance

对于已经被 policy 保留的 segment (g_j)，计算删除它后的影响：

[
I^{drop}_{j,l}
==============

## \log P(a \mid C_t^{\pi_\theta})

\log P(a \mid C_t^{\pi_\theta} \setminus g_{j,l})
]

如果 (I^{drop}_{j,l}) 大，说明该 segment 在当前 compressed cache state 下是必要的。

### 5.4.2 Restore importance

对于已经被 policy 删除的 segment (g_j)，计算恢复它后的收益：

[
I^{restore}_{j,l}
=================

## \log P(a \mid C_t^{\pi_\theta} \cup g_{j,l})

\log P(a \mid C_t^{\pi_\theta})
]

如果 (I^{restore}_{j,l}) 大，说明 policy 错删了该 segment。

### 5.4.3 Oracle label

将 importance 转化为 oracle keep probability：

[
y^*_{j,l}
=========

\sigma(I_{j,l}/\tau)
]

其中 (\tau) 为温度参数。

也可以使用 ranking label：

[
g^+ \succ g^-
]

即 oracle importance 高的 segment 应该排在 importance 低的 segment 前面。

---

## 5.5 On-Policy Retention Distillation

### 5.5.1 为什么需要 on-policy

如果只用 full-KV 状态生成 oracle label，则 policy 学到的是静态 full-context importance。但在线推理时，policy 面对的是自己压缩后的 cache state。某些 segment 在 full-KV 下不重要，但在其他 evidence 被删掉后可能变得重要；某些 segment 在 full-KV 下重要，但在当前 compressed state 中可能已经失去作用。

因此，本研究采用 on-policy distillation：

[
C_t^{\pi_\theta}
================

\text{Evict}(C_t, \pi_\theta, B)
]

让 policy 在自己的 compressed-cache state 上产生 trajectory，再用 oracle 修正它在这些真实状态下的错误。

### 5.5.2 Policy definition

定义轻量 retention policy：

[
p_{j,l}
=======

\pi_\theta(a_{j,l}=1 \mid \phi_{j,l})
]

其中 (a_{j,l}=1) 表示保留第 (j) 个 segment 在第 (l) 层的 KV entries。

policy 可以采用两种形式：

#### Direct keep-probability policy

[
p_{j,l}=\sigma(MLP(\phi_{j,l}))
]

#### Adaptive gated scoring policy

[
\mathbf{w}*{j,l}=G*\theta(\phi_{j,l})
]

[
S_{j,l}
=======

\sum_{k=1}^{5}
w_{j,l,k}
\cdot
\text{Pool}_{i\in g_j}(s_k(i))
]

[
p_{j,l}=\sigma(S_{j,l})
]

第二种更可解释，因为可以分析不同任务、不同层、不同 budget 下 policy 更依赖哪类 signal。

---

## 5.6 KV-TIP: Entropy-Divergence Taxonomy for Retention Decisions

借鉴 on-policy distillation 中的 entropy-divergence token importance 思想，本研究提出 KV-TIP taxonomy。

### 5.6.1 Retention entropy

policy 对 segment 的 keep/drop 决策有概率：

[
p_{j,l} = \pi_\theta(\text{keep} \mid \phi_{j,l})
]

定义 retention entropy：

[
h_{j,l}
=======

*

## p_{j,l}\log p_{j,l}

(1-p_{j,l})\log(1-p_{j,l})
]

高 entropy 表示 policy 不确定该 segment 是否应该保留。

### 5.6.2 Oracle-policy divergence

oracle label 为 (y^*_{j,l})，定义 oracle-policy divergence：

[
\delta_{j,l}
============

D_{KL}
(
\text{Bern}(p_{j,l})
\Vert
\text{Bern}(y^*_{j,l})
)
]

也可以用 BCE 近似：

[
\delta_{j,l}
============

*

## y^**{j,l}\log p*{j,l}

(1-y^**{j,l})\log(1-p*{j,l})
]

高 divergence 表示 policy 和 oracle 分歧大。

### 5.6.3 Four quadrants

根据 retention entropy 和 oracle-policy divergence，将 segment decisions 分为四类：

| 区域 | Retention Entropy | Oracle-Policy Divergence | 含义                                        |
| -- | ----------------- | ------------------------ | ----------------------------------------- |
| Q1 | 高                 | 高                        | policy 不确定且判断错，强训练信号                      |
| Q2 | 高                 | 低                        | policy 不确定但判断尚可，边界稳定信号                    |
| Q3 | 低                 | 高                        | policy 自信但错误，尤其是 confident wrong eviction |
| Q4 | 低                 | 低                        | policy 已经会了，训练价值低                         |

其中最重要的是 Q3。它对应于 policy 很自信地删除了重要 evidence，或者很自信地保留了大量无用 distractors。这类错误在 multi-hop reasoning、agentic planning 和 RAG 中尤其危险。

### 5.6.4 Soft-OR informative decision selection

归一化：

[
\hat h_{j,l} = \text{Normalize}(h_{j,l})
]

[
\hat \delta_{j,l} = \text{Normalize}(\delta_{j,l})
]

定义 Soft-OR score：

[
z_{j,l}
=======

\hat h_{j,l}
+
\hat \delta_{j,l}
-----------------

\hat h_{j,l}\hat \delta_{j,l}
]

选择 top-(\rho) decisions 进行训练：

[
\mathcal{T}
===========

\text{TopK}*{j,l}(z*{j,l}, \rho)
]

这样可以保留两类 informative decisions：

1. policy 不确定的 decisions；
2. policy 自信但与 oracle 分歧大的 decisions。

同时跳过大量低 entropy、低 divergence 的 solved decisions，降低训练成本。

### 5.6.5 Trust-Region Oracle Reweighting

实验中需要区分 **diagnostic filtering** 和 **full-method reweighting**：

* **Filtering**：只保留 support rows 与 Soft-OR top-rho oracle rows。它适合作为消融，用来回答“高 entropy / 高 divergence / Q3 是否包含有效训练信号”。但它会改变训练分布，容易使 policy 过度关注少数强 oracle rows，导致 multi-needle evidence coverage 下降。
* **Reweighting**：保留完整 on-policy dataset，只对 Soft-OR top-rho oracle rows 提高权重。它对应完整方法，因为普通 rows 仍然提供 calibration、negative examples 和 heuristic support，而 oracle outliers 通过更高 loss weight 得到强化。

借鉴 TrOPD 的 trust-region 思想，本研究将 oracle-policy divergence 低的 rows 视为 trust-region support，将 divergence 高的 rows 视为 oracle outliers。对这些 outliers 不直接丢弃，也不单独训练，而是在完整数据上乘以权重：

[
w'_{j,l}
=
w_{j,l}
\cdot
m_{selected}
\cdot
m_{Q3}^{\mathbb{1}[Q3]}
\cdot
m_{missed}^{\mathbb{1}[\text{missed-keep}]}
]

其中：

* (m_{selected})：Soft-OR top-rho oracle row 的基础增强系数；
* (m_{Q3})：policy 自信但 oracle 认为错误时的额外增强；
* (m_{missed})：restore oracle 发现 missed-keep 时的额外增强；
* 权重最终会被 (w_{max}) clamp，避免少数样本主导训练。

这一设计对应当前代码中的 `select_op_oracle_dataset.py --method-preset trust_region_reweight`。它把 Soft-OR 从“训练集过滤规则”改成“oracle outlier weighting rule”，更符合 on-policy distillation 中“只在可靠区域直接学习、对 outlier 采取保守处理”的思想。

---

## 5.7 Training Objective

训练目标由三部分构成。

### 5.7.1 Oracle distillation loss

[
\mathcal{L}_{distill}
=====================

\sum_{(j,l)\in\mathcal{T}}
\text{BCE}(y^**{j,l},p*{j,l})
]

### 5.7.2 Ranking loss

对于 oracle importance 高的 segment (g^+) 和低的 segment (g^-)：

[
\mathcal{L}_{rank}
==================

-\log \sigma(S_{g^+,l}-S_{g^-,l})
]

### 5.7.3 Budget regularization

约束预期保留 KV 数量不超过 budget：

[
\mathcal{L}_{budget}
====================

\max(0,\sum_j p_{j,l}|g_j|-B_l)^2
]

最终 loss：

[
\mathcal{L}
===========

\mathcal{L}*{distill}
+
\lambda \mathcal{L}*{rank}
+
\mu \mathcal{L}_{budget}
]

可选加入 latency-aware regularization：

[
\mathcal{L}_{latency}
=====================

c_1 \cdot #\text{segments}
+
c_2 \cdot #\text{eviction calls}
+
c_3 \cdot \text{policy cost}
]

---

## 5.8 Layer-wise Budget Allocation

不同 layer 对不同类型信息的依赖不同，因此 OP-SieveKV 设计 layer-wise budget：

[
B_l = \rho_l B_{total}
]

其中 (\rho_l) 可由 layer-level statistics 决定：

[
\rho_l =
\text{softmax}
(
aH_l + bQ_l + cD_l
)
]

其中：

* (H_l)：第 (l) 层 attention entropy；
* (Q_l)：query-relevant attention mass；
* (D_l)：long-range dependency score。

实际实现中可以先采用 layer-group 方案：

[
l \in {\text{lower}, \text{middle}, \text{upper}}
]

这样只需要三套 retention masks，降低工程复杂度。

---

## 5.9 Uncertainty-Aware Dynamic Eviction

在 decode-time，eviction 不应该以固定频率和固定强度执行。OP-SieveKV 根据 uncertainty 动态调整 eviction aggressiveness。

### 5.9.1 Uncertainty score

定义综合不确定性：

[
U_t =
\lambda_1 H_y(t)
+
\lambda_2 H_A(t)
+
\lambda_3 \frac{1}{M_t+\epsilon}
]

其中：

* (H_y(t))：生成 token 分布熵；
* (H_A(t))：attention entropy；
* (M_t)：top-1 和 top-2 segment score margin。

### 5.9.2 Dynamic budget

根据 (U_t) 调整 budget：

[
b_t =
\begin{cases}
b_{high}, & U_t > \tau_h \
b_{mid}, & \tau_l \le U_t \le \tau_h \
b_{low}, & U_t < \tau_l
\end{cases}
]

高不确定性时更保守，保留更多 KV；低不确定性时更激进，压缩更多 KV。

### 5.9.3 Amortized eviction

eviction 不需要每步执行，可以每 (n) 步更新一次：

[
t \mod n = 0
]

例如 (n=8) 或 (n=16)。这样可以显著降低 policy 和 selection 的平均 per-step overhead。

---

## 6. 算法流程

### 6.1 Training Algorithm

```
Algorithm 1: On-Policy Oracle Distillation for OP-SieveKV

Input:
  Training prompts D
  Initial retention policy πθ
  LLM model M
  Global KV budget B
  Informative decision retention ratio ρ

1. Initialize πθ from original SieveKV heuristic or offline oracle distillation.

2. For each training iteration:
   a. Sample prompt x, query q, answer a from D.
   b. Run prefill and construct semantic segments G = {g1, ..., gm}.
   c. Compute static features:
        information density,
        query relevance,
        factual likelihood,
        role,
        position,
        query features.
   d. Use current policy πθ to predict p_{j,l}.
   e. Allocate layer-wise budget B_l.
   f. Execute eviction under πθ and obtain compressed cache C^{πθ}.
   g. Decode under compressed cache and obtain on-policy trajectory.
   h. Select candidate segments for oracle evaluation.
   i. For kept candidates, compute drop importance.
   j. For dropped candidates, compute restore importance.
   k. Convert importance to oracle label y*_{j,l}.
   l. Compute retention entropy h_{j,l}.
   m. Compute oracle-policy divergence δ_{j,l}.
   n. Compute Soft-OR score z_{j,l}.
   o. Select top-ρ oracle outlier decisions within each budget.
   p. Keep the full on-policy dataset and multiply weights for selected outliers.
   q. Apply extra multipliers for Q3 confident-wrong and missed-keep rows.
   r. Clamp weights to avoid a few oracle rows dominating optimization.
   s. Update πθ using weighted distillation loss, ranking loss, and budget regularization.

Output:
  Lightweight retention policy πθ.
```

### 6.2 Online Inference Algorithm

```
Algorithm 2: OP-SieveKV Online Serving

Input:
  Prompt x, query q, trained policy πθ, budget B

1. Run prefill.
2. Segment prompt into semantic segments.
3. Compute static semantic features once.
4. Compute request-level and segment-level features.
5. Predict keep probabilities p_{j,l} with πθ.
6. Allocate layer-wise budget B_l.
7. Select retained segments for each layer.
8. Build KV masks and prune cache.
9. During decoding:
     a. Update attention-derived statistics every n steps.
     b. Compute uncertainty U_t.
     c. Adjust eviction budget and frequency.
     d. Update retention masks if needed.
10. Generate final answer.
```

---

## 7. 实验计划

### 7.1 Models

建议从较小模型开始，逐步扩大：

1. Qwen2.5-3B-Instruct
2. Llama-3.2-3B-Instruct
3. Qwen2.5-7B-Instruct 或 Qwen3-8B
4. 若资源允许，扩展到 14B 级别模型

这样可以验证：

* 小模型上的方法可行性；
* 不同架构之间的 transfer；
* larger model 上的 scalability。

### 7.2 Benchmarks

实验不应只停留在 RULER NIAH，需要扩展到更多任务。

#### Retrieval-oriented tasks

* RULER Needle-in-a-Haystack
* Multi-needle NIAH
* Variable-depth retrieval
* Variable-length retrieval

#### Multi-hop reasoning

* HotpotQA-style long-context setting
* MuSiQue-style multi-hop QA
* 自构造 multi-hop needle benchmark

#### Long-context understanding

* LongBench
* NoLiMa
* Multi-document QA

#### Summarization

* GovReport
* QMSum
* Multi-document summarization subsets

#### RAG / Agentic planning

* 多文档 RAG QA
* DeepPlanning-style constrained planning
* Tool-use trajectory compression if available

### 7.3 Baselines

需要和以下方法比较：

1. Full KV
2. StreamingLLM
3. H2O
4. SnapKV
5. KVzip
6. Original SieveKV
7. Offline Oracle Distillation
8. OP-SieveKV without Soft-OR
9. OP-SieveKV full version

如果能实现，也可以加入：

* KIVI / KVQuant as orthogonal quantization baseline
* Eviction + quantization hybrid variant

### 7.4 Main Metrics

评估指标分三类。

#### Quality metrics

* Exact match
* Retrieval accuracy
* F1
* ROUGE / BLEU for summarization
* Constraint satisfaction rate for planning
* Multi-hop answer correctness

#### Memory metrics

* KV cache memory
* Peak GPU memory
* Compression ratio
* Maximum supported context length
* Maximum batch size under fixed GPU memory

#### Latency / system metrics

* Prefill latency
* Decode latency
* Eviction overhead per step
* Tokens per second
* Time-to-first-token
* Time-per-output-token
* P50 / P95 / P99 latency
* End-to-end serving throughput

### 7.5 Core Experiments

#### Experiment 1: Main accuracy-memory trade-off

比较不同 cache budget 下的性能：

[
b \in {50%, 30%, 20%, 10%, 5%}
]

观察 OP-SieveKV 是否在相同 memory budget 下优于 SieveKV 和其他 baselines。

#### Experiment 2: Offline vs On-policy distillation

比较：

* Fixed SieveKV
* Offline oracle-distilled policy
* On-policy oracle-distilled policy

验证 on-policy 是否缓解 distribution mismatch。

#### Experiment 3: KV-TIP taxonomy effectiveness

比较训练 decision selection 方法：

* all decisions
* entropy-only
* divergence-only
* Q3-only
* Soft-OR

验证 Soft-OR 是否能更好选择 informative retention decisions。

#### Experiment 4: Q3 overconfident eviction analysis

分析 policy 自信但错误的 cases：

* confident drop important evidence
* confident keep distractor
* multi-hop intermediate evidence deletion
* low lexical overlap but high causal importance segment

展示具体 case study。

#### Experiment 5: Semantic segment vs fixed block

比较：

* 16-token block
* 32-token block
* sentence segment
* paragraph segment
* RAG chunk
* hierarchical segment

验证 semantic segment 是否优于固定 block。

#### Experiment 6: Layer-wise budget ablation

比较：

* shared-index retention
* same budget per layer
* layer-group budget
* full layer-wise adaptive budget

验证 per-layer retention 的必要性。

#### Experiment 7: Uncertainty-aware dynamic eviction

比较：

* fixed budget
* fixed frequency
* dynamic budget
* dynamic frequency
* dynamic budget + dynamic frequency

观察是否能改善 worst-case performance 和 latency trade-off。

#### Experiment 8: System overhead analysis

重点回答审稿人关心的问题：

* learned policy 是否增加 latency？
* amortized eviction 是否有效？
* policy overhead 占 decode time 的比例是多少？
* 是否能通过更大 batch size 抵消单请求 overhead？
* 在固定显存下 throughput 是否提升？

---

## 8. 消融实验设计

### 8.1 Feature ablation

移除不同特征：

* remove attention mass
* remove information density
* remove head entropy
* remove query relevance
* remove factual likelihood
* remove role features
* remove position features
* remove layer features
* remove budget features
* remove query features

观察各部分贡献。

### 8.2 Policy form ablation

比较：

* linear model
* 2-layer MLP
* adaptive gated scoring
* direct keep probability
* decision tree / rule-distilled policy

目标是找到 accuracy 与 latency 的最佳平衡。

### 8.3 Oracle type ablation

比较：

* full-KV leave-one-out oracle
* on-policy drop oracle
* on-policy restore oracle
* drop + restore combined oracle
* approximate oracle with sampled candidates

### 8.4 Training selection ablation

比较：

* train on all decisions
* entropy-only selection
* divergence-only selection
* Soft-OR selection
* Q3-only selection
* random selection

### 8.5 Layer granularity ablation

比较：

* global policy
* layer group policy
* full layer-wise policy
* per-head policy if feasible

---

## 9. 预期贡献

本研究预期形成以下贡献：

### Contribution 1: On-Policy Retention Distillation

提出一种面向 KV cache retention 的 on-policy oracle distillation 框架。不同于传统 full-KV 静态 oracle，本方法在当前 policy 自己生成的 compressed-cache state 上构造 counterfactual oracle label，从而缓解训练和推理状态分布不一致的问题。

### Contribution 2: KV-TIP Taxonomy

提出 retention entropy 与 oracle-policy divergence 两轴分类方法，将 KV retention decisions 分为 Q1/Q2/Q3/Q4 四类，尤其识别 policy 自信但错误的 Q3-type eviction cases。该分析为理解 learned KV retention policy 的错误模式提供了新的视角。

### Contribution 3: Adaptive Semantic Segment Retention

将固定 token block retention 升级为 semantic segment retention，并通过轻量 policy 结合多信号特征、role、position、query、budget 和 layer 信息预测 keep probability，提高对复杂任务的泛化能力。

### Contribution 4: Layer-wise and Uncertainty-aware KV Budgeting

提出 layer-wise budget allocation 和 uncertainty-aware dynamic eviction，使不同 layer 和不同 decode state 下的 KV budget 能够动态调整，从而改善 accuracy-memory-latency trade-off。

### Contribution 5: Comprehensive System Evaluation

在 retrieval、multi-hop reasoning、long-context understanding、summarization 和 agentic planning 等任务上进行系统评估，并同时报告 accuracy、memory、latency、throughput 和 worst-case performance。

---

## 10. 研究创新点总结

本研究相对于原始 SieveKV 的升级可以概括为：

| 维度          | SieveKV                            | OP-SieveKV                            |
| ----------- | ---------------------------------- | ------------------------------------- |
| 权重          | 人工固定                               | oracle-distilled adaptive policy      |
| 粒度          | fixed 16-token block               | semantic / hierarchical segment       |
| 训练          | 无训练 heuristic                      | on-policy oracle distillation         |
| 选择依据        | 多信号加权                              | retention entropy + oracle divergence |
| 错误分析        | 无专门建模                              | 识别 confident wrong eviction           |
| 层策略         | shared-index                       | layer-wise / layer-group budget       |
| eviction 强度 | fixed budget                       | uncertainty-aware dynamic budget      |
| serving 目标  | memory saving + retrieval accuracy | accuracy-memory-latency trade-off     |

---

## 11. 可行性分析

### 11.1 工程可行性

本研究可以基于现有 SieveKV 代码继续开发。最小可行版本不需要一开始实现所有模块：

1. 先使用 fixed block 而非 semantic segment；
2. 先做 global policy 而非 layer-wise policy；
3. 先做 offline oracle，再加入 on-policy；
4. 先使用 sentence-level segment，再扩展到 RAG chunk 和 code segment；
5. 先用 3B 模型跑通，再扩展到 7B / 8B。

### 11.2 训练成本控制

oracle masking 成本较高，但可以通过以下方式降低：

1. 只对 candidate segments 做 oracle；
2. 对 segment 进行采样；
3. 先用 heuristic score 过滤 top-k 和 bottom-k；
4. 使用 restore/drop 近似；
5. 缓存 static features 和 oracle labels；
6. 使用 replay buffer 复用历史 trajectory。

### 11.3 推理开销控制

online serving 阶段不使用 oracle，只使用轻量 policy。并且：

1. static semantic features 在 prefill 后一次性计算；
2. policy 可以采用 request-level 或 segment-level MLP；
3. dynamic features 每 (n) 步更新一次；
4. eviction amortized 执行；
5. layer-wise mask 先采用 3 个 layer group，减少复杂度。

---

## 12. 风险与应对策略

### Risk 1: Learned policy 增加 latency

应对：

* 使用轻量 MLP 或 linear gate；
* request-level gate + segment-level score；
* amortized eviction；
* static feature caching；
* 详细报告 policy overhead；
* 必要时将 MLP 蒸馏成线性模型或规则表。

### Risk 2: Oracle masking 成本太高

应对：

* candidate sampling；
* top-k / bottom-k subset；
* drop/restore approximation；
* replay buffer；
* 只在训练阶段使用 oracle；
* 分阶段训练，先 offline 再 on-policy fine-tune。

### Risk 3: Semantic segment 不稳定

应对：

* 提供 fallback 到 fixed block；
* 使用 hierarchical segment；
* 对不同任务使用不同 segmenter；
* 对 segment 长度做上限控制。

### Risk 4: On-policy training 不稳定

应对：

* 使用 SieveKV heuristic warmup；
* 混合 offline oracle data 和 on-policy data；
* 使用 replay buffer；
* 控制 policy update rate；
* 使用 conservative budget schedule。

### Risk 5: Layer-wise mask 工程复杂

应对：

* 先实现 layer-group mask；
* 只分 lower/middle/upper 三组；
* shared-index 作为 baseline；
* 在效果明显后再尝试 full layer-wise 或 per-head retention。

---

## 13. 实施路线图

### Phase 1: Reproduce and strengthen SieveKV baseline

时间：第 1–2 周

任务：

1. 整理现有 SieveKV 代码；
2. 复现实验结果；
3. 加入更多 baseline；
4. 完成 latency / memory profiling；
5. 准备固定 block、signal score、pinning、budget control 的稳定实现。

产出：

* 稳定 SieveKV baseline；
* baseline result table；
* profiling script。

---

### Phase 2: Offline oracle distillation MVP

时间：第 3–5 周

任务：

1. 实现 segment-level oracle masking；
2. 构造 oracle importance label；
3. 训练轻量 MLP policy；
4. 比较 fixed SieveKV 与 offline distilled policy；
5. 完成 basic ablation。

产出：

* offline oracle label dataset；
* first learned retention policy；
* initial accuracy-memory result。

---

### Phase 3: On-policy retention distillation

时间：第 6–8 周

任务：

1. 让当前 policy 执行 compressed-cache rollout；
2. 在 on-policy cache state 上做 drop/restore oracle；
3. 实现 retention entropy 和 oracle-policy divergence；
4. 实现 Soft-OR decision selection；
5. 比较 entropy-only、divergence-only、Q3-only、Soft-OR。

产出：

* OP-SieveKV core method；
* KV-TIP taxonomy result；
* Q3 case analysis。

---

### Phase 4: Semantic segment and layer-wise budget

时间：第 9–11 周

任务：

1. 实现 sentence / paragraph / RAG chunk segmentation；
2. 实现 hierarchical segment fallback；
3. 实现 layer-group budget；
4. 比较 fixed block vs semantic segment；
5. 比较 shared-index vs layer-group retention。

产出：

* semantic segment retention result；
* layer-wise budget ablation；
* visualization of retained segments by layer。

---

### Phase 5: Uncertainty-aware dynamic eviction and system optimization

时间：第 12–14 周

任务：

1. 实现 generation entropy、attention entropy、score margin；
2. 实现 dynamic budget；
3. 实现 amortized eviction；
4. 分析 policy overhead；
5. 优化 selection 和 mask update。

产出：

* latency-aware OP-SieveKV；
* accuracy-memory-latency trade-off curves；
* serving-level profiling。

---

### Phase 6: Large-scale evaluation and paper writing

时间：第 15–18 周

任务：

1. 扩展 benchmark；
2. 扩展模型；
3. 完成所有 ablation；
4. 整理 case studies；
5. 撰写论文；
6. 准备 appendix 和 reproducibility checklist。

产出：

* 完整论文初稿；
* 完整实验表格；
* 代码和复现实验文档。

---

## 14. 预期论文结构

### Abstract

介绍 KV cache bottleneck、现有 heuristic eviction 的局限、OP-SieveKV 的 on-policy oracle distillation 框架、KV-TIP taxonomy、semantic segment retention 和 layer-wise dynamic budgeting，以及主要实验结果。

### 1. Introduction

* 长上下文 LLM serving 中 KV cache 的显存瓶颈；
* 现有 eviction 方法的不足；
* SieveKV 的多信号启发式有效但仍然固定；
* 本文提出 OP-SieveKV；
* 总结贡献。

### 2. Background and Related Work

* KV cache eviction；
* KV cache quantization / offloading；
* long-context retrieval；
* on-policy distillation；
* token importance / entropy-divergence selection；
* learned cache policy。

### 3. Motivation

* fixed weights 不适应不同任务；
* fixed block 切断语义；
* full-KV oracle 存在 off-policy mismatch；
* confident wrong eviction 的案例分析。

### 4. Method

* semantic segment construction；
* feature design；
* on-policy oracle construction；
* KV-TIP taxonomy；
* Soft-OR informative decision selection；
* training objective；
* layer-wise budget；
* uncertainty-aware eviction；
* online serving algorithm。

### 5. Experimental Setup

* models；
* datasets；
* baselines；
* metrics；
* implementation details。

### 6. Results

* main results；
* offline vs on-policy；
* Soft-OR vs entropy/divergence；
* semantic segment；
* layer-wise budget；
* uncertainty-aware eviction；
* latency and memory profiling。

### 7. Analysis

* Q3 case study；
* retained segment visualization；
* layer-specific retention patterns；
* failure cases；
* overhead breakdown。

### 8. Limitations

* oracle cost；
* large-model scalability；
* segmentation dependency；
* highly open-ended generation；
* production integration.

### 9. Conclusion

总结 OP-SieveKV 的意义：将 KV eviction 从固定 heuristic 推进到 on-policy、oracle-distilled、自适应、语义片段级的 memory-efficient serving framework。

---

## 15. 预期实验结果假设

本研究希望验证以下假设：

1. OP-SieveKV 在相同 cache budget 下，比原始 SieveKV 获得更高 retrieval / reasoning accuracy。
2. On-policy oracle distillation 优于 offline oracle distillation，尤其在 aggressive budget 和 multi-hop tasks 下。
3. Soft-OR selection 优于 entropy-only 和 divergence-only selection，能够有效捕获 confident wrong eviction cases。
4. Semantic segment retention 优于 fixed block retention，尤其在 RAG、多文档 QA 和代码任务中。
5. Layer-wise budget 优于 shared-index retention，能够在相同 memory 下提升质量。
6. Uncertainty-aware dynamic eviction 能改善 worst-case accuracy，并减少误删关键证据。
7. 在合理工程优化后，learned policy 的额外 latency 可以控制在可接受范围内，并通过 memory saving、larger batch size 或 longer context support 获得整体 serving 收益。

---

## 16. 最小可行版本

为了降低风险，可以先实现一个 MVP：

**OP-SieveKV-Lite**

1. 使用原始 SieveKV 作为初始 policy；
2. segment 暂时使用 16-token block 或 sentence；
3. 不做 full layer-wise，仅做 global retention；
4. 只在部分 candidate block 上做 drop/restore oracle；
5. 使用 MLP 预测 keep probability；
6. 实现 retention entropy + oracle divergence；
7. 使用 Soft-OR 选择训练样本；
8. 在 RULER NIAH、multi-needle 和一个 multi-hop benchmark 上验证。

MVP 目标：

* 证明 on-policy distillation 比 fixed SieveKV 更强；
* 证明 Q3 confident wrong eviction 存在；
* 证明 Soft-OR selection 比 entropy-only 更有效；
* 控制 policy overhead 不显著增加。

---

## 17. 最终定位

OP-SieveKV 的核心定位不是单纯“加一个小模型”，而是：

**将 KV cache eviction 从人工启发式规则，升级为由 full-KV oracle 指导、在 policy 自身 compressed-cache trajectory 上学习、能够识别自信错误并动态适应任务与层级结构的 semantic retention framework。**

它可以作为一篇更高水平会议论文的主线，因为它不仅提出一个新的 scoring rule，而是提出一个更一般的问题和框架：

1. KV retention policy 是否存在 off-policy supervision mismatch？
2. 如何定义 retention decisions 中的 informative samples？
3. 如何识别 policy 自信但错误的 eviction？
4. 如何在 memory、quality、latency 三者之间进行自适应优化？

如果实验充分，这个方向有潜力从原始 SieveKV 的 C 会工作升级为面向 B/A 级别会议的完整研究。
