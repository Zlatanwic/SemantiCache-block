# SieveKV 面试/答辩讲稿

目标时长：含自我介绍约 15 分钟；自我介绍约 1 分钟，技术汇报约 14 分钟。  
主线：长上下文 KV cache 内存压力 -> query-agnostic eviction 的缺陷 -> SieveKV 的 query-aware 多信号打分 -> 实验验证。  
口径提醒：final paper 主结果是 `semantic / SieveKV` 单层 eviction，不是 `tiered_semantic` 的 hot/warm/cold CPU offload 版本。`tiered_semantic` 可以作为扩展实现或 Q&A 说明。
计时说明：Slide A-B 是面试开场自我介绍；Slide 1 起的技术页按原 14 分钟版本写，实际汇报时按 Backup 时间表压缩到总计 15 分钟。

---

## Slide A - 自我介绍：研究背景（0:00-0:30）

### PPT 内容

- 姓名 / 学校 / 专业方向
- 研究兴趣：
  - LLM inference systems
  - long-context serving
  - KV cache compression / eviction
- 当前关注问题：
  - 如何在有限显存下保留长上下文里的关键证据

### 讲稿

各位老师好，我先简单介绍一下自己。我目前主要关注大模型推理系统，尤其是长上下文服务里的 KV cache 管理问题。我的兴趣点在于：当上下文越来越长、显存预算越来越紧时，系统应该如何判断哪些历史 token 真的值得保留。今天汇报的 SieveKV，就是围绕这个问题做的一次尝试。

---

## Slide B - 自我介绍：项目定位与汇报路线（0:30-1:00）

### PPT 内容

- 项目定位：
  - 面向 retrieval-heavy long-context 场景
  - 在 tight cache budget 下保持答案证据 span
- 我的贡献侧重：
  - query-aware 多信号 scoring
  - role-aware pinning
  - block-level retention
  - 系统实现与实验评估
- 汇报路线：
  - problem -> method -> experiments -> SieveKV-v2

### 讲稿

这个项目的定位是 retrieval-heavy 的长上下文任务，比如问答、RAG 和多证据检索。我的核心关注不是让模型生成得更花哨，而是在 KV cache 预算受限时，尽可能保住和答案有关的证据 span。接下来我会按四步讲：先讲问题和现有方法的不足，再讲 SieveKV 的多信号方法，然后讲实验结果，最后讲我认为可以往 CCF-A/B 继续推进的 SieveKV-v2 方向。

---

## Slide 1 - 标题页（0:00-0:30）

### PPT 内容

- **SieveKV: Semantic Multi-Signal Eviction for Long-Context Retrieval**
- Query-aware KV cache eviction for retrieval-heavy prompts
- 关键词：KV cache / query-aware / multi-signal / distractor filtering
- 姓名 / 单位 / 日期

### 讲稿

各位老师好，我今天汇报的是 SieveKV，一个面向长上下文检索任务的 KV cache 驱逐策略。它的核心想法是：在检索类 prompt 里，query 本身已经告诉我们“哪些上下文 token 更可能是答案”，所以 eviction 不应该只看通用注意力重要性，而应该把 query relevance 和 factual prior 一起纳入打分。最终在 RULER NIAH 上，我们在只保留 20% KV cache 的情况下达到 70% 准确率，在紧预算下仍然保持了较强检索能力，并明显优于 SnapKV 等 eviction baseline。

---

## Slide 2 - 背景：KV Cache 是长上下文推理的内存瓶颈（0:30-1:25）

### PPT 内容

- KV cache 保存每层 attention 的 Key/Value，避免 decode 时重复计算历史 token
- 内存随序列长度线性增长：
  - `layers x kv_heads x seq_len x head_dim x 2(K,V) x dtype`
- 3B 模型 + 几千 token prompt 时，KV cache 可以达到百 MB 级
- 当模型权重已经占据显存，KV cache 会限制：
  - context length
  - batch size
  - serving cost

### 讲稿

先说问题背景。LLM 在自回归生成时，每一步都要 attend 到前面所有 token。如果每次都重新算历史 token 的 Key 和 Value，代价会非常高，所以推理系统会把每一层的 K/V 缓存下来，这就是 KV cache。它的好处是省计算，但代价是显存随上下文长度线性增长。对长上下文任务来说，真正限制可服务长度和 batch size 的，经常不是模型参数本身，而是不断膨胀的 KV cache。

---

## Slide 3 - 现有 Eviction 的共同弱点：Query-Agnostic（1:25-2:25）

### PPT 内容

| 方法 | 核心信号 | 类型 | Query-aware? |
|---|---|---|---|
| StreamingLLM | attention sink + recent window | progressive | No |
| H2O | cumulative attention | progressive | No |
| SnapKV | prefill observation-window attention | one-shot | No |
| KVzip | reconstruction scoring | one-shot | No |

- 这些方法能回答“哪个 token 一般重要”
- 但难回答“哪个 token 对当前 query 是答案”

### 讲稿

已有方法大致分两类。一类是 progressive，比如 StreamingLLM 和 H2O，在 decode 过程中持续删 token；另一类是 one-shot，比如 SnapKV 和 KVzip，在 prefill 后做一次 pruning。它们都很有价值，但有一个共同限制：基本是 query-agnostic 的。也就是说，它们能判断一个 token 在模型计算里是不是普遍重要，却不知道这个 token 对当前问题是不是答案。检索任务里，这个差别很关键，因为答案通常只是几千个 distractor 中的一小段事实。

---

## Slide 4 - 核心洞察：检索 Prompt 里的 Query 是天然抓手（2:25-3:25）

### PPT 内容

- NIAH 示例：
  - Context: Paul Graham essays
  - Needle: “One of the magic numbers for Kate is: 24592.”
  - Query: “What is the magic number for Kate?”
- Query token: `Kate / magic / number`
- Answer span 同时具有：
  - query overlap
  - factual pattern: number / entity / date / unit
  - local information density

### 讲稿

我们的洞察很直接：在检索任务中，prompt 通常是结构化的，最后有一个明确 query。比如问题问 Kate 的 magic number，那上下文里和 Kate、magic、number 相关的 span，就比普通散文句子更值得保留。再比如答案里常常有数字、日期、实体名、单位，这些 factual pattern 也能提供先验。因此 SieveKV 不只问“模型注意过谁”，还问“谁更像当前 query 的答案”。

---

## Slide 5 - 位置基础：Prompt Token 和 KV Slot 一一同构（3:25-4:25）

### PPT 内容

```text
prefill input_ids[i]
      -> layer l 的 K[l][:, :, i, :]
      -> layer l 的 V[l][:, :, i, :]
```

- SemanticAnalyzer 给的是逻辑位置 `i` 的语义分数
- prefill 后，`token[i]` 与 `KV pair[i]` 在序列维度上一一对应
- 后续 pruning 后也通过位置映射维持绑定关系
- 论文主结果：单层 SieveKV eviction
- `tiered_semantic`：代码扩展，不是 final paper 主结果

### 讲稿

这里有一个实现上的基础：我们不是抽象地给“词”打分，而是给 prompt 的逻辑位置打分。prefill 时，输入的 `token[i]` 会在每一层生成同一序列位置上的 K/V，因此语义分数 `score[i]` 可以绑定到第 `i` 个 KV pair。只要后续 eviction 时同步裁剪 token 分数和 KV slot，这个绑定就保持成立。如果做更复杂的分层或重排，则需要额外维护位置映射，比如代码里的 `origin_positions` 和 `hot_positions`。但 final paper 的主结果是普通 SieveKV eviction，不依赖 CPU warm tier。

---

## Slide 6 - 方法总览：Prefill 静态分析 + Decode 动态更新（4:25-5:25）

### PPT 内容

- **Prefill 阶段**
  - 模型 forward 产生初始 KV cache 和 prefill attention
  - SemanticAnalyzer 一次性计算静态信号：
    - information density
    - query relevance
    - factual likelihood
    - role tags for pinning
- **Decode 阶段**
  - 每步累积 attention mass 和 head entropy
  - cache 超预算时，合成多信号分数
  - 选择高价值 block 保留，低价值 block 驱逐

### 讲稿

整体流程分两阶段。prefill 阶段模型正常 forward，产生 KV cache 和 attention。同时，SemanticAnalyzer 基于 prompt token 和最新 query 计算静态信号，这些信号不依赖生成过程，所以只算一次。decode 阶段，每生成一个 token，我们持续累积 attention 相关的动态信号。如果当前 cache 超过预算，就把动态信号和静态信号合成分数，再做 block-level retention。严格按代码说，SemanticAnalyzer 当前是在 prefill forward 结束后、decode 前调用；逻辑上属于 prefill sidecar，工程上也可以与 GPU prefill 做 overlap。

---

## Slide 7 - 五个信号：2 个动态 + 3 个静态（5:25-7:15）

### PPT 内容

**动态 attention-derived**

1. `s_alpha`: cumulative attention mass  
   `sum_l sum_h sum_t A_l,h(t,i)`
2. `s_gamma`: head entropy  
   `- sum_h p_h(i) log p_h(i)`

**静态 content/query-derived**

3. `s_beta`: information density，32-token 局部窗口  
   `0.3 digit + 0.2 code - 0.2 punct + 0.3 unique`
4. `s_q`: query relevance，窗口 token-id 与 query token-id overlap
5. `s_f`: factual likelihood  
   `0.25 digit + 0.12 upper + 0.28 month + 0.28 unit + 0.45 entity`

### 讲稿

五个信号可以分成两类。第一类是动态 attention-derived。`attention mass` 是每个 token 在 prefill 和 decode 中累计收到多少注意力，类似 H2O 的 heavy-hitter 思路；`head entropy` 看这个 token 是否被多个 KV head 广泛关注，而不是只被一个 head 偶然关注。

第二类是静态信号。`information density` 对每个 token 取以它为中心的 32-token 窗口，看这个局部区域是否包含数字、代码符号、独特 token，同时惩罚纯标点。`query relevance` 是窗口内 token id 与最新 query token id 的重叠比例。`factual likelihood` 则抓数字、大写、月份、单位、实体形态这些答案常见模式。注意这里不是看 KV 向量内容，而是先给 token 逻辑位置打分，再绑定到对应 KV pair。

---

## Slide 8 - 分数组合、Pinning 与 Block Retention（7:15-8:40）

### PPT 内容

- Retention score 视角：

```text
s(i) = alpha*s_alpha + beta*s_beta + gamma*s_gamma
       + w_q*s_q + w_f*s_f

alpha=0.4, beta=0.3, gamma=0.3
w_q >= 0.30, w_f >= 0.40
```

- Role-aware pinning：永不驱逐
  - system prompt
  - latest user query tail
  - recent decode tokens
- Block retention：
  - unpinned token 按连续 block 选择
  - 一个关键 token 可以保护周围上下文

### 讲稿

合成时，论文里用 retention score 表达，也就是分数越高越该保留。base score 来自 attention mass、information density 和 head entropy；query relevance 和 factual likelihood 作为 bonus 加进去，并设置权重下限，防止它们因为原始数值范围较小而被 attention 信号淹没。

除此之外还有两个工程机制。第一是 role-aware pinning，system prompt、最新 user query 的尾部，以及最近生成 token 会被硬保护。第二是 block retention。逐 token 删除会破坏局部连续性，所以 SieveKV 按连续 block 保留；这样一个关键答案 token 可以带着它周围的上下文一起留下来。实现里代码内部用 eviction pressure，方向和论文 retention score 相反，但选择目标是一致的。

---

## Slide 9 - 走一遍 NIAH 例子：为什么紧预算仍然有效（8:40-9:50）

### PPT 内容

- 场景：
  - prompt 约 3700 tokens
  - budget `b=20%`，约保留 742 tokens
  - needle 在 50% depth
- Needle block 的信号：
  - factual likelihood: `0.25` vs haystack `<0.05`
  - information density: `0.7` vs haystack `0.3`
  - query relevance: `0.7` vs haystack `~0`
- 结果：
  - needle block 排名靠前
  - 大量 distractor 被移除

### 讲稿

用一个 NIAH 例子解释机制。假设 prompt 大约 3700 token，20% 预算只允许保留 742 个 token。needle 是 Kate 的 magic number 24592。这个 span 会同时触发 factual signal，因为它包含实体和数字；触发 query relevance，因为它和 query 共享 Kate、magic、number；也触发 information density，因为它比普通散文更局部密集。因此 needle block 排名靠前，而大量 Paul Graham essay 的普通句子被删掉。这里的重点不是“删得越多一定越好”，而是 query-aware scoring 让紧预算优先花在答案相关 span 上，从而减少无关上下文对检索的干扰。

---

## Slide 10 - 实验设置：模型、Benchmark、Baselines（9:50-11:00）

### PPT 内容

- 模型：
  - Qwen2.5-3B-Instruct：主实验，36 layers，2 KV heads
  - Llama-3.2-3B-Instruct：跨架构，28 layers，8 KV heads
  - BitsAndBytes 4-bit NF4 权重量化，单张 48GB GPU
- Benchmark：
  - RULER NIAH，Paul Graham essay haystack + factual needle
  - Qwen prompt 约 3700 tokens，Llama 约 3400 tokens
  - exact substring match / `string_match_all`
- Baselines：
  - Full KV, StreamingLLM, H2O, SnapKV, KVzip
- Protocol：
  - greedy decoding
  - cache budget `b` = 每层保留 KV entries 的比例

### 讲稿

实验主要使用两个 3B instruct 模型。Qwen2.5-3B 是主实验，Llama-3.2-3B 用来验证跨架构迁移。两个模型都用 BitsAndBytes 4-bit NF4 加载权重，注意这里量化的是模型权重，不是 SieveKV 主结果里的 KV cache。

Benchmark 是 RULER NIAH：把随机 factual needle 插入 Paul Graham essay haystack，再问模型对应问题。评分是严格的 substring match，答案关键词必须出现在输出里。这个任务的好处是 ground truth 明确，能比较干净地评估 eviction 是否保住答案。Baseline 覆盖 full KV、StreamingLLM、H2O、SnapKV 和 KVzip。

---

## Slide 11 - 主结果：紧预算下保持稳健检索（11:00-12:10）

### PPT 内容

- Qwen2.5-3B-Instruct / RULER NIAH

| Policy | b=50% | b=30% | b=20% |
|---|---:|---:|---:|
| Full KV | 58% | 58% | 58% |
| SnapKV | 60% | 42% | 30% |
| KVzip | 38% | 32% | 12% |
| H2O | 22% | 8% | 8% |
| StreamingLLM | 28% | 20% | 10% |
| **SieveKV** | **60%** | **62%** | **70%** |

- `b=10%` budget sensitivity：SieveKV 达到 80%，baseline 全部低于 40%
- 解释：distractor filtering

### 讲稿

主结果最核心的数字是这张表。在 Qwen 的 RULER NIAH 上，full KV 是 58%。SnapKV 在 50% 预算接近 full KV，但到 20% 预算掉到 30%。H2O 和 StreamingLLM 掉得更明显。SieveKV 则从 50% 的 60% 提升到 20% 的 70%，是唯一一个预算收紧反而变好的方法。更细的 budget sensitivity 中，在 10% 预算下 SieveKV 达到 80%，所有 baseline 都低于 40%。这说明对于 retrieval-heavy 场景，紧预算不必然意味着检索失败；如果删掉的是 distractor，模型反而更容易集中到答案 span。

---

## Slide 12 - 稳健性与系统开销（12:10-13:30）

### PPT 内容

- Multi-needle：
  - `k=4`: SieveKV 92.5%，Full KV 80%，SnapKV 70%
- Ablation：
  - 去掉单一信号：最多下降 4 points
  - 去掉全部 semantic signals：70% -> 8%
- Cross-architecture：
  - Llama-3.2-3B @ b=20%: SieveKV 84%，Full KV 90%
- Systems：
  - eviction/step: 8.86 ms，约 decode time 的 3.9%
  - throughput: 3.4 tok/s vs Full KV 6.5 tok/s
  - memory: 131 MB -> 26 MB at b=20%
- 小结：
  - query-aware 多信号 eviction 在 retrieval-heavy long context 中有效

### 讲稿

最后看稳健性和系统开销。多 needle 任务里，4 个 needle 全部答对才算正确，SieveKV 达到 92.5%，表现出对多个稀疏事实 span 的保留能力。消融实验说明，单独去掉某个信号最多下降 4 个点，但如果去掉所有 semantic signals，只剩 attention，准确率从 70% 掉到 8%，说明多信号是协同工作的。

跨架构上，同一套权重直接迁移到 Llama-3.2-3B，SieveKV 在 20% 预算下达到 84%，距离 full KV 的 90% 只差 6 个点。系统层面，SieveKV 每步 eviction 开销 8.86 ms，占 decode 时间不到 4%，吞吐从 full KV 的 6.5 tok/s 降到 3.4 tok/s，这是当前局限；收益是 20% 预算下 KV cache 从 131 MB 降到 26 MB，节省 80%。所以当前版本证明了 query-aware 多信号 eviction 的有效性，但它仍然是固定权重、shared-index、固定 block 的启发式策略，这也自然引出下一步的改进方向。

---

## Slide 13 - 后续方向：SieveKV-v2（13:30-14:45）

### PPT 内容

- **SieveKV-v2: Oracle-Distilled Adaptive Semantic Retention**
- 从手工 heuristic 升级为 learned cache policy
- Oracle trace：
  - full KV 运行样本
  - 记录 attention contribution
  - masking / leave-one-block-out 测量删除 block 是否影响答案
- 轻量 policy：
  - 输入：五个信号 + role + position + budget + query features
  - 输出：segment/block keep probability
- Retention unit：
  - 从固定 16-token block 升级为 semantic segment
  - sentence / paragraph / code function / RAG chunk

### 讲稿

如果继续往 CCF-A/B 会推进，我会把后续工作聚焦成一个方向：SieveKV-v2，Oracle-Distilled Adaptive Semantic Retention。当前 SieveKV 的优势是简单、可解释，但本质上还是手工组合五个信号。下一步可以先构造 oracle：对同一个 prompt 跑 full KV，记录每层每个 token 或 block 的 attention contribution，再通过 masking 或 leave-one-block-out 看删除某个 block 是否导致答案错误。这样就能得到更接近真实任务影响的 block importance 标签。

然后训练一个轻量 policy，而不是训练大模型。这个 policy 的输入可以包括现在已有的五个信号、role tag、位置、cache budget 和 query features，输出每个 segment 或 block 的 keep probability。这样 serving 时仍然很轻量，但方法叙事从“人工 heuristic”升级成了“从 oracle retention trace 蒸馏出的 learned cache policy”。

同时，retention unit 也可以从固定 16-token block 升级为 semantic segment，例如句子、段落、代码函数或者 RAG chunk。这样可以避免只保留数字却丢掉对应实体的问题，也更容易扩展到多跳问答、代码和真实 RAG 场景。

---

## 结束页 / Q&A（14:45-15:00）

### PPT 内容

- **Thank you**
- Q&A
- 三个可重复 takeaway：
  1. Query 是检索任务里最强的 retention hint
  2. token 逻辑位置与 KV slot 位置同构，使 semantic score 可以绑定到 KV pair
  3. SieveKV 用五个轻量信号 + role pinning + block retention 实现 distractor filtering

### 讲稿

我的汇报到这里结束。最后总结三点：第一，检索任务里的 query 是天然的保留提示；第二，prompt token 和 KV slot 的位置同构让我们可以把语义分数绑定到 KV pair；第三，多信号加 role pinning 和 block retention 可以有效过滤 distractor，在 tight budget 下保住答案 span。后续如果继续推进，我会优先做 oracle-distilled adaptive semantic retention。谢谢各位老师。

---

# Backup Slides

## Backup 1 - 五个信号的精确定义

### PPT 内容

- `s_alpha`: cumulative attention mass
  - prefill + decode 累加 attention
- `s_gamma`: head entropy
  - 多 KV head 分布越均匀，熵越高
- `s_beta`: information density
  - 以 token `i` 为中心的 32-token 窗口，不是最近 32 个 token
- `s_q`: query relevance
  - 当前代码是 token-id set overlap
- `s_f`: factual likelihood
  - digit / uppercase / month / unit / entity-ish

### 讲稿

如果老师问五个信号怎么算，可以按这页回答。重点是三个静态信号都只依赖 prompt token 和 query，不依赖 KV 向量；两个动态信号来自 attention tracker，会随 decode 更新。

---

## Backup 2 - `tiered_semantic` 和 Final Paper 的关系

### PPT 内容

- Final paper 主结果：
  - `semantic / SieveKV`
  - 单层 retained KV，其他直接 prune
- 代码扩展：
  - `tiered_semantic`
  - hot: GPU full precision
  - warm: CPU int8 quantized candidate tier
  - cold: drop
- 不把 tiered 结果作为 paper 主结论

### 讲稿

如果被问到 hot/warm/cold，需要明确：这是代码里后来实现的系统扩展，不是 final paper 主表结果。final paper 讨论的是 SieveKV 的 eviction policy 本身；量化和 tiering 更适合作为 future work 或系统扩展。

---

## Backup 3 - 实验为什么主要用 NIAH

### PPT 内容

- NIAH 优点：
  - ground truth 明确
  - needle position 可控
  - needle count 可控
  - exact match deterministic
- 避免开放生成 benchmark 的归因困难：
  - summarization 风格差异
  - decode randomness
  - answer paraphrase
- 限制：
  - 结论主要适用于 retrieval-heavy long-context

### 讲稿

我们不是说 SieveKV 已经覆盖所有长上下文任务。NIAH 的价值是把 eviction quality 单独隔离出来看：答案有没有被保住，输出里有没有包含事实值。LongBench 这类开放任务更接近真实场景，但评价会混入生成风格和 paraphrase 的影响，所以放在后续工作更合理。

---

## Backup 4 - 多轮对话支持与 Role-Aware 6 级分层

### PPT 内容

- 多轮对话原生支持
  - 输入：标准 `messages` = `[system, user, assistant, user, ...]`
  - 解析：扫 ChatML `<|im_start|>...<|im_end|>`（Qwen）或 Llama header 切 segment
  - 入口：`run_generation.py:977-1004` / `semantic_analyzer.py:197-235`
- 6 级 RoleTag 优先级（高 → 低，`semantic_analyzer.py:17-25`）
  - `SYSTEM = 5`
  - `USER_LATEST = 4`：最新一轮 user
  - `USER_HISTORY = 3`：历史轮 user
  - `CONTEXT = 2`
  - `ASSISTANT = 1`
  - `FILLER = 0`
- 最新 user 判定（`semantic_analyzer.py:217-229`）
  - 扫所有 segment，记录 `last_user_idx`
  - 仅最后一个 user segment 标 `USER_LATEST`，其它 user 全部降级 `USER_HISTORY`
- 不平等打分：四层叠加
  - **Layer 1 — Pin mask**（`semantic_analyzer.py:552-575`，`config.py:80-81`）
    - `SYSTEM` + `USER_LATEST` 永不 evict（pin → -inf）
    - `USER_HISTORY` 不 pin，可被 evict
  - **Layer 2 — Role bonus**（`eviction_policies.py:731-740`）
    - `USER_HISTORY +0.42` > `USER_LATEST +0.18`
    - `ASSISTANT -0.70`，`FILLER -0.35`，`CONTEXT +0.08`
  - **Layer 3 — 信息信号乘数**（`eviction_policies.py:751-753`）
    - `USER_HISTORY × 0.75`（query/factual/authority 强放大）
    - `USER_LATEST × 0.28`（弱放大，因已 pin）
    - `ASSISTANT × -0.55`（反向打压）
  - **Layer 4 — 历史 assistant 强 evict**（`eviction_policies.py:770-774`）
    - `scores[ASSISTANT_HISTORY] = -inf`
    - 例外：最近 `semantic_generated_retention_window=12` 个生成 token
- Promotion（warm → hot）也分层（`eviction_policies.py:810-815`）
  - `USER_HISTORY +0.90`（优先召回）
  - `USER_LATEST -0.20`（已在 hot）
  - `SYSTEM = -inf`（promotion 不参与）

### 讲稿

如果老师问 SieveKV 是否支持多轮对话、历史轮和最新轮 token 是否平等打分，可以按这页回答。

第一，多轮对话原生支持。输入是标准 messages 列表，含 system / user / assistant 多轮历史。SemanticAnalyzer 扫 ChatML 或 Llama 的 header 边界把 prompt 切成 segment，给每段打 role tag。

第二，打分完全不平等，分四层叠加。最新一轮 user 是 `USER_LATEST`，被硬 pin 锁死永不 evict；历史轮 user 是 `USER_HISTORY`，不 pin，但 role bonus 反而加得更高（+0.42 vs +0.18）——因为它没有 pin 兜底，需要靠分数活下来。同时三个信息信号 query relevance、factual、authority 在 USER_HISTORY 上乘以 0.75 强放大，在 USER_LATEST 上只乘 0.28 弱放大。历史 assistant 通常是模型的复述，信息冗余度最高，除了最近 12 个生成窗口之外直接打成 -inf 强 evict。Promotion 阶段同样优先把信息性强的历史 user 召回 hot 层。

这套机制就是 paper 里 role-aware pinning 的完整落地，颗粒度做到每个 role 不同的 bonus、不同的乘数、不同的 promotion 偏置。

---

## Backup 5 - 精度堆栈与 Warm 层 int8 量化

### PPT 内容

- 三层精度堆栈（互相正交）
  - **模型权重**：4-bit NF4（BitsAndBytes，`config.py:14-18`）
    - `bnb_4bit_quant_type="nf4"`
    - `bnb_4bit_use_double_quant=True`
  - **计算 / KV cache**：float16
    - `torch_dtype="float16"`，`bnb_4bit_compute_dtype="float16"`
    - `attn_implementation="eager"`（不用 flash，要 hook attention 算 5 个 signal）
  - **Warm tier（tiered 扩展）**：int8 per-vector
    - `semantic_warm_bits=8`，`semantic_warm_device="cpu"`（`config.py:62-64`）
- Warm 层量化算法（`quantized_tier_cache.py:21-44`）
  - 对称量化，**per-token-per-head** 沿 `head_dim` 维度（不是 per-tensor / per-channel）
  - 每个 `(layer, token, head)` 向量自己一个 scale
  - 关键三行：
    ```python
    working = tensor.to(torch.float32)
    scale   = working.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 127.0
    q       = torch.round(working / scale).clamp(-127, 127).to(torch.int8)
    ```
  - scale 单独存 fp32；dequant：`q.fp32 * scale → fp16`
- 压缩比（`head_dim=128`）
  - 原始：`128 × fp16 = 256` bytes
  - 量化后：`128 × int8 + 1 × fp32 scale = 132` bytes
  - **≈ 1.94×**（不是简单 2×，因为 scale 占位）
- Tiered 拓扑
  - Hot：fp16 GPU `DynamicCache`
  - Warm：int8 + fp32 scale CPU `QuantizedWarmTier`
  - 召回：`materialize_positions` 选择性 dequant，`materialize` 全量合并

### 讲稿

如果老师追问精度细节，要分清楚两件完全独立的事。

第一，模型权重量化。两个评测模型 Qwen2.5-3B 和 Llama-3.2-3B 都用 BitsAndBytes 的 4-bit NF4 加载，配 double quantization 进一步压缩 scale，计算时 dequant 到 fp16 走前向。这部分是 BNB 库提供的标准能力，不是 SieveKV 的贡献。注意 attention implementation 用 eager 不用 flash，因为 SieveKV 要 hook attention weights 算 cumulative attention mass 和 head entropy 两个动态信号。

第二，KV cache 精度。final paper 主结果里 KV cache 原生是 fp16，我们做的是 token-level eviction，不动数据精度。warm 层的 int8 per-vector 量化是 tiered_semantic 扩展实现的一部分——把 evict 出来的 fp16 KV 向量沿 head_dim 维度做对称 absmax 量化，每个 token-head 自己一个 fp32 scale，存到 CPU 上。promotion 召回时再 dequant 成 fp16 上 GPU。

压缩比大约 1.94 倍——不是 2 倍，因为每个向量要带一个 fp32 scale。如果之后做 KIVI 风格的 2-bit、或者 group-wise scale 共享，可以进一步把比值往 8× 推。这块只在 Backup 2 提到的 tiered 模式下才用，不在 final paper 主表里。

---

## Backup 6 - 时间控制表

| 阶段 | Slides | 累计时间 | 提醒 |
|---|---:|---:|---|
| 自我介绍 | A-B | 1:00 | 控制在 1 分钟内 |
| 技术开场 | 1 | 1:20 | 标题页只给一句核心结果 |
| 背景 + 痛点 | 2-4 | 4:00 | 不展开太多相关工作 |
| 方法 | 5-9 | 10:00 | Slide 7 和 8 是重点，Slide 9 可压缩 |
| 实验 + 开销 | 10-12 | 13:45 | 数字必须准确 |
| 后续方向 | 13 | 14:45 | 聚焦 SieveKV-v2 一条路线 |
| Q&A | 结束页 | 15:00 | 正好收束 |

必背数字：

- `70% / 58% / 30%`: b=20% SieveKV / Full KV / SnapKV
- `80%`: b=10% SieveKV
- `92.5%`: 4-needle SieveKV
- `84%`: Llama cross-architecture
- `8%`: 去掉 semantic signals
- `8.86 ms`: 每步 eviction overhead
- `131 MB -> 26 MB`: b=20% KV memory
