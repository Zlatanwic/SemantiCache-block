# OP-SieveKV: On-Policy Oracle-Distilled KV Cache Retention

下面是一版可直接做 PPT 的详细讲稿，按四个 Part 组织。整体故事线是：

> KV cache 很贵 -> 传统 heuristic eviction 有 off-policy mismatch -> 我们提出 on-policy oracle distillation -> 发现 Q3/missed-keep -> 训练 learned retention policy -> 实验显示优势与边界 -> 最后讨论 limitation 和下一步。

---

## PART 1 Introduction and Related Work

### Slide 1. 背景：长上下文 LLM 的 KV cache 瓶颈

大家好，我的项目关注的是长上下文大语言模型推理中的一个核心系统问题：KV cache。

在 Transformer 自回归生成时，模型会缓存每一层 attention 的 key 和 value。这样生成下一个 token 时不用重新计算历史上下文，速度会快很多。但问题是，随着上下文长度增加，KV cache 的显存开销线性增长。对于长文档问答、多轮对话、RAG、多跳推理这类任务，KV cache 很快会成为推理成本和可部署性的瓶颈。

所以一个自然的问题是：我们能不能只保留真正重要的 KV token，删除不重要的部分，同时尽量不影响最终回答质量？

### Slide 2. 现有 KV eviction 方法

已有方法大致可以分成几类。

第一类是固定窗口或 StreamingLLM，只保留最近 token 和少量 attention sink token。这类方法简单高效，但容易丢掉中间的重要证据。

第二类是 attention-based 方法，比如 H2O、SnapKV，它们根据 attention score 或 observation window 来估计 token 重要性。这比固定窗口更自适应，但仍然主要依赖启发式信号。

第三类是压缩或合并类方法，比如 KVzip，尝试在保留较少 cache 的同时维持信息量。

还有原始 SieveKV / semantic baseline，它用语义、角色、query relevance 等信号做规则式 retention。相比纯 attention 方法，它更适合对话和结构化上下文，但权重仍然是人工设计的。

### Slide 3. 现有方法的关键问题

我认为原方法主要有三个问题。

第一，很多方法是 heuristic 的，无法知道某个 segment 对最终答案是否真的重要。

第二，很多 oracle 或训练标签是 off-policy 的。也就是说，它们在完整上下文里判断 token 重要性，但实际推理时，模型面对的是已经被当前 policy 压缩过的 KV cache。完整上下文下不重要的 token，在压缩状态下可能变得非常关键。

第三，普通方法容易忽略一种危险错误：policy 非常自信地删除了某个看似无关、但实际上对回答很重要的 evidence。我把这类错误称为 Q3 confident-wrong eviction。

### Slide 4. 项目目标

所以本项目的目标不是简单调一个新的 heuristic score，而是把 KV retention 从人工规则推进到一个 learned policy。

具体来说，我希望实现：

1. 让 policy 先在自己的 compressed cache 状态下执行 eviction。
2. 再用 counterfactual oracle 判断保留或恢复某个 segment 是否会影响 gold answer。
3. 用这些 oracle label 训练一个轻量 MLP retention policy。
4. 用 KV-TIP taxonomy 分析哪些决策最有训练价值。
5. 通过 Soft-OR / reweighting 选择或强化关键训练样本。

最终目标是在相同 cache budget 下，比 H2O、SnapKV、KVzip、StreamingLLM 和原始 SieveKV 更稳。

---

## PART 2 Methodology and Innovations

### Slide 5. 方法总览：OP-SieveKV

我的方法叫 OP-SieveKV，全称是 On-Policy Oracle-Distilled Adaptive Semantic KV Retention。

整体流程可以分成两阶段。

训练阶段：

输入 prompt、query 和 gold answer。当前 retention policy 先执行 eviction，得到自己的 compressed KV cache。然后我对候选 segment 做 drop / restore oracle，得到它们对答案 log-probability 的真实边际贡献。最后用这些标签训练 retention policy。

推理阶段：

模型 prefill 后，把上下文切成 semantic segments，计算每个 segment 的特征，然后用 MLP 预测 keep probability，根据 budget 保留得分最高的 segment。

### Slide 6. Innovation 1：On-policy compressed-cache oracle

第一个创新点是 on-policy oracle。

传统 drop oracle 通常是在完整 prompt 上做：删除某个 segment，看答案概率是否下降。但这不是 policy 实际推理时遇到的状态。

我的 collector 先让当前 policy 在给定 budget 下压缩 KV cache，得到 `C^{pi}`。然后对两类 segment 做 counterfactual：

如果 segment 已经被保留，就做 drop oracle：

删除它，看 gold answer log-probability 是否下降。

如果 segment 已经被删除，就做 restore oracle：

把它加回 compressed cache，看 gold answer log-probability 是否上升。

这个 restore 分支非常关键，因为它能发现 policy 已经删掉、但其实应该保留的 missed-keep evidence。

### Slide 7. Innovation 2：Drop + Restore oracle

具体来说，我定义两个重要性：

对于已保留 segment：

```text
I_drop = logP(answer | C) - logP(answer | C \ g)
```

如果删除后答案概率下降，说明这个 segment 是有用的。

对于已删除 segment：

```text
I_restore = logP(answer | C union g) - logP(answer | C)
```

如果恢复后答案概率上升，说明 policy 原来删错了。

这比只做 drop oracle 更适合 on-policy learning，因为在压缩状态下，重要性会随已保留内容变化。

### Slide 8. Innovation 3：KV-TIP taxonomy

第三个创新点是 KV-TIP taxonomy。

我借鉴 on-policy distillation 里 entropy + divergence 的思想，把 retention decision 分成四类。

横轴是 oracle-policy divergence，也就是 oracle label 和 policy probability 的差距。

纵轴是 retention entropy，也就是 policy 对 keep/drop 的不确定性。

最重要的是 Q3：

> low entropy + high divergence

也就是 policy 很自信，但 oracle 认为它错了。比如 policy 很自信地 drop 了一个 query overlap 不高、但对 multi-hop answer 很关键的 evidence。

这个 taxonomy 不只是分析工具，也可以指导训练样本选择。

### Slide 9. Innovation 4：Soft-OR selection and reweighting

一开始我实现了 Soft-OR filtering：

```text
z = h + d - h * d
```

其中 `h` 是 normalized entropy，`d` 是 normalized divergence。Soft-OR 的直觉是：只要不确定性高，或者和 oracle 分歧大，这个样本就值得训练。

但实验发现，直接 filtering 会造成严重 distribution shift：选出来的数据几乎全是 positive keep labels，模型会学成过度保留，multi-needle 直接崩掉。

所以最终完整方法改成 Soft-OR reweighting：

不删除原始数据，而是在完整 on-policy dataset 上，对 Soft-OR top-rho 的 oracle rows 提高权重。这样既保留负类和普通 teacher rows 的分布，又强化 Q3/missed-keep 信号。

### Slide 10. Lightweight retention policy

训练的 policy 是一个轻量 MLP。输入特征包括：

- attention / entropy / density / query / factual / authority / recency 等多信号统计
- segment position 和 length
- budget ratio
- role features，比如 system、latest user、context
- pinned/query/template/boundary 等结构特征
- heuristic keep probability

输出是每个 segment 的 keep probability。推理时根据 keep probability 排序，在 budget 内选择保留 segment。

---

## PART 3 Experiments and Results

### Slide 11. 实验设置

实验主要使用 Qwen2.5-3B-Instruct。

任务包括：

1. Single-needle NIAH
2. Multi-needle NIAH
3. 后续扩展 LongBench hotpotqa / gov_report

Baselines 包括：

- Full cache
- StreamingLLM
- H2O
- SnapKV
- KVzip
- Semantic / original SieveKV
- On-policy learned OP-SieveKV
- Soft-OR reweight OP-SieveKV

为了减少小样本波动，我实现了 fixed manifest evaluation。也就是说，所有 policy 在完全相同的一批样本上评估，并计算 paired delta 和 95% CI。

### Slide 12. On-policy oracle 数据收集结果

在 full student-policy on-policy oracle collection 中，我们得到：

- 7896 rows
- 1887 oracle-scored rows
- drop oracle: 618
- restore oracle: 1269
- missed-keep restore candidates: 227
- Q3 low-prob confident-wrong subset: 160

这说明 restore branch 确实发现了传统 prompt-level oracle 看不到的错误。

也就是说，policy 在自己的 compressed cache 状态下，确实会自信地删掉一些重要 evidence。这直接支持了本项目的核心假设。

### Slide 13. Single-needle NIAH 结果

在 NIAH 120 fixed-manifest 上，on-policy learned policy 表现很强：

- 30% budget: 100%
- 20% budget: 99.2%
- 10% budget: 100%
- 5% budget: 45%

尤其在 5% budget 下，semantic baseline 是 0%，onpolicy 是 45%。这说明 on-policy oracle 在 aggressive budget 下确实能纠正一部分 critical evidence deletion。

不过在 10% 以上，semantic baseline 本身也接近 100%，所以 single-needle 任务区分度有限。

### Slide 14. Multi-needle 主结果

Multi-needle 更能体现多证据保留能力。

在 multi-needle 90 fixed-manifest 中，semantic baseline 仍然很强。onpolicy 和 softor_rw 大多接近 semantic，并明显强于 H2O、Streaming、KVzip，大多数情况下也强于 SnapKV。

例如 softor_rw：

- 20% k=2: 75.0
- 20% k=4: 68.3
- 20% k=8: 82.5
- 30% k=4: 70.0，比 semantic 68.3 略高
- 5% k=2: 55.0，比 semantic 48.3 高

但它不是全面胜出。在 10% k=4/k=8 和 5% k=8 上，softor_rw 低于 semantic。

### Slide 15. 一个重要负结果：Filtering Soft-OR 会失败

这个项目里一个很重要的发现是：naive Soft-OR filtering 失败了。

filtering-only 的 `softor_z` 在 debug multi 中平均只有 19.4%，例如：

- 30% k=2: 0%
- 30% k=8: 12.5%

原因是 filtering 后的数据分布严重偏正，模型几乎学成“什么都想 keep”。但在固定 budget 下，真正重要的是排序能力，而不是单纯提高 keep probability。

这也是为什么我后来把完整方法改成 reweighting，而不是 filtering。

### Slide 16. Soft-OR reweighting 修复稳定性

Soft-OR reweighting 保留全部 on-policy dataset，只提高 selected rows 的训练权重。

结果显示，它明显修复了 filtering 的崩坏：

- filtering Soft-OR debug multi: 19.4%
- reweight Soft-OR debug multi: 65.3%
- reweight Soft-OR large multi: 61.7%

这说明 Soft-OR 的核心信号是有用的，但必须避免改变训练分布。对于深度学习项目来说，这也是一个很好的 lesson：sample selection 不只是选高价值样本，还要保持整体数据分布稳定。

### Slide 17. 和 baseline 的比较

从主表可以看到，learned OP-SieveKV 系列相比外部 baselines 有明显优势。

H2O 在 multi-needle 上经常只有 10-40%。

KVzip 在 k=8 上表现很弱，比如 30% k=8 只有 2.9%。

StreamingLLM 对 multi-needle 也不稳定，常常只能保留最近信息。

SnapKV 在部分设置不错，但 k=8 和低 budget 下明显下降。

OP-SieveKV 的优势在于，它不只依赖 attention 或窗口，而是利用 oracle distillation 学到哪些 semantic segment 对最终答案更关键。

### Slide 18. 当前结果的诚实解读

目前结果不是“全面碾压 semantic baseline”。

更准确的结论是：

1. On-policy oracle 确实发现了真实的 missed-keep / Q3 错误。
2. Learned policy 在 single-needle aggressive budget 下有明显收益。
3. 在 multi-needle 上，OP-SieveKV 显著优于多数外部 baseline。
4. 但相比强 semantic baseline，softor_rw 多数是持平、小幅提升或局部下降。
5. Soft-OR filtering 失败，reweighting 才是稳定版本。

这说明方法主线成立，但还需要更强的 benchmark 和更细的 policy design 才能达到论文级全面优势。

---

## PART 4 Conclusion and Discussions

### Slide 19. 总结贡献

这个项目的贡献可以总结为四点。

第一，提出并实现了 on-policy compressed-cache oracle。它比 prompt-level oracle 更接近真实 serving 状态。

第二，实现了 drop + restore oracle，发现了 missed-keep 和 Q3 confident-wrong eviction。

第三，训练了轻量 OP-SieveKV retention policy，把原始 heuristic SieveKV 推进到 learned adaptive policy。

第四，系统比较了 filtering Soft-OR 和 reweighting Soft-OR，发现 reweighting 更稳定，也更适合作为完整方法。

### Slide 20. 对原方法的改进

相对原始 SieveKV / semantic baseline，本项目的改进是：

- 从人工权重变成 learned MLP policy
- 从 off-policy full-context importance 变成 on-policy compressed-cache importance
- 从只看保留的重要性，扩展到被删除 segment 的 restore importance
- 从普通 supervised labels，加入 KV-TIP / Soft-OR 的 informative decision weighting
- 从普通 sweep，升级为 fixed-manifest paired evaluation

这些改进使方法从“一个启发式 cache eviction rule”变成“一个 oracle-distilled learned retention framework”。

### Slide 21. Limitations

当然，目前还有几个限制。

第一，当前 on-policy oracle 是 post-prefill compressed-cache oracle，还不是 decode 多步 trajectory oracle。

第二，layer-wise budget 和 dynamic eviction 还没有实现。现在仍然是 shared-index global retention。

第三，semantic segmentation 还没有扩展到 RAG chunk、code function、table row 等复杂结构。

第四，LongBench 和 profiling 还在补充。当前主要结果集中在 NIAH 和 multi-needle。

第五，Soft-OR reweight 没有全面超过 semantic baseline，只是在部分 budget 和 needle count 下持平或小幅改善。

### Slide 22. Future Work

后续可以沿三个方向继续。

第一，扩展 benchmark。除了 NIAH 和 multi-needle，还需要 HotpotQA-style long-context QA、LongBench summarization、RAG planning 等任务。

第二，做 layer-wise / layer-group budget。不同层对信息的需求不同，低层可能需要局部结构，高层更关注 evidence。现在 shared-index 可能不是最优。

第三，实现 decode-time dynamic eviction。根据 generation entropy、attention entropy 和 policy margin 动态调整 budget 或 eviction frequency，减少误删关键 evidence 的风险。

### Slide 23. Final Takeaway

最后，我想用一句话总结这个项目：

> KV cache eviction 不应该只是一个固定 heuristic；它可以被看作一个 on-policy decision learning problem。

我们让 policy 在自己的压缩状态下犯错，再用 oracle 去纠正它。这个过程揭示了 Q3 confident-wrong eviction，也说明了 learned KV retention 的潜力和挑战。

当前结果已经证明：on-policy oracle signal 是真实存在的，learned OP-SieveKV 可以在 aggressive budget 和多证据任务中超过多数外部 baseline；但要全面超过强 semantic baseline，还需要更好的 reweighting、benchmark 泛化和系统优化。

### Slide 24. Q&A

谢谢大家，我的汇报到这里。欢迎提问。
