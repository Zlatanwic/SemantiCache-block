# 当前 SemantiCache 策略报告

## 先解释：`gauntlet` 代表什么

在这个项目里，`gauntlet` 不是一个算法名，而是一套“更难、更毒的评测关卡”。

它的意思可以理解成：

- 强化压力测试
- 晋级关卡
- 当前的主评测门槛

之所以叫 `gauntlet`，是因为它不再只是普通的 Needle-in-a-Haystack 检索，而是故意加入了：

- 很像正确答案的干扰事实
- 时间、时长、频道号、代号这类容易串台的属性
- 更低的 cache budget
- 更长的 haystack

所以现在的实验逻辑是：

- `frontier` 已经基本跑满了
- `gauntlet` 才是当前真正决定 keep / discard 的主 benchmark

一句话说，`gauntlet` 就是“当前策略必须闯过去的难关”。

## 执行摘要

当前仓库里 checked-in 的 SemantiCache frontier 是 commit `8aa4074`。

这套策略不是一次性拍脑袋设计出来的，而是通过 Autoresearch 风格的 keep / discard 实验循环，一步步收敛出来的。它主要沿着三个方向演化：

1. 提高 factual token 和 query-relevant token 的保护强度
2. 改进 block 级别的保留排序方式
3. 在预算不够时，更精细地选择 block 内部应该保留哪些 token

当前策略的代表性结果是：

- `frontier` 套件：`1.000000` accuracy，`1.000000` hard accuracy，`2.318179s`
- `gauntlet` 套件：`0.933333` accuracy，也就是 `84/90`，`1.000000` hard accuracy，也就是 `30/30`，`7.262574s`

这说明当前策略在一般检索上已经很稳，在 hardest slice 上也很稳，但在最紧预算下仍然会出现少量“属性绑定错误”。

## 这套策略想解决什么问题

SemantiCache 的核心问题不是“模型能不能回答”，而是：

在 KV cache 预算固定的前提下，应该保留哪些 token，才能最大概率保住后续事实恢复能力？

因此当前策略不是在做一个抽象的最优压缩器，而是在做一个工程上可用的、面向 benchmark 优化的 retention policy。

它遵循的是一个很明确的原则：

优先保住最可能成为后续事实锚点的 token，同时保证最近的生成尾部不会因为 eviction 过猛而塌掉。

## 当前策略的整体结构

当前实现主要在这两个文件里：

- [eviction_policies.py](d:/semanticache/eviction_policies.py)
- [semantic_analyzer.py](d:/semanticache/semantic_analyzer.py)

### 1. 核心打分信号

当前 eviction score 来自 5 个压力项：

- attention pressure
- information density pressure
- head entropy pressure
- query relevance pressure
- factual pressure

实现上，先把每个信号当成“越大越值得保留”的 keep-oriented signal，再做一次反向归一化，把它转成“越大越该被删”的 eviction pressure：

`eviction_pressure = 1 - normalized_keep_signal`

最后总分是：

`score = alpha * attn + beta * density + gamma * entropy + query_weight * query + factual_weight * factual`

当前代码里的关键下限是：

- `alpha = 0.4`
- `beta = 0.3`
- `gamma = 0.3`
- `query_weight >= 0.30`
- `factual_weight >= 0.40`

这意味着当前 frontier 相比早期版本，更明显地偏向：

- 保住与问题相关的 token
- 保住像事实的 token

### 2. 被硬保护的 token

有一些 token 不参加普通的 eviction 竞争，而是直接被保护：

- 所有 `system` token
- 最新一轮 user message 的尾部 token
- 最近生成窗口内的 token

当前代码里的关键保护参数是：

- latest user tail cap：`56`
- recent decode window：默认路径下仍然保留较强的 recent window 保护

这部分保护来自实验中的一个很稳定的结论：

- latest-user tail 放得太松会掉稳定性
- recent decode tail 保护不够时，生成质量会明显变脆

### 3. 角色感知语义标注

`semantic_analyzer.py` 会给 token 打上角色标签：

- `FILLER`
- `ASSISTANT`
- `CONTEXT`
- `USER_HISTORY`
- `USER_LATEST`
- `SYSTEM`

这样策略就能区分：

- 必须保的 system 指令
- 高优先级的最新用户问题
- 可能仍有价值但优先级较低的历史对话

### 4. 信息密度信号

信息密度是一个局部窗口里的启发式分数，它偏好：

- 数字多的区域
- code-like symbol 多的区域
- 词汇多样性更高的区域

并对标点占比高的区域做惩罚。

这可以近似理解为：

一个局部区域越像“信息浓度高的内容”，越值得保留。

### 5. Query relevance 信号

query relevance 的实现比较简单，但很实用：

- 取最终 user question
- tokenize
- 统计局部窗口与问题 token 的重叠程度

它不是 embedding 级语义检索，而是一个轻量的 question-alignment bias。

当前 frontier 在 `exp-037` 中把 query weight 的下限提高到了 `0.30`，并最终保留了这个改动。

### 6. Factual bonus 信号

当前 factual bonus 主要偏向这些 token 模式：

- digit token
- uppercase token
- month token
- 时间 / 单位 token，比如 `AM`、`UTC`、`hour`、`minutes`、`degrees`、`Celsius`

这个信号的目的，是优先保住这些“看起来像事实”的局部区域，尤其是：

- 日期
- 时间
- 时长
- 数值约束

当前 frontier 在 `exp-038` 中把 factual weight 的下限提高到了 `0.40`。

## 当前的保留机制是怎么工作的

真正影响效果最大的，其实不是“打分公式”本身，而是 `select_keep_indices` 的演化。

### 第一步：先保住 pinned token 和 recent token

策略先把这两类 token 标记为 protected：

- semantic pinned token
- recent decode token

如果这些 token 已经把预算占满了，那就直接保留它们，不再进入后面的 block 选择逻辑。

### 第二步：把剩余 token 切成 block

没有被保护的 token 会被切成固定大小的 block。

当前 frontier 的 block size cap 是 `3`。

这是一个很关键的演化点，因为更大的 block 会带来两个问题：

- 一个重要 token 会被周围噪声稀释
- 低预算下 block 粒度太粗，不利于精细保留

### 第三步：block 按“最重要 token”排序，而不是按均值排序

早期版本用的是：

- block 内所有 token 的平均 eviction score

当前 frontier 改成了：

- block 内最小的 eviction score，也就是“这个 block 里最值得保留的 token”

实际效果是：

只要一个 block 里有一个很关键的 token，这个 block 就不会因为周围噪声太多而整体掉队。

这个改动来自 `exp-022`，它在不丢分的前提下，大幅降低了 `gauntlet` 的平均时间。

### 第四步：如果 block 装不下，就在 block 内挑最该保留的 token

早期版本还有一个问题：

- 当一个 block 只能部分放进预算时，它默认保留 block 的最后几个 token

这个行为非常容易把事实 span 截错位置。

当前 frontier 改成了：

- 如果 block 只能部分保留，就在这个 block 内按 score 再挑一次
- 只保留最值得保留的那些 token
- 最后再按原始顺序排回去

这个改动来自 `exp-023`，它把 `gauntlet` 从 `83/90` 推到了 `84/90`。

## 这套策略是怎么一步步演化出来的

关键的 keep 节点如下：

1. `exp-009`
   把 semantic block size 从 `16` 收紧到 `8`

2. `exp-011`
   把 block size 从 `8` 进一步收紧到 `4`

3. `exp-015`
   把 latest-user tail pinning 从 `64` 降到 `56`

4. `exp-019`
   引入更难的 `gauntlet` baseline，正式把主评测切到 adversarial harder tests

5. `exp-022`
   block 排序从“均值”改成“按最安全 token 排序”

6. `exp-023`
   partial block 不再机械保尾部 token，而是按 token score 选最该保留的 token

7. `exp-037`
   把 query weight floor 提到 `0.30`

8. `exp-038`
   把 factual weight floor 提到 `0.40`

## 当前性能总结

### Frontier suite

`frontier` 套件实际上已经基本打满：

- `exp-015`：`1.000000` accuracy，`1.000000` hard accuracy，`2.318179s`

这也是为什么后面需要引入 `gauntlet`，因为继续在 `frontier` 上优化已经很难得到有区分度的反馈。

### Gauntlet suite

`gauntlet` 是当前真正有区分度的门槛。

几个关键里程碑：

- `exp-019` baseline：`83/90`，`1.000000` hard accuracy，`8.223390s`
- `exp-022`：仍然是 `83/90`，但时间降到 `6.821975s`
- `exp-023`：提高到 `84/90`，`7.340104s`
- `exp-037`：同样 `84/90`，但时间进一步降到 `6.932372s`
- `exp-038`：仍然 `84/90`，`7.262574s`

这里有一个很重要的备注：

当前 checked-in frontier 是 `exp-038` 路线，也就是 commit `8aa4074`。  
但如果只看同准确率下的时间，`exp-037` 其实更快。

所以严格说，当前代码并不是“所有 tie 情况下时间最优”的唯一版本，而是当前被保留下来的最新 frontier。

## 当前策略擅长什么

当前这版策略已经明显擅长：

- 在 hardest long-context slice 上保持稳定
- 在大多数情况下保住日期、时间和数字事实
- 在更低 cache budget 下避免整体崩盘
- 比早期的 block-average retention 策略更稳地保住关键 span

真正起效果的不是某一个单点技巧，而是下面这些东西一起工作：

- role pinning
- recent tail protection
- factual weighting
- query weighting
- 更细粒度的 block 选择逻辑

## 当前还剩下哪些失败模式

当前 frontier 的失败已经非常集中。

从 [exp-038.json](d:/semanticache/results/autoresearch/exp-038.json) 看，6 个错例全部来自：

- `haystack_length = 2000`
- `budget = 0.1`

也就是最紧预算、但不是最长上下文的那一档。

这些错例基本分成两类。

### 1. migration 的 duration 绑定错误

模型经常会回答：

- 日期对了
- 时间对了
- 时长错了

典型表现是把正确答案里的 `4 hours` 漂成干扰项中的 `1 hour`。

这说明当前策略并不是完全没保住相关区域，而是没有把 authoritative duration 和 nearby distractor duration 足够稳定地区分开。

### 2. incident bridge 的属性混绑错误

模型会把这些属性串台：

- channel 号错
- duration 错
- open time 却经常还是对的

典型表现包括：

- `Delta-24`
- `24`
- `42`
- `30 minutes`
- `1 hour`
- `2 hours`

这些都说明当前 benchmark 的难点已经不只是“能不能找回主题”，而是：

能不能在 eviction 压力下，保住一组精确绑定在一起的属性。

## 如何理解当前瓶颈

当前 SemantiCache 的问题已经不是：

模型完全忘记了相关事实。

而更像是：

模型保住了“正确主题附近的部分证据”，但没有稳定保住“正确属性组合”。

也就是说，现在的瓶颈更接近：

- 局部 span cohesion 不够
- fact tuple 保留不够完整
- authoritative span 和 confusable distractor span 还没有被拉开足够大差距

## 下一步最值得做的实验方向

接下来最有价值的方向不是再粗暴加大 recent window，而是更针对性地处理“事实 tuple”。

我认为最值得试的方向有：

1. 增强局部 tuple 保留
   不是只保一个最强 token，而是保住它附近短小但完整的 factual span

2. 增强短距离 factual cohesion
   如果一个 block 里出现多个很重要的 factual token，就优先把它们作为一个局部簇一起留下

3. 专门处理结构化 identifier
   像 `Delta-42` 这类混合字母数字 identifier，当前 factual bonus 仍然不够稳

4. 更保守地引入“authoritative”信号
   `exp-020` 那种大而泛的 authority-marker 设计太激进了，但收缩版也许仍然有价值

## 最后的结论

当前 SemantiCache 策略可以概括为：

一套 role-aware、query-aware、fact-aware 的 KV cache retention policy，并且已经通过持续的 keep / discard 实验，证明它不是拍脑袋 heuristic，而是一个被 benchmark 反复筛选过的工程策略。

它当前最重要的三个特征是：

- 它是被稳定 harness 驱动出来的，而不是开放式乱试
- 它已经显著优于早期的 block-average 版本
- 它的失败已经被压缩到一个很小、很可解释的角落

所以这套策略还没有“完全解决问题”，但它已经是一个可信的、可解释的、可继续优化的 frontier。 
