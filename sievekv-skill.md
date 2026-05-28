## 背景与启发
这篇工作其实来讲，还是偏向检索场景的，也是受了一片北大DAIR lab的论文的启发（PQCache），核心思想是：注意力机制打分本来就是基于“相似度”的选择，然后顺着这个思路我们就可以向信息检索领域的一些问题的方向上去思考，我强化了query在token选择上的重要性

## token打分机制的物理和逻辑基础
程序维护输入prompt的token[i]的逻辑位置与prefill之后的kv cache的K[layer][:, :, i, :]和V[layer][:, :, i, :]的逻辑位置是位置同构一一对应的。SemanticAnalyzer 给的是 prompt 逻辑位置 i 的语义分数，比如 density、query relevance、factual bonus、role tag。这就决定了，我们是可以在prefill的同时对token进行静态信号打分的。

只要kv cache没被重排，score[i]就可以找到对应的slot[i]的（这里我们没有引入类似于paged attention的重排机制，当然如果引入也只是多做一层映射的问题）。后续即使经过eviction以及offloading，我们也维护了一个映射，比如`origin_position`和`hot_position`等等，总而言之，我们可以通过对prompt token的逻辑位置的三个静态信号打分可以对应到物理的kv pair。

## 五个信号
### 两个动态信号
1. cumulative attention mass：每次forward的累加注意力分数(H2O)
2. head entropy：衡量token[i]是否在多个head的注意力分数都比较高
### 三个静态信号
3. information density:
取一个 32-token 滑动窗口，统计局部内容密度：
```
s_beta(i)
= 0.3 * digit_ratio
+ 0.2 * code_ratio
- 0.2 * punct_ratio
+ 0.3 * unique_ratio
```
其中：

digit_ratio：窗口里含数字 token 的比例
code_ratio：代码/符号类 token 比例，比如 { } [ ] = # _
punct_ratio：纯标点 token 比例，负贡献
unique_ratio：窗口内不同 token 数 / 窗口长度
4. query relevance: 先把latest user token=16保存起来，然后在32-token的滑动窗口里计算与query的重合度
5. factual likelihood / factual bonus：
同样取 32-token 窗口，统计事实型特征：
```
s_f(i)
= 0.25 * digit_ratio
+ 0.12 * upper_ratio
+ 0.28 * month_ratio
+ 0.28 * fact_unit_ratio
+ 0.45 * entity_ratio
```
其中：

digit_ratio：数字
upper_ratio：大写 / acronym
month_ratio：月份名
fact_unit_ratio：单位、时间词，如 utc, hour, degree
entity_ratio：像人名、项目名、标签、缩写的 token
