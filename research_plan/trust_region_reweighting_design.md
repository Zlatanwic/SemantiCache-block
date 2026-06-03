# Trust-Region Oracle Reweighting for OP-SieveKV

本文档把 TrOPD 的 trust region / outlier estimation 思路迁移到 OP-SieveKV，用于指导当前代码实现和后续实验。它不是替代 on-policy oracle，而是说明：在收集到 on-policy oracle label 后，应该如何更稳定地使用这些 label 训练 retention policy。

## 1. 核心直觉

TrOPD 的关键观点是：on-policy distillation 中，student 自己生成的 token 并不总是适合被 teacher 直接监督。当 teacher 和 student 分布差距很大时，teacher 在这些 token 上给出的反向 KL 信号可能会变成不可靠梯度。因此 TrOPD 把 token 分成：

- trust region：teacher/student 足够一致，监督信号相对可靠；
- outlier region：teacher/student 差异大，直接训练可能不稳定，但里面仍然可能包含重要信息。

OP-SieveKV 中也存在同构问题。当前 policy 在自己的 compressed-cache state 上做 eviction，oracle 再对 kept/dropped segments 做 drop/restore counterfactual。这个 oracle label 是 on-policy 的，但并不意味着所有 oracle rows 都应该以同样方式训练。

对于 KV retention，我们将其解释为：

- trust-region decision：policy keep probability 与 oracle keep label 接近，说明当前策略和 oracle 基本一致；
- oracle-outlier decision：policy 与 oracle 分歧大，尤其是 policy 自信但 oracle 认为它错了；
- missed-keep outlier：policy 倾向 drop，但 restore oracle 发现恢复该 segment 能提高答案概率；
- over-keep outlier：policy 倾向 keep，但 drop oracle 发现删除该 segment 不伤害答案，甚至更好。

这解释了我们已有实验现象：naive Soft-OR filtering 会破坏原训练分布，导致 multi-needle 崩溃；而 Soft-OR reweighting 保留全量数据，只强化 oracle outliers，训练更稳定。

## 2. 方法定义

对每个 segment decision 记：

- policy keep probability: `p`
- oracle keep label: `y`
- retention entropy: `h = -p log p - (1-p) log(1-p)`
- oracle-policy divergence: `d = |y - p|`

Soft-OR score 使用归一化后的 entropy 和 divergence：

```text
h_hat = normalize(h)
d_hat = normalize(d)
z = h_hat + d_hat - h_hat * d_hat
```

`z` 高表示该 decision 至少满足一种条件：

- policy 不确定，值得训练边界；
- policy 与 oracle 分歧大，值得纠错。

完整方法不再把 top-rho rows 单独过滤出来训练，而是：

1. 保留完整 on-policy dataset，作为 trust-region support；
2. 计算所有 oracle rows 的 `z`；
3. 在每个 budget 内选 top-rho oracle rows；
4. 对 selected oracle rows 乘以额外训练权重；
5. 对 Q3 confident-wrong 和 missed-keep rows 进一步加权；
6. 训练时仍使用原有 `train_op_policy.py` 的 weighted BCE。

## 3. 与 filtering 的区别

Filtering ablation:

```text
output rows = support rows + selected oracle rows
```

用途是验证“高 entropy / 高 divergence / Q3 rows 是否有信息量”。它不应该作为最终方法，因为它会删除大量普通负类、teacher rows 和稳定 support rows，导致 policy calibration 失衡。

Trust-region reweighting:

```text
output rows = all rows
output weight_i = base_weight_i * multiplier_i
```

用途是最终方法。它保留原始分布，只增强 oracle-discovered outliers。这样可以避免 multi-needle 中出现“只学会保护少数强 evidence，忘记覆盖多个 evidence”的问题。

## 4. 当前代码入口

数据选择脚本：

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

等价显式写法：

```bash
python select_op_oracle_dataset.py \
  --dataset results/op_oracle_dataset_onpolicy_student.pt \
  --output results/op_oracle_dataset_onpolicy_trust_region_rw.pt \
  --mode trust_region_soft_or \
  --selection-action reweight \
  --rho 0.30 \
  --score-normalization zscore_sigmoid \
  --selected-weight-multiplier 2.0 \
  --q3-weight-multiplier 1.5 \
  --missed-keep-weight-multiplier 1.5 \
  --max-weight 20
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

评测主表：

```bash
python run_v2_main_table.py \
  --benchmarks niah multi \
  --niah-manifest results/v2/manifests/niah_120.json \
  --multi-manifest results/v2/manifests/multi_90.json \
  --budgets 0.3 0.2 0.1 0.05 \
  --op-ckpt onpolicy=results/op_policy_onpolicy_student.pt \
  --op-ckpt trust_rw=results/op_policy_onpolicy_trust_region_rw.pt \
  --output-dir results/v2/main_table_trust_region_rw \
  --append-to results/v2/experiment_summary.md
```

## 5. 代码字段约定

`select_op_oracle_dataset.py` 会在 metadata 中补充：

- `selection_policy_prob`: 用于 selection 的 policy keep probability；
- `retention_entropy`: binary retention entropy；
- `oracle_policy_divergence`: `abs(oracle_label - policy_prob)`；
- `selection_entropy`: 归一化 entropy；
- `selection_divergence`: 归一化 divergence；
- `soft_or_score`: Soft-OR score；
- `decision_type`: `missed_keep` / `over_keep` / `aligned_keep` / `aligned_drop`；
- `kvtip_quadrant`: `Q1_confident_aligned` 等四象限标签；
- `trust_region_probability`: 近似 trust-region agreement diagnostic；
- `trust_region_type`: `trust_region` 或 `oracle_outlier`；
- `selected_for_training`: 是否被 top-rho 选中；
- `selection_weight_multiplier`: reweight multiplier；
- `training_treatment`: `oracle_outlier_reweight` 或 `trust_region_support`。

训练脚本不需要知道这些字段。它只读取 `features`、`labels`、`weights`。这些 metadata 主要用于诊断、复现实验和写论文。

## 6. 实现伪代码

```text
load dataset
for each row:
    p = row.policy_keep_prob
    y = row.oracle_label
    h = binary_entropy(p)
    d = abs(y - p)

normalize h and d across oracle rows
z = h_hat + d_hat - h_hat * d_hat

for each budget:
    selected = top rho oracle rows by z

weights = base_weights
for row in selected:
    m = selected_weight_multiplier
    if row is Q3_confident_wrong:
        m *= q3_weight_multiplier
    if row is missed_keep:
        m *= missed_keep_weight_multiplier
    weights[row] = clamp(weights[row] * m, max_weight)

save full dataset with updated weights
```

## 7. 推荐消融

为了把故事讲清楚，建议保留以下 ablations：

- all-decisions：不做 selection/reweight；
- entropy-only filter：只看不确定性；
- divergence-only filter：只看 oracle-policy disagreement；
- Q3-only filter：只看 confident-wrong；
- missed-keep-only filter：只看 restore positives；
- Soft-OR filter：证明 naive filtering 会造成分布破坏；
- trust-region reweight：最终方法。

报告时要强调：filtering 是诊断性 ablation，reweighting 才是部署前训练方法。

## 8. 实验判断标准

不要只看 single-needle NIAH。trust-region reweight 的目标是稳定保留多证据，所以判断顺序建议是：

1. multi-needle k=4/k=8 是否不崩；
2. 10%/20% budget 是否接近或超过 semantic baseline；
3. 5% budget 是否相比 semantic 或 onpolicy 有局部改善；
4. NIAH 单针是否保持 10% 以上接近满分；
5. 输出权重 max/mean 是否没有过度放大。

如果出现 NIAH 很高但 multi-needle 大幅下降，通常说明权重太集中，应该降低 `selected-weight-multiplier` 或 `q3/missed_keep` multiplier，或者提高 `rho` 以覆盖更多普通 oracle rows。

## 9. 后续扩展

下一步可以把当前 binary trust-region diagnostic 升级成更接近 TrOPD 的形式：

- 用当前 policy 的 selected/drop action probability 表示 student confidence；
- 用 oracle soft label 或 answer log-prob delta 表示 teacher/oracle support；
- 使用 `min(oracle_support / policy_action_prob, 1)` 作为 trust probability；
- 对 trust rows 使用常规 BCE，对 outlier rows 使用 conservative reweight 或 ranking loss；
- 对 restore/missed-keep outliers 添加 evidence coverage regularization，避免只保护单个 needle。

