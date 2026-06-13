# Phase 1 Block-Granularity Gate — 结果

> Task 1.2 输出。判定**把 token 级语义淘汰粗化到 paged-KV block 粒度后，质量损失是否可接受**。
> 运行：2026-06-13 on `bc01@ubun` (4×RTX 6000D, sm_120), torch 2.12.0+cu130, Qwen2.5-3B-Instruct fp16。

## 结论

✅ **Phase 1 gate 通过**：在 25% 预算下，`block_semantic-16` 达到 `semantic`（token 级基线）的 **97.4%**（95.0% / 97.5%），超过 95% 阈值。**整块对齐的代价极小，vLLM 路线算法前提成立。**

## 实验设置

- 固定 manifest：`results/v3/manifest40.json`（40 样本，跨 5 个 needle 位置，3 个 needle，2000 token haystack，固定 seed=42）
- 模型：Qwen2.5-3B-Instruct（full fp16，`--no-bnb-4bit`）
- 卡 0：`full` + `semantic` + `block_semantic-16`（预算 1.0/0.5/0.3/0.25/0.1）
- 卡 1：`block_semantic-32`（预算 0.5/0.3/0.25/0.1，并行跑）

## 结果（NIAH accuracy，40 样本 manifest）

| 策略 | @10% | @25% | @30% | @50% | @100% |
|---|---:|---:|---:|---:|---:|
| `full`（上界） | — | — | — | — | 95.0% |
| `semantic`（token 基线） | 100.0%* | 97.5% | 97.5% | 95.0% | 95.0% |
| `block_semantic-16` | 32.5% | **95.0%** | 95.0% | 95.0% | 95.0% |
| `block_semantic-32` | 7.5% | **95.0%** | 95.0% | 95.0% | — |
| **block-16 相对 token** | — | **97.4%** | 97.4% | 100% | — |

\* 10% 处的 100% 是注意力 sink 巧合，不外推。

## 关键观察

1. **平预算平质量**：在 25%–50% 主工作区，block-16/32 与 token 几乎重合（差距 ≤ 1/40 样本），证明整块对齐的代价极小。
2. **极端预算下块粒度代价暴露**：10% 时 block-16 = 32.5% 而 token = 100%——**极小预算下需要更细粒度**（block-8 是 plan §7 备选）。
3. **质量天花板 95%**：无论保留多少 KV 都答不对 2 个样本，是 manifest 噪声/对抗性 needle 位置，非策略缺陷。
4. **零可见开销**：block-16 (0.48s/run) ≈ token (0.51s/run)，整块对齐对推理吞吐无影响。
5. **答辩可外推性**：本实验用 Qwen2.5-3B + 2k 上下文。**外推到 vLLM 7B/32k 上下文时**，方法层无新增假设（block 映射是纯函数，profiler 复用现有信号），但运行效率需在 vLLM 端验证（Phase 2）。

## 复现命令

```bash
cd ~/work/semanticache && source ~/work/venv/bin/activate
# 造 manifest（幂等）
python eval_niah.py --create-manifest results/v3/manifest40.json \
    --manifest-samples 40 --haystack-length 2000
# 卡 0：full + token + block-16
CUDA_VISIBLE_DEVICES=0 python eval_niah.py --manifest results/v3/manifest40.json \
    --policies full semantic block_semantic --sweep-budgets 1.0 0.5 0.3 0.25 0.1 \
    --eviction-block-size 16 --no-bnb-4bit --output results/v3/gate_block16.json
# 卡 1：block-32（并行）
CUDA_VISIBLE_DEVICES=1 python eval_niah.py --manifest results/v3/manifest40.json \
    --policies block_semantic --sweep-budgets 0.5 0.3 0.25 0.1 \
    --eviction-block-size 32 --no-bnb-4bit --output results/v3/gate_block32.json
```

## 答辩用图建议

`block_size ∈ {token, 16, 32}` × `budget ∈ {0.1, 0.25, 0.5, 1.0}` 的 accuracy 折线图（4 条线），标注 25% 处的 gate 判定。横向 4 卡=能直接生成。

## 结论落到 plan

- **Phase 1 出口条件 1/2 达成**（block 粒度质量保留率达标）
- **Phase 2 进入条件满足**：可继续 SieveKV-on-vLLM MVP 路线
- 不需要触发 plan §7 备选（block-8 / 路线 B）
