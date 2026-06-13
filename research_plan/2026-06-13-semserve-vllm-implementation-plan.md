# SemServe: 基于 vLLM 的语义感知多用户 KV Cache 分配与调度 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 vLLM v1 engine 上实现"token/block 级语义价值驱动的跨请求 KV cache 显存分配与调度"系统（SemServe），在多用户竞争场景下用语义信息密度决定谁被压缩、谁被降级、谁被抢占，目标投稿 MLSys / EuroSys / ATC。

**Architecture:** 三层结构：(1) **Profiler 层**（复用 semanticache 的静态语义信号 + OP policy MLP，在请求 admission 时异步计算 segment→block 级语义价值）；(2) **Priority + Allocator 层**（OS 进程调度类比：租户动态优先级（语义价值 + SLO + aging）→ cgroup 式租户 budget；全局显存压力下按边际语义价值做 water-filling 块回收，输出 demote/evict/preempt 指令）；(3) **Runtime 层**（修改 vLLM v1 KVCacheManager 支持 block-table 压缩式部分淘汰 + CPU int8 warm tier 降级，修改 scheduler 抢占路径接入 allocator）。RetentionPlan IR 作为算法与系统的桥梁保留。

**Tech Stack:** vLLM v1 engine（源码 fork，pin 单一 commit）、PyTorch、FlashAttention/FlashInfer paged kernel、现有 semanticache 代码库（HF transformers 层，继续作为离线验证 testbed）。硬件：4× RTX 6000D（Linux 服务器）。本地 Windows 只做语法检查（`uv run python -m py_compile`），所有运行在 GPU 服务器。

---

## 0. 定位（一段话，写给未来的自己）

已有工作占掉的坑：**KVShare**（arXiv 2503.16525，跨请求语义复用/去重）、**Semantic Scheduling**（2506.12204，请求级语义排序）、**SAECache**（2605.18825，prefix cache 语义淘汰）、**RoleKV**（角色级淘汰）、**Continuum/CacheTTL**（2511.02230，TTL 保留）。SemServe 的差异化主张必须始终是：

> 显存竞争下的 **token/block 级语义价值跨请求分配**：vLLM 在显存压力下只会整请求抢占（recompute/swap）；SemServe 把抢占变成渐进式语义压缩——低语义密度请求先被压缩/降级到 int8 warm tier，高密度请求保持全精度，并用 SLO 约束做公平性兜底。评估引入 **quality-aware goodput**（SLO 达标 ∧ 答案正确）这一新指标。

任何阶段如果发现实现内容滑向"语义复用"或"请求级排序"，停下来重新对齐。

### 0.1 OS 类比框架（论文 framing 与设计词汇表）

故事主线：**PagedAttention 把虚拟内存的分页机制搬进了 KV cache，prefix caching 是 COW 共享页；但 OS 内存子系统的其余部分——进程优先级、cgroup 配额、页面回收、压缩交换、OOM killer——在 KV cache 世界还没有对应物。SemServe 补全这个子系统，并用语义价值替代访问时近性（recency）作为回收信号。**

| OS 概念 | SemServe 对应物 | 实现位置 |
|---|---|---|
| 虚拟内存 / 分页 | PagedAttention | vLLM 已有 |
| COW 共享页 | prefix caching | vLLM 已有 |
| 进程创建 + 初始工作集 | 租户 admission 时的初始 KV 分配（"fork"） | Profiler + Scheduler 入口 |
| 动态优先级调度（含 aging） | 语义价值 + SLO 等级 + 等待时间 aging → 租户优先级 | `semserve/priority.py` (Task 3.0) |
| cgroup 内存配额 | 优先级 → 租户级 KV budget | `semserve/allocator.py` |
| 页面回收（active/inactive LRU） | budget 超限时按 block 语义分数回收 | Allocator + KVCacheManager.compact_request |
| zswap / zram 压缩内存 | int8 warm tier（`quantized_tier_cache.py` 复用） | `semserve/tier_manager.py` |
| swap to disk | demote 到 CPU 内存 | `semserve/tier_manager.py` |
| OOM killer | 整请求抢占（最后手段，fallback） | vLLM 原生路径 |
| IPC 共享内存 | 跨租户语义复用（KVShare 方向） | Phase 5 可选扩展，不做主贡献 |

使用纪律：**类比是 framing 不是 contribution**。论文中每个映射必须有真实机制和实验支撑；类比表只作为组织语言。设计上有两个由类比直接导出的硬约束：(1) 优先级**必须带 aging**，纯语义优先级会饿死低密度租户（OS 调度的经典 starvation 教训，写进设计反而强化类比的严肃性）；(2) "fork" 初始分配按租户首请求的语义画像给初始 budget，之后 budget 随优先级动态调整，**租户（tenant）是预算主体，请求（request）是回收执行单元**——两级结构：tenant priority → tenant budget（cgroup 层），block semantic score → 租户内回收顺序（页面回收层）。

---

## 1. 仓库与环境策略

```
D:\semanticache\              # 现有 repo（算法 testbed + 论文），保持不动
  semserve/                   # 新 Python 包：profiler / scorer / allocator / tier（纯算法，不依赖 vLLM）
  bench/                      # 多租户负载生成器 + 指标采集（依赖 vLLM 仅作为 client）
  tests/semserve/             # 单元测试（本地可跑，CPU only）
  research_plan/              # 本计划

<GPU 服务器> ~/work/
  semanticache/               # git clone 本 repo
  vllm/                       # fork: github.com/Zlatanwic/vllm，分支 semserve/main
```

原则：
- **`semserve/` 包不 import vllm**。所有 vLLM 内部修改放在 fork 里，fork 通过 `from semserve import ...` 调用本包（pip install -e）。这样算法代码可以在本地 Windows 上用 CPU 单测开发，vLLM 集成代码只在服务器上改。
- vLLM fork **pin 死一个 release tag**（Phase 0 确定具体版本），整个项目周期不追 upstream，论文写完前不 rebase。
- GPU 服务器环境：CUDA 版本需匹配 RTX 6000D（Blackwell 架构需要 CUDA 12.8+ / 近期 torch），Phase 0 第一件事就是确认。

---

## Phase 0：环境与 vLLM 源码勘察（约 1–2 周）

目标：在 4×6000D 上跑通 vanilla vLLM serving 基线，产出一份"vLLM v1 内部结构勘察笔记"，确定后续所有修改的锚点。**这是后续计划的事实基础，Phase 2 开始前必须根据勘察结果回到本文档修订 Phase 2–4 的文件路径。**

### Task 0.1: GPU 服务器环境与 vLLM 安装

**Files:**
- Create: `docs/semserve/00-environment.md`（环境记录）

- [ ] **Step 1: 确认硬件与驱动**

```bash
nvidia-smi                                    # 确认 4 卡、显存容量、驱动版本
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability(0))"
```
预期：4 张卡可见；记录 compute capability（Blackwell 应为 sm_120 系）。若 torch 不支持该 CC，安装 nightly 或对应 CUDA 12.8 wheel。

- [ ] **Step 2: 安装 pinned vLLM（先 pip 装验证可用性，再切源码）**

```bash
pip install vllm                              # AutoDL 习惯：bare pip；记录装到的版本号
export HF_ENDPOINT=https://hf-mirror.com      # 国内镜像
vllm serve Qwen/Qwen2.5-7B-Instruct --max-model-len 32768 --port 8000
```
预期：server 启动，`curl localhost:8000/v1/models` 返回模型。记录使用的 attention backend（启动日志里有，如 `FLASH_ATTN` / `FLASHINFER`）。

- [ ] **Step 3: fork 并源码安装**

```bash
git clone https://github.com/Zlatanwic/vllm ~/work/vllm && cd ~/work/vllm
git checkout <step2 安装的版本对应 tag>        # pin！写入 00-environment.md
git checkout -b semserve/main
VLLM_USE_PRECOMPILED=1 pip install -e .       # 复用预编译 kernel，只改 Python 层时无需重编
```
预期：`python -c "import vllm; print(vllm.__file__)"` 指向 ~/work/vllm。重跑 Step 2 的 serve 验证。

- [ ] **Step 4: 把环境信息写入 `docs/semserve/00-environment.md` 并 commit**（vLLM tag、torch/CUDA 版本、attention backend、单卡可承载的 max-model-len）

### Task 0.2: vLLM v1 内部结构勘察笔记

**Files:**
- Create: `docs/semserve/01-vllm-internals.md`

逐个模块读源码 + 加 print 跑通一条请求，回答以下问题清单（这些是 Phase 2–4 修改的全部锚点）。预期文件位置以 v1 engine 为准（`vllm/v1/` 下），**以实际源码为准修订**：

- [ ] **Step 1: Scheduler**（预期 `vllm/v1/core/sched/scheduler.py`）
  - 显存不足时的抢占路径在哪个函数？抢占顺序是什么（默认 LIFO/优先级）？
  - 被抢占请求是 recompute 还是 swap？v1 是否还有 swap 路径？
  - `priority` 参数从 request 传到 scheduler 的链路。
- [ ] **Step 2: KVCacheManager / BlockPool**（预期 `vllm/v1/core/kv_cache_manager.py`、`block_pool.py`）
  - block_size 默认值；free block 不足的判定点。
  - 每个 running request 的 block table 数据结构；逻辑 block → 物理 block 的映射在哪一层。
  - prefix caching 的 block hash 机制——部分淘汰一个请求的中段 block 会破坏什么不变量？
- [ ] **Step 3: Attention backend**
  - block_table 和 seq_lens 如何传入 kernel；**如果把某请求 block_table 中段的项移除并前移压缩、同时减小 seq_len，kernel 是否仍正确**（这是整个项目的 go/no-go 问题，Task 2.1 做实验验证）。
  - RoPE 是 pre-cache 应用（cache 中 K 已带位置）还是 post-cache？（决定淘汰中段后位置是否自洽；HF 路线我们已验证 pre-cache 可行。）
- [ ] **Step 4: 请求入口 hook 点**
  - prompt token ids 在哪一层最早可拿到（用于 profiler）；request-level 自定义字段如何附加（`Request` 对象 or extra_args）。
- [ ] **Step 5: 把回答整理进 `01-vllm-internals.md`，commit**。**然后回到本计划，修订 Phase 2–4 中的文件路径与函数名。**

### Task 0.3: vanilla serving 基线测量

**Files:**
- Create: `bench/baseline_smoke.py`

- [ ] **Step 1: 写最小压测脚本**（OpenAI client，N 并发，记录 TTFT/TPOT/总吞吐）

```python
"""Smoke benchmark: N concurrent long-context requests against a vLLM server."""
import argparse
import asyncio
import json
import time

from openai import AsyncOpenAI


async def one_request(client: AsyncOpenAI, model: str, prompt: str) -> dict:
    t0 = time.perf_counter()
    first_token_t = None
    n_tokens = 0
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        stream=True,
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if first_token_t is None:
                first_token_t = time.perf_counter()
            n_tokens += 1
    t1 = time.perf_counter()
    return {
        "ttft": first_token_t - t0 if first_token_t else None,
        "tpot": (t1 - first_token_t) / max(n_tokens - 1, 1) if first_token_t else None,
        "total": t1 - t0,
        "tokens": n_tokens,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--prompt-tokens", type=int, default=8000)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    args = parser.parse_args()

    client = AsyncOpenAI(base_url=args.base_url, api_key="EMPTY")
    filler = "The quick brown fox jumps over the lazy dog. " * (args.prompt_tokens // 9)
    prompt = filler + "\n\nSummarize the above in one sentence."

    results = await asyncio.gather(
        *[one_request(client, args.model, prompt) for _ in range(args.concurrency)]
    )
    ttfts = sorted(r["ttft"] for r in results if r["ttft"])
    print(json.dumps({
        "concurrency": args.concurrency,
        "ttft_p50": ttfts[len(ttfts) // 2],
        "ttft_p99": ttfts[int(len(ttfts) * 0.99)],
        "mean_tpot": sum(r["tpot"] for r in results) / len(results),
    }, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: 跑出三组数据并记录**：concurrency ∈ {4, 16, 64}，prompt 8k tokens，记录显存压力下是否出现抢占（vLLM 日志 grep "preempt"）。找到**能稳定触发抢占的负载点**——这就是后面所有对比实验的工况。
- [ ] **Step 3: Commit** `bench/baseline_smoke.py` + 数据到 `docs/semserve/00-environment.md`。

**Phase 0 出口条件**：vanilla 基线数据在手；`01-vllm-internals.md` 五个问题全部有答案；本计划 Phase 2–4 的路径已按实际源码修订。

---

## Phase 1：block 粒度离线验证（约 1–2 周，与 Phase 0 并行，本地 + 服务器均可）

目标：**在动 vLLM 之前，先在现有 HF testbed 上回答核心算法假设**——把 token 级语义淘汰粗化到 16-token block 粒度后，质量损失是否可接受。如果 block 粒度质量崩了，整个 vLLM 路线要换设计（见第 7 节风险表），所以这一步必须最先做。

### Task 1.1: segment 分数 → block 分数映射器（纯函数，本地可开发）

**Files:**
- Create: `semserve/__init__.py`
- Create: `semserve/block_mapping.py`
- Test: `tests/semserve/test_block_mapping.py`

- [ ] **Step 1: 写失败测试**

```python
"""Tests for segment-score -> block-score mapping."""
import torch

from semserve.block_mapping import segment_scores_to_block_scores


def test_uniform_segment_covers_blocks():
    # 一个 segment 覆盖 token [0, 32)，分数 0.8，block_size=16 -> 两个 block 都是 0.8
    seg_bounds = [(0, 32)]
    seg_scores = [0.8]
    out = segment_scores_to_block_scores(seg_bounds, seg_scores, seq_len=32, block_size=16)
    assert out.shape == (2,)
    assert torch.allclose(out, torch.tensor([0.8, 0.8]))


def test_partial_overlap_uses_token_weighted_mean():
    # segment A [0,8) 分数 1.0，segment B [8,16) 分数 0.0 -> block0 = 0.5
    out = segment_scores_to_block_scores([(0, 8), (8, 16)], [1.0, 0.0], seq_len=16, block_size=16)
    assert torch.allclose(out, torch.tensor([0.5]))


def test_pinned_segment_forces_max_score():
    # pinned segment（如 system prompt / 最新 user query）所在 block 强制 1.0
    out = segment_scores_to_block_scores(
        [(0, 16), (16, 32)], [0.2, 0.3], seq_len=32, block_size=16,
        pinned_bounds=[(0, 16)],
    )
    assert torch.allclose(out, torch.tensor([1.0, 0.3]))


def test_tail_block_padding():
    # seq_len=20, block_size=16 -> 2 个 block，尾块按实际 4 个 token 加权
    out = segment_scores_to_block_scores([(0, 20)], [0.6], seq_len=20, block_size=16)
    assert out.shape == (2,)
```

- [ ] **Step 2: 运行确认失败** `uv run pytest tests/semserve/test_block_mapping.py -v` → FAIL (module not found)

- [ ] **Step 3: 最小实现**

```python
"""Map segment-level retention scores to paged-attention block scores.

This is the bridge between the RetentionPlan IR (segment decisions) and
vLLM's block-granular KV cache: a block's score is the token-weighted mean
of the segment scores covering it, with pinned segments forcing 1.0.
"""
from __future__ import annotations

import torch


def segment_scores_to_block_scores(
    segment_bounds: list[tuple[int, int]],
    segment_scores: list[float],
    *,
    seq_len: int,
    block_size: int,
    pinned_bounds: list[tuple[int, int]] | None = None,
) -> torch.Tensor:
    """Return per-block scores of shape (ceil(seq_len / block_size),)."""
    token_scores = torch.zeros(seq_len, dtype=torch.float32)
    token_weight = torch.zeros(seq_len, dtype=torch.float32)
    for (start, end), score in zip(segment_bounds, segment_scores):
        start, end = max(0, start), min(seq_len, end)
        token_scores[start:end] += score
        token_weight[start:end] += 1.0
    token_scores = token_scores / token_weight.clamp(min=1.0)

    num_blocks = (seq_len + block_size - 1) // block_size
    pad = num_blocks * block_size - seq_len
    padded = torch.nn.functional.pad(token_scores, (0, pad))
    mask = torch.nn.functional.pad(torch.ones(seq_len), (0, pad))
    block_scores = (padded.view(num_blocks, block_size).sum(-1)
                    / mask.view(num_blocks, block_size).sum(-1).clamp(min=1.0))

    if pinned_bounds:
        for start, end in pinned_bounds:
            b0, b1 = start // block_size, (max(start, end - 1)) // block_size
            block_scores[b0 : b1 + 1] = 1.0
    return block_scores
```

- [ ] **Step 4: 测试通过** → **Step 5: Commit** `feat: add segment-to-block score mapping for paged KV`

### Task 1.2: HF testbed 上的 block 粒度淘汰策略

**Files:**
- Modify: `eviction_policies.py`（新增 `BlockSemantiCachePolicy(SemantiCachePolicy)`，复用现有 segment 打分，只把最终 keep mask 按 16-token block 对齐：一个 block 内任一 token 被保留则整块保留 → 用 Task 1.1 的 block score + block 级 top-k 选择）
- Test: `test_block_policy_lite.py`（模仿现有 `test_opsievekv_lite.py` 的写法做 smoke test）

- [ ] **Step 1**: 实现 `BlockSemantiCachePolicy`：在 `SemantiCachePolicy` 的 token 分数输出后插入 `segment_scores_to_block_scores` → 按 block 分数排序 → 在 token budget 约束下整块选取（`budget_blocks = budget_tokens // block_size`）。
- [ ] **Step 2**: smoke test 通过（CPU 小模型或服务器小规模跑通）。
- [ ] **Step 3**: 服务器上跑 NIAH 对比（复用 `eval_niah.py` 的 harness）：

```bash
python eval_niah.py --policy semanticache --budget 0.25 --out results/v3/niah_token.json
python eval_niah.py --policy block_semanticache --block-size 16 --budget 0.25 --out results/v3/niah_block16.json
python eval_niah.py --policy block_semanticache --block-size 32 --budget 0.25 --out results/v3/niah_block32.json
```

- [ ] **Step 4: 判定 gate 并记录**到 `docs/semserve/02-block-granularity.md`。

**Phase 1 出口条件（go/no-go gate）**：block-16 粒度在 NIAH/multi-needle 上达到 token 级质量的 ≥95%。达标 → Phase 2；不达标 → 看 block-8（需改 vLLM block_size 启动参数）或转向第 7 节备选路线 B。

---

## Phase 2：单请求语义淘汰进 vLLM（"SieveKV-on-vLLM" MVP，约 3–4 周）

目标：单条请求 prefill 结束后，按语义 block 分数淘汰其 KV block，decode 正常进行，质量与 HF 实现对齐。**这本身就是一个独立可发表的 artifact**（现有 learned eviction 工作几乎都停在 HF 层）。

> 注意：本 Phase 的 vLLM 内部文件路径/函数名以 Task 0.2 勘察结果为准，下面写的是预期锚点。

### Task 2.1: 【SPIKE】block_table 压缩可行性实验（最高优先级，1–2 天）

**Files:**
- Create: `~/work/vllm/examples/semserve_spike_blocktable.py`（fork 内一次性脚本）

- [ ] **Step 1**: 写一个直接驱动 `LLMEngine`（非 server 模式）的脚本：提交一条 2048-token 请求，prefill 完成后**手动**从其 block table 中删除中段 25% 的 block（前移压缩剩余项、修正该请求的 seq_len / num_computed_tokens），继续 decode 32 token。
- [ ] **Step 2**: 验证三件事：(a) 不 crash；(b) 输出与"HF 实现按相同 mask 淘汰"语义一致（不要求 bit-exact，要求生成内容合理且与 HF 版淘汰结果可比）；(c) prefix caching 开启时是否破坏 hash 不变量（预期需要对被修改请求禁用 prefix cache 复用——记录处理方式）。
- [ ] **Step 3**: 把结论（包括当前 attention backend 下 kernel 是否接受压缩后的 block_table、RoPE 是否 pre-cache）写入 `01-vllm-internals.md`。**这是项目的 go/no-go 点**：若当前 backend 不可行，测试 `VLLM_ATTENTION_BACKEND=FLASHINFER`；都不可行 → 第 7 节路线 B。

### Task 2.2: 【SPIKE】prefill 后 query-window 打分可行性（替代 AttentionTracker，1–2 天）

背景：`attention_tracker.py` 依赖 `output_attentions`，vLLM 永远不会提供。替代方案 = SnapKV 式观察窗：prefill 结束后，取最后 W=64 个 token 的 query 状态，与 cache 中所有 K 做一次 batched matmul 得到注意力 proxy 分数（只在选定的 2–4 层做，控制开销）。

- [ ] **Step 1**: 在 fork 中找到 prefill 完成后能同时拿到 (a) 最后 W 个位置的 hidden states 或 q 投影、(b) 该请求 KV cache 物理 block 的 hook 点（候选：model_runner 的 forward 之后、attention layer 的 forward hook）。
- [ ] **Step 2**: 实现一次性的打分函数，输出 per-token attention proxy 分数；与 HF 实现里 AttentionTracker 在相同 prompt 上的累计分数做 Spearman 相关性对比，目标 ρ > 0.7。
- [ ] **Step 3**: 测量该打分的额外延迟（目标 < 5% prefill 时间）。结论写入 `01-vllm-internals.md`。
- [ ] **降级路径**：若 hook 拿不到 q 状态，则 MVP 退化为**纯静态信号 + OP policy MLP**（`op_policy_model.py` 的 7 信号中 attn/entropy 两项置零或用 density/query 代理填充——OP-SieveKV 已有的 ablation 数据可以告诉我们这两个信号去掉损失多少，先查 `results/` 里的 ablation 结果再决定）。

### Task 2.3: SemServe profiler（请求 admission 时的语义画像，semserve 包内，本地可开发）

**Files:**
- Create: `semserve/profiler.py`
- Test: `tests/semserve/test_profiler.py`

- [ ] **Step 1: 写失败测试**（接口约定）

```python
"""Profiler turns raw prompt token ids into a RequestSemanticProfile."""
from transformers import AutoTokenizer

from semserve.profiler import SemanticProfiler


def test_profile_chat_prompt():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": "You are helpful."},
         {"role": "user", "content": "What is the capital of France? " + "filler " * 200}],
        tokenize=False, add_generation_prompt=True,
    )
    token_ids = tok.encode(prompt)
    profiler = SemanticProfiler(tok, policy_checkpoint=None)  # None -> heuristic 分数
    profile = profiler.profile(token_ids)

    assert profile.num_tokens == len(token_ids)
    assert len(profile.segment_bounds) == len(profile.segment_scores) >= 2
    assert 0.0 <= min(profile.segment_scores) <= max(profile.segment_scores) <= 1.0
    assert profile.pinned_bounds  # system prompt 和最新 user query 必须被 pin
    assert profile.mean_density >= 0.0  # 请求级聚合密度，供 allocator 使用
```

- [ ] **Step 2: 确认失败** → **Step 3: 实现**

```python
"""Per-request semantic profiling at admission time.

Wraps the existing SemanticAnalyzer (static signals) and optionally the
OP-SieveKV SegmentRetentionMLP into a vLLM-independent profile object.
Runs on CPU; designed to overlap with request queueing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import PreTrainedTokenizer

from semantic_analyzer import SemanticAnalyzer, RoleTag
from op_policy_model import load_policy_checkpoint


@dataclass
class RequestSemanticProfile:
    num_tokens: int
    segment_bounds: list[tuple[int, int]]
    segment_scores: list[float]          # [0,1] keep 价值
    pinned_bounds: list[tuple[int, int]]
    mean_density: float                  # 请求级语义密度（allocator 的核心输入）
    metadata: dict = field(default_factory=dict)


class SemanticProfiler:
    def __init__(self, tokenizer: PreTrainedTokenizer,
                 policy_checkpoint: str | Path | None = None):
        self.analyzer = SemanticAnalyzer(tokenizer)
        self.policy = None
        if policy_checkpoint is not None:
            self.policy, self.feat_mean, self.feat_std, _ = load_policy_checkpoint(policy_checkpoint)

    def profile(self, token_ids: list[int]) -> RequestSemanticProfile:
        ids = torch.tensor(token_ids)
        role_tags = self.analyzer.compute_role_tags(ids)          # 现有接口，名称以实际为准
        density = self.analyzer.compute_density_scores(ids)       # 现有接口，名称以实际为准
        bounds = self._segment_by_role_and_punct(ids, role_tags)  # 复用现有 segmentation 逻辑
        scores = self._score_segments(ids, bounds, role_tags, density)
        pinned = [b for b, tag in zip(bounds, self._segment_roles(bounds, role_tags))
                  if tag in (RoleTag.SYSTEM, RoleTag.USER_LATEST)]
        return RequestSemanticProfile(
            num_tokens=len(token_ids),
            segment_bounds=bounds,
            segment_scores=scores,
            pinned_bounds=pinned,
            mean_density=float(density.mean()),
        )
    # _segment_by_role_and_punct / _score_segments: 从 SemantiCachePolicy
    # (eviction_policies.py:483) 中提取现有 segmentation 与打分代码——这是一次
    # 重构任务：把"打分"从"淘汰执行"中拆出来，使其不依赖 DynamicCache。
```

实现要点：`eviction_policies.py` 中 `SemantiCachePolicy` 的 segmentation + 打分逻辑（不含 cache 操作）抽到独立函数，原 policy 改为调用该函数——保证 HF testbed 与 vLLM 路线用**同一份打分代码**，论文里两边数字才可比。

- [ ] **Step 4: 测试通过**（本地 CPU 即可，0.5B tokenizer 下载小）→ **Step 5: Commit** `refactor: extract scoring from SemantiCachePolicy; add SemanticProfiler`

### Task 2.4: vLLM 集成 — 单请求淘汰 MVP

**Files（fork 内，路径按 Task 0.2 修订）:**
- Create: `~/work/vllm/vllm/semserve_integration/__init__.py`（profiler 调用 + 分数缓存，按 request_id 索引）
- Modify: scheduler/kv_cache_manager 中 prefill 完成的回调点：注入"按 block 分数淘汰至 budget"逻辑（复用 Task 2.1 spike 验证过的 block_table 压缩操作，工程化：封装成 `KVCacheManager.compact_request(request_id, keep_block_mask)`）
- Create: 启动参数 `--semserve-budget 0.25 --semserve-policy <ckpt path>`（环境变量起步也可：`SEMSERVE_BUDGET=0.25`）

- [ ] **Step 1**: profiler 在 admission 时异步跑（线程池），结果存 `request_id -> RequestSemanticProfile` 字典。
- [ ] **Step 2**: prefill 完成回调中：`block_scores = segment_scores_to_block_scores(...)`（+ Task 2.2 的 attention proxy 若可用）→ top-k 保留 → `compact_request`。
- [ ] **Step 3: 正确性验证**——同一 prompt、同一 budget 下，vLLM 版与 HF 版（`BlockSemantiCachePolicy`）保留的 block 集合 Jaccard ≥ 0.9（打分代码同源，差异只来自 attention proxy）。写成脚本 `bench/verify_vllm_vs_hf.py`。
- [ ] **Step 4: 质量验证**——`eval_niah.py` 加 `--backend vllm-server` 模式（HTTP client 打 SemServe server），NIAH 分数与 HF block 版差距 < 2 分。
- [ ] **Step 5: 性能验证**——`bench/baseline_smoke.py` 对比 vanilla vs SemServe（budget=0.25）：长 prompt 高并发下吞吐应**提升**（cache 占用降为 1/4 → 并发容量上升），单请求开销 < 5%。
- [ ] **Step 6: Commit + tag** `semserve-mvp-v0.1`。

**Phase 2 出口条件**：MVP 三项验证（正确性/质量/性能）全过。此时已有一个可写 workshop paper 的 artifact。

---

## Phase 3：跨租户语义分配器（项目核心贡献，约 3–4 周）

目标：两级结构。**租户层**（cgroup 类比）：动态优先级 → 租户 KV budget；**块回收层**（页面回收类比）：budget 超限/全局压力时按 block 语义分数决定压缩谁、降级谁、抢占谁，替换 vLLM 默认的"整请求抢占"。

### Task 3.0: 租户动态优先级模块（纯算法，本地可开发 + 单测）

**Files:**
- Create: `semserve/priority.py`
- Test: `tests/semserve/test_priority.py`

优先级定义（v1，论文可消融各项）：

```
priority(t) = w_sem * mean_density_t        # 语义价值（profiler 聚合，随新请求/新 turn 更新）
            + w_slo * slo_class_t           # 租户 SLO 等级（静态配置：0/1/2）
            + w_age * age(t)                # aging：自上次获得 budget 提升以来的等待时间，防饿死
budget_t   = clamp(B_total * priority_t / Σ priority, floor_t, ceil_t)
```

- [ ] **Step 1: 写失败测试**

```python
"""Tests for dynamic tenant priority and budget assignment."""
from semserve.priority import TenantPriorityTracker


def test_high_density_tenant_gets_larger_budget():
    tr = TenantPriorityTracker(total_blocks=100)
    tr.update_tenant("A", mean_density=0.9, slo_class=1)
    tr.update_tenant("B", mean_density=0.1, slo_class=1)
    budgets = tr.compute_budgets()
    assert budgets["A"] > budgets["B"]
    assert budgets["A"] + budgets["B"] <= 100


def test_aging_prevents_starvation():
    tr = TenantPriorityTracker(total_blocks=100, w_age=0.05)
    tr.update_tenant("A", mean_density=0.9, slo_class=1)
    tr.update_tenant("B", mean_density=0.1, slo_class=1)
    b0 = tr.compute_budgets()["B"]
    for _ in range(50):           # B 长期拿不到提升，aging 累积
        tr.tick()
    b1 = tr.compute_budgets()["B"]
    assert b1 > b0                # 饿死保护生效


def test_floor_and_ceiling_respected():
    tr = TenantPriorityTracker(total_blocks=100, floor_blocks=10, ceil_blocks=80)
    tr.update_tenant("A", mean_density=1.0, slo_class=2)
    tr.update_tenant("B", mean_density=0.0, slo_class=0)
    budgets = tr.compute_budgets()
    assert budgets["B"] >= 10 and budgets["A"] <= 80


def test_priority_is_dynamic_across_turns():
    # 同一租户新 turn 带来高密度请求 -> 优先级上升
    tr = TenantPriorityTracker(total_blocks=100)
    tr.update_tenant("A", mean_density=0.2, slo_class=1)
    p0 = tr.priorities()["A"]
    tr.update_tenant("A", mean_density=0.9, slo_class=1)
    assert tr.priorities()["A"] > p0
```

- [ ] **Step 2: 确认失败** `uv run pytest tests/semserve/test_priority.py -v` → FAIL
- [ ] **Step 3: 最小实现**（`TenantPriorityTracker`：dict 维护 per-tenant `(ema_density, slo_class, age_ticks)`；`update_tenant` 用 EMA 更新密度并把 age 清零的时机定义为"budget 相对份额上升"；`tick()` 全员 age+1；`compute_budgets` 按公式归一化后做 floor/ceil clamp，再把因 clamp 产生的剩余按优先级二次分配）
- [ ] **Step 4: 测试通过** → **Step 5: Commit** `feat: dynamic tenant priority with aging (cgroup-style budgets)`

### Task 3.1: 块回收算法（纯算法，本地可开发 + 单测）

**Files:**
- Create: `semserve/allocator.py`
- Test: `tests/semserve/test_allocator.py`

核心抽象：每个 running request r 有一条**价值-预算曲线** V_r(b) = "保留 b 个 block 时的累计语义价值"（block 分数降序的前缀和，pinned block 永远在前）。全局可用 block 数 B 不足时，求解：

```
max Σ_r V_r(b_r)   s.t.  Σ_r b_r ≤ B,  b_r ≥ pinned_r,  b_r ≥ floor_r
```

其中 `floor_r` 来自 Task 3.0 的租户 budget：同一租户的所有请求共享该租户的 budget 下限按请求长度比例分摊。两级耦合方式：租户层先定各租户可用 block 总量（慢时钟，每次 admission/完成时更新），块回收层在全局压力时刻做细粒度 water-filling（快时钟，每次 free block 不足时触发）。

V_r 是凹函数（分数降序前缀和），所以贪心 water-filling 即最优：每次把一个 block 配额给当前边际价值最高的请求。逆操作（回收）同理：每次从边际价值最低的请求收走一个 block。

- [ ] **Step 1: 写失败测试**

```python
"""Tests for the cross-request water-filling allocator."""
import torch

from semserve.allocator import SemanticAllocator, RequestCacheState


def _req(rid, scores, pinned=1, floor=0):
    return RequestCacheState(
        request_id=rid,
        block_scores=torch.tensor(scores),
        num_pinned_blocks=pinned,
        floor_blocks=floor,
    )


def test_pressure_reclaims_from_low_value_request():
    # A 高密度（全 0.9），B 低密度（全 0.1），各持有 4 block，需回收 3 个
    alloc = SemanticAllocator()
    plan = alloc.reclaim(
        [_req("A", [1.0, 0.9, 0.9, 0.9]), _req("B", [1.0, 0.1, 0.1, 0.1])],
        num_blocks_needed=3,
    )
    # B 的 3 个低分 block 全部被收走，A 不动
    assert plan.reclaim_per_request == {"B": 3}


def test_pinned_and_floor_are_never_reclaimed():
    alloc = SemanticAllocator()
    plan = alloc.reclaim(
        [_req("A", [1.0, 0.2, 0.2, 0.2], pinned=1, floor=2)],
        num_blocks_needed=4,
    )
    # floor=2 -> 最多收走 2 个；剩余缺口标记为需要抢占
    assert plan.reclaim_per_request == {"A": 2}
    assert plan.unmet_blocks == 2
    assert plan.preempt_candidates == ["A"]


def test_marginal_value_interleaving():
    # A=[1.0,0.8,0.3], B=[1.0,0.5,0.4]，收 2 个 -> 先收 A 的 0.3，再收 B 的 0.4
    alloc = SemanticAllocator()
    plan = alloc.reclaim(
        [_req("A", [1.0, 0.8, 0.3]), _req("B", [1.0, 0.5, 0.4])],
        num_blocks_needed=2,
    )
    assert plan.reclaim_per_request == {"A": 1, "B": 1}
```

- [ ] **Step 2: 确认失败** → **Step 3: 实现**（堆上做 lazy 贪心，每请求维护"下一个被收走的 block 边际分数"指针；输出 `ReclaimPlan{reclaim_per_request, unmet_blocks, preempt_candidates}`）→ **Step 4: 测试通过** → **Step 5: Commit**

- [ ] **Step 6: 接入 Task 3.0**：`floor_blocks` 由 `TenantPriorityTracker.compute_budgets()` 给出（替代写死常数）。论文 ablation 点：纯语义 vs 语义+aging vs 语义+aging+SLO 三档。

### Task 3.2: 降级层 — CPU int8 warm tier（替代直接丢弃）

**Files:**
- Create: `semserve/tier_manager.py`（复用 `quantized_tier_cache.py` 的 `QuantizedTensor`，按 block 为单位 quantize→CPU / fetch→dequantize→GPU）
- Test: `tests/semserve/test_tier_manager.py`（round-trip 误差 < 1%，CPU 可测）

- [ ] **Step 1–4**: TDD 同上模式。关键接口：`demote_blocks(request_id, physical_block_ids, kv_tensors) -> None` 和 `promote_blocks(request_id, logical_block_ids) -> kv_tensors`。
- [ ] **Step 5（fork 内）**: 接入点 = allocator 发出 reclaim 指令时，先 demote 再释放物理 block；请求后续若出现质量风险信号（可选：decode 阶段对被淘汰区域的 proxy 需求检测，二期再做），promote 回来。**MVP 先做 demote-only（被收走的 block 进 warm tier 但不自动召回），promote 作为 Phase 5 扩展**——这样不阻塞主线，且 demote-only 已经构成"渐进压缩 vs 整请求抢占"的完整故事（被压缩请求质量受损但不重算，抢占请求 TTFT 爆炸）。
- [ ] **Step 6: Commit**

### Task 3.3: scheduler 接入 + 端到端

**Files（fork 内）:**
- Modify: scheduler 的显存不足处理路径：free blocks 不够时先问 `SemanticAllocator.reclaim`，按 plan 执行 compact/demote；仅当 `unmet_blocks > 0` 才走原生抢占（按 preempt_candidates 排序）。

- [ ] **Step 1**: 接入并通过 Task 0.3 的压测（确认不 crash、不死锁——注意 plan 全 pinned 时必须能 fall through 到原生抢占，避免活锁）。
- [ ] **Step 2: 关键对照实验**（项目的 money shot，先跑小规模确认信号存在）：

```
工况：Phase 0 找到的抢占触发负载（长 prompt 高并发）
对照：vanilla vLLM（整请求抢占） vs SemServe（语义渐进压缩）
预期：SemServe 的 TTFT p99 大幅下降（无重算），吞吐持平或更高，
      质量探针请求的正确率下降可控（< 5 pt）
```

- [ ] **Step 3**: 数据写入 `docs/semserve/03-allocator-results.md`，commit + tag `semserve-v0.2`。

**Phase 3 出口条件**：money shot 实验信号确认存在（TTFT p99 改善 > 30% 且质量损失可控）。信号不存在 → 回到 allocator 设计迭代，而不是继续往下铺。

---

## Phase 4：多租户评估体系（约 3–4 周，harness 可与 Phase 3 并行开发）

### Task 4.1: 负载生成器

**Files:**
- Create: `bench/workload.py`（trace 生成）、`bench/runner.py`（回放 + 指标采集）、`bench/metrics.py`

负载构成（每个 trace 是 (arrival_time, tenant, prompt, expected_answer?) 列表）：
- **chat 租户**：ShareGPT 采样，短 prompt 高频率；
- **RAG/长文租户**：LongBench 风格 8k–32k prompt；
- **质量探针**：NIAH 式请求按固定比例混入（已知答案 → 可测 contention 下的正确率），复用 `eval_niah.py` 的 needle 构造代码；
- 到达过程：Poisson，扫 arrival rate 找 SLO 拐点。

- [ ] 指标（`bench/metrics.py`）：TTFT p50/p99、TPOT、吞吐、抢占次数、**quality-aware goodput** = 比例(SLO 达标 ∧ 探针答对)、租户间公平性（Jain index）。
- [ ] TDD：trace 生成的确定性（固定 seed）、指标计算各写单测。

### Task 4.2: 基线实现

全部以"vLLM 启动参数/小 patch"形式实现，跑同一 trace：

| 基线 | 实现方式 |
|---|---|
| vanilla vLLM | 默认（FCFS + 抢占重算） |
| uniform 压缩 | 每请求无差别 budget=0.5（SnapKV/StreamingLLM 式，per-request 应用，证明"语义跨请求分配"优于"无差别压缩"） |
| VTC 公平调度 | 复现 token counter 调度（只动 scheduler 排序，量不大） |
| oracle 上界 | 用离线 drop/restore oracle 标注的 block 重要性（复用 `collect_op_onpolicy_oracle_data.py` 思路）替代 policy 分数 |

- [ ] 每条基线 + SemServe 跑完整 trace 矩阵（3 个 arrival rate × 2 模型规模），结果入 `results/semserve/`。

### Task 4.3: 规模实验

- [ ] 单卡 7B（主矩阵）→ TP=4 大模型（Qwen2.5-72B 或 32B，按 6000D 实测显存定）验证结论随规模保持。4 卡也可拆成 4 个独立单卡 server 并行扫参数，加速主矩阵。

---

## Phase 5：消融、扩展与论文（约 3–4 周）

- [ ] 消融：(a) 去掉 OP policy MLP 用纯 heuristic；(b) 去掉公平下限；(c) 去掉 warm tier（直接丢弃）；(d) block_size 16/32/64；(e) attention proxy 有无。
- [ ] 扩展（按时间取舍）：warm tier promote 召回；多轮会话跨 turn 保留（衔接现有 role-aware 多轮工作）；**跨租户语义复用（IPC/共享内存类比）**——在 prefix caching 的精确匹配之外做语义近似匹配复用（KVShare 方向），作为系统完整性的可选章节而非主贡献，时间不够直接砍。
- [ ] Overhead 剖析：profiler CPU 时间、allocator 决策时间、demote 带宽占用（复用 `eval_overhead.py` 的报告格式）。
- [ ] 论文：MLSys / EuroSys / ATC（查最近的截稿日历后定）。故事线直接继承 `op_sievekv_research_story_and_system_plan.md` 第 3 节，把"RetentionPlan runtime"升级为"multi-tenant SemServe runtime"。

---

## 7. 风险表与备选路线

| 风险 | 触发点 | 备选 |
|---|---|---|
| block 粒度质量崩 | Phase 1 gate | block-8（vLLM 支持 block_size=8）；或路线 B |
| block_table 压缩在所有 backend 都不可行 | Task 2.1 | **路线 B**：放弃 token/block 级淘汰，做"请求级语义压缩档位"——压力下把低密度请求整体 swap 到 CPU int8（粗粒度但工程量小一个量级，故事改讲 semantic-aware preemption/swap ordering） |
| attention proxy 拿不到 | Task 2.2 | 纯静态信号 + OP MLP（先查现有 ablation 确认损失） |
| money shot 信号弱 | Phase 3 出口 | 加大 contention 强度 / 提高长文租户占比；若仍弱，说明 vanilla 抢占在该工况不够痛，需重新选工况（如 agent 多轮、超长 RAG） |
| vLLM 版本内部结构与预期不符 | Task 0.2 | 本计划 Phase 2–4 路径全部以勘察笔记为准修订，计划是活文档 |

## 8. 近期里程碑：课堂答辩包（答辩日 = 6月17日，4 天崩溃倒排）

课堂答辩不需要 vLLM 改造完成，需要的是**完整的故事 + 一个实证 slide + 设计蓝图**。今天 = 6月13日，只剩 4 天，原"1.5–2 周"路径必须压缩。核心取舍：

- **保底实证**押在 **HF testbed block 粒度实验**（Task 1.1 + 1.2）：跑的是已在 AutoDL 验证过的现有代码（`eval_niah.py`），换机器即可，风险低，**几乎不可能产不出结果**。这是"我们的方案有依据"那页。
- **vLLM 抢占病理图**（Task 0.3）作为**高价值 bonus**：需在 Blackwell 上全新装 vLLM，CUDA/torch 匹配是 4 天内的主要不确定性。拿到就是最强动机页；卡住则用 vLLM 已知抢占行为 + PagedAttention 论文论证动机，**不阻塞答辩**。
- **故事 + 设计蓝图**：零依赖，立即可做。OS 类比表（0.1 节）+ 三层架构图（Profiler / Priority+Allocator / Runtime）。课堂听众懂 OS，类比表是全场最强的一页。

### 4 天倒排

| 日期 | 必做（保底） | 尝试（bonus） |
|---|---|---|
| 6/13 周六（今天） | ① 现有代码在 6000D 服务器跑通（拉起 `eval_niah.py` 冒烟）；② 本地写 `BlockSemantiCachePolicy`（Task 1.1 + 1.2 Step 1–2，纯 CPU TDD）；③ slide 骨架：OS 类比 + 架构图 | 服务器并行装 vLLM（Task 0.1） |
| 6/14 周日 | 服务器跑 block 粒度 NIAH 对比（token vs block-16/32），出质量保留曲线 | vLLM 装通则跑 `baseline_smoke.py` 找抢占触发点（Task 0.3） |
| 6/15 周一 | 把 block 曲线做成图进 slide；正文与讲稿成形 | vLLM 抢占病理图（arrival rate ↑ → 抢占数/TTFT p99 ↑）若拿到则进 slide |
| 6/16 周二 | **实验硬冻结**（不再开新实验）；slide 定稿 + 排练；导出 | — |
| 6/17 周三 | 答辩 | — |

产出物：`research_plan/ppt_semserve/` 下的 slides（用现有 ppt 工作流生成）。

**红线**：6/16 之后不碰任何新实验；若 6/15 仍未拿到任何实证图，答辩降级为纯设计提案（故事 + 蓝图 + 已知文献动机），这在课堂场景完全可接受。

## 9. 里程碑总览

| 周 | 里程碑 |
|---|---|
| W1–2 | Phase 0 环境 + 勘察笔记；Phase 1 block gate 结果；**课堂答辩包素材就绪** |
| W3–4 | Task 2.1/2.2 两个 spike 结论（项目 go/no-go） |
| W5–8 | SieveKV-on-vLLM MVP（tag v0.1，可出 workshop artifact） |
| W9–12 | 租户优先级 + 跨请求 allocator + money shot 实验（tag v0.2） |
| W13–16 | 多租户评估矩阵 + 基线 |
| W17–20 | 消融 + 论文初稿 |

执行纪律：每个 Phase 出口条件不满足时**不进入下一 Phase**；每完成一个 Task 用 Conventional Commits 提交；GPU 实验前先报预计耗时（长任务用 nohup + 日志）。
