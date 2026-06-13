# SemServe 环境记录（00-environment）

> Task 0.1 产出物。记录 GPU 服务器环境、依赖版本与关键决策，作为后续所有实验的事实基础。
> 最后更新：2026-06-13。

## 硬件

| 项 | 值 |
|---|---|
| GPU | 4× NVIDIA RTX 6000D（Blackwell） |
| 单卡显存 | 85651 MiB ≈ **85.6 GB**（总 ≈342 GB） |
| Compute capability | **sm_120** = `(12, 0)` |
| 驱动 | NVIDIA-SMI 595.80 |
| 驱动 CUDA | 13.2 |
| 主机 | `bc01@ubun`（Ubuntu，GCC 13.3.0） |

## Python / 依赖（HF testbed 轨道）

| 项 | 值 |
|---|---|
| 系统 Python | 3.12.3（仅 `python3`，无 `python`；`python3.12-venv` 未装） |
| 环境管理 | **uv venv**（`~/work/venv`）—— `python3 -m venv` 因缺 `ensurepip` 失败，uv 自带 bootstrap 绕开 |
| torch | **2.12.0+cu130**（`pip install torch --index-url https://download.pytorch.org/whl/cu130`） |
| torch CUDA | 13.0 |
| 其余 | transformers, accelerate, numpy, matplotlib, seaborn, tqdm, modelscope, pytest |
| **未安装** | `vllm`（bonus 轨道，待 Task 0.1 Step 2–3）；`bitsandbytes`（被 `--no-bnb-4bit` 绕过，不需要） |

### 复现命令
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
cd ~/work/semanticache
uv venv ~/work/venv --python 3.12 && source ~/work/venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu130
uv pip install transformers accelerate numpy matplotlib seaborn tqdm modelscope pytest
```
> ⚠️ 每开新终端需 `source ~/work/venv/bin/activate`。

## 关键决策

1. **不用量化**：85 GB×4 显存充裕，`eval_niah.py` 一律加 `--no-bnb-4bit` 跑 full fp16，绕开 `bitsandbytes>=0.43` 不支持 sm_120 的问题。
2. **单卡跑单模型**：HF testbed 实验用 `CUDA_VISIBLE_DEVICES=0` 固定单卡；其余 3 卡留作并行扫参（plan Task 4.3）。
3. **模型来源**：ModelScope（`run_generation.py` 用 `snapshot_download`），缓存默认 `~/.cache/modelscope/`（AutoDL 专用路径 `/root/autodl-tmp` 不存在，已优雅 fallback）。
4. **大显存对项目的影响**：触发 vLLM 抢占（money-shot 工况）需要很高并发 / 极长上下文才能填满 85 GB KV——Task 0.3 选工况时需特别加大 contention。

## vLLM 轨道（系统实验，Task 0.1 Step 2–3）

| 项 | 值 |
|---|---|
| 环境 | **独立 venv `~/work/vllm-venv`**（与 HF 轨道的 `~/work/venv` 分开，互不污染） |
| vllm | **0.23.0**（预编译 wheel，无需 nvcc/源码编译） |
| torch | **2.11.0+cu128**（vLLM 自带匹配,非 HF 轨道的 2.12+cu130） |
| Compute capability | `(12, 0)` ✔ Blackwell 已识别（CUDA13 驱动向后兼容 cu128 二进制） |
| 实验模型 | **Qwen2.5-14B-Instruct** |

### Blackwell vLLM 安装坑（踩过，记下来免得重来）

1. **不能复用 HF 轨道的 venv**：那个 venv 里 torch 2.12+cu130 太新,没有任何发布版 vLLM 支持 → uv 一路回退到远古 vllm 0.1.3/0.2.5 试图源码编译 → 缺 `CUDA_HOME` 失败（此机无 nvcc）。
2. **repo 的 `pyproject.toml` 有 `[[tool.uv.index]] = cu130`**：在 repo 目录里 `uv pip install` 会把 `nvidia-cudnn/cublas` 锁死在 cu130 索引 → cu128 torch 链不可满足。
3. **解法**：`cd ~`(离开 repo 索引)+ 全新 venv + 下面这条:

```bash
uv venv ~/work/vllm-venv --python 3.12 --seed && source ~/work/vllm-venv/bin/activate
cd ~     # 关键:避开 repo pyproject 的 cu130 索引
uv pip install "vllm>=0.9" --torch-backend=cu128 --index-strategy unsafe-best-match
uv pip install openai      # loadgen 客户端依赖
python -c "import torch, vllm; print(vllm.__version__, torch.__version__, torch.cuda.get_device_capability(0))"
# 期望: 0.23.0 2.11.0+cu128 (12, 0)
```
> `>=0.9` 堵死回退到远古版;`unsafe-best-match` 让 nvidia-* 从 cu128 索引取 linux-x86_64 wheel。

### 待补（serve 后回填）
- [ ] vLLM 启动日志里的 attention backend（FLASH_ATTN / FLASHINFER / 默认）
- [ ] 单卡 14B 可承载的 max-model-len 与 `--gpu-memory-utilization` 触发抢占的阈值
- [ ] v1 engine 抢占路径：recompute 还是 swap（`/metrics` 的 `vllm:num_preemption*` 计数器名）
