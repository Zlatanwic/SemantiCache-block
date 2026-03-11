# SemantiCache: Semantics-Aware KV Cache Eviction for LLM Inference

## 项目结构

```
semanticache/
├── README.md                 # 本文件
├── requirements.txt          # 依赖
├── config.py                 # 全局配置
├── kv_cache_manager.py       # KV Cache 管理器（核心）
├── eviction_policies.py      # 驱逐策略（基线 + SemantiCache）
├── semantic_analyzer.py      # 语义信号计算
├── attention_tracker.py      # Attention score 追踪
├── run_generation.py         # 带 KV Cache 驱逐的生成入口
├── eval_niah.py              # Needle-in-a-Haystack 评测
├── eval_multiturn.py         # 多轮对话回忆评测
└── visualize.py              # Attention heatmap 可视化
```

## 环境配置

```bash
# 使用 uv 管理依赖
uv init semanticache
cd semanticache

# 安装 PyTorch (CUDA 13.x, 使用 cu130 wheel)
uv add torch torchvision --extra-index-url https://download.pytorch.org/whl/cu130

# 安装其他依赖
uv add transformers autoawq accelerate numpy matplotlib seaborn tqdm

# 验证环境
uv run python run_generation.py --test
```

## 快速开始

```bash
# 第一步：验证模型加载和 KV Cache hook
python run_generation.py --test

# 第二步：对比不同驱逐策略
python run_generation.py --policy full       # 无驱逐（baseline upper bound）
python run_generation.py --policy window     # Local Window
python run_generation.py --policy streaming  # StreamingLLM
python run_generation.py --policy h2o        # H2O
python run_generation.py --policy semantic   # SemantiCache（你的方法）

# 第三步：跑 Needle-in-a-Haystack 评测
python eval_niah.py --budget 0.5  # 50% cache budget
```

## 硬件要求

- GPU: RTX 5060 Laptop (8GB VRAM), CUDA 13.2
- 模型: Qwen2.5-3B-Instruct-AWQ (~1.8GB VRAM)
- 剩余 ~6GB 用于 KV Cache
