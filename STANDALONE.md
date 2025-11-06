# RL-Qwen3 独立版本说明

## 概述

`rl-qwen3` 现在是一个**完全独立的项目**，不再依赖 `openpipe-art` 库。所有必要的核心功能都已集成到项目中。

## 主要变更

### 1. 移除的依赖
- ✅ 已移除对 `openpipe-art` 的依赖
- ✅ 所有代码现在完全自包含

### 2. 新增的模块

#### `qwen3_agent/core/`
包含从 ART 库复制的核心类型：
- `types.py` - 基础类型定义（Message, Tools, TrainConfig 等）
- `trajectories.py` - Trajectory 和 TrajectoryGroup 类
- `model.py` - Model 和 TrainableModel 类
- `gather.py` - 轨迹收集和聚合功能

#### `qwen3_agent/utils/`
包含必要的工具函数：
- `limit_concurrency.py` - 并发控制装饰器
- `iterate_dataset.py` - 数据集迭代工具
- `litellm_utils.py` - LiteLLM 与 OpenAI 格式转换

### 3. 简化的训练逻辑

`qwen3_agent/train.py` 已经重写为独立版本：
- ✅ 不再依赖 `art.LocalAPI`
- ✅ 使用简化的训练循环
- ✅ 保留核心功能：数据加载、轨迹生成、评估

**注意**：当前版本专注于数据生成和评估。如需完整的强化学习训练（梯度更新），需要额外集成训练后端（如 vLLM + LoRA）。

## 安装

```bash
# 进入项目目录
cd examples/rl-qwen3

# 使用 uv 安装依赖（推荐）
uv sync

# 或使用 pip
pip install -e .
```

## 使用方法

### 1. 配置环境变量

创建 `.env` 文件（参考 `env.example`）：

```bash
# Model configuration
MODEL_NAME=Qwen/Qwen2.5-14B-Instruct
RUN_ID=001

# Training configuration
TRAJECTORIES_PER_GROUP=6
GROUPS_PER_STEP=8
LEARNING_RATE=1.2e-5
EVAL_STEPS=30
TRAINING_DATASET_SIZE=4000
VAL_SET_SIZE=100
NUM_EPOCHS=2

# Inference
INFERENCE_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=dummy

# Optional
VERBOSE=true
```

### 2. 运行训练

```bash
# 使用训练脚本
./scripts/train.sh

# 或直接运行
.venv/bin/python -m qwen3_agent.train
```

### 3. 运行评估

```bash
# 使用评估脚本
./scripts/quick_eval.sh

# 或直接运行
.venv/bin/python -m qwen3_agent.benchmark
```

## 项目结构

```
rl-qwen3/
├── qwen3_agent/
│   ├── core/              # 核心类型和功能（从 ART 移植）
│   │   ├── __init__.py
│   │   ├── types.py
│   │   ├── trajectories.py
│   │   ├── model.py
│   │   └── gather.py
│   ├── utils/             # 工具函数
│   │   ├── __init__.py
│   │   ├── limit_concurrency.py
│   │   ├── iterate_dataset.py
│   │   └── litellm_utils.py
│   ├── data/              # 数据加载
│   ├── config.py          # 配置类
│   ├── rollout.py         # 推理逻辑
│   ├── benchmark.py       # 评估逻辑
│   ├── train.py           # 训练逻辑（简化版）
│   └── tools.py           # 工具函数
├── scripts/               # 便捷脚本
├── pyproject.toml         # 项目配置（已更新）
└── README.md              # 项目文档
```

## 限制和未来工作

### 当前限制
1. **训练后端**：当前版本不包含完整的梯度更新逻辑。需要额外实现：
   - LoRA 微调集成
   - 模型检查点管理
   - Weights & Biases 日志记录

2. **推理后端**：需要单独启动推理服务器（如 vLLM）

### 扩展方向
1. 集成 vLLM 作为推理和训练后端
2. 添加完整的 RL 训练流程（PPO/GRPO）
3. 添加模型检查点保存和加载
4. 集成 W&B 进行实验跟踪

## 与原 ART 库的区别

| 功能 | ART 库 | RL-Qwen3 独立版 |
|------|--------|----------------|
| 核心类型 | ✅ | ✅ |
| 轨迹收集 | ✅ | ✅ |
| 数据迭代 | ✅ | ✅ |
| 评估功能 | ✅ | ✅ |
| LocalAPI | ✅ | ❌ (已简化) |
| 训练后端 | ✅ | ⚠️ (需额外实现) |
| W&B 集成 | ✅ | ❌ (可选添加) |
| S3 同步 | ✅ | ❌ |

## 贡献

欢迎贡献！特别是以下方面：
- 完整的训练后端集成
- 更多的评估指标
- 文档改进

## 许可证

参见 LICENSE 文件。

