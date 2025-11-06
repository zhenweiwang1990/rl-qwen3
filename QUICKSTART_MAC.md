# 快速开始 - macOS (CPU 版本)

本指南适用于在 macOS 上使用 CPU 进行推理和训练。

## 重要说明

在 macOS 上，我们**不安装** GPU 加速库（如 `vllm`, `unsloth`），因为它们依赖于 CUDA 和 `bitsandbytes`，这些库在 macOS 上不可用。

macOS 版本适用于：
- 🧪 测试和开发
- 📊 数据处理和分析
- 🎯 使用外部 API 进行推理（如 OpenAI、vLLM 服务器）
- 🔬 小规模实验和原型验证

**注意**：大规模训练和高性能推理建议使用 Linux + GPU 环境。

## 安装步骤

### 1. 安装 uv（如果尚未安装）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 安装依赖（自动平台检测）

```bash
# 在 macOS 和 Linux 上都使用相同的命令
uv sync

# ✨ uv 会自动检测你的平台：
# - macOS: 跳过 GPU 库（vllm, unsloth, bitsandbytes）
# - Linux: 自动安装 GPU 库
```

**重要**：项目使用**平台标记**（platform markers），`uv.lock` 文件在 macOS 和 Linux 之间共享不会有冲突！

### 3. 设置环境变量

```bash
cp env.example .env
# 编辑 .env 文件，填入必要的 API keys
```

关键配置：
```bash
# 使用 CPU 设备（Apple Silicon 可以使用 MPS）
DEVICE=cpu  # 或 mps（用于 Apple Silicon）
TORCH_DEVICE=cpu  # 或 mps

# 使用外部推理服务
# 如果你有远程 GPU 服务器运行 vLLM
INFERENCE_BASE_URL=http://your-gpu-server:8000/v1
INFERENCE_API_KEY=your_api_key
```

## 使用方式

### 使用本地 CPU 推理

虽然速度较慢，但可用于测试：

```bash
# 使用小型模型
uv run python qwen3_agent/rollout.py
```

**警告**：在 CPU 上运行 14B 模型会非常慢。建议使用外部推理服务。

### 使用外部推理服务（推荐）

最佳实践是在 GPU 服务器上运行推理，Mac 上只运行训练逻辑：

```bash
# 在 GPU 服务器上启动 vLLM
# ssh to-gpu-server
# ./scripts/start_vllm.sh

# 在 Mac 上设置 .env
INFERENCE_BASE_URL=http://your-gpu-server:8000/v1
INFERENCE_MODEL_NAME=OpenPipe/Qwen3-14B-Instruct

# 运行训练（使用远程推理）
uv run python qwen3_agent/train.py
```

### 数据处理和基准测试

这些操作在 CPU 上运行良好：

```bash
# 生成数据集
./scripts/generate_database.sh

# 运行基准测试（使用远程推理）
uv run python qwen3_agent/benchmark.py
```

## PyTorch 设备设置

在 macOS 上，PyTorch 会自动选择可用的设备：

- **Apple Silicon (M1/M2/M3)**: 可以使用 MPS (Metal Performance Shaders) 加速
- **Intel Mac**: 使用 CPU

```python
import torch

# 自动检测
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 或在环境变量中设置
# DEVICE=mps
# TORCH_DEVICE=mps
```

## 局限性

在 macOS CPU 环境下：

❌ **不支持**：
- vLLM 推理（需要 CUDA）
- Unsloth 加速训练（需要 CUDA 和 bitsandbytes）
- 4-bit/8-bit 量化（需要 bitsandbytes）
- 大规模并行训练

✅ **支持**：
- 使用远程 API 的推理
- 数据处理和准备
- 小模型本地推理（速度较慢）
- 轨迹采样和处理
- 评估和基准测试
- 使用远程推理的 RL 训练

## 性能对比

| 操作 | macOS CPU | macOS MPS | Linux GPU |
|------|-----------|-----------|-----------|
| 14B 模型推理 | 🐌 很慢 | 🚶 较慢 | 🚀 快速 |
| 数据处理 | ✅ 正常 | ✅ 正常 | ✅ 正常 |
| 训练（使用远程推理） | ✅ 可用 | ✅ 可用 | ✅ 推荐 |
| 训练（本地推理） | ❌ 不推荐 | ⚠️ 勉强可用 | ✅ 推荐 |

## 故障排除

### 问题：`bitsandbytes` 安装失败

**解决方案**：这是预期的。我们已将 GPU 依赖移到可选依赖组。只运行 `uv sync` 即可。

### 问题：MPS 设备错误

如果在使用 MPS 时遇到问题：

```bash
# 改用 CPU
export DEVICE=cpu
export TORCH_DEVICE=cpu
```

### 问题：模型太大，内存不足

- 使用更小的模型
- 或使用远程推理服务
- 减少 batch size 和并发数

## 推荐工作流程

**最佳实践**：Mac 用于开发，GPU 服务器用于训练

1. 在 Mac 上开发和测试代码
2. 使用小数据集在本地验证
3. 在 GPU 服务器上运行实际训练
4. 使用 W&B 监控训练进度
5. 在 Mac 上分析结果

## 相关文档

- [Linux 快速开始](QUICKSTART.md) - GPU 训练
- [独立部署](QUICKSTART_STANDALONE.md) - 生产部署
- [W&B 集成](docs/WANDB.md) - 训练监控

## 需要帮助？

如果你遇到 macOS 特定的问题，请查看：
1. PyTorch MPS 支持文档
2. 项目的 Issue tracker
3. 环境变量配置（`env.example`）

