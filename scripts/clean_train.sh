#!/bin/bash
# 在完全干净的环境中启动训练

set -e

echo "=========================================="
echo "清洁环境训练启动器"
echo "=========================================="
echo ""

cd /workspace/rl-qwen3

echo "1. 加载环境配置..."
# 加载.env文件
if [ -f ".env" ]; then
    set -a  # 自动导出变量
    source .env
    set +a
    echo "✓ .env 已加载"
else
    echo "⚠️  .env 文件不存在"
fi
echo ""

echo "2. 检查GPU..."
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
echo ""

echo "3. 使用干净的环境启动Python..."
echo ""

# 检查虚拟环境
if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN="$(pwd)/.venv/bin/python"
    echo "使用虚拟环境: $PYTHON_BIN"
else
    PYTHON_BIN="python3"
    echo "使用系统Python: $PYTHON_BIN"
fi
echo ""

# 保存必要的环境变量
SAVED_WANDB_API_KEY="${WANDB_API_KEY}"
SAVED_WANDB_PROJECT="${WANDB_PROJECT:-qwen3_email_agent}"
SAVED_WANDB_MODE="${WANDB_MODE:-online}"
SAVED_HF_TOKEN="${HF_TOKEN}"
SAVED_HOME="${HOME}"

echo "环境变量:"
echo "  WANDB_API_KEY: ${SAVED_WANDB_API_KEY:0:10}..."
echo "  WANDB_PROJECT: $SAVED_WANDB_PROJECT"
echo "  WANDB_MODE: $SAVED_WANDB_MODE"
echo ""

# 使用env -i创建完全干净的环境，只保留必要的变量
env -i \
    HOME="$SAVED_HOME" \
    PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    PYTHONUNBUFFERED=1 \
    WANDB_API_KEY="$SAVED_WANDB_API_KEY" \
    WANDB_PROJECT="$SAVED_WANDB_PROJECT" \
    WANDB_MODE="$SAVED_WANDB_MODE" \
    HF_TOKEN="$SAVED_HF_TOKEN" \
    "$PYTHON_BIN" -u qwen3_agent/train.py "$@"

