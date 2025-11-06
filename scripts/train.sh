#!/bin/bash
set -e

echo "=========================================="
echo "Qwen3 Email Agent - Training"
echo "=========================================="
echo ""

# Check if virtual environment exists (check both venv and .venv)
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found. Please run 'uv sync' first."
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Error: .env file not found. Please create one based on .env.example"
    exit 1
fi

# Load environment variables
set -a
source .env
set +a

# Set default values if not provided
export RUN_ID="${RUN_ID:-001}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-14B-Instruct}"
export VERBOSE="${VERBOSE:-false}"

echo "Configuration:"
echo "  - Run ID: $RUN_ID"
echo "  - Model: $MODEL_NAME"
echo "  - Verbose: $VERBOSE"
echo ""

# Run training
echo "Starting training..."
echo ""

python -m qwen3_agent.train

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="

