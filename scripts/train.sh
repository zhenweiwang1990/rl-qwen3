#!/bin/bash
set -e

echo "=========================================="
echo "Qwen3 Email Agent - Training"
echo "=========================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please run ./scripts/setup.sh first."
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

uv run python -m qwen3_agent.train

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="

