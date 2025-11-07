#!/bin/bash
set -e

# Start vLLM server for Qwen3 14B model
# Usage: ./scripts/start_vllm.sh [model_name] [port]
# Reference: https://huggingface.co/OpenPipe/Qwen3-14B-Instruct

MODEL_NAME="${1:-OpenPipe/Qwen3-14B-Instruct}"
PORT="${2:-8000}"

echo "=========================================="
echo "Starting vLLM Server for Qwen3-14B"
echo "Model: $MODEL_NAME"
echo "Port: $PORT"
echo "=========================================="
echo ""

echo "Starting vLLM server..."
echo "Access the server at: http://localhost:$PORT/v1"
echo "To stop the server, press Ctrl+C"
echo ""

# Start vLLM server with basic settings
uv run vllm serve "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
