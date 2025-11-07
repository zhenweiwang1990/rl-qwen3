#!/bin/bash
# Quick training script for SB3-based training (small scale for testing)

set -e

echo "================================================"
echo "  Quick SB3 Training (Test Mode)"
echo "================================================"
echo ""

# Quick training configuration (small scale)
export MODEL_NAME="OpenPipe/Qwen3-14B-Instruct"
export OUTPUT_DIR="./output_sb3_test"
export NUM_EPISODES="100"  # Small number for testing
export BATCH_SIZE="2"
export MAX_TURNS="5"
export LEARNING_RATE="1e-5"
export EVAL_FREQUENCY="20"
export SAVE_FREQUENCY="50"
export TRAINING_DATASET_SIZE="100"  # Small dataset
export VAL_SET_SIZE="20"
export USE_LORA="true"
export USE_WANDB="false"  # Disable wandb for quick testing
export VERBOSE="true"

echo "Quick training configuration:"
echo "  - Small model: ${MODEL_NAME}"
echo "  - Episodes: ${NUM_EPISODES}"
echo "  - Dataset size: ${TRAINING_DATASET_SIZE}"
echo "  - LoRA: enabled"
echo "  - Wandb: disabled"
echo ""

# Run training
uv run python -m qwen3_agent.train_sb3

echo ""
echo "Quick training complete!"
echo "Check results in: ${OUTPUT_DIR}"

