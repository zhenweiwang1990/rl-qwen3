#!/bin/bash
# Training script for SB3-based local training on Mac

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Qwen3 Email Agent - SB3 Training (Mac)${NC}"
echo -e "${BLUE}================================================${NC}"

# Default configuration
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
export OUTPUT_DIR="${OUTPUT_DIR:-./output_sb3}"
export NUM_EPISODES="${NUM_EPISODES:-1000}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export MAX_TURNS="${MAX_TURNS:-10}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export EVAL_FREQUENCY="${EVAL_FREQUENCY:-50}"
export SAVE_FREQUENCY="${SAVE_FREQUENCY:-100}"
export TRAINING_DATASET_SIZE="${TRAINING_DATASET_SIZE:-1000}"
export VAL_SET_SIZE="${VAL_SET_SIZE:-100}"
export USE_LORA="${USE_LORA:-true}"
export USE_WANDB="${USE_WANDB:-true}"
export VERBOSE="${VERBOSE:-false}"

# Device configuration (auto-detected in code)
export DEVICE="${DEVICE:-auto}"

echo -e "\n${GREEN}Configuration:${NC}"
echo "  Model: ${MODEL_NAME}"
echo "  Output Directory: ${OUTPUT_DIR}"
echo "  Number of Episodes: ${NUM_EPISODES}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Max Turns: ${MAX_TURNS}"
echo "  Learning Rate: ${LEARNING_RATE}"
echo "  Use LoRA: ${USE_LORA}"
echo "  Device: ${DEVICE}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Check if database exists
if [ ! -f "enron_emails.db" ]; then
    echo -e "${BLUE}Generating email database...${NC}"
    python -c "from qwen3_agent.data.local_email_db import generate_database; generate_database()"
fi

# Start training
echo -e "\n${GREEN}Starting training...${NC}"
echo ""

uv run python -m qwen3_agent.train_sb3

echo -e "\n${GREEN}Training complete!${NC}"
echo -e "Model saved to: ${OUTPUT_DIR}"
echo ""

