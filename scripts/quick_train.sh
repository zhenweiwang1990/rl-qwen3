#!/bin/bash
set -e

echo "=========================================="
echo "Quick Training Test - Minimal Configuration"
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

# Override with minimal test configuration
export RUN_ID="${RUN_ID:-quick-test}"
export MODEL_NAME="${MODEL_NAME:-OpenPipe/Qwen3-14B-Instruct-Instruct}"
export VERBOSE="true"

# Minimal training config for quick iteration
export TRAJECTORIES_PER_GROUP=2      # Reduced from 6
export GROUPS_PER_STEP=2              # Reduced from 8
export LEARNING_RATE="1.2e-5"
export EVAL_STEPS=1                   # Evaluate every step (was 30)
export VAL_SET_SIZE=5                 # Tiny validation set (was 100)
export TRAINING_DATASET_SIZE=10       # Tiny training set (was 4000)
export NUM_EPOCHS=2                   # Keep 2 epochs

# Execution settings
export MAX_TURNS=5                    # Reduced from 10
export MAX_TOKENS=512                 # Reduced from 2048
export BENCH_SEQUENTIAL=true          # Sequential for stability
export ROLLOUT_CONCURRENCY=1          # Single concurrency
export LITELLM_TIMEOUT=60
export LITELLM_MAX_RETRIES=2

echo "Quick Test Configuration:"
echo "  - Run ID: $RUN_ID"
echo "  - Model: $MODEL_NAME"
echo "  - Training dataset: $TRAINING_DATASET_SIZE samples"
echo "  - Validation set: $VAL_SET_SIZE samples"
echo "  - Trajectories per group: $TRAJECTORIES_PER_GROUP"
echo "  - Groups per step: $GROUPS_PER_STEP"
echo "  - Eval every: $EVAL_STEPS steps"
echo "  - Epochs: $NUM_EPOCHS"
echo "  - Max turns: $MAX_TURNS"
echo "  - Max tokens: $MAX_TOKENS"
echo ""
echo "This should complete in ~5-10 minutes"
echo ""

# Run training
echo "Starting quick training test..."
echo ""

uv run python -m qwen3_agent.train

echo ""
echo "=========================================="
echo "Quick Training Test Complete!"
echo "=========================================="
echo ""
echo "Training loop executed successfully with:"
echo "  - Initial evaluation (step 0)"
echo "  - Multiple training steps with trajectory generation"
echo "  - Periodic evaluations"
echo "  - Final evaluation"
echo ""
echo "Next steps:"
echo "  1. Check that checkpoints were saved"
echo "  2. Run full benchmark: TEST_SET_SIZE=20 ./scripts/benchmark.sh"
echo "  3. Scale up for real training: ./scripts/train.sh"

