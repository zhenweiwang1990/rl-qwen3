#!/bin/bash

# ============================================================================
# Qwen3 Email Agent - Full RL Training Script
# ============================================================================
# This script runs complete reinforcement learning training with gradient
# updates, checkpoint management, and S3 backup.
#
# Usage:
#   ./scripts/train_with_rl.sh [RUN_ID]
#
# Example:
#   ./scripts/train_with_rl.sh 001
#
# Prerequisites:
#   1. Install dependencies: pip install -r requirements.txt
#   2. Configure .env file with API keys and S3 settings
#   3. Generate email database: ./scripts/generate_database.sh
# ============================================================================

set -e  # Exit on error

# Change to script directory
cd "$(dirname "$0")/.."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Qwen3 Email Agent - Full RL Training${NC}"
echo -e "${BLUE}================================================${NC}"

# Load environment variables
if [ -f .env ]; then
    echo -e "${GREEN}✓ Loading .env file${NC}"
    export $(cat .env | grep -v '^#' | xargs)
else
    echo -e "${RED}✗ .env file not found${NC}"
    echo "Please create a .env file from env.example:"
    echo "  cp env.example .env"
    echo "  vim .env"
    exit 1
fi

# Set RUN_ID from argument or default
if [ -n "$1" ]; then
    export RUN_ID=$1
fi

echo -e "\n${BLUE}Configuration:${NC}"
echo "  RUN_ID: ${RUN_ID:-001}"
echo "  MODEL: ${MODEL_NAME:-OpenPipe/Qwen3-14B-Instruct}"
echo "  EPOCHS: ${NUM_EPOCHS:-4}"
echo "  GROUPS_PER_STEP: ${GROUPS_PER_STEP:-8}"
echo "  TRAJECTORIES_PER_GROUP: ${TRAJECTORIES_PER_GROUP:-6}"
echo "  LEARNING_RATE: ${LEARNING_RATE:-1.2e-5}"
echo "  EVAL_STEPS: ${EVAL_STEPS:-30}"
echo "  BACKUP_BUCKET: ${BACKUP_BUCKET:-not configured}"

# Check required environment variables
REQUIRED_VARS=("OPENAI_API_KEY")
MISSING_VARS=()

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=("$var")
    fi
done

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    echo -e "\n${RED}✗ Missing required environment variables:${NC}"
    for var in "${MISSING_VARS[@]}"; do
        echo "  - $var"
    done
    echo ""
    echo "Please set these in your .env file."
    exit 1
fi

# Check S3 configuration
if [ -z "$BACKUP_BUCKET" ]; then
    echo -e "\n${YELLOW}⚠ Warning: BACKUP_BUCKET not configured${NC}"
    echo "  Checkpoints will not be backed up to S3."
    echo "  To enable S3 backup, set BACKUP_BUCKET, AWS_ACCESS_KEY_ID,"
    echo "  and AWS_SECRET_ACCESS_KEY in your .env file."
    read -p "Continue without S3 backup? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check WandB configuration
if [ -z "$WANDB_API_KEY" ]; then
    echo -e "\n${YELLOW}⚠ Warning: WANDB_API_KEY not configured${NC}"
    echo "  Training metrics will not be logged to Weights & Biases."
    echo "  To enable WandB logging, set WANDB_API_KEY in your .env file"
    echo "  or run: wandb login"
fi

# Check if database exists
if [ ! -f "qwen3_agent/data/enron_emails.db" ]; then
    echo -e "\n${YELLOW}⚠ Email database not found${NC}"
    echo "Generating database..."
    ./scripts/generate_database.sh
fi

echo -e "\n${BLUE}Starting training...${NC}"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo -e "${GREEN}✓ Activating virtual environment${NC}"
    source venv/bin/activate
fi

# Run training with proper error handling
echo -e "\n${BLUE}================================================${NC}"
echo -e "${BLUE}  Starting RL Training Loop${NC}"
echo -e "${BLUE}================================================${NC}\n"

python -m qwen3_agent.train

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n${GREEN}================================================${NC}"
    echo -e "${GREEN}  ✓ Training completed successfully!${NC}"
    echo -e "${GREEN}================================================${NC}"
    
    # Print checkpoint location
    echo -e "\nModel checkpoints saved to:"
    echo "  .art/qwen3_email_agent/models/qwen3-email-agent-${RUN_ID}"
    
    if [ -n "$BACKUP_BUCKET" ]; then
        echo -e "\nBackup location:"
        echo "  s3://${BACKUP_BUCKET}/qwen3_email_agent/models/qwen3-email-agent-${RUN_ID}"
    fi
    
    # Suggest next steps
    echo -e "\n${BLUE}Next steps:${NC}"
    echo "  1. View training metrics: https://wandb.ai"
    echo "  2. Run benchmark: ./scripts/benchmark.sh"
    echo "  3. Compare models: ./scripts/compare_models.sh"
else
    echo -e "\n${RED}================================================${NC}"
    echo -e "${RED}  ✗ Training failed with exit code $EXIT_CODE${NC}"
    echo -e "${RED}================================================${NC}"
    exit $EXIT_CODE
fi

