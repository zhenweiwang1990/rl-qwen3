# Quick Start Guide

This guide will help you get started with training and evaluating the Qwen3 Email Agent in minutes.

## Prerequisites

- **Python 3.10, 3.11, or 3.12** (Python 3.13+ is not supported)
- (Optional) NVIDIA GPU with CUDA for faster training
- (Optional) Docker for containerized deployment  
- OpenAI API key for evaluation

⚠️ **Python Version**: If you have Python 3.13+, you'll need to install Python 3.12 or use the Docker container instead.

## 5-Minute Setup

### 1. Initial Setup

```bash
# Navigate to project directory
cd examples/rl-qwen3

# Run setup script
./scripts/setup.sh

# This will:
# - Install uv (if not already installed)
# - Sync all dependencies using uv
# - Check for required files
```

**Note**: This project uses [uv](https://github.com/astral-sh/uv) for dependency management, which is significantly faster than pip.

### 2. Configure Environment

```bash
# Copy environment template
cp env.example .env

# Edit with your API keys
nano .env  # or vim .env, or use any text editor
```

**Minimum required configuration:**
```bash
# In .env file, set at minimum:
OPENAI_API_KEY=sk-your-actual-openai-key-here
```

### 3. Test Installation

```bash
# Run import test with uv
uv run python test_import.py

# Expected output: "✓ All tests passed!"
```

## Quick Test Run

Test the system with a single scenario:

```bash
# Test with GPT-4o (baseline)
./scripts/quick_eval.sh gpt-4o

# Test with local Qwen3 (requires vLLM server running)
./scripts/quick_eval.sh Qwen/Qwen3-14B

# Show full trajectory details (YAML)
SHOW_TRAJECTORY_DETAILS=true ./scripts/quick_eval.sh

# You should see:
# - The test scenario details
# - Each turn's tool calls and responses (verbose)
# - Detailed judge evaluation process (GPT-4o)
# - Comprehensive summary tables:
#   * Key metrics (answer correctness, reward, tokens)
#   * Agent behavior details
#   * Error tracking
#   * Full metrics grid
```

## Start Training

### Option A: Local Training (Recommended for testing)

```bash
# Quick training run (2 epochs, small batch)
RUN_ID=test GROUPS_PER_STEP=2 NUM_EPOCHS=2 VERBOSE=true ./scripts/train.sh

# Or use uv directly
RUN_ID=test GROUPS_PER_STEP=2 NUM_EPOCHS=2 VERBOSE=true uv run python -m qwen3_agent.train

# This will:
# - Download the email database (first time only)
# - Train the model for 2 epochs
# - Save checkpoints
# - Run evaluation
```

### Option B: Docker Training (Recommended for production)

```bash
# Build and start container
docker-compose up -d qwen3-train-gpu

# Enter container
docker exec -it qwen3-email-agent-train-gpu /bin/bash

# Inside container, run training
./scripts/train.sh

# Monitor from outside
./scripts/monitor_training.sh
```

## Run Evaluation

### Quick Evaluation with Base Model

Before training, you can test the base Qwen3 14B model on a single scenario:

**Step 1: Install vLLM**

```bash
pip install 'vllm>=0.8.5'
```

**Step 2: Start vLLM Server (in a separate terminal)**

```bash
./scripts/start_vllm.sh
# This will start serving Qwen/Qwen3-14B on http://localhost:8000
# Reference: https://huggingface.co/Qwen/Qwen3-14B
```

**Step 3: Run Quick Evaluation**

```bash
# In another terminal, run quick evaluation
./scripts/quick_eval.sh

# This will:
# - Use Qwen3 14B (via vLLM) as the agent to search emails
# - Use GPT-4o as judge to evaluate if the answer is correct
# - Show detailed output of the agent's reasoning
```

### Full Benchmark

After training (or to evaluate a pretrained model):

```bash
# Make sure vLLM server is running first
# Then benchmark on test set
RUN_ID=test TEST_SET_SIZE=50 ./scripts/benchmark.sh

# Results will be saved to benchmark_results_test.csv
```

## Common Commands

### Training Commands

```bash
# Full training run
./scripts/train.sh

# Custom configuration
RUN_ID=experiment1 NUM_EPOCHS=4 LEARNING_RATE=8e-6 ./scripts/train.sh

# Verbose mode for debugging
VERBOSE=true ./scripts/train.sh
```

### Evaluation Commands

```bash
# Quick evaluation (10 scenarios)
TEST_SET_SIZE=10 ./scripts/benchmark.sh

# Full evaluation (100 scenarios)
TEST_SET_SIZE=100 ./scripts/benchmark.sh

# Single scenario test
./scripts/quick_eval.sh qwen3-email-agent-test
```

### Docker Commands

```bash
# Start GPU training
docker-compose up -d qwen3-train-gpu

# View logs
docker-compose logs -f qwen3-train-gpu

# Stop training
docker-compose down

# Clean up everything
docker-compose down -v
```

## Understanding the Output

### Training Progress

```
=== Epoch 1/2, Step 0, Global Step 0 ===
Generating 2 trajectory groups...
Training on 2 trajectory groups...
Completed step 0

--- Evaluating at Global Step 0 ---
validation qwen3-email-agent-test: 100%|████| 100/100
```

### Evaluation Metrics

Key metrics to watch:
- **answer_correct**: % of correct answers (most important)
- **reward**: Average reward (-2 to 2)
- **num_turns**: Average turns taken
- **sources_correct**: % of correct source citations

### Example Results

```
┌────────────────────┬────────┐
│ Metric             │ Value  │
├────────────────────┼────────┤
│ answer_correct     │ 0.75   │
│ sources_correct    │ 0.68   │
│ reward             │ 1.23   │
│ num_turns          │ 3.5    │
│ duration           │ 12.3   │
└────────────────────┴────────┘
```

## Troubleshooting

### "Email database not found"

The database will be automatically generated on first run. To manually generate:

```bash
./scripts/generate_database.sh
```

### "CUDA out of memory"

Reduce batch size:

```bash
GROUPS_PER_STEP=2 TRAJECTORIES_PER_GROUP=4 ./scripts/train.sh
```

### "OpenAI API key not found"

Make sure you've set it in `.env`:

```bash
echo "OPENAI_API_KEY=sk-your-key" >> .env
```

### Python version incompatibility

If you see errors like `no validator found for <class 'pydantic.v1.fields.UndefinedType'>`, you're likely using Python 3.13+:

**Option 1: Use Docker (recommended)**
```bash
docker-compose up qwen3-train-gpu  # Uses Python 3.10 in container
```

**Option 2: Install Python 3.12**
```bash
# macOS with Homebrew
brew install python@3.12

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Use uv to create environment and install dependencies
uv sync
```

### Import errors

Reinstall dependencies:

```bash
# With uv (recommended)
uv sync --refresh

# This will update all dependencies and reinstall the package
```

## Next Steps

1. **Experiment with configurations**: Try different hyperparameters
2. **Compare models**: Use `./scripts/compare_models.sh`
3. **Analyze results**: Check the generated CSV files
4. **Read full documentation**: See `README.md` for detailed information

## Getting Help

- Check `README.md` for detailed documentation
- Review troubleshooting section
- Check logs in `logs/` directory
- Run with `VERBOSE=true` for detailed output

## Example Workflow

Here's a complete example workflow:

```bash
# 1. Setup
cd examples/rl-qwen3
./scripts/setup.sh
cp env.example .env
# Edit .env with your keys

# 2. Test installation
source venv/bin/activate
python test_import.py

# 3. Quick test
./scripts/quick_eval.sh gpt-4o

# 4. Small training run
RUN_ID=test1 NUM_EPOCHS=2 VERBOSE=true ./scripts/train.sh

# 5. Evaluate
RUN_ID=test1 TEST_SET_SIZE=50 ./scripts/benchmark.sh

# 6. Compare with baseline
./scripts/compare_models.sh

# 7. Full training (if test looks good)
RUN_ID=full NUM_EPOCHS=4 ./scripts/train.sh
```

That's it! You're now ready to train and evaluate your Qwen3 Email Agent. 🚀

