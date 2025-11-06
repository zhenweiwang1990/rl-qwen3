# Qwen3 Email Agent

A complete reinforcement learning project for training **OpenPipe/Qwen3-14B-Instruct** as an email search agent using the **OpenPipe ART** library. The agent learns to search through email databases using three tools (search, read, and answer) to answer user queries about their emails.

## Features

- 🔥 **Full RL Training** with gradient updates using OpenPipe ART (GRPO/PPO)
- 💾 **Checkpoint Management** - automatic save/load with S3 backup
- 📊 **Comprehensive Evaluation** with detailed metrics and benchmarking
- 📈 **WandB Integration** for training monitoring and visualization
- 🐳 **Docker Support** for easy deployment on CUDA-enabled Linux servers
- 🍎 **Apple Silicon Support** for training on Mac (MPS backend)
- 📝 **Detailed Logging** with verbose mode for debugging
- 🔧 **Flexible Configuration** via environment variables
- ☁️ **S3 Integration** for model checkpoint backup and recovery

## Project Structure

```
rl-qwen3/
├── qwen3_agent/           # Main package
│   ├── config.py          # Configuration classes
│   ├── tools.py           # Email search tools
│   ├── rollout.py         # Agent rollout logic
│   ├── train.py           # Training script
│   ├── benchmark.py       # Evaluation utilities
│   └── data/              # Data loading modules
├── scripts/               # Utility scripts
│   ├── setup.sh           # Environment setup
│   ├── train.sh           # Training script
│   ├── benchmark.sh       # Benchmark script
│   └── quick_eval.sh      # Quick single scenario test
├── Dockerfile             # CUDA Docker image
├── Dockerfile.cpu         # CPU-only Docker image
├── docker-compose.yml     # Docker Compose configuration
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project metadata
└── README.md              # This file
```

## Requirements

- **Python 3.10, 3.11, or 3.12** (Python 3.13+ is not supported due to Pydantic v1 dependencies)
- **NVIDIA GPU with CUDA** (recommended for training) or Apple Silicon with MPS
- **OpenAI API key** for GPT-4o evaluation
- **WandB account** (optional but recommended) for training monitoring
- **AWS S3 bucket** (optional) for checkpoint backup
- (Optional) Docker for containerized deployment

⚠️ **Important**: If you're using Python 3.13+, please use Docker or install Python 3.12.

## Quick Start

### 1. Local Setup (macOS or Linux)

```bash
# Clone the repository (if not already done)
cd examples/rl-qwen3

# Run setup script (installs uv and syncs dependencies)
./scripts/setup.sh

# Create .env file from template
cat > .env << EOF
# OpenAI API Key (for evaluation with GPT-4o)
OPENAI_API_KEY=your_openai_api_key_here

# AWS S3 Configuration (optional, for checkpoint backup)
BACKUP_BUCKET=your_s3_bucket_name
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key

# Training Configuration
RUN_ID=001
VERBOSE=false
MODEL_NAME=Qwen/Qwen2.5-14B-Instruct
EOF

# Edit the .env file with your actual keys
vim .env

# Generate email database
./scripts/generate_database.sh
```

**Note**: This project now uses [uv](https://github.com/astral-sh/uv) for dependency management, which is significantly faster than pip. The setup script will automatically install uv if it's not already present.

## Usage

### Full RL Training (with Gradient Updates)

This project now includes **complete reinforcement learning training** with real gradient updates, checkpoint management, and S3 backup using the OpenPipe ART library.

#### Prerequisites

1. **Install dependencies**:
```bash
./scripts/setup.sh  # Installs uv and syncs all dependencies
```

2. **Configure environment**:
```bash
cp env.example .env
vim .env
```

Required settings in `.env`:
- `OPENAI_API_KEY` - for GPT-4o evaluation
- `WANDB_API_KEY` - for training monitoring (get from https://wandb.ai)
- `BACKUP_BUCKET` - S3 bucket for checkpoints (optional but recommended)
- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` - AWS credentials

3. **Generate email database**:
```bash
./scripts/generate_database.sh
```

#### Run Training

```bash
# Run full RL training with default config
./scripts/train_with_rl.sh

# Or with custom RUN_ID
./scripts/train_with_rl.sh 002

# Or run directly with custom parameters (using uv)
RUN_ID=002 VERBOSE=true NUM_EPOCHS=2 uv run python -m qwen3_agent.train
```

#### What Happens During Training

1. **Initialization**: 
   - ART LocalAPI initializes training backend (vLLM + LoRA)
   - Restores checkpoints from S3 if available
   - Loads training/validation datasets

2. **Training Loop** (for each step):
   - Generate trajectory groups via rollout
   - **Calculate advantages and rewards**
   - **Perform gradient updates using GRPO/PPO**
   - **Update model weights**
   - Log metrics to WandB

3. **Evaluation** (every N steps):
   - Run benchmark on validation set
   - Log evaluation metrics
   - Save checkpoint
   - Clean up old checkpoints (keep best + latest)
   - Push to S3

4. **Completion**:
   - Final evaluation
   - Save final checkpoint
   - Push to S3

### Benchmarking

#### Local Benchmark

```bash
# Run benchmark on test set
./scripts/benchmark.sh

# Or with custom parameters
RUN_ID=001 TEST_SET_SIZE=50 VERBOSE=true ./scripts/benchmark.sh
```

#### Docker Benchmark

```bash
docker-compose up qwen3-benchmark
```

### Quick Evaluation

Test a single scenario with detailed logging:

```bash
# Test with GPT-4o
./scripts/quick_eval.sh gpt-5-mini

# Test with your trained model
./scripts/quick_eval.sh qwen3-email-agent-001

# Show full trajectory details (YAML)
SHOW_TRAJECTORY_DETAILS=true ./scripts/quick_eval.sh gpt-4o
```

The quick evaluation script will:
- Display the test scenario details
- Show each turn's tool calls and responses  
- Display detailed judge evaluation process (GPT-4o)
- Present a comprehensive summary table with:
  - Key metrics (answer correctness, reward, token usage)
  - Agent behavior details
  - Error tracking
  - Full metrics in a grid format

## 🤖 About OpenPipe/Qwen3-14B-Instruct

This project uses [OpenPipe/Qwen3-14B-Instruct](https://huggingface.co/OpenPipe/Qwen3-14B-Instruct), which is specifically optimized for agent training workflows:

- **Optimized for ART**: Fine-tuned to work seamlessly with the ART training framework
- **Superior Tool Calling**: Enhanced function calling capabilities for multi-tool workflows  
- **Strong Reasoning**: Excellent performance on multi-step reasoning tasks
- **Efficient Training**: Optimized for LoRA fine-tuning with GRPO
- **Community Proven**: Successfully used in production agent systems

According to the [OpenPipe ART examples](https://github.com/OpenPipe/ART), Qwen3 models have demonstrated excellent results in various agent tasks including:
- 📧 Email search and retrieval (this project)
- 🎮 Game playing (2048, Tic Tac Toe, Codenames)
- 🔧 Tool use with MCP servers
- 🧠 Multi-step reasoning tasks

## Configuration

### Environment Variables

All configuration is done via environment variables in the `.env` file:

#### API Keys
- `OPENAI_API_KEY`: OpenAI API key for GPT-4o evaluation (required)
- `OPENPIPE_API_KEY`: OpenPipe API key for logging (optional)
- `AWS_ACCESS_KEY_ID`: AWS credentials for S3 backup (optional)
- `AWS_SECRET_ACCESS_KEY`: AWS credentials for S3 backup (optional)
- `BACKUP_BUCKET`: S3 bucket name for checkpoints (optional)

#### Model Configuration
- `MODEL_NAME`: Base model to use (default: `Qwen/Qwen2.5-14B-Instruct`)
- `RUN_ID`: Unique identifier for training run (default: `001`)
- `MAX_TURNS`: Maximum turns for agent (default: `10`)
- `MAX_TOKENS`: Maximum tokens per response (default: `2048`)

#### Training Parameters
- `TRAJECTORIES_PER_GROUP`: Trajectories per group (default: `6`)
- `GROUPS_PER_STEP`: Groups per training step (default: `8`)
- `LEARNING_RATE`: Learning rate (default: `1.2e-5`)
- `EVAL_STEPS`: Evaluation frequency (default: `30`)
- `VAL_SET_SIZE`: Validation set size (default: `100`)
- `TRAINING_DATASET_SIZE`: Training dataset size (default: `4000`)
- `NUM_EPOCHS`: Number of epochs (default: `4`)

#### System Configuration
- `VERBOSE`: Enable detailed logging (default: `false`)
- `DEVICE`: Device to use (`cuda`, `mps`, or `cpu` - auto-detected if not set)

### Training Configurations

Create different training runs by adjusting parameters:

```bash
# Fast training for testing
RUN_ID=quick GROUPS_PER_STEP=2 NUM_EPOCHS=1 ./scripts/train.sh

# High-quality training
RUN_ID=hq GROUPS_PER_STEP=16 NUM_EPOCHS=4 LEARNING_RATE=8e-6 ./scripts/train.sh

# Long context training
RUN_ID=long MAX_TURNS=30 ./scripts/train.sh
```

## Hardware Support

### CUDA (Linux)

The project fully supports NVIDIA GPUs via CUDA:

```bash
# Check CUDA availability
docker run --rm --runtime=nvidia qwen3-train-gpu python -c "import torch; print(torch.cuda.is_available())"
```

### Apple Silicon (macOS)

Training on Apple Silicon Macs uses the MPS backend:

```bash
# The device is automatically detected as 'mps'
VERBOSE=true ./scripts/train.sh
```

### CPU

CPU training is supported but significantly slower:

```bash
DEVICE=cpu ./scripts/train.sh
```

## Docker Commands

### Build Images

```bash
# Build GPU image
docker build -t qwen3-agent:gpu -f Dockerfile .

# Build CPU image
docker build -t qwen3-agent:cpu -f Dockerfile.cpu .
```

### Run Containers

```bash
# Interactive GPU training
docker-compose run --rm qwen3-train-gpu /bin/bash

# Run benchmark
docker-compose run --rm qwen3-benchmark

# Run with custom command
docker-compose run --rm qwen3-train-gpu python -m qwen3_agent.train
```

### Container Management

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f qwen3-train-gpu

# Remove volumes
docker-compose down -v
```

## Dataset

This project uses the Enron email dataset with synthetic questions:
- **Training set**: ~7,000 synthetic questions
- **Test set**: ~1,500 synthetic questions
- **Source**: `corbt/enron_emails_sample_questions` on Hugging Face

The dataset is automatically downloaded on first run.

## Evaluation Metrics

The agent is evaluated on multiple metrics:

- `answer_correct`: Whether the answer is semantically correct (evaluated by GPT-4o)
- `sources_correct`: Whether the correct source email was cited
- `num_turns`: Number of turns taken to answer
- `reward`: Final reward (-2 to 2)
- `duration`: Time taken for rollout
- `prompt_tokens`: Number of prompt tokens used
- `completion_tokens`: Number of completion tokens used

Additional diagnostic metrics:
- `ever_found_right_email`: Whether search found the relevant email
- `ever_read_right_email`: Whether agent read the relevant email
- `returned_i_dont_know`: Whether agent gave up
- `bad_tool_call_*`: Various tool calling errors

## Reward Function

The reward function encourages correct answers with proper citations:

- **Correct answer with sources**: 1.0 to 2.0
  - +0.3 for correct sources
  - +0.1 for efficiency (fewer sources)
  - +0.1 for speed (fewer turns)
  
- **No answer / "I don't know"**: 0.0 to 1.0 (with partial credit)

- **Wrong answer**: -1.0 to 0.0 (with partial credit)

- **Tool calling errors**: -2.0 to -1.0

Partial credit is given for:
- Finding the right email (+0.1)
- Reading the right email (+0.1)
- Not attempting invalid emails (+0.1)
- Correct sources even if answer is wrong (+0.1)

## Understanding the Training Process

### Model Checkpoints

Checkpoints are saved at:
- **Local**: `.art/qwen3_email_agent/models/qwen3-email-agent-{RUN_ID}/`
- **S3**: `s3://{BACKUP_BUCKET}/qwen3_email_agent/models/qwen3-email-agent-{RUN_ID}/`

Each checkpoint includes:
- LoRA adapter weights
- Training state (optimizer, step count)
- Model configuration
- Training history

### Monitoring Training

#### WandB Dashboard
View real-time metrics at https://wandb.ai:
- `train/avg_reward` - Average reward per training batch
- `train/policy_loss` - Policy gradient loss
- `train/kl_div` - KL divergence from reference model
- `val/reward` - Validation reward
- `val/answer_correct` - Validation accuracy

#### Local Logs
```bash
# Watch training progress
tail -f training.log

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Configuration Deep Dive

Key training hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRAJECTORIES_PER_GROUP` | 6 | Samples per scenario (for advantage calculation) |
| `GROUPS_PER_STEP` | 8 | Scenarios per training step |
| `LEARNING_RATE` | 1.2e-5 | Gradient descent learning rate |
| `EVAL_STEPS` | 30 | Evaluation frequency |
| `NUM_EPOCHS` | 4 | Complete passes through dataset |

**Batch size per step** = `TRAJECTORIES_PER_GROUP` × `GROUPS_PER_STEP` = 48 trajectories

## Key Differences from art-e

This project is a **standalone version** of the `art-e` project with several improvements:

1. **Independent Package**: Works as a standalone project, imports ART as a dependency
2. **GPT-4o Evaluation**: Uses OpenAI's GPT-4o for answer correctness evaluation
3. **Cleaner Structure**: More organized code structure with better separation of concerns
4. **Better Logging**: Enhanced verbose logging for debugging
5. **Docker Support**: Full Docker and docker-compose support for easy deployment
6. **Apple Silicon**: Explicit MPS backend support for Mac training
7. **Environment Config**: All configuration via environment variables
8. **Better Scripts**: Improved shell scripts for common operations

## Troubleshooting

### Database Not Found

```bash
# The database is automatically generated on first training run
# Or manually generate:
./scripts/generate_database.sh
```

### CUDA Out of Memory

Reduce batch size or model parameters:

```bash
GROUPS_PER_STEP=4 TRAJECTORIES_PER_GROUP=4 ./scripts/train.sh
```

### Docker GPU Not Working

Ensure NVIDIA Docker runtime is installed:

```bash
# Install nvidia-docker2
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### MPS Not Available on Mac

Ensure you have PyTorch with MPS support:

```bash
pip install --upgrade torch torchvision torchaudio
python -c "import torch; print(torch.backends.mps.is_available())"
```

## Development

### Running Tests

```bash
# pytest is already included in dependencies
# Run tests with uv
uv run pytest tests/
```

### Adding New Features

1. Update the relevant module in `qwen3_agent/`
2. Update tests if needed
3. Update documentation
4. Test locally and in Docker

### Code Style

We follow PEP 8 style guidelines. Format code with:

```bash
uv run black qwen3_agent/
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project inherits the license from the parent ART project.

## Acknowledgments

- Based on the [ART (Agent Rollout Training)](https://github.com/openpipe/art) framework
- Uses the Enron email dataset
- Evaluation powered by OpenAI GPT-4o
- Inspired by the `art-e` project

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review the troubleshooting section above

## Citation

If you use this project in your research, please cite:

```bibtex
@software{qwen3_email_agent,
  title={Qwen3 Email Agent: Reinforcement Learning for Email Search},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/ART2}
}
```

