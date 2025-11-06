# Changelog

All notable changes to the Qwen3 Email Agent project will be documented in this file.

## [0.1.0] - 2025-11-06

### Added

#### Core Features
- Initial project setup for Qwen3 14B email agent training
- Reinforcement learning training loop using ART framework
- Email search agent with three tools (search, read, answer)
- Comprehensive evaluation system with GPT-4o-based answer checking

#### Code Structure
- `qwen3_agent/config.py` - Configuration management with environment variable support
- `qwen3_agent/tools.py` - Email search and reading tools
- `qwen3_agent/rollout.py` - Agent rollout and evaluation logic
- `qwen3_agent/train.py` - Training loop implementation
- `qwen3_agent/benchmark.py` - Benchmarking utilities
- `qwen3_agent/data/` - Data loading modules for Enron email dataset

#### Scripts
- `scripts/setup.sh` - Environment setup and dependency installation
- `scripts/train.sh` - Training execution script
- `scripts/benchmark.sh` - Benchmark execution script
- `scripts/quick_eval.sh` - Quick single-scenario evaluation
- `scripts/generate_database.sh` - Email database generation
- `scripts/compare_models.sh` - Multi-model comparison
- `scripts/monitor_training.sh` - Training progress monitoring
- `docker-run.sh` - Docker convenience wrapper

#### Docker Support
- `Dockerfile` - CUDA-enabled image for GPU training
- `Dockerfile.cpu` - CPU-only image for development
- `docker-compose.yml` - Multi-service orchestration
- Support for NVIDIA GPU runtime

#### Documentation
- `README.md` - Comprehensive project documentation
- `QUICKSTART.md` - Quick start guide for new users
- `CHANGELOG.md` - This file
- Code comments and docstrings throughout

#### Configuration
- `pyproject.toml` - Project metadata and dependency specification
- `uv.lock` - Locked dependencies for reproducible builds
- `env.example` - Environment variable template
- `.gitignore` - Git ignore rules
- `.dockerignore` - Docker ignore rules

#### Hardware Support
- CUDA support for NVIDIA GPUs
- Apple Silicon (MPS) support for Mac training
- CPU fallback for development
- Auto-detection of best available device

### Features vs art-e

#### Improvements over art-e
1. **GPT-4o Evaluation** - Uses OpenAI GPT-4o instead of Gemini for answer correctness
2. **Cleaner Structure** - Better organized code with clear separation of concerns
3. **Enhanced Logging** - Verbose mode with detailed progress tracking
4. **Docker Support** - Full containerization with docker-compose
5. **Apple Silicon** - Explicit MPS backend support for Mac
6. **Environment Config** - All configuration via environment variables
7. **Better Scripts** - More comprehensive shell scripts for common tasks
8. **Documentation** - Enhanced documentation with quick start guide

#### Maintained from art-e
- Same Enron email dataset
- Same tool interface (search, read, answer)
- Compatible reward function
- Same training methodology
- Same evaluation metrics

### Technical Details

#### Model Configuration
- Base model: Qwen/Qwen2.5-14B-Instruct (configurable)
- Default max turns: 10
- Default max tokens: 2048
- Tool-based function calling

#### Training Configuration
- Default trajectories per group: 6
- Default groups per step: 8
- Default learning rate: 1.2e-5
- Default evaluation steps: 30
- Default epochs: 4

#### Evaluation Metrics
- Answer correctness (GPT-4o evaluated)
- Source correctness
- Turn efficiency
- Token usage
- Success/failure modes
- Duration tracking

### Dependencies

#### Core Dependencies
- openpipe-art - Training framework
- datasets - Hugging Face datasets
- litellm - Multi-provider LLM interface
- polars - Fast dataframe library
- transformers - Model loading
- langchain-core - Function calling utilities

#### Development Dependencies
- pytest - Testing framework
- python-dotenv - Environment management
- tqdm - Progress bars
- panza - Additional utilities

### Known Issues

None at initial release.

### Future Plans

- [ ] Support for additional base models (Qwen3, Llama 3, etc.)
- [ ] Multi-GPU training support
- [ ] Wandb integration for experiment tracking
- [ ] More sophisticated reward shaping
- [ ] Additional email tools (e.g., date filtering)
- [ ] Streaming inference support
- [ ] Model quantization support
- [ ] Curriculum learning support

### Contributors

- Initial implementation based on art-e project
- Docker and infrastructure improvements
- Enhanced logging and monitoring
- Documentation and quick start guide

---

## Version History

- **0.1.0** - Initial release (2025-11-06)

