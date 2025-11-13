# LLM Provider Support

The People Search Agent benchmark now supports multiple LLM providers, making it easy to compare models from different sources.

## Supported Providers

### 1. Ollama (Local Inference)
Run models locally on your own hardware.

**Pros:**
- Free, no API costs
- Full control and privacy
- No internet required after model download

**Cons:**
- Requires GPU for good performance
- Limited to locally available models
- Need to manage model downloads

### 2. OpenRouter (Cloud API)
Access hundreds of models through a single API.

**Pros:**
- Access to many state-of-the-art models
- No GPU required
- Easy model comparison
- Pay only for what you use

**Cons:**
- Requires API key and payment
- Costs per token
- Requires internet connection

## Quick Start

### Using Ollama

```bash
# Start Ollama server (if not running)
ollama serve

# Pull a model
ollama pull qwen3:14b

# Run benchmark
python -m qwen3_agent.agents.people_search_agent.benchmark \
    --provider ollama \
    --model qwen3:14b \
    --num-samples 10
```

### Using OpenRouter

```bash
# Set your API key
export OPENROUTER_API_KEY="sk-or-v1-..."

# Run benchmark
python -m qwen3_agent.agents.people_search_agent.benchmark \
    --provider openrouter \
    --model qwen3-30b-a3b-instruct-2507 \
    --num-samples 10
```

Or use the convenience script:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
./scripts/benchmark_openrouter.sh qwen3-30b-a3b-instruct-2507 10
```

## Configuration

### Environment Variables

```bash
# Provider selection (default: ollama)
export LLM_PROVIDER="openrouter"  # or "ollama"

# Model selection
export LLM_MODEL="qwen3-30b-a3b-instruct-2507"

# OpenRouter specific
export OPENROUTER_API_KEY="sk-or-v1-..."
export OPENROUTER_SITE_URL="https://yoursite.com"  # Optional
export OPENROUTER_SITE_NAME="Your Site"  # Optional
```

### Command-Line Arguments

Arguments override environment variables:

```bash
python -m qwen3_agent.agents.people_search_agent.benchmark \
    --provider openrouter \
    --model qwen3-30b-a3b-instruct-2507 \
    --num-samples 100 \
    --max-turns 20
```

### Priority Order

1. Command-line arguments (highest priority)
2. Environment variables
3. Default values (lowest priority)

## Model Names

### Ollama Models

Use Ollama model tags directly:
- `qwen3:14b`
- `qwen3:7b`
- `llama3.1:70b`

### OpenRouter Models

The system supports short names that map to full OpenRouter model IDs:

| Short Name | Full Model ID |
|------------|---------------|
| `qwen3-14b` | `qwen/qwen-2.5-14b-instruct` |
| `qwen3-30b-a3b-instruct-2507` | `qwen/qwen3-30b-a3b-instruct-2507` |
| `qwen2.5-72b` | `qwen/qwen-2.5-72b-instruct` |

You can also use full OpenRouter model IDs directly:
```bash
--model "qwen/qwen3-30b-a3b-instruct-2507"
```

See all available models at [OpenRouter Models](https://openrouter.ai/models).

## Usage Examples

### Example 1: Quick Test

```bash
# Test with Ollama (5 samples)
python -m qwen3_agent.agents.people_search_agent.benchmark \
    --provider ollama \
    --model qwen3:14b \
    --num-samples 5 \
    --debug

# Test with OpenRouter (5 samples)
export OPENROUTER_API_KEY="sk-or-v1-..."
python -m qwen3_agent.agents.people_search_agent.benchmark \
    --provider openrouter \
    --model qwen3-30b-a3b-instruct-2507 \
    --num-samples 5 \
    --debug
```

### Example 2: Full Benchmark

```bash
# Full benchmark with 200 samples
python -m qwen3_agent.agents.people_search_agent.benchmark \
    --provider openrouter \
    --model qwen3-30b-a3b-instruct-2507 \
    --num-samples 200 \
    --max-turns 20
```

### Example 3: Model Comparison

```bash
# Test multiple models
for model in qwen3:14b qwen3:7b llama3.1:70b; do
    python -m qwen3_agent.agents.people_search_agent.benchmark \
        --provider ollama \
        --model "$model" \
        --num-samples 50 \
        --output-dir "./results/$model"
done

# Compare OpenRouter models
export OPENROUTER_API_KEY="sk-or-v1-..."
for model in qwen3-14b qwen3-30b-a3b-instruct-2507 qwen2.5-72b; do
    python -m qwen3_agent.agents.people_search_agent.benchmark \
        --provider openrouter \
        --model "$model" \
        --num-samples 50 \
        --output-dir "./results/$model"
done
```

### Example 4: Python API

```python
import asyncio
import os
from qwen3_agent.agents.people_search_agent.benchmark import benchmark_agent

async def compare_providers():
    """Compare Ollama and OpenRouter."""
    
    # Benchmark with Ollama
    print("Testing Ollama...")
    df_ollama = await benchmark_agent(
        provider="ollama",
        model="qwen3:14b",
        num_samples=10,
    )
    
    # Benchmark with OpenRouter
    print("\nTesting OpenRouter...")
    os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-..."
    df_openrouter = await benchmark_agent(
        provider="openrouter",
        model="qwen3-30b-a3b-instruct-2507",
        num_samples=10,
    )
    
    # Compare results
    print("\nResults:")
    print(f"Ollama avg reward: {df_ollama['reward'].mean():.3f}")
    print(f"OpenRouter avg reward: {df_openrouter['reward'].mean():.3f}")

asyncio.run(compare_providers())
```

## Debug Mode

Enable debug mode to see detailed information about API requests and responses:

```bash
python -m qwen3_agent.agents.people_search_agent.benchmark \
    --provider openrouter \
    --model qwen3-30b-a3b-instruct-2507 \
    --num-samples 10 \
    --debug \
    --debug-count 3  # Show details for first 3 tasks
```

This will:
- Print full request/response payloads
- Save detailed JSON traces to `benchmark_results/debug/`
- Show verbose output for each step

## Output Files

The benchmark saves results to `./benchmark_results/`:

```
benchmark_results/
├── benchmark_qwen3_30b_a3b_instruct_2507_10samples_20251112_120000.csv
├── benchmark_qwen3_30b_a3b_instruct_2507_10samples_20251112_120000_summary.json
└── debug/
    ├── task_1_trace.json
    ├── task_2_trace.json
    └── ...
```

The CSV contains per-task metrics, and the JSON summary contains aggregated statistics including the provider used.

## Extending Support

To add a new provider:

1. Create a new class in `llm_provider.py` that inherits from `BaseLLM`
2. Implement the `chat()` method that returns responses in Ollama format
3. Add the provider to the `create_llm()` factory function
4. Update documentation

Example:

```python
class MyProviderLLM(BaseLLM):
    def __init__(self, model: str, api_key: str, debug: bool = False):
        self.model = model
        self.api_key = api_key
        self.debug = debug
    
    def chat(self, messages, tools=None, temperature=0.7):
        # Implement your provider's API call
        # Convert response to Ollama format
        return {
            "message": {
                "role": "assistant",
                "content": "...",
                "tool_calls": [...]
            }
        }
```

## Troubleshooting

### Ollama Connection Error
```
Error calling Ollama API: Connection refused
```
**Solution:** Make sure Ollama server is running: `ollama serve`

### OpenRouter API Key Error
```
OpenRouter API key required
```
**Solution:** Set the environment variable: `export OPENROUTER_API_KEY="sk-or-v1-..."`

### Model Not Found
```
Error: Model not found
```
**Solution:** 
- For Ollama: Pull the model first: `ollama pull qwen3:14b`
- For OpenRouter: Check model name at https://openrouter.ai/models

### Import Error
```
ImportError: cannot import name 'create_llm'
```
**Solution:** Make sure you're using the latest code and have all dependencies installed.

## Cost Estimation (OpenRouter)

OpenRouter charges per token. Typical costs:

| Model | Cost per 1M tokens | Est. cost per 100 tasks |
|-------|-------------------|------------------------|
| Qwen 2.5 14B | ~$0.20 | ~$0.03 |
| Qwen3 30B | ~$0.60 | ~$0.09 |
| Qwen 2.5 72B | ~$0.90 | ~$0.14 |

*Estimates based on ~1500 tokens per task. Actual costs may vary.*

Use `--num-samples` to control benchmark size and costs.

## Additional Resources

- [OpenRouter Documentation](https://openrouter.ai/docs)
- [OpenRouter Models](https://openrouter.ai/models)
- [Ollama Documentation](https://ollama.ai/docs)
- [Ollama Models](https://ollama.ai/library)

For detailed OpenRouter setup, see [OPENROUTER_USAGE.md](./OPENROUTER_USAGE.md).

