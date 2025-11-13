# OpenRouter Integration Guide

This guide explains how to use OpenRouter as an LLM provider for the People Search Agent benchmark.

## Setup

### 1. Get OpenRouter API Key

Sign up at [OpenRouter](https://openrouter.ai/) and get your API key.

### 2. Set Environment Variables

```bash
# Required: Your OpenRouter API key
export OPENROUTER_API_KEY="sk-or-v1-..."

# Optional: Configure default provider and model
export LLM_PROVIDER="openrouter"
export LLM_MODEL="qwen3-30b-a3b-instruct-2507"

# Optional: For rankings on openrouter.ai
export OPENROUTER_SITE_URL="https://yoursite.com"
export OPENROUTER_SITE_NAME="Your Site Name"
```

## Usage

### Method 1: Using Environment Variables

Set the environment variables as shown above, then run:

```bash
# Run benchmark with environment variables
python -m qwen3_agent.agents.people_search_agent.benchmark \
    --num-samples 10 \
    --max-turns 20

# Or use the script
./scripts/benchmark_people_search.sh
```

### Method 2: Using Command-Line Arguments

```bash
# Specify provider and model via CLI
python -m qwen3_agent.agents.people_search_agent.benchmark \
    --provider openrouter \
    --model qwen3-30b-a3b-instruct-2507 \
    --num-samples 10 \
    --max-turns 20

# With debug mode to see first task in detail
python -m qwen3_agent.agents.people_search_agent.benchmark \
    --provider openrouter \
    --model qwen3-30b-a3b-instruct-2507 \
    --num-samples 10 \
    --debug \
    --debug-count 3
```

### Method 3: Using Python API

```python
import asyncio
import os
from qwen3_agent.agents.people_search_agent.benchmark import benchmark_agent

# Set API key
os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-..."

async def run_benchmark():
    df = await benchmark_agent(
        provider="openrouter",
        model="qwen3-30b-a3b-instruct-2507",
        num_samples=10,
        max_turns=20,
        debug=True,
        debug_count=1,
    )
    print(f"Results saved to benchmark_results/")
    return df

# Run
asyncio.run(run_benchmark())
```

## Supported Models

### OpenRouter Model Names

The system supports short model names that are automatically mapped to full OpenRouter model identifiers:

| Short Name | Full OpenRouter Name |
|------------|---------------------|
| `qwen3-14b` | `qwen/qwen-2.5-14b-instruct` |
| `qwen3-30b-a3b-instruct-2507` | `qwen/qwen3-30b-a3b-instruct-2507` |
| `qwen2.5-72b` | `qwen/qwen-2.5-72b-instruct` |

You can also use full model names directly:
```bash
--model "qwen/qwen3-30b-a3b-instruct-2507"
```

To add more model mappings, edit `llm_provider.py` in the `create_llm()` function.

## Provider Comparison

### Ollama (Local)
- ✅ Free, runs locally
- ✅ No API key needed
- ✅ Full control over resources
- ❌ Requires local GPU
- ❌ Limited to locally available models

### OpenRouter (Cloud)
- ✅ Access to many models
- ✅ No GPU required
- ✅ Easy to compare different models
- ❌ Requires API key
- ❌ Costs per API call
- ❌ Requires internet connection

## Examples

### Example 1: Compare Ollama vs OpenRouter

```bash
# Benchmark with Ollama (local)
export LLM_PROVIDER="ollama"
export LLM_MODEL="qwen3:14b"
python -m qwen3_agent.agents.people_search_agent.benchmark \
    --num-samples 100 \
    --output-dir ./results/ollama

# Benchmark with OpenRouter
export LLM_PROVIDER="openrouter"
export LLM_MODEL="qwen3-30b-a3b-instruct-2507"
export OPENROUTER_API_KEY="sk-or-v1-..."
python -m qwen3_agent.agents.people_search_agent.benchmark \
    --num-samples 100 \
    --output-dir ./results/openrouter
```

### Example 2: Quick Test with Debug

```bash
# Test with just 3 samples and detailed debug output
export OPENROUTER_API_KEY="sk-or-v1-..."

python -m qwen3_agent.agents.people_search_agent.benchmark \
    --provider openrouter \
    --model qwen3-30b-a3b-instruct-2507 \
    --num-samples 3 \
    --debug \
    --debug-count 3
```

### Example 3: Large Benchmark Run

```bash
# Full benchmark with 200 samples
export OPENROUTER_API_KEY="sk-or-v1-..."

python -m qwen3_agent.agents.people_search_agent.benchmark \
    --provider openrouter \
    --model qwen3-30b-a3b-instruct-2507 \
    --num-samples 200 \
    --max-turns 20 \
    --output-dir ./benchmark_results
```

## Output Files

The benchmark generates:
- `benchmark_<model>_<samples>samples_<timestamp>.csv` - Detailed per-task results
- `benchmark_<model>_<samples>samples_<timestamp>_summary.json` - Summary statistics
- `debug/task_<id>_trace.json` - Debug traces (if `--debug` enabled)

## Troubleshooting

### "OpenRouter API key required" Error
Make sure you've set the `OPENROUTER_API_KEY` environment variable:
```bash
export OPENROUTER_API_KEY="your-key-here"
```

### "Unknown provider" Error
Check that you're using either `"ollama"` or `"openrouter"` (lowercase):
```bash
--provider openrouter  # ✅ Correct
--provider OpenRouter  # ❌ Will be converted to lowercase
```

### Model Not Found
If you get a model not found error:
1. Check the model name is correct on [OpenRouter Models](https://openrouter.ai/models)
2. Use the full model identifier (e.g., `qwen/qwen3-30b-a3b-instruct-2507`)
3. Add a mapping in `llm_provider.py` if you want to use short names

### Connection Timeout
OpenRouter requests have a 120-second timeout. If you're getting timeouts:
- The model might be cold-starting (first request can be slow)
- Try again in a few seconds
- Check your internet connection

## Cost Considerations

OpenRouter charges per token. To estimate costs:
1. Check model pricing at [OpenRouter Pricing](https://openrouter.ai/docs#models)
2. Each task typically uses 500-2000 tokens (depending on complexity)
3. Example: 100 samples × 1500 tokens avg × $0.002 per 1K tokens = ~$0.30

Use `--num-samples` to control the number of tasks evaluated.

