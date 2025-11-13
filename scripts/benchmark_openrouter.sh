#!/bin/bash
# Benchmark People Search Agent using OpenRouter
#
# Usage:
#   export OPENROUTER_API_KEY="sk-or-v1-..."
#   ./scripts/benchmark_openrouter.sh [model] [num_samples]
#
# Examples:
#   ./scripts/benchmark_openrouter.sh qwen3-30b-a3b-instruct-2507 10
#   ./scripts/benchmark_openrouter.sh qwen2.5-72b 100

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
MODEL="${1:-qwen3-30b-a3b-instruct-2507}"
NUM_SAMPLES="${2:-10}"
MAX_TURNS="${3:-20}"

# Check API key
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "❌ Error: OPENROUTER_API_KEY environment variable is not set"
    echo ""
    echo "Please set your OpenRouter API key:"
    echo "  export OPENROUTER_API_KEY=\"sk-or-v1-...\""
    echo ""
    echo "Get your API key at: https://openrouter.ai/"
    exit 1
fi

echo "🚀 Starting OpenRouter Benchmark"
echo "=================================="
echo "Provider: openrouter"
echo "Model: $MODEL"
echo "Samples: $NUM_SAMPLES"
echo "Max turns: $MAX_TURNS"
echo "=================================="
echo ""

# Run benchmark
cd "$PROJECT_ROOT"

python -m qwen3_agent.agents.people_search_agent.benchmark \
    --provider openrouter \
    --model "$MODEL" \
    --num-samples "$NUM_SAMPLES" \
    --max-turns "$MAX_TURNS" \
    --output-dir ./benchmark_results

echo ""
echo "✅ Benchmark complete!"
echo "Results saved to: benchmark_results/"

