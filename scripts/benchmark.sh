#!/bin/bash
set -e

echo "=========================================="
echo "Qwen3 Email Agent - Benchmark"
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

# Set defaults
export RUN_ID="${RUN_ID:-001}"
export TEST_SET_SIZE="${TEST_SET_SIZE:-100}"
export VERBOSE="${VERBOSE:-true}"

echo "Configuration:"
echo "  - Run ID: $RUN_ID"
echo "  - Test set size: $TEST_SET_SIZE"
echo "  - Verbose: $VERBOSE"
echo ""

# Create benchmark script
cat > /tmp/run_benchmark.py << 'EOF'
import asyncio
import os
from dotenv import load_dotenv
from qwen3_agent.config import PolicyConfig, get_device
from qwen3_agent.benchmark import benchmark_model
from qwen3_agent.core import Model
import polars as pl

load_dotenv()

async def main():
    run_id = os.environ.get("RUN_ID", "001")
    test_set_size = int(os.environ.get("TEST_SET_SIZE", "100"))
    verbose = os.environ.get("VERBOSE", "false").lower() == "true"
    device = get_device()

    model_name = os.environ.get("MODEL_NAME", "OpenPipe/Qwen3-14B-Instruct")
    # Prefer explicit VLLM_BASE_URL if provided, else use INFERENCE_BASE_URL
    vllm_base_url = os.environ.get("VLLM_BASE_URL") or os.environ.get("INFERENCE_BASE_URL", "http://localhost:8000/v1")

    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print("")

    # Pure standalone model using LiteLLM + OpenAI-compatible server
    if "Qwen" in model_name or "qwen" in model_name:
        model = Model(
            name=model_name,
            project="qwen3_email_agent",
            config=PolicyConfig(
                litellm_model_name=f"openai/{model_name}",
                verbose=verbose,
            ),
            inference_api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
            inference_base_url=vllm_base_url,
            inference_model_name=model_name,
        )
    else:
        # External OpenAI model
        model = Model(
            name=model_name,
            project="qwen3_email_agent",
            config=PolicyConfig(
                litellm_model_name=f"openai/{model_name}",
                verbose=verbose,
            ),
            inference_api_key=os.getenv("OPENAI_API_KEY"),
            inference_base_url="https://api.openai.com/v1",
            inference_model_name=model_name,
        )

    results = await benchmark_model(model, limit=test_set_size, verbose=verbose)

    print("\n" + "="*60)
    print("Benchmark Results")
    print("="*60)
    print(results)

    # Save results
    results.write_csv(f"benchmark_results_{run_id}.csv")
    print(f"\nResults saved to benchmark_results_{run_id}.csv")

asyncio.run(main())
EOF

# Run benchmark
uv run python /tmp/run_benchmark.py

# Cleanup
rm /tmp/run_benchmark.py

echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="

