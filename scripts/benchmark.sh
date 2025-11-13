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

# Run benchmark using the CLI module
uv run python -m qwen3_agent.benchmark_cli

echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="

