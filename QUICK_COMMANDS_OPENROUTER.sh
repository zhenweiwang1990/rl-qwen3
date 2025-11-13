#!/bin/bash
# Quick Commands for OpenRouter Benchmarking
# Copy and paste these commands to get started

# =============================================================================
# SETUP (Run once)
# =============================================================================

# Set your OpenRouter API key (get it from https://openrouter.ai/)
export OPENROUTER_API_KEY="sk-or-v1-YOUR_KEY_HERE"

# =============================================================================
# QUICK TEST (5 samples with debug)
# =============================================================================

# Test qwen3-30b-a3b-instruct-2507 with 5 samples
python -m qwen3_agent.agents.people_search_agent.benchmark \
    --provider openrouter \
    --model qwen3-30b-a3b-instruct-2507 \
    --num-samples 5 \
    --debug

# =============================================================================
# SMALL BENCHMARK (10 samples)
# =============================================================================

# Using the convenience script
./scripts/benchmark_openrouter.sh qwen3-30b-a3b-instruct-2507 10

# Or directly with Python
python -m qwen3_agent.agents.people_search_agent.benchmark \
    --provider openrouter \
    --model qwen3-30b-a3b-instruct-2507 \
    --num-samples 10 \
    --max-turns 20

# =============================================================================
# MEDIUM BENCHMARK (50 samples)
# =============================================================================

python -m qwen3_agent.agents.people_search_agent.benchmark \
    --provider openrouter \
    --model qwen3-30b-a3b-instruct-2507 \
    --num-samples 50 \
    --max-turns 20

# =============================================================================
# FULL BENCHMARK (100 samples)
# =============================================================================

python -m qwen3_agent.agents.people_search_agent.benchmark \
    --provider openrouter \
    --model qwen3-30b-a3b-instruct-2507 \
    --num-samples 100 \
    --max-turns 20

# =============================================================================
# LARGE BENCHMARK (200 samples)
# =============================================================================

python -m qwen3_agent.agents.people_search_agent.benchmark \
    --provider openrouter \
    --model qwen3-30b-a3b-instruct-2507 \
    --num-samples 200 \
    --max-turns 20

# =============================================================================
# USING ENVIRONMENT VARIABLES (Set once, use everywhere)
# =============================================================================

# Set environment variables
export LLM_PROVIDER="openrouter"
export LLM_MODEL="qwen3-30b-a3b-instruct-2507"
export OPENROUTER_API_KEY="sk-or-v1-YOUR_KEY_HERE"

# Now run without specifying provider/model
python -m qwen3_agent.agents.people_search_agent.benchmark --num-samples 10

# =============================================================================
# COMPARE MULTIPLE MODELS
# =============================================================================

# Test different Qwen models
for model in qwen3-14b qwen3-30b-a3b-instruct-2507 qwen2.5-72b; do
    echo "Testing $model..."
    python -m qwen3_agent.agents.people_search_agent.benchmark \
        --provider openrouter \
        --model "$model" \
        --num-samples 50 \
        --output-dir "./results/$model"
done

# =============================================================================
# CUSTOM OUTPUT DIRECTORY
# =============================================================================

python -m qwen3_agent.agents.people_search_agent.benchmark \
    --provider openrouter \
    --model qwen3-30b-a3b-instruct-2507 \
    --num-samples 10 \
    --output-dir ./my_benchmark_results

# =============================================================================
# DEBUG MODE (See detailed API calls)
# =============================================================================

# Show details for first 3 tasks
python -m qwen3_agent.agents.people_search_agent.benchmark \
    --provider openrouter \
    --model qwen3-30b-a3b-instruct-2507 \
    --num-samples 10 \
    --debug \
    --debug-count 3

# =============================================================================
# COMPARE WITH OLLAMA (if you have it running locally)
# =============================================================================

# OpenRouter
python -m qwen3_agent.agents.people_search_agent.benchmark \
    --provider openrouter \
    --model qwen3-30b-a3b-instruct-2507 \
    --num-samples 50 \
    --output-dir ./results/openrouter

# Ollama (local)
python -m qwen3_agent.agents.people_search_agent.benchmark \
    --provider ollama \
    --model qwen3:14b \
    --num-samples 50 \
    --output-dir ./results/ollama

# Compare results
echo "OpenRouter results:"
cat ./results/openrouter/benchmark_*_summary.json | grep reward_mean
echo "Ollama results:"
cat ./results/ollama/benchmark_*_summary.json | grep reward_mean

