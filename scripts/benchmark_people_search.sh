#!/bin/bash
# Benchmark People Search Agent

cd "$(dirname "$0")/.."

echo "🧪 Running People Search Agent Benchmark..."
echo ""

uv run python -m qwen3_agent.agents.people_search_agent.benchmark "$@"

