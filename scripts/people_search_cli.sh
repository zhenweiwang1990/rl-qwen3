#!/bin/bash
# Convenience script to run People Search Agent CLI

cd "$(dirname "$0")/.."

echo "🚀 Starting People Search Agent CLI..."
echo ""

uv run python -m qwen3_agent.agents.people_search_agent.cli "$@"

