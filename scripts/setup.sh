#!/bin/bash
set -e

echo "=========================================="
echo "Setting up Qwen3 Email Agent Project"
echo "=========================================="
echo ""

# Check if we're in the correct directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found. Please run this script from the project root."
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "✓ uv installed"
    echo ""
    echo "Please restart your terminal or run:"
    echo "  source $HOME/.cargo/env"
    echo ""
    echo "Then run this script again."
    exit 0
fi

echo "Using uv version: $(uv --version)"
echo ""

# Sync dependencies using uv
echo "Syncing dependencies with uv..."
uv sync

echo ""
echo "✓ Dependencies synced"

# Check for .env file
if [ ! -f ".env" ]; then
    echo ""
    echo "WARNING: .env file not found!"
    echo "Please create a .env file with your configuration."
    echo "You can copy .env.example as a starting point:"
    echo "  cp .env.example .env"
    echo "Then edit .env with your API keys and settings."
else
    echo ""
    echo "✓ .env file found"
fi

# Download email database if needed
echo ""
echo "Checking for email database..."
if [ ! -f "enron_emails.db" ]; then
    echo "Email database not found. You need to generate it."
    echo "The database will be automatically generated on first training run."
    echo "Or you can run: ./scripts/generate_database.sh"
else
    echo "✓ Email database found"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "🚀 Your project is now using uv for dependency management!"
echo ""
echo "To run commands with uv, use:"
echo "  uv run python -m qwen3_agent.train"
echo "  uv run python -m qwen3_agent.benchmark"
echo ""
echo "Or use the convenience scripts:"
echo "  ./scripts/train_with_rl.sh"
echo "  ./scripts/benchmark.sh"
echo ""

