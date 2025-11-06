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

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✓ Dependencies installed"

# Install the project in editable mode
echo ""
echo "Installing qwen3_agent package..."
pip install -e .

echo ""
echo "✓ qwen3_agent package installed"

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
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start training, run:"
echo "  ./scripts/train.sh"
echo ""
echo "To run benchmarks, run:"
echo "  ./scripts/benchmark.sh"
echo ""

