#!/bin/bash
# One-click setup script for training database

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================="
echo "Training Database Setup"
echo "=================================="
echo ""

# Check prerequisites
echo "📋 Checking prerequisites..."

if [ ! -f "data/profile_detail.csv" ]; then
    echo "❌ Error: data/profile_detail.csv not found"
    echo "   Please ensure the profile data file exists"
    exit 1
fi

if [ ! -f "data/10000-training-linkedin-handle.csv" ]; then
    echo "❌ Error: data/10000-training-linkedin-handle.csv not found"
    echo "   Please ensure the LinkedIn handle list exists"
    exit 1
fi

echo "✅ Prerequisites OK"
echo ""

# Step 1: Create training database
echo "📊 Step 1/3: Creating training database..."
echo "   This may take 2-3 minutes..."
python3 create_training_db.py

if [ ! -f "profiles_training.db" ]; then
    echo "❌ Error: Failed to create training database"
    exit 1
fi

echo "✅ Training database created"
echo ""

# Step 2: Filter benchmark queries
echo "🔍 Step 2/3: Filtering benchmark queries..."
python3 filter_benchmark.py

if [ ! -f "data/benchmark-queries-training.csv" ]; then
    echo "❌ Error: Failed to create filtered benchmark"
    exit 1
fi

echo "✅ Benchmark queries filtered"
echo ""

# Step 3: Test database
echo "🧪 Step 3/3: Testing database..."
python3 test_training_db.py

if [ $? -ne 0 ]; then
    echo "❌ Error: Database test failed"
    exit 1
fi

echo ""
echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "📁 Created files:"
echo "   - profiles_training.db (339MB, 10,000 profiles)"
echo "   - data/benchmark-queries-training.csv (872 queries)"
echo ""
echo "🚀 Next steps:"
echo "   1. Run benchmark:    python3 benchmark.py -n 10"
echo "   2. Try CLI:          python3 cli.py"
echo "   3. Start training:   python3 ../../train.py --agent people_search"
echo ""
echo "📚 Documentation:"
echo "   - QUICKSTART_TRAINING.md - Quick start guide"
echo "   - TRAINING_DB_GUIDE.md   - Detailed usage guide"
echo "   - TRAINING_DB_SETUP.md   - Technical summary"
echo ""

