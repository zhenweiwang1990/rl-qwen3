#!/bin/bash
set -e

echo "=========================================="
echo "Generating Email Database"
echo "=========================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please run ./scripts/setup.sh first."
    exit 1
fi

# Check if database already exists
if [ -f "enron_emails.db" ]; then
    echo "Database already exists at enron_emails.db"
    read -p "Do you want to regenerate it? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing database."
        exit 0
    fi
    echo "Removing existing database..."
    rm enron_emails.db
fi

echo "Generating database from Enron email dataset..."
echo "This may take several minutes..."
echo ""

# Create a Python script to generate the database
cat > /tmp/generate_db.py << 'EOF'
import sys
import os

# Add qwen3_agent to path
sys.path.insert(0, os.path.abspath('.'))

from qwen3_agent.data.local_email_db import generate_database

print("Starting database generation...")
generate_database()
print("\n✓ Database generated successfully!")
print(f"Database location: {os.path.abspath('enron_emails.db')}")
EOF

# Run the generation script
uv run python /tmp/generate_db.py

# Cleanup
rm /tmp/generate_db.py

echo ""
echo "=========================================="
echo "Database Generation Complete!"
echo "=========================================="

