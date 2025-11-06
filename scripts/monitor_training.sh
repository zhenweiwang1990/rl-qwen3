#!/bin/bash

# Monitor training progress by watching logs
# Usage: ./scripts/monitor_training.sh [container_name]

CONTAINER_NAME="${1:-qwen3-email-agent-train-gpu}"

echo "=========================================="
echo "Monitoring Training Progress"
echo "Container: $CONTAINER_NAME"
echo "=========================================="
echo ""
echo "Press Ctrl+C to stop monitoring"
echo ""

# Check if running in Docker
if command -v docker &> /dev/null && docker ps | grep -q "$CONTAINER_NAME"; then
    echo "Monitoring Docker container logs..."
    docker logs -f "$CONTAINER_NAME"
elif [ -d "logs" ]; then
    echo "Monitoring local log files..."
    tail -f logs/*.log
else
    echo "No Docker container or log files found."
    echo "If training locally, use VERBOSE=true to see detailed output."
fi

