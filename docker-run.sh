#!/bin/bash
set -e

# Convenience script for running Docker commands
# Usage: ./docker-run.sh [command]

COMMAND=${1:-train}

case $COMMAND in
  train)
    echo "Starting GPU training in Docker..."
    docker-compose up qwen3-train-gpu
    ;;
  
  train-cpu)
    echo "Starting CPU training in Docker..."
    docker-compose up qwen3-train-cpu
    ;;
  
  benchmark)
    echo "Running benchmark in Docker..."
    docker-compose up qwen3-benchmark
    ;;
  
  shell)
    echo "Opening shell in GPU container..."
    docker-compose run --rm qwen3-train-gpu /bin/bash
    ;;
  
  shell-cpu)
    echo "Opening shell in CPU container..."
    docker-compose run --rm qwen3-train-cpu /bin/bash
    ;;
  
  build)
    echo "Building Docker images..."
    docker-compose build
    ;;
  
  logs)
    echo "Showing logs..."
    docker-compose logs -f
    ;;
  
  stop)
    echo "Stopping containers..."
    docker-compose down
    ;;
  
  clean)
    echo "Cleaning up containers and volumes..."
    docker-compose down -v
    ;;
  
  *)
    echo "Usage: ./docker-run.sh [command]"
    echo ""
    echo "Available commands:"
    echo "  train        - Start GPU training"
    echo "  train-cpu    - Start CPU training"
    echo "  benchmark    - Run benchmark"
    echo "  shell        - Open shell in GPU container"
    echo "  shell-cpu    - Open shell in CPU container"
    echo "  build        - Build Docker images"
    echo "  logs         - Show container logs"
    echo "  stop         - Stop containers"
    echo "  clean        - Remove containers and volumes"
    exit 1
    ;;
esac

