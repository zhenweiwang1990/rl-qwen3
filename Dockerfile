# Dockerfile for Qwen3 Email Agent Training
# Supports both CUDA (Linux) and CPU/MPS (macOS)

# Base image with Python and CUDA support
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY qwen3_agent/ ./qwen3_agent/
COPY scripts/ ./scripts/
COPY pyproject.toml .
COPY README.md .

# Install the package in development mode
RUN pip install -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Create directories for data and models
RUN mkdir -p /workspace/data /workspace/models /workspace/logs

# Make scripts executable
RUN chmod +x /workspace/scripts/*.sh

# Default command
CMD ["/bin/bash"]

