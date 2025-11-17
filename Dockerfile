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

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy project files
COPY pyproject.toml .
COPY uv.lock .
COPY README.md .
COPY qwen3_agent/ ./qwen3_agent/
COPY scripts/ ./scripts/

# Install dependencies using uv
RUN uv sync --frozen

# Set environment variables
ENV PYTHONUNBUFFERED=1
# Note: CUDA_VISIBLE_DEVICES is NOT set here to allow runtime detection
# Set it at runtime if needed: docker run -e CUDA_VISIBLE_DEVICES=0 ...

# Create directories for data and models
RUN mkdir -p /workspace/data /workspace/models /workspace/logs

# Make scripts executable
RUN chmod +x /workspace/scripts/*.sh

# Default command
CMD ["/bin/bash"]

