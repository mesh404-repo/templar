FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 and build dependencies for flash-attn
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    ca-certificates \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /uvx /bin/

# Configure uv for production
ENV UV_COMPILE_BYTECODE=0
ENV UV_LINK_MODE=copy
ENV UV_PYTHON=python3.12

# Create non-root user for security
RUN useradd -m -s /bin/bash -u 2000 runner

# Set up working directory
WORKDIR /home/runner

# Create virtual environment and install dependencies
# Install PyTorch with CUDA 12.8 support (matching base image)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /home/runner/.venv && \
    uv pip install --python=/home/runner/.venv/bin/python \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch torchvision && \
    uv pip install --python=/home/runner/.venv/bin/python \
    transformers \
    datasets \
    accelerate \
    "torchtitan @ git+https://github.com/pytorch/torchtitan.git"

# Install flash-attn dependencies first (required for compilation)
# See: https://github.com/Dao-AILab/flash-attention
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python=/home/runner/.venv/bin/python \
    packaging ninja wheel setuptools

# Install flash-attn (requires compilation)
# MAX_JOBS=4 prevents OOM during compilation on machines with many cores
# --no-build-isolation is required per flash-attn docs
ENV MAX_JOBS=4
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python=/home/runner/.venv/bin/python \
    flash-attn --no-build-isolation

# Set PATH to use venv
ENV PATH="/home/runner/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/home/runner/.venv"

# Copy runner script
COPY runner.py /home/runner/runner.py
RUN chown -R runner:runner /home/runner

# Switch to non-root user
USER runner

# Entry point
ENTRYPOINT ["python", "/home/runner/runner.py"]
