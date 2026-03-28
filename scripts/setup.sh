#!/bin/bash
set -e

echo "=== LLM Pretrainer Setup for RunPod ==="

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Ensure uv is in PATH for current session
export PATH="$HOME/.local/bin:$PATH"

# Create venv and install torch first (flash-attn needs it for building)
echo "Installing torch first..."
uv venv
uv pip install --python .venv/bin/python torch==2.4.1

# Now install flash-attn (precompiled wheel for torch 2.4.1)
echo "Installing flash-attn..."
uv pip install --python .venv/bin/python flash-attn==2.7.4.post1 --no-build-isolation

# Install remaining dependencies
echo "Installing remaining dependencies..."
uv sync

# Disable wandb by default (set WANDB_API_KEY to enable)
if [ -z "$WANDB_API_KEY" ]; then
    echo "WANDB_API_KEY not set, disabling wandb..."
    export WANDB_MODE=disabled
fi
