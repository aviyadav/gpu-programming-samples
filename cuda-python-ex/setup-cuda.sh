#!/usr/bin/env bash
# setup-cuda.sh
#
# One-time post-sync setup for numba CUDA support with uv on Linux.
# Run this once after `uv sync` (and again if you delete/recreate .venv).
#
# What this does:
#   1. Creates .venv/cuda_home/ — a minimal CUDA_HOME directory tree that
#      numba's path-discovery code expects, populated with symlinks into the
#      PyPI-installed nvidia-cuda-* packages.
#   2. Installs .venv/lib/.../sitecustomize.py — sets CUDA_HOME automatically
#      at Python startup so `uv run` works without any extra env vars.
#
# Usage:
#   uv sync
#   bash setup-cuda.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
SITE_PACKAGES=$(uv run python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)

# ---------------------------------------------------------------------------
# Verify that the required PyPI packages were installed by uv sync
# ---------------------------------------------------------------------------
NVCC_LIB="$SITE_PACKAGES/nvidia/cuda_nvcc"
CUDART_LIB="$SITE_PACKAGES/nvidia/cuda_runtime"

if [ ! -d "$NVCC_LIB" ]; then
  echo "ERROR: nvidia-cuda-nvcc-cu12 not found at $NVCC_LIB"
  echo "       Run 'uv sync' first."
  exit 1
fi

if [ ! -d "$CUDART_LIB" ]; then
  echo "ERROR: nvidia-cuda-runtime-cu12 not found at $CUDART_LIB"
  echo "       Run 'uv sync' first."
  exit 1
fi

# ---------------------------------------------------------------------------
# Build the cuda_home staging tree
# ---------------------------------------------------------------------------
CUDA_HOME="$VENV/cuda_home"

echo "Creating CUDA_HOME staging directory at $CUDA_HOME ..."

mkdir -p "$CUDA_HOME/nvvm/lib64"
mkdir -p "$CUDA_HOME/nvvm/libdevice"
mkdir -p "$CUDA_HOME/lib64"

# libnvvm.so — numba's find_lib() requires a versioned name (libnvvm.so.N)
ln -sf "$NVCC_LIB/nvvm/lib64/libnvvm.so" "$CUDA_HOME/nvvm/lib64/libnvvm.so"
ln -sf "$NVCC_LIB/nvvm/lib64/libnvvm.so" "$CUDA_HOME/nvvm/lib64/libnvvm.so.4"

# libdevice bitcode (needed by numba to compile device math functions)
ln -sf "$NVCC_LIB/nvvm/libdevice/libdevice.10.bc" \
       "$CUDA_HOME/nvvm/libdevice/libdevice.10.bc"

# libcudart (CUDA runtime — numba reads its version to pick supported CCs)
ln -sf "$CUDART_LIB/lib/libcudart.so.12" "$CUDA_HOME/lib64/libcudart.so.12"

echo "  nvvm/lib64/libnvvm.so.4     -> $(readlink "$CUDA_HOME/nvvm/lib64/libnvvm.so.4")"
echo "  nvvm/libdevice/libdevice.bc -> $(readlink "$CUDA_HOME/nvvm/libdevice/libdevice.10.bc")"
echo "  lib64/libcudart.so.12       -> $(readlink "$CUDA_HOME/lib64/libcudart.so.12")"

# ---------------------------------------------------------------------------
# Install sitecustomize.py into the venv's site-packages
# ---------------------------------------------------------------------------
SITECUSTOMIZE="$SITE_PACKAGES/sitecustomize.py"

echo ""
echo "Installing sitecustomize.py at $SITECUSTOMIZE ..."

cat > "$SITECUSTOMIZE" << 'EOF'
import os
import pathlib

# Point numba to the unified CUDA_HOME staging directory that contains symlinks
# to libraries from nvidia-cuda-nvcc-cu12 and nvidia-cuda-runtime-cu12.
# Structure:
#   cuda_home/nvvm/lib64/libnvvm.so.4    -> nvidia/cuda_nvcc/nvvm/lib64/libnvvm.so
#   cuda_home/nvvm/libdevice/libdevice.10.bc -> nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc
#   cuda_home/lib64/libcudart.so.12      -> nvidia/cuda_runtime/lib/libcudart.so.12
_cuda_home = pathlib.Path(__file__).parent.parent.parent.parent / "cuda_home"
if _cuda_home.exists() and "CUDA_HOME" not in os.environ:
    os.environ["CUDA_HOME"] = str(_cuda_home)
EOF

echo ""
echo "Setup complete. Run the program with:"
echo "  uv run main.py"
