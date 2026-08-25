#!/usr/bin/env bash
# 从 GitHub 克隆 mamba 并按 GPU 架构本地编译。
# 请先安装 causal-conv1d（见 install_causal_conv1d.sh）。
#
# Usage:
#   bash install_mamba_ssm.sh h100     # NVIDIA H100, sm_90, tag v2.2.5
#   bash install_mamba_ssm.sh 5090     # GeForce RTX 5090, sm_120, tag v2.2.6
#
# Optional env:
#   CUDA_HOME   default /usr/local/cuda-12.8
#   MAX_JOBS    compile parallelism (default 2)
#   SRC_DIR     existing clone path; if unset, clone into ./third_party/mamba
set -euo pipefail

GPU="${1:-}"
if [[ "$GPU" != "h100" && "$GPU" != "5090" ]]; then
  echo "Usage: bash $0 h100|5090"
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="https://github.com/state-spaces/mamba.git"

if [[ "$GPU" == "h100" ]]; then
  GIT_TAG="v2.2.5"
  ARCH_LIST="9.0"
else
  GIT_TAG="v2.2.6"
  ARCH_LIST="12.0"
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export MAMBA_FORCE_BUILD=TRUE
export TORCH_CUDA_ARCH_LIST="$ARCH_LIST"
export MAX_JOBS="${MAX_JOBS:-2}"
export CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS"
export MAKEFLAGS="-j${MAX_JOBS}"
export Ninja_NUM_JOBS="$MAX_JOBS"

SRC_DIR="${SRC_DIR:-$ROOT/third_party/mamba}"
if [[ ! -d "$SRC_DIR/.git" ]]; then
  mkdir -p "$(dirname "$SRC_DIR")"
  git clone --branch "$GIT_TAG" --depth 1 "$REPO_URL" "$SRC_DIR"
else
  git -C "$SRC_DIR" fetch --tags --depth 1 origin "refs/tags/${GIT_TAG}:refs/tags/${GIT_TAG}" || true
  git -C "$SRC_DIR" checkout "$GIT_TAG"
fi

cd "$SRC_DIR"
rm -rf build dist *.egg-info
echo "[mamba-ssm] GPU=$GPU tag=$GIT_TAG arch=$ARCH_LIST CUDA_HOME=$CUDA_HOME"
pip install -v --no-build-isolation .
