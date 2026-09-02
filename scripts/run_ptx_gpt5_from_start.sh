#!/usr/bin/env bash
# GPT-5 + PTX + profiling from start (27-kernel cohort).
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate
source /etc/profile.d/cuda.sh 2>/dev/null || true

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY first}"

export CUTEGEN_BASE_PATH="$PWD"
export CUTLASS_BASE_PATH="$PWD/cutegen/cutlass"
export CUTLASS_INCLUDE_PATH="$PWD/cutegen/cutlass/include"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export NSIGHT_COMPUTE_BIN="${NSIGHT_COMPUTE_BIN:-$CUDA_HOME/bin/ncu}"

export KERNEL_BACKEND=ptx
export USE_PROFILING=true
export PROFILING_START_DEPTH=-1
export GPU_REQ_SPACE=0
export CUTEGEN_SAVE_DIR_BASE="$PWD/saved_nodes/ptx/level1-profiled-from-start-gpt5"
export TOKEN_USAGE_CSV_PATH="$PWD/openai_gpt5_ptx_from_start_token_usage.csv"
export CUTEGEN_KERNEL_IDS=1,4,6,9,14,21,22,33,40,49,50,53,54,55,58,59,61,70,75,80,83,88,99,102,103,105,107

mkdir -p "$CUTEGEN_SAVE_DIR_BASE"

nohup python -u scripts/resume_interrupted_kernels.py --kernel-ids "$CUTEGEN_KERNEL_IDS" \
  > ptx_gpt5_from_start.log 2>&1 &
echo $! > ptx_gpt5_from_start.pid
echo "PID=$(cat ptx_gpt5_from_start.pid) log=ptx_gpt5_from_start.log save=$CUTEGEN_SAVE_DIR_BASE"
