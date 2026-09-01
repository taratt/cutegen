#!/usr/bin/env bash
# GPT-5 + Triton + delayed profiling (27-kernel cohort). k1 assumed done in save tree.
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY first}"

export CUTEGEN_BASE_PATH="$PWD"
export CUTLASS_BASE_PATH="$PWD/cutegen/cutlass"
export CUTLASS_INCLUDE_PATH="$PWD/cutegen/cutlass/include"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export NSIGHT_COMPUTE_BIN="${NSIGHT_COMPUTE_BIN:-$CUDA_HOME/bin/ncu}"
export KERNEL_BACKEND=triton
export USE_PROFILING=true
export CUTEGEN_SAVE_DIR_BASE="$PWD/saved_nodes/triton/level1-profiled-gpt5"
export TOKEN_USAGE_CSV_PATH="$PWD/openai_gpt5_triton_delayed.csv"
export GPU_REQ_SPACE=0

nohup bash -c '
  CUTEGEN_KERNEL_IDS=4,9,102 PROFILING_START_DEPTH=6 python -u -m cutegen.main &&
  CUTEGEN_KERNEL_IDS=21,22,88 PROFILING_START_DEPTH=2 python -u -m cutegen.main &&
  CUTEGEN_KERNEL_IDS=54,55,58,59,80,83,103 PROFILING_START_DEPTH=6 python -u -m cutegen.main &&
  CUTEGEN_KERNEL_IDS=33,40,49,53 PROFILING_START_DEPTH=5 python -u -m cutegen.main &&
  CUTEGEN_KERNEL_IDS=99 PROFILING_START_DEPTH=3 python -u -m cutegen.main &&
  CUTEGEN_KERNEL_IDS=6,14 PROFILING_START_DEPTH=6 python -u -m cutegen.main &&
  CUTEGEN_KERNEL_IDS=50,61,70,75,107 PROFILING_START_DEPTH=6 python -u -m cutegen.main &&
  CUTEGEN_KERNEL_IDS=105 PROFILING_START_DEPTH=5 python -u -m cutegen.main
' > triton_gpt5_delayed.log 2>&1 &
echo $! > triton_gpt5_delayed.pid
echo "PID=$(cat triton_gpt5_delayed.pid) log=triton_gpt5_delayed.log"
