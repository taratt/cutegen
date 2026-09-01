#!/usr/bin/env bash
# Resume CuTe/Kimi from-start (level1_from_start). Run ONE job at a time on local GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate

: "${MOONSHOT_API_KEY:?Set MOONSHOT_API_KEY first}"

export CUTEGEN_BASE_PATH="$PWD"
export CUTLASS_BASE_PATH="$PWD/cutegen/cutlass"
export CUTLASS_INCLUDE_PATH="$PWD/cutegen/cutlass/include"
export NSIGHT_COMPUTE_BIN="$(which ncu)"
export KERNEL_BACKEND=cute
export USE_PROFILING=true
export PROFILING_START_DEPTH=-1
export CUTEGEN_SAVE_DIR_BASE="$PWD/saved_nodes/cute/level1_from_start"
export TOKEN_USAGE_CSV_PATH="$PWD/kimi_token_usage_cute_from_start.csv"

nohup python -u scripts/resume_interrupted_kernels.py --auto \
  > cute_kimi_from_start_resume.log 2>&1 &
echo $! > cute_kimi_from_start_resume.pid
echo "PID=$(cat cute_kimi_from_start_resume.pid) log=cute_kimi_from_start_resume.log"
tail -f cute_kimi_from_start_resume.log
