#!/usr/bin/env bash
# Run cutegen experiments via rust.cat router (gpt-5 / gpt-5.6-sol / gpt-5.6-luna).
#
# Usage:
#   export OPENAI_API_KEY='sk-...'
#   ./scripts/run_rustcat_openai.sh gpt-5.6-sol triton delayed
#
# Args:
#   $1 model:      gpt-5 | gpt-5.6-sol | gpt-5.6-luna  (default: gpt-5.6-sol)
#   $2 backend:    triton | ptx | cuda | cute           (default: triton)
#   $3 profiling:  delayed | from-start | nopf          (default: delayed)
#
# Env overrides:
#   CUTEGEN_KERNEL_IDS   comma-separated kernel ids (default: full 27-kernel cohort script)
#   CUTEGEN_SAVE_DIR_BASE  override save dir
set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY first}"

MODEL="${1:-gpt-5.6-sol}"
BACKEND="${2:-triton}"
PROFILE="${3:-delayed}"

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://rust.cat/v1}"
export OPENAI_MODEL="$MODEL"
export OPENAI_REASONING_EFFORT="${OPENAI_REASONING_EFFORT:-high}"

export CUTEGEN_BASE_PATH="$PWD"
export CUTLASS_BASE_PATH="$PWD/cutegen/cutlass"
export CUTLASS_INCLUDE_PATH="$PWD/cutegen/cutlass/include"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export NSIGHT_COMPUTE_BIN="${NSIGHT_COMPUTE_BIN:-$CUDA_HOME/bin/ncu}"
export KERNEL_BACKEND="$BACKEND"
export GPU_REQ_SPACE=0

case "$PROFILE" in
  delayed)
    export USE_PROFILING=true
    export PROFILING_START_DEPTH="${PROFILING_START_DEPTH:-6}"
    SAVE_SUFFIX="profiled"
    ;;
  from-start)
    export USE_PROFILING=true
    export PROFILING_START_DEPTH=0
    SAVE_SUFFIX="profiled-from-start"
    ;;
  nopf)
    export USE_PROFILING=false
    SAVE_SUFFIX="no-profile"
    ;;
  *)
    echo "Unknown profiling mode: $PROFILE (expected delayed|from-start|nopf)" >&2
    exit 1
    ;;
esac

MODEL_SLUG="${MODEL//./-}"
export CUTEGEN_SAVE_DIR_BASE="${CUTEGEN_SAVE_DIR_BASE:-$PWD/saved_nodes/$BACKEND/level1-${SAVE_SUFFIX}-${MODEL_SLUG}}"
export TOKEN_USAGE_CSV_PATH="${TOKEN_USAGE_CSV_PATH:-$PWD/openai_${MODEL_SLUG}_${BACKEND}_${PROFILE}.csv}"

LOG="rustcat_${MODEL_SLUG}_${BACKEND}_${PROFILE}.log"
PIDFILE="rustcat_${MODEL_SLUG}_${BACKEND}_${PROFILE}.pid"

if [[ -n "${CUTEGEN_KERNEL_IDS:-}" ]]; then
  RUN_CMD="CUTEGEN_KERNEL_IDS=$CUTEGEN_KERNEL_IDS python -u -m cutegen.main"
else
  RUN_CMD='
    CUTEGEN_KERNEL_IDS=4,9,102 PROFILING_START_DEPTH=6 python -u -m cutegen.main &&
    CUTEGEN_KERNEL_IDS=21,22,88 PROFILING_START_DEPTH=2 python -u -m cutegen.main &&
    CUTEGEN_KERNEL_IDS=54,55,58,59,80,83,103 PROFILING_START_DEPTH=6 python -u -m cutegen.main &&
    CUTEGEN_KERNEL_IDS=33,40,49,53 PROFILING_START_DEPTH=5 python -u -m cutegen.main &&
    CUTEGEN_KERNEL_IDS=99 PROFILING_START_DEPTH=3 python -u -m cutegen.main &&
    CUTEGEN_KERNEL_IDS=6,14 PROFILING_START_DEPTH=6 python -u -m cutegen.main &&
    CUTEGEN_KERNEL_IDS=50,61,70,75,107 PROFILING_START_DEPTH=6 python -u -m cutegen.main &&
    CUTEGEN_KERNEL_IDS=105 PROFILING_START_DEPTH=5 python -u -m cutegen.main
  '
fi

echo "model=$MODEL backend=$BACKEND profile=$PROFILE"
echo "OPENAI_BASE_URL=$OPENAI_BASE_URL"
echo "save_dir=$CUTEGEN_SAVE_DIR_BASE"
echo "token_csv=$TOKEN_USAGE_CSV_PATH"

nohup bash -c "$RUN_CMD" > "$LOG" 2>&1 &
echo $! > "$PIDFILE"
echo "PID=$(cat "$PIDFILE") log=$LOG"
