import os

CUTEGEN_BASE_PATH = os.environ.get("CUTEGEN_BASE_PATH", "/home/user/cutegen") # base path for the cutegen project
BUILD_DIRECTORY_BASE = f"{CUTEGEN_BASE_PATH}/build/" # base path for the build directory
GPU_LOCK_FILE = "/tmp/gpu_flock.lock" # file for the GPU lock
CUTLASS_BASE_PATH = os.environ.get("CUTLASS_BASE_PATH")
CUTLASS_INCLUDE_PATH = os.environ.get("CUTLASS_INCLUDE_PATH")
INITIAL_PROMPT_FILE = f"{CUTEGEN_BASE_PATH}/cutegen/prompts/cute_initial_kernelbench_prompt.txt"
OPTIMIZE_PROMPT_FILE=f"{CUTEGEN_BASE_PATH}/cutegen/prompts/cute_optimized_kernelbench_prompt.txt"
EDIT_PROMPT_FILE=f"{CUTEGEN_BASE_PATH}/cutegen/prompts/cute_optimized_kernelbench_prompt.txt"
TUNE_PROMPT_FILE=f"{CUTEGEN_BASE_PATH}/cutegen/prompts/cute_tuning_kernelbench_prompt.txt"
DEBUG_PRINT = True # whether to print debug messages

LOAD_MODEL_BACKOFF_TIME = 1.0 # time to wait before retrying a failed operation
RUN_MODEL_BACKOFF_TIME = 2.0 # time to wait before retrying a failed operation
GPU_REQ_SPACE = 0.85 # 1/4 GPU space reserved.
CPU_REQ_SPACE = 70.0 # 70% CPU space reserved.

COMPILE_LOG_CHARS = 10000 # maximum number of characters to pass into an llm to fix a compile error

MAX_FIX_ATTEMPTS = int(os.environ.get("SK_MAX_FIX_ATTEMPTS", 5)) # maximum number of attempts to fix a compile or correctness error
MAX_RETRY_ATTEMPTS = 3 # maximum number of attempts to regenerate a node based on ref and prev_src
FIX_COMPILE_MODE = "edits"
FIX_CORRECT_MODE = "edits"
FIX_RETRIEVE = True

CODEGEN_INITIAL_MODE = "original" # mode to generate the initial code, options: original, edits
CODEGEN_OPTIMIZE_MODE = "edits" # mode to optimize the code, options: original, edits

MAX_CONCURRENT_PROCESSES = 1 # maximum number of concurrent processes for node execution
EVAL_RUN_TIMEOUT = 300.0 # maximum time to wait for a child process to finish run_in_subprocess
MAX_CONCURRENT_PROBLEMS = 1 # maximum number of problems to run concurrently on the coordinator
MAX_DEPTH = 10 # maximum depth of the search
MAX_RETRIES_PER_DEPTH = 0

FEEDBACK_MODE = "iterative"
USE_PROFILING = True
PROFILING_START_DEPTH = 7
NSIGHT_COMPUTE_BIN = os.environ.get("NSIGHT_COMPUTE_BIN", "/usr/local/cuda/bin/ncu")
NSIGHT_COMPUTE_SET = "basic"
BENCHMARK_TORCH_COMPILE = False # whether to benchmark the torch compile
TORCH_COMPILE_MODE = "default" # mode to benchmark the torch compile, options: default, max-autotune
ALWAYS_OPTIMIZE_BEST = False
EVAL_COLD_CACHE = False # whether to evaluate the cold cache

from cutegen.llm_api import LLMConfig

# code generation model selection
LLM_CONFIG_CODEGEN = [
    # LLMConfig(server_type="openai", model_name="o4-mini-2025-04-16", temperature=0.5, is_reasoning_model=True, max_completion_tokens=100000),
    #LLMConfig(server_type="openai", model_name="o3-2025-04-16", temperature=0.5, is_reasoning_model=True, max_completion_tokens=100000),
    # LLMConfig(server_type="percepta", model_name="Qwen/Qwen3-32B", temperature=0.0, max_tokens=100000)
    # LLMConfig(server_type="google", model_name="gemini-2.5-pro", temperature=0.5, max_tokens=100000),
    LLMConfig(server_type="openai", model_name="gpt-5", temperature=0.5, is_reasoning_model=True, max_completion_tokens=100000)
]
