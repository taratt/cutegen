import os
import re
from cutegen.node import Node
from cutegen.util import read_file, fetch_kernelbench_problem_ref
from cutegen.coordinator import Coordinator

from cutegen.config import CUTEGEN_BASE_PATH, CUTLASS_BASE_PATH, CUTLASS_INCLUDE_PATH

LEVEL1_DIR = f"{CUTEGEN_BASE_PATH}/KernelBench/level2"
SAVE_DIR_BASE = f"{CUTEGEN_BASE_PATH}/saved_nodes/cute/level2"
from cutegen.llm_api import save_token_usage_csv, set_current_kernel_name

if __name__ == "__main__":
    num_prefix = re.compile(r"^(\d+)_.*\.py$")
    numbered_files = []
    for f in os.listdir(LEVEL1_DIR):
            m = num_prefix.match(f)
            if m:
                num = int(m.group(1))
                numbered_files.append((num, f))

    target_files = [f for num, f in sorted(numbered_files) if num <=60 and num>=31 ][0:]

    print(target_files)
    for fname in target_files:
        fpath = os.path.join(LEVEL1_DIR, fname)
        ref = read_file(fpath)
        save_folder = os.path.join(SAVE_DIR_BASE, fname)

        nodes = [Node(ref=ref, src="", save_folder_path=save_folder)]
        coordinator = Coordinator(nodes)
        coordinator.codegen_initial_addendum = f"Your task is to optimize using CUTE framework. If the operation can be implement using CUTE operators, USE CUTE instead of writting CUDA from scratch! If cute layout and tensors allow more optimization, use them. DO NOT under any circumstances generate CUTLASS templated code. The include cute location is found in {CUTLASS_INCLUDE_PATH}/cute/. You should add this include path directly in code in load_inline. DO NOT read from environment variables. Start with generating the simplest implementation in CuTE that is correct. For convolution-like kernels, do NOT start from a naive direct one-thread-per-output kernel when output spatial dimensions are large; start from a cooperative tiled implementation in CUTE instead, but do not jump immediately to a fragile implicit-GEMM rewrite. Pay close attention to the correct convolution kernel example given to you. Notice that CUTE tuples don’t support operator[]; you must use cute::get<Idx>(...). Pay close attention to the matrix operands dimensions and how they are compared to each other and base your implementation on what suits best for those relations. If there are a sequence of operations, you can try fusing them in the kernel."
       # coordinator.code_optimize_addendum = f"You MUST attempt to use CUTE code to optimize and write correct code"
        set_current_kernel_name(fname)
        coordinator.run()
