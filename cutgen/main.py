import os
import re
from cutgen.node import Node
from cutgen.util import read_file, fetch_kernelbench_problem_ref
from cutgen.coordinator import Coordinator

from cutgen.config import SKXOSS_BASE_PATH

LEVEL1_DIR = "/home/tarasaba/PycharmProjects/cutgen/KernelBench/level1"
SAVE_DIR_BASE = f"{SKXOSS_BASE_PATH}/saved_nodes/cute/attempt_15"

if __name__ == "__main__":
    num_prefix = re.compile(r"^(\d+)_.*\.py$")
    numbered_files = []
    for f in os.listdir(LEVEL1_DIR):
            m = num_prefix.match(f)
            if m:
                num = int(m.group(1))
                numbered_files.append((num, f))

    target_files = [f for num, f in sorted(numbered_files) if num >=19 and num<=32][0:]
    print(target_files)
    # ref = read_file("/home/tarasaba/PycharmProjects/cutgen/KernelBench/attempt_0/5_Matrix_scalar_multiplication.py")
    # nodes = [Node(ref=ref, src="", save_folder_path=f"{SKXOSS_BASE_PATH}/saved_nodes/attempt_0/5_Matrix_scalar_multiplication.py") for i in range(1)]
    # coordinator = Coordinator(nodes)
    # coordinator.codegen_initial_addendum = f"Your task is to optimize use CUTLASS framework. If the operation can be implement using CUTLASS operators, USE CUTLASS instead of writting CUDA from scratch!"
    # coordinator.run()
    for fname in target_files:
        fpath = os.path.join(LEVEL1_DIR, fname)
        ref = read_file(fpath)
        save_folder = os.path.join(SAVE_DIR_BASE, fname)

        nodes = [Node(ref=ref, src="", save_folder_path=save_folder)]
        coordinator = Coordinator(nodes)
        coordinator.codegen_initial_addendum = f"Your task is to optimize using CUTE framework. If the operation can be implement using CUTE operators, USE CUTE instead of writting CUDA from scratch! If cute layout and tensors allow more optimization, use them. DO NOT under any circumstances generate CUTLASS templated code. The include cute location is found in /home/tarasaba/cutlass/include/cute/. You should add this include path directly in code in load_inline. DO NOT read from environment variables. Start with generating the simplest implementation but correct. Notice that CUTE tuples don’t support operator[]; you must use cute::get<Idx>(...). Pay close attention to the matrix operands dimensions and how they are compared to each other and base your implementation on what suits best for those relations."
       # coordinator.code_optimize_addendum = f"You MUST attempt to use CUTE code to optimize and write correct code"
        coordinator.run()