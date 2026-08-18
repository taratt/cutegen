import os
import re
from cutegen.node import Node
from cutegen.util import read_file, fetch_kernelbench_problem_ref
from cutegen.coordinator import Coordinator

from cutegen.config import (
    CUTEGEN_BASE_PATH,
    CUTLASS_BASE_PATH,
    CUTLASS_INCLUDE_PATH,
    KERNEL_BACKEND,
)

LEVEL1_DIR = f"{CUTEGEN_BASE_PATH}/KernelBench/level1"
SAVE_DIR_BASE = os.environ.get(
    "CUTEGEN_SAVE_DIR_BASE",
    f"{CUTEGEN_BASE_PATH}/saved_nodes/{KERNEL_BACKEND}/level1-delayed-profile",
)
from cutegen.llm_api import save_token_usage_csv, set_current_kernel_name

if __name__ == "__main__":
    num_prefix = re.compile(r"^(\d+)_.*\.py$")
    numbered_files = []
    for f in os.listdir(LEVEL1_DIR):
            m = num_prefix.match(f)
            if m:
                num = int(m.group(1))
                numbered_files.append((num, f))

    target_ids = {
        1, 4, 9, 102,
        21, 22, 88,
        54, 55, 58, 59, 80, 83, 103,
        33, 40, 49, 53,
        99,
    }
    target_ids_from_env = os.environ.get("CUTEGEN_KERNEL_IDS")
    if target_ids_from_env:
        target_ids = {
            int(kernel_id.strip())
            for kernel_id in target_ids_from_env.split(",")
            if kernel_id.strip()
        }
    target_files = [
        filename
        for number, filename in sorted(numbered_files)
        if number in target_ids
    ]

    print(target_files)
    for fname in target_files:
        fpath = os.path.join(LEVEL1_DIR, fname)
        ref = read_file(fpath)
        save_folder = os.path.join(SAVE_DIR_BASE, fname)

        nodes = [Node(ref=ref, src="", save_folder_path=save_folder)]
        nodes[0].metadata["kernel_backend"] = KERNEL_BACKEND
        coordinator = Coordinator(nodes)
        if KERNEL_BACKEND == "ptx":
            coordinator.codegen_initial_addendum = (
                "Use only PTX loaded through cutegen.ptx_runtime.PtxModule. "
                "Preserve the validate_generated_code hook and do not emit CUDA C++."
            )
        elif KERNEL_BACKEND == "cute":
            coordinator.codegen_initial_addendum = (
                f"Use CUTE headers from {CUTLASS_INCLUDE_PATH}/cute/ and do not "
                "generate CUTLASS templated code."
            )
        else:
            coordinator.codegen_initial_addendum = (
                "Your task is to optimize using CUDA. DO NOT under any "
                "circumstances generate CUTLASS templated code. Pay close "
                "attention to the matrix operands dimensions and how they are "
                "compared to each other and base your implementation on what "
                "suits best for those relations. If there are a sequence of "
                "operations, you can try fusing them in the kernel."
            )
       # coordinator.code_optimize_addendum = f"You MUST attempt to use CUTE code to optimize and write correct code"
        set_current_kernel_name(fname)
        coordinator.run()
