import os

from skxoss.node import Node
from skxoss.util import read_file, fetch_kernelbench_problem_ref
from skxoss.coordinator import Coordinator

from skxoss.config import SKXOSS_BASE_PATH

if __name__ == "__main__":

    # ref = read_file(f"{SKXOSS_BASE_PATH}/KernelBench/level1/16_Matmul_with_transposed_A.py")
    # nodes = [Node(ref=ref, src="", save_folder_path=f"{SKXOSS_BASE_PATH}/saved_nodes/16_Matmul_with_transposed_A") for i in range(1)]
    # ref = read_file(f"{SKXOSS_BASE_PATH}/KernelBench/level1/17_Matmul_with_transposed_B.py")
    # nodes = [Node(ref=ref, src="", save_folder_path=f"{SKXOSS_BASE_PATH}/saved_nodes/17_Matmul_with_transposed_B") for i in range(1)]
    ref = read_file(f"{SKXOSS_BASE_PATH}/KernelBench/hopper_gemm/13_llama_feedforward.py")
    nodes = [Node(ref=ref, src="", save_folder_path=f"{SKXOSS_BASE_PATH}/saved_nodes/13_llama_feedforward") for i in range(1)]
    coordinator = Coordinator(nodes)
    coordinator.codegen_initial_addendum = f"Your task is to optimize use CUTLASS framework. If the operation can be implement using CUTLASS operators, USE CUTLASS instead of writting CUDA from scratch!"
    coordinator.run()
