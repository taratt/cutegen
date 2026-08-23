"""End-to-end Triton test through Cutegen correctness and timing."""

import tempfile
from pathlib import Path

from cutegen.evaluate import evaluate
from cutegen.node import ErrorType, Node


REFERENCE_SOURCE = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, a, b):
        return a + b

def get_inputs():
    return [torch.randn(65536), torch.randn(65536)]

def get_init_inputs():
    return []
"""


GENERATED_SOURCE = """
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(a_ptr, b_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, a + b, mask=mask)

def validate_generated_code():
    a = torch.empty(256, device="cuda")
    b = torch.empty_like(a)
    output = torch.empty_like(a)
    vector_add_kernel[(1,)](a, b, output, 256, BLOCK_SIZE=256)

class ModelNew(nn.Module):
    def forward(self, a, b):
        output = torch.empty_like(a)
        n_elements = output.numel()
        grid = (triton.cdiv(n_elements, 256),)
        vector_add_kernel[grid](a, b, output, n_elements, BLOCK_SIZE=256)
        return output
"""


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cutegen_triton_eval_") as directory:
        save_directory = Path(directory) / "node"
        save_directory.mkdir()
        (save_directory / "best_time.txt").write_text("999999.0")

        node = Node(
            ref=REFERENCE_SOURCE,
            src=GENERATED_SOURCE,
            save_folder_path=str(save_directory),
        )
        node.metadata["kernel_backend"] = "triton"
        result = evaluate(
            node,
            get_time=True,
            get_profile=False,
            torch_compile=False,
        )

        if result is None or node.error_type != ErrorType.PASS:
            raise AssertionError(
                f"Triton evaluator failed: error={node.error_type}, "
                f"metadata={node.metadata}"
            )
        if not node.time or node.time.get("num_trials") != 100:
            raise AssertionError(
                f"Triton timing statistics are incomplete: {node.time}"
            )

        print("Triton evaluator correctness: PASS")
        print(f"Triton evaluator CUDA-event timing: {node.time}")


if __name__ == "__main__":
    main()
