"""End-to-end PTX test through Cutegen's compile/correctness/timing pipeline."""

import tempfile
from pathlib import Path

from cutegen.evaluate import check_compile, evaluate
from cutegen.node import ErrorType, Node
from scripts.smoke_test_ptx_runtime import PTX_SOURCE


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


def generated_source(ptx_source: str) -> str:
    return f"""
import torch
import torch.nn as nn
from cutegen.ptx_runtime import PtxModule, u32

PTX_SOURCE = {ptx_source!r}
_ptx_module = None

def _get_ptx_module():
    global _ptx_module
    if _ptx_module is None:
        _ptx_module = PtxModule(PTX_SOURCE)
    return _ptx_module

def validate_generated_code():
    _get_ptx_module().function("vector_add_f32")

class ModelNew(nn.Module):
    def forward(self, a, b):
        output = torch.empty_like(a)
        block = 256
        grid = (a.numel() + block - 1) // block
        _get_ptx_module().launch(
            "vector_add_f32",
            grid=grid,
            block=block,
            arguments=[a, b, output, u32(a.numel())],
            stream=torch.cuda.current_stream(a.device),
        )
        return output
"""


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cutegen_ptx_eval_") as directory:
        root = Path(directory)

        invalid_metadata = {"compile": "", "correct": ""}
        invalid_source = generated_source(
            PTX_SOURCE.replace("add.f32 %f3, %f1, %f2;", "not_an_instruction;")
        )
        invalid_result = check_compile(
            REFERENCE_SOURCE,
            invalid_source,
            invalid_metadata,
            build_directory=str(root / "invalid_build"),
        )
        if invalid_result:
            raise AssertionError("Invalid PTX unexpectedly passed compile validation")
        compile_error = invalid_metadata.get("compile", "")
        if (
            "cuModuleLoadDataEx" not in compile_error
            or "PTX JIT error log:" not in compile_error
            or "not_an_instruction" not in compile_error
        ):
            raise AssertionError(
                "Invalid PTX did not surface detailed JIT diagnostics as compile metadata"
            )
        print("PTX invalid-source classification: PASS")

        save_directory = root / "node"
        save_directory.mkdir()
        (save_directory / "best_time.txt").write_text("999999.0")
        node = Node(
            ref=REFERENCE_SOURCE,
            src=generated_source(PTX_SOURCE),
            save_folder_path=str(save_directory),
        )
        node.metadata["kernel_backend"] = "ptx"
        result = evaluate(
            node,
            get_time=True,
            get_profile=False,
            torch_compile=False,
        )

        if result is None or node.error_type != ErrorType.PASS:
            raise AssertionError(
                f"PTX evaluator failed: error={node.error_type}, metadata={node.metadata}"
            )
        if not node.time or node.time.get("num_trials") != 100:
            raise AssertionError(f"PTX timing statistics are incomplete: {node.time}")

        print("PTX evaluator correctness: PASS")
        print(f"PTX evaluator timing: {node.time}")


if __name__ == "__main__":
    main()
