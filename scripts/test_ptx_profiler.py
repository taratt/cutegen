"""Nsight Compute integration test for Driver-API PTX kernels."""

import tempfile

import torch

from cutegen.evaluate import run_nsight_profile
from scripts.smoke_test_ptx_runtime import PTX_SOURCE
from scripts.test_ptx_evaluator import REFERENCE_SOURCE, generated_source


def main() -> None:
    metadata = {}
    with tempfile.TemporaryDirectory(prefix="cutegen_ptx_ncu_") as directory:
        result = run_nsight_profile(
            REFERENCE_SOURCE,
            generated_source(PTX_SOURCE),
            metadata,
            build_directory=directory,
            device=torch.device("cuda:0"),
            num_warmups=1,
            num_iters=1,
        )

    if result is None or metadata.get("profile_returncode") != 0:
        raise AssertionError(f"PTX Nsight profiling failed: {metadata}")
    metrics = metadata.get("nsight_metrics", {})
    if metrics.get("kernel") != "vector_add_f32":
        raise AssertionError(f"PTX kernel name was not parsed: {metrics}")

    print("PTX Nsight profiling: PASS")
    print(f"PTX Nsight kernel: {metrics['kernel']}")


if __name__ == "__main__":
    main()
