"""Nsight Compute integration test for Triton JIT kernels."""

import tempfile

import torch

from cutegen.evaluate import run_nsight_profile
from scripts.test_triton_evaluator import GENERATED_SOURCE, REFERENCE_SOURCE


def main() -> None:
    metadata = {}
    with tempfile.TemporaryDirectory(prefix="cutegen_triton_ncu_") as directory:
        result = run_nsight_profile(
            REFERENCE_SOURCE,
            GENERATED_SOURCE,
            metadata,
            build_directory=directory,
            device=torch.device("cuda:0"),
            num_warmups=1,
            num_iters=1,
        )

    if result is None or metadata.get("profile_returncode") != 0:
        raise AssertionError(f"Triton Nsight profiling failed: {metadata}")
    metrics = metadata.get("nsight_metrics", {})
    if not metrics.get("kernel"):
        raise AssertionError(f"Triton kernel name was not parsed: {metrics}")

    print("Triton Nsight profiling: PASS")
    print(f"Triton Nsight kernel: {metrics['kernel']}")


if __name__ == "__main__":
    main()
