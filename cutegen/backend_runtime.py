"""Per-node backend helpers for optional cute→PTX handoff.

Defaults preserve existing behavior. Activation requires:
  KERNEL_BACKEND=cute
  CUTE_TO_PTX=1
"""

from __future__ import annotations

import os
from typing import Any

from cutegen.config import CUTEGEN_BASE_PATH, KERNEL_BACKEND

CUTE_TO_PTX = os.environ.get("CUTE_TO_PTX", "").lower() in {"1", "true", "yes", "on"}

_BACKEND_PROMPTS = {
    "cuda": (
        "cuda_initial_kernelbench_prompt.txt",
        "cuda_only_optimization_prompt.txt",
        "cuda_coding_debugging_guide.txt",
    ),
    "cute": (
        (
            "cute_initial_kernelbench_prompt_minimal.txt"
            if os.environ.get("CUTE_INITIAL_PROMPT", "full").lower() == "minimal"
            else "cute_initial_kernelbench_prompt.txt"
        ),
        "cute_optimized_kernelbench_prompt.txt",
        "cuda_coding_debugging_guide.txt",
    ),
    "ptx": (
        "ptx_initial_kernelbench_prompt.txt",
        "ptx_optimization_prompt.txt",
        "ptx_coding_debugging_guide_sm89.txt",
    ),
    "triton": (
        "triton_initial_kernelbench_prompt.txt",
        "triton_optimization_prompt.txt",
        "triton_coding_debugging_guide.txt",
    ),
}


def cute_to_ptx_enabled() -> bool:
    flag = os.environ.get("CUTE_TO_PTX", "").lower() in {"1", "true", "yes", "on"}
    # Prefer live env so launcher scripts that set env before import still work
    # if this helper is re-checked later; fall back to imported config.
    backend = os.environ.get("KERNEL_BACKEND", KERNEL_BACKEND).lower()
    return flag and backend == "cute"


def effective_backend(metadata: dict[str, Any] | None = None) -> str:
    """Backend used for eval/codegen/fix.

    When cute→PTX is off, always return the process KERNEL_BACKEND.
    When on, prefer node.metadata['kernel_backend'] after the handoff.
    """
    process_backend = os.environ.get("KERNEL_BACKEND", KERNEL_BACKEND).lower()
    if not cute_to_ptx_enabled():
        return process_backend if process_backend in _BACKEND_PROMPTS else KERNEL_BACKEND
    if metadata:
        backend = str(metadata.get("kernel_backend") or "").lower()
        if backend in _BACKEND_PROMPTS:
            return backend
    return "cute" if process_backend == "cute" else process_backend


def prompt_files_for_backend(backend: str) -> tuple[str, str, str]:
    initial, optimize, debug = _BACKEND_PROMPTS[backend]
    base = f"{CUTEGEN_BASE_PATH}/cutegen/prompts"
    return (
        f"{base}/{initial}",
        f"{base}/{optimize}",
        f"{base}/{debug}",
    )


def ptx_optimize_addendum() -> str:
    return (
        "Continue optimizing as NVIDIA PTX loaded through cutegen.ptx_runtime."
        " PtxModule. Preserve validate_generated_code. Do not emit CUDA C++, "
        "CuTe, load_inline, or CUTLASS."
    )
