#!/usr/bin/env python3
"""Launch a cute→PTX handoff experiment (opt-in; does not change defaults).

Depth 0: CuTe generation + eval.
After first PASS: extract PTX from the build, LLM-translate to PtxModule form,
then continue depths as PTX optimization.

Example:
  KERNEL_BACKEND=cute CUTE_TO_PTX=1 USE_PROFILING=false \\
  CUTEGEN_KERNEL_IDS=21 \\
  CUTEGEN_SAVE_DIR_BASE=$PWD/saved_nodes/cute-ptx/level1-nopf-sonnet5 \\
  python -u -m cutegen.main
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("KERNEL_BACKEND", "cute")
    os.environ["CUTE_TO_PTX"] = "1"
    os.environ.setdefault("USE_PROFILING", "false")
    os.environ.setdefault("CUTEGEN_BASE_PATH", str(project_root))
    os.environ.setdefault(
        "CUTLASS_BASE_PATH",
        str(project_root / "cutegen" / "cutlass"),
    )
    os.environ.setdefault(
        "CUTLASS_INCLUDE_PATH",
        str(project_root / "cutegen" / "cutlass" / "include"),
    )
    os.environ.setdefault(
        "CUTEGEN_SAVE_DIR_BASE",
        str(project_root / "saved_nodes" / "cute-ptx" / "level1-nopf"),
    )
    if "CUTEGEN_KERNEL_IDS" not in os.environ:
        print("Set CUTEGEN_KERNEL_IDS (e.g. 21) before running.", file=sys.stderr)
        sys.exit(2)

    # Import after env is set so config picks KERNEL_BACKEND/CUTE_TO_PTX.
    from cutegen.backend_runtime import cute_to_ptx_enabled
    from cutegen.config import KERNEL_BACKEND

    if not cute_to_ptx_enabled():
        print(
            f"cute→PTX not enabled (KERNEL_BACKEND={KERNEL_BACKEND!r}). "
            "Export KERNEL_BACKEND=cute and CUTE_TO_PTX=1.",
            file=sys.stderr,
        )
        sys.exit(2)

    print(
        f"cute→PTX run: backend={KERNEL_BACKEND} "
        f"ids={os.environ['CUTEGEN_KERNEL_IDS']} "
        f"save={os.environ['CUTEGEN_SAVE_DIR_BASE']}"
    )
    os.chdir(project_root)
    os.execvp(sys.executable, [sys.executable, "-u", "-m", "cutegen.main"])


if __name__ == "__main__":
    main()
