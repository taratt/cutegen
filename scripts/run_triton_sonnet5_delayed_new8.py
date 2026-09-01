"""Run the new-8 kernels for Sonnet 5 Triton delayed profiling."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

KERNEL_BATCHES = (
    ("matmul-attention", (6, 14), 6),
    ("convolution", (50, 61, 70, 75, 107), 6),
    ("reduction-norm", (105,), 5),
)
TOKEN_USAGE_CSV = "sonnet5_token_usage_triton_delayed_new8.csv"


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY must be set before starting the run.")

    project_root = Path(__file__).resolve().parents[1]
    base_env = os.environ.copy()
    base_env["CUTEGEN_BASE_PATH"] = str(project_root)
    base_env["KERNEL_BACKEND"] = "triton"
    base_env["USE_PROFILING"] = "true"
    base_env["CUTEGEN_SAVE_DIR_BASE"] = str(
        project_root / "saved_nodes" / "triton" / "level1-profiled-sonnet5"
    )
    base_env["TOKEN_USAGE_CSV_PATH"] = str(project_root / TOKEN_USAGE_CSV)

    for label, kernel_ids, profiling_start_depth in KERNEL_BATCHES:
        env = base_env.copy()
        env["CUTEGEN_KERNEL_IDS"] = ",".join(str(k) for k in kernel_ids)
        env["PROFILING_START_DEPTH"] = str(profiling_start_depth)
        print(
            f"\n=== Triton Sonnet delayed {label}: "
            f"kernels={env['CUTEGEN_KERNEL_IDS']} "
            f"profiling_start_depth={profiling_start_depth} ===",
            flush=True,
        )
        subprocess.run(
            [sys.executable, "-u", "-m", "cutegen.main"],
            cwd=project_root,
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
