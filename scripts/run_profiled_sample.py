"""Run each kernel category with its own profiling start depth."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


KERNEL_GROUPS = {
    "matmul-attention": {
        "kernel_ids": (1, 4, 9, 102),
        "profiling_start_depth": 6,
    },
    "activation": {
        "kernel_ids": (21, 22, 88),
        "profiling_start_depth": 2,
    },
    "convolution": {
        "kernel_ids": (54, 55, 58, 59, 80, 83, 103),
        "profiling_start_depth": 6,
    },
    "reduction-norm": {
        "kernel_ids": (33, 40, 49, 53),
        "profiling_start_depth": 5,
    },
    "loss": {
        "kernel_ids": (99,),
        "profiling_start_depth": 3,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the sampled kernels with category-specific profiling depths."
    )
    parser.add_argument(
        "categories",
        nargs="*",
        choices=KERNEL_GROUPS,
        help="Categories to run; omit to run every category.",
    )
    parser.add_argument(
        "--backend",
        choices=("cuda", "cute", "ptx"),
        default=os.environ.get("KERNEL_BACKEND", "cuda"),
        help="Generated kernel backend (default: KERNEL_BACKEND or cuda).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    categories = args.categories or list(KERNEL_GROUPS)
    project_root = Path(__file__).resolve().parents[1]

    base_env = os.environ.copy()
    base_env["CUTEGEN_BASE_PATH"] = str(project_root)
    base_env["KERNEL_BACKEND"] = args.backend
    base_env["CUTLASS_BASE_PATH"] = base_env.get(
        "CUTLASS_BASE_PATH", str(project_root / "cutegen" / "cutlass")
    )
    base_env["CUTLASS_INCLUDE_PATH"] = base_env.get(
        "CUTLASS_INCLUDE_PATH",
        str(project_root / "cutegen" / "cutlass" / "include"),
    )
    base_env["CUTEGEN_SAVE_DIR_BASE"] = base_env.get(
        "CUTEGEN_SAVE_DIR_BASE",
        str(project_root / "saved_nodes" / args.backend / "level1-profiled"),
    )
    base_env["USE_PROFILING"] = "true"

    for category in categories:
        group = KERNEL_GROUPS[category]
        env = base_env.copy()
        env["CUTEGEN_KERNEL_IDS"] = ",".join(
            str(kernel_id) for kernel_id in group["kernel_ids"]
        )
        env["PROFILING_START_DEPTH"] = str(group["profiling_start_depth"])

        print(
            f"\n=== Running {category}: kernels={env['CUTEGEN_KERNEL_IDS']} "
            f"profiling_start_depth={env['PROFILING_START_DEPTH']} ===",
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
