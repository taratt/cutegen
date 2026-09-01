"""Run the additional KernelBench sample with Sonnet 5 and no profiling."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


KERNEL_IDS = (6, 105, 75, 14, 70, 61, 107, 50)
BACKENDS = ("cuda", "ptx", "cute")
TOKEN_USAGE_FILES = {
    "cuda": "sonnet5_token_usage_cuda_nopf.csv",
    "ptx": "sonnet5_token_usage_ptx_nopf.csv",
    "cute": "sonnet5_token_usage_cute_nopf.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run kernels 6, 105, 75, 14, 70, 61, 107, and 50 with "
            "Sonnet 5 on CUDA, PTX, and CuTe without profiling."
        )
    )
    parser.add_argument(
        "backends",
        nargs="*",
        default=list(BACKENDS),
        metavar="BACKEND",
        help="Backends to run in order (default: cuda ptx cute).",
    )
    args = parser.parse_args()
    invalid_backends = [
        backend for backend in args.backends if backend not in BACKENDS
    ]
    if invalid_backends:
        parser.error(
            f"invalid backends: {invalid_backends}; "
            f"choose from {', '.join(BACKENDS)}"
        )
    return args


def validate_setup(project_root: Path) -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY must be set before starting the run.")

    level1_directory = project_root / "KernelBench" / "level1"
    available_ids = {
        int(path.name.split("_", 1)[0])
        for path in level1_directory.glob("*.py")
        if path.name.split("_", 1)[0].isdigit()
    }
    missing_ids = sorted(set(KERNEL_IDS) - available_ids)
    if missing_ids:
        raise SystemExit(f"Missing KernelBench level1 kernel IDs: {missing_ids}")

    cutlass_include = project_root / "cutegen" / "cutlass" / "include"
    if not cutlass_include.is_dir():
        raise SystemExit(f"CUTLASS include directory is missing: {cutlass_include}")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    validate_setup(project_root)

    kernel_ids = ",".join(str(kernel_id) for kernel_id in KERNEL_IDS)
    base_env = os.environ.copy()
    base_env["CUTEGEN_BASE_PATH"] = str(project_root)
    base_env["CUTLASS_BASE_PATH"] = base_env.get(
        "CUTLASS_BASE_PATH", str(project_root / "cutegen" / "cutlass")
    )
    base_env["CUTLASS_INCLUDE_PATH"] = base_env.get(
        "CUTLASS_INCLUDE_PATH",
        str(project_root / "cutegen" / "cutlass" / "include"),
    )
    base_env["CUTEGEN_KERNEL_IDS"] = kernel_ids
    base_env["USE_PROFILING"] = "false"

    for backend in args.backends:
        env = base_env.copy()
        env["KERNEL_BACKEND"] = backend
        env["CUTEGEN_SAVE_DIR_BASE"] = str(
            project_root / "saved_nodes" / backend / "level1-no-profile-sonnet5"
        )
        env["TOKEN_USAGE_CSV_PATH"] = str(
            project_root / TOKEN_USAGE_FILES[backend]
        )

        print(
            f"\n=== Sonnet 5 no-profile: backend={backend} "
            f"kernels={kernel_ids} "
            f"save_dir={env['CUTEGEN_SAVE_DIR_BASE']} ===",
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
