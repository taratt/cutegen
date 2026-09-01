#!/usr/bin/env python3
"""Resume kernel searches that stopped on API/timeout (or similar) mid-run.

Examples:
  # Resume specific kernels in a Triton no-profile tree
  KERNEL_BACKEND=triton USE_PROFILING=false \\
  CUTEGEN_SAVE_DIR_BASE=$PWD/saved_nodes/triton/level1-no-profile-sonnet5 \\
  python -u scripts/resume_interrupted_kernels.py --kernel-ids 70,61,107,6,14,50

  # Auto-pick incomplete dirs under the save base
  python -u scripts/resume_interrupted_kernels.py --auto

  # Dry-run: show what would be resumed
  python -u scripts/resume_interrupted_kernels.py --auto --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# Allow `python scripts/...` without editable install quirks.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _bootstrap_env_from_argv() -> None:
    """Set KERNEL_BACKEND before cutegen.config is imported (it reads env once)."""
    # Lightweight pre-parse so --backend / env win before config import.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--backend", default=None)
    known, _ = pre.parse_known_args()
    backend = (known.backend or os.environ.get("KERNEL_BACKEND") or "cuda").lower()
    os.environ["KERNEL_BACKEND"] = backend
    os.environ.setdefault("CUTEGEN_BASE_PATH", str(PROJECT_ROOT))


_bootstrap_env_from_argv()

from cutegen.config import CUTEGEN_BASE_PATH, KERNEL_BACKEND, MAX_DEPTH  # noqa: E402
from cutegen.coordinator import Coordinator  # noqa: E402
from cutegen.llm_api import save_token_usage_csv, set_current_kernel_name  # noqa: E402
from cutegen.resume import (  # noqa: E402
    backend_codegen_addendum,
    build_resume_node,
    find_kernel_dir,
    list_incomplete_kernel_dirs,
    summarize_kernel_dir,
)
from cutegen.util import read_file  # noqa: E402


LEVEL1_DIR = Path(CUTEGEN_BASE_PATH) / "KernelBench" / "level1"


def _parse_ids(raw: str | None) -> list[int]:
    if not raw:
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _resolve_ref(kernel_dirname: str) -> str:
    ref_path = LEVEL1_DIR / kernel_dirname
    if not ref_path.is_file():
        # Tolerate slight naming drift: match by leading id.
        m = re.match(r"^(\d+)_", kernel_dirname)
        if not m:
            raise FileNotFoundError(f"No KernelBench ref for {kernel_dirname}")
        kid = int(m.group(1))
        matches = sorted(LEVEL1_DIR.glob(f"{kid}_*.py"))
        if not matches:
            raise FileNotFoundError(f"No KernelBench file for id {kid}")
        ref_path = matches[0]
    return read_file(str(ref_path))


def _configure_coordinator(coordinator: Coordinator, backend: str) -> None:
    coordinator.codegen_initial_addendum = backend_codegen_addendum(backend)


def resume_one(
    save_folder: Path,
    *,
    backend: str,
    dry_run: bool,
    force_from_scratch: bool,
) -> str:
    summary = summarize_kernel_dir(save_folder)
    print(
        f"\n=== {save_folder.name}: jsons={summary['n_json']} "
        f"pass={summary['n_pass']} max_pass_depth={summary['max_pass_depth']} "
        f"best={summary['best']} incomplete={summary['incomplete']} ==="
    )
    if not summary["incomplete"] and not force_from_scratch:
        print("  skip: looks complete")
        return "skipped_complete"

    ref = _resolve_ref(save_folder.name)
    node = build_resume_node(
        save_folder,
        ref,
        backend=backend,
        force_from_scratch=force_from_scratch,
    )
    if node is None:
        return "skipped_complete"

    print(
        f"  starting node depth={node.depth} uuid={node.uuid} "
        f"resumed_from={node.metadata.get('resumed_from')}"
    )
    if dry_run:
        return "dry_run"

    set_current_kernel_name(save_folder.name)
    coordinator = Coordinator([node])
    _configure_coordinator(coordinator, backend)
    coordinator.run()
    return "ran"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Continue LLM optimize/eval for kernels interrupted by API/timeout "
            f"(continues until MAX_DEPTH={MAX_DEPTH})."
        )
    )
    p.add_argument(
        "--save-dir",
        default=os.environ.get(
            "CUTEGEN_SAVE_DIR_BASE",
            str(Path(CUTEGEN_BASE_PATH) / "saved_nodes" / KERNEL_BACKEND / "level1"),
        ),
        help="Experiment directory containing per-kernel folders.",
    )
    p.add_argument(
        "--kernel-ids",
        default=os.environ.get("CUTEGEN_KERNEL_IDS", ""),
        help="Comma-separated kernel ids to resume (e.g. 70,61,107).",
    )
    p.add_argument(
        "--auto",
        action="store_true",
        help="Resume every incomplete kernel dir under --save-dir.",
    )
    p.add_argument(
        "--backend",
        default=os.environ.get("KERNEL_BACKEND", KERNEL_BACKEND),
        choices=("cuda", "cute", "ptx", "triton"),
        help="Kernel backend (must match the save tree; sets prompts via env).",
    )
    p.add_argument(
        "--force-from-scratch",
        action="store_true",
        help="Ignore existing PASSes and restart at depth 0 in the same folder.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resume plan only; do not call the LLM.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir).resolve()
    backend = args.backend
    if backend != KERNEL_BACKEND:
        raise SystemExit(
            f"--backend={backend} but config loaded KERNEL_BACKEND={KERNEL_BACKEND}. "
            "Set KERNEL_BACKEND in the environment (or pass --backend before import "
            "via the script's argv bootstrap)."
        )

    print(
        f"Resume backend={backend} MAX_DEPTH={MAX_DEPTH} save_dir={save_dir}",
        flush=True,
    )

    targets: list[Path] = []
    if args.auto:
        incomplete = list_incomplete_kernel_dirs(save_dir)
        print(f"Found {len(incomplete)} incomplete kernel dirs under {save_dir}")
        for item in incomplete:
            print(
                f"  - {item['kernel']}: pass={item['n_pass']} "
                f"max_pass_depth={item['max_pass_depth']} best={item['best']}"
            )
            targets.append(Path(item["path"]))
    else:
        ids = _parse_ids(args.kernel_ids)
        if not ids:
            raise SystemExit(
                "Provide --kernel-ids or --auto "
                "(or set CUTEGEN_KERNEL_IDS)."
            )
        for kid in ids:
            found = find_kernel_dir(save_dir, kid)
            if found is None:
                # Create folder name from KernelBench file so empty stubs can start.
                matches = sorted(LEVEL1_DIR.glob(f"{kid}_*.py"))
                if not matches:
                    print(f"WARNING: no KernelBench file for id {kid}; skipping")
                    continue
                found = save_dir / matches[0].name
                if args.dry_run:
                    print(f"Would create empty save dir {found}")
                else:
                    found.mkdir(parents=True, exist_ok=True)
                    print(f"Created empty save dir {found}")
            targets.append(found)

    if not targets:
        raise SystemExit("No kernels to resume.")

    results = {}
    for folder in targets:
        try:
            results[folder.name] = resume_one(
                folder,
                backend=backend,
                dry_run=args.dry_run,
                force_from_scratch=args.force_from_scratch,
            )
        except Exception as exc:
            print(f"ERROR resuming {folder.name}: {exc}")
            results[folder.name] = f"error:{exc}"

    if not args.dry_run:
        try:
            save_token_usage_csv()
        except Exception as exc:
            print(f"WARNING: could not flush token usage CSV: {exc}")

    print("\n=== Resume summary ===")
    for name, status in results.items():
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
