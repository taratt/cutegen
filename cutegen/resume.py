"""Resume interrupted kernel searches from saved_nodes directories.

Used when a run died on API/timeout (or similar) and left either:
- an empty stub (best_time=999999, no JSONs), or
- some PASS nodes but depth < MAX_DEPTH.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from cutegen.config import (
    ALWAYS_OPTIMIZE_BEST,
    CUTLASS_INCLUDE_PATH,
    KERNEL_BACKEND,
    MAX_DEPTH,
)
from cutegen.node import ErrorType, Node


def _mean_time(time_obj: Any, default: float = 999999.0) -> float:
    if time_obj is None:
        return default
    if isinstance(time_obj, (int, float)):
        return float(time_obj)
    if isinstance(time_obj, dict):
        return float(time_obj.get("mean", default))
    return default


def _is_pass(error_type: Any) -> bool:
    if error_type == ErrorType.PASS or error_type == 1:
        return True
    if isinstance(error_type, str):
        return error_type in {"PASS", "ErrorType.PASS"}
    return False


def load_saved_node_dicts(save_folder: str | Path) -> list[dict]:
    folder = Path(save_folder)
    nodes: list[dict] = []
    if not folder.is_dir():
        return nodes
    for path in sorted(folder.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            print(f"Skipping unreadable node {path}: {exc}")
            continue
        data["_path"] = str(path)
        nodes.append(data)
    return nodes


def summarize_kernel_dir(save_folder: str | Path) -> dict:
    folder = Path(save_folder)
    best = None
    best_path = folder / "best_time.txt"
    if best_path.exists():
        try:
            best = float(best_path.read_text().strip())
        except Exception:
            best = best_path.read_text().strip()

    node_dicts = load_saved_node_dicts(folder)
    passes = [d for d in node_dicts if _is_pass(d.get("error_type"))]
    max_pass_depth = max((int(d.get("depth", -1)) for d in passes), default=-1)
    best_pass = None
    if passes:
        best_pass = min(passes, key=lambda d: _mean_time(d.get("time")))

    placeholder = isinstance(best, float) and best >= 999998.0
    incomplete = False
    if not node_dicts and (placeholder or best is None):
        incomplete = True
    elif passes and max_pass_depth < MAX_DEPTH:
        incomplete = True
    elif node_dicts and not passes:
        incomplete = True

    return {
        "path": str(folder),
        "kernel": folder.name,
        "best": best,
        "n_json": len(node_dicts),
        "n_pass": len(passes),
        "max_pass_depth": max_pass_depth,
        "placeholder_best": placeholder,
        "incomplete": incomplete,
        "best_pass_uuid": (best_pass or {}).get("uuid"),
        "best_pass_time": _mean_time((best_pass or {}).get("time")) if best_pass else None,
    }


def backend_codegen_addendum(backend: str | None = None) -> str:
    backend = (backend or KERNEL_BACKEND or "cuda").lower()
    if backend == "ptx":
        return (
            "Use only PTX loaded through cutegen.ptx_runtime.PtxModule. "
            "Preserve the validate_generated_code hook and do not emit CUDA C++."
        )
    if backend == "cute":
        return (
            "Your task is to optimize using CUTE framework. If the operation can be implement "
            "using CUTE operators, USE CUTE instead of writting CUDA from scratch! If cute layout "
            "and tensors allow more optimization, use them. DO NOT under any circumstances generate "
            "CUTLASS templated code. The include cute location is found in "
            f"{CUTLASS_INCLUDE_PATH}/cute/. You should add this include path directly in code in "
            "load_inline. DO NOT read from environment variables. Start with generating the simplest "
            "implementation in CuTe that is correct. For convolution-like kernels, do NOT start from "
            "a naive direct one-thread-per-output kernel when output spatial dimensions are large; "
            "start from a cooperative tiled implementation in CUTE instead, but do not jump immediately "
            "to a fragile implicit-GEMM rewrite. Pay close attention to the correct convolution kernel "
            "example given to you. Notice that CUTE tuples don’t support operator[]; you must use "
            "cute::get<Idx>(...). Pay close attention to the matrix operands dimensions and how they "
            "are compared to each other and base your implementation on what suits best for those "
            "relations. If there are a sequence of operations, you can try fusing them in the kernel."
        )
    if backend == "triton":
        return (
            "Use only a custom @triton.jit kernel and Python launch glue. "
            "Do not emit CUDA C++, CuTe, PTX text, torch.compile, or "
            "PyTorch computational fallbacks. Preserve the full reference "
            "semantics in one Triton kernel, use explicit masks and "
            "strides, choose a fixed reproducible launch configuration, "
            "and preserve the validate_generated_code JIT-validation hook."
        )
    return (
        "Your task is to optimize using CUDA. DO NOT under any "
        "circumstances generate CUTLASS templated code. Start with generating the simplest "
        "implementation in CUDA that is correct. Pay close "
        "attention to the matrix operands dimensions and how they are "
        "compared to each other and base your implementation on what "
        "suits best for those relations. If there are a sequence of "
        "operations, you can try fusing them in the kernel."
    )


def _pick_continuation_baseline(node_dicts: list[dict]) -> tuple[dict | None, dict | None]:
    """Return (deepest_pass, best_time_pass)."""
    passes = [d for d in node_dicts if _is_pass(d.get("error_type"))]
    if not passes:
        return None, None
    deepest = max(passes, key=lambda d: (int(d.get("depth", -1)), -_mean_time(d.get("time"))))
    best = min(passes, key=lambda d: _mean_time(d.get("time")))
    return deepest, best


def build_resume_node(
    save_folder: str | Path,
    ref: str,
    *,
    backend: str | None = None,
    force_from_scratch: bool = False,
) -> Node | None:
    """
    Build a Coordinator starting node that continues this kernel directory.

    Returns:
      - depth-0 empty node if no PASS exists (fresh start / empty stub)
      - depth-(max_pass+1) optimize node if PASSes exist and max_pass < MAX_DEPTH
      - None if already complete (unless force_from_scratch)
    """
    folder = Path(save_folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    backend = (backend or KERNEL_BACKEND or "cuda").lower()
    node_dicts = load_saved_node_dicts(folder)

    if force_from_scratch or not node_dicts:
        node = Node(ref=ref, src="", save_folder_path=str(folder), depth=0)
        node.metadata["kernel_backend"] = backend
        node.metadata["resumed_from"] = "scratch"
        return node

    deepest_pass, best_pass = _pick_continuation_baseline(node_dicts)
    if deepest_pass is None:
        # JSONs exist but none passed — restart from depth 0.
        node = Node(ref=ref, src="", save_folder_path=str(folder), depth=0)
        node.metadata["kernel_backend"] = backend
        node.metadata["resumed_from"] = "no_pass_restart"
        return node

    last_depth = int(deepest_pass.get("depth", 0))
    if last_depth >= MAX_DEPTH:
        print(
            f"{folder.name}: already at max PASS depth {last_depth} "
            f"(MAX_DEPTH={MAX_DEPTH}); nothing to resume"
        )
        return None

    meta_src = deepest_pass.get("metadata") or {}
    best_meta = (best_pass or {}).get("metadata") or {}

    # Prefer recorded best_passed_* when present; fall back to computed.
    best_src = (
        meta_src.get("best_passed_src")
        or best_meta.get("best_passed_src")
        or (best_pass or {}).get("src")
        or deepest_pass.get("src")
        or ""
    )
    best_depth = int(
        meta_src.get("best_passed_depth")
        or best_meta.get("best_passed_depth")
        or (best_pass or {}).get("depth")
        or last_depth
    )
    best_time = (
        meta_src.get("best_passed_time")
        or best_meta.get("best_passed_time")
        or (best_pass or {}).get("time")
    )
    best_nsight = (
        meta_src.get("best_passed_nsight_metrics")
        or best_meta.get("best_passed_nsight_metrics")
        or (best_pass or {}).get("metadata", {}).get("nsight_metrics")
        or {}
    )

    last_src = deepest_pass.get("src") or meta_src.get("last_passed_src") or best_src
    if ALWAYS_OPTIMIZE_BEST and best_src:
        prev_src = best_src
        prev_time = best_time
        prev_nsight = best_nsight
    else:
        prev_src = last_src
        prev_time = deepest_pass.get("time") or meta_src.get("previous_src_time")
        prev_nsight = meta_src.get("nsight_metrics") or meta_src.get("prev_nsight_metrics") or {}

    next_depth = last_depth + 1
    node = Node(
        ref=ref,
        src="",
        prev_src=prev_src,
        ref_time=deepest_pass.get("ref_time") or best_pass.get("ref_time"),
        save_folder_path=str(folder),
        depth=next_depth,
    )
    node.metadata = {
        "kernel_backend": backend,
        "resumed_from": deepest_pass.get("uuid"),
        "resumed_from_path": deepest_pass.get("_path"),
        "last_passed_src": last_src,
        "last_passed_depth": last_depth,
        "best_passed_src": best_src,
        "best_passed_depth": best_depth,
        "best_passed_time": best_time,
        "best_passed_nsight_metrics": best_nsight,
        "previous_src_time": prev_time,
        "prev_nsight_metrics": prev_nsight,
        "retries_by_depth": dict(meta_src.get("retries_by_depth") or {}),
        "transient_retries": 0,
    }
    print(
        f"{folder.name}: resume from PASS {deepest_pass.get('uuid')} "
        f"(depth {last_depth}) -> continue at depth {next_depth} "
        f"(best so far depth={best_depth}, time={_mean_time(best_time)})"
    )
    return node


def find_kernel_dir(save_dir_base: str | Path, kernel_id: int) -> Path | None:
    base = Path(save_dir_base)
    if not base.is_dir():
        return None
    matches = sorted(
        p for p in base.iterdir() if p.is_dir() and re.match(rf"^{kernel_id}_", p.name)
    )
    return matches[0] if matches else None


def list_incomplete_kernel_dirs(save_dir_base: str | Path) -> list[dict]:
    base = Path(save_dir_base)
    if not base.is_dir():
        return []
    out = []
    for path in sorted(base.iterdir()):
        if not path.is_dir():
            continue
        if not re.match(r"^\d+_", path.name):
            continue
        summary = summarize_kernel_dir(path)
        if summary["incomplete"]:
            out.append(summary)
    return out
