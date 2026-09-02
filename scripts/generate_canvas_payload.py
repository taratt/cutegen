#!/usr/bin/env python3
"""Scan saved_nodes and emit JSON payload for Cutegen experiment canvases."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_selection import (
    ALL27,
    COHORT25,
    NEW8,
    SAMPLE19,
    experiment_method,
    experiment_model,
    model_tag as model_tag_for,
    paths_in_progress,
    paths_superseded_by_minimal,
)
from token_usage_agg import build_tokens_by_col

ROOT = Path(__file__).resolve().parents[1] / "saved_nodes"
ALL_KERNELS = ALL27
MAX_DEPTH = 10
SENTINEL = 1e5

KERNEL_TYPE = {
    1: "matmul",
    4: "matmul",
    9: "matmul",
    6: "matmul",
    14: "matmul",
    21: "activation",
    22: "activation",
    88: "activation",
    33: "norm",
    40: "norm",
    105: "norm",
    49: "reduction",
    53: "reduction",
    54: "conv",
    55: "conv",
    58: "conv",
    59: "conv",
    61: "conv",
    70: "conv",
    75: "conv",
    80: "conv",
    83: "conv",
    103: "conv",
    107: "conv",
    50: "conv",
    99: "loss",
    102: "attention",
}

KERNEL_NAMES: dict[int, str] = {}

# Verified I/O precision cheats: fp32 reference but generated kernel uses lower precision.
PRECISION_IO_CHEATS = [
    {
        "path": "cute/level1-no-profile-sonnet5",
        "kernel": 49,
        "note": "AT_DISPATCH kHalf/kBFloat16 on fp32 Max reduction ref",
    },
    {
        "path": "triton/level1-no-profile-sonnet5",
        "kernel": 33,
        "note": "fp16 compute path on fp32 BatchNorm ref",
    },
]


def parse_time(v):
    if v is None:
        return None
    if isinstance(v, dict):
        for k in ("mean", "min"):
            if k in v and v[k] is not None:
                try:
                    t = float(v[k])
                    return None if t >= SENTINEL else t
                except (TypeError, ValueError):
                    pass
        return None
    try:
        t = float(v)
        return None if t >= SENTINEL else t
    except (TypeError, ValueError):
        return None


def kid(name: str) -> int | None:
    m = re.match(r"^(\d+)_", name)
    return int(m.group(1)) if m else None


def label_experiment(exp: Path) -> dict:
    rel = exp.relative_to(ROOT)
    parts = list(rel.parts)
    backend = parts[0]
    leaf = parts[-1]
    rel_s = str(rel)
    model = experiment_model(rel_s, leaf, backend)
    method = experiment_method(leaf)
    backend_label = {
        "cuda": "CUDA",
        "cute": "CuTe",
        "cute-ptx": "CuTe→PTX",
        "ptx": "PTX",
        "triton": "Triton",
    }.get(backend, backend.upper())
    if method == "Minimal":
        tag = "min"
    elif method == "Minimal from start":
        tag = "minstart"
    elif method == "No profiling":
        tag = "nopf"
    elif method == "Delayed profiling":
        tag = "delay"
    elif method == "Profiling from start":
        tag = "start"
    else:
        tag = leaf.replace("level1-", "")
    mtag = model_tag_for(model)
    col = f"{backend_label}/{tag}/{mtag}"
    label = f"{backend_label} · {method} · {model}"
    return {
        "col": col,
        "path": rel_s,
        "label": label,
        "backend": backend_label,
        "method": method,
        "model": model,
        "leaf": leaf,
    }


def curve_for_kernel(ker_dir: Path) -> list[float | None]:
    by_depth: dict[int, float | None] = {}
    ref = None
    for jp in ker_dir.glob("*.json"):
        try:
            j = json.loads(jp.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        d = j.get("depth")
        if d is None or d > MAX_DEPTH:
            continue
        r = parse_time(j.get("ref_time"))
        if r is not None:
            ref = r
        meta = j.get("metadata") or {}
        gen = parse_time(j.get("time"))
        compile_err = bool(str(meta.get("compile") or "").strip())
        correct_err = bool(str(meta.get("correct") or "").strip())
        ok = gen is not None and not compile_err and not correct_err and ref and gen > 0
        sp = round(ref / gen, 4) if ok else None
        prev = by_depth.get(d)
        if prev is None or (sp is not None and (prev is None or sp > prev)):
            by_depth[d] = sp
    k = kid(ker_dir.name)
    if k is not None:
        KERNEL_NAMES[k] = ker_dir.name.replace(".py", "")
    return [by_depth.get(d) for d in range(MAX_DEPTH + 1)]


def main() -> None:
    collected: list[tuple[dict, dict[str, list[float | None]]]] = []
    seen_cols: set[str] = set()

    for exp in sorted(
        {bt.parent.parent for bt in ROOT.rglob("best_time.txt") if re.match(r"^\d+_", bt.parent.name)},
        key=str,
    ):
        if exp.name == "kimi-k3":
            continue
        meta = label_experiment(exp)
        col = meta["col"]
        if col in seen_cols:
            # disambiguate duplicate col with leaf suffix
            col = f"{col}@{meta['leaf']}"
            meta = {**meta, "col": col}
        seen_cols.add(col)
        exp_curves: dict[str, list[float | None]] = {}
        for ker in exp.iterdir():
            if not ker.is_dir() or not re.match(r"^\d+_", ker.name):
                continue
            k = kid(ker.name)
            if k is None:
                continue
            points = curve_for_kernel(ker)
            if any(p is not None for p in points):
                exp_curves[str(k)] = points
        if exp_curves:
            collected.append((meta, exp_curves))

    available_paths = {meta["path"] for meta, _ in collected}
    exclude_paths = paths_superseded_by_minimal(available_paths) | set(paths_in_progress())

    exps: list[dict] = []
    curves: dict[str, dict[str, list[float | None]]] = {}
    for meta, exp_curves in collected:
        if meta["path"] in exclude_paths:
            continue
        exps.append({k: meta[k] for k in ("col", "path", "label", "backend", "method", "model", "leaf")})
        curves[meta["col"]] = exp_curves

    kernel_meta = []
    for k in ALL_KERNELS:
        kernel_meta.append(
            {
                "id": k,
                "name": KERNEL_NAMES.get(k, f"{k}_"),
                "ktype": KERNEL_TYPE.get(k, "other"),
                "cohort": "sample19" if k in SAMPLE19 else ("new8" if k in NEW8 else "other"),
            }
        )

    tokens = build_tokens_by_col(exps)

    path_to_col = {e["path"]: e["col"] for e in exps}
    precision_cheats = []
    for cheat in PRECISION_IO_CHEATS:
        col = path_to_col.get(cheat["path"])
        if not col:
            continue
        precision_cheats.append(
            {
                "col": col,
                "kernel": cheat["kernel"],
                "path": cheat["path"],
                "note": cheat["note"],
            }
        )

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "maxDepth": MAX_DEPTH,
        "depthLabels": [str(i) for i in range(MAX_DEPTH + 1)],
        "sample19Ids": SAMPLE19,
        "new8Ids": NEW8,
        "all27Ids": ALL27,
        "cohort25Ids": COHORT25,
        "kernelTypes": sorted(set(KERNEL_TYPE.values())),
        "exps": exps,
        "curves": curves,
        "kernels": kernel_meta,
        "tokens": tokens,
        "precisionCheats": precision_cheats,
    }
    out = Path("/tmp/cutegen_canvas_payload.json")
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    skipped = sorted(paths_in_progress() & available_paths)
    if skipped:
        print(f"excluded in-progress ({len(skipped)}):")
        for p in skipped:
            print(f"  - {p}")
    for e in exps:
        print(f"  {e['col']:28} {e['path']}")


if __name__ == "__main__":
    main()
