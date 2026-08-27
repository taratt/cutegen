#!/usr/bin/env python3
"""Aggregate LLM token-usage CSVs and attach them to experiment paths."""

from __future__ import annotations

import csv
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Explicit map: saved_nodes relative experiment path → token CSV filenames.
# Multiple files are summed (e.g. sample-19 + new-8 runs). Avoid overlapping
# kernel coverage across files for the same experiment.
TOKEN_CSV_BY_PATH: dict[str, list[str]] = {
    "cuda/level1-no-profile": [
        "kimi_token_usage_cuda_nopf.csv",
        "kimi_token_usage_cuda_nopf_new8.csv",
    ],
    "cuda/level1-no-profile-sonnet5": ["sonnet5_token_usage_cuda_nopf.csv"],
    "cuda/level1-profiled": ["kimi_token_usage_cuda_delayed_120.csv"],
    "cuda/level1-profiled-sonnet5": [
        "sonnet5_cuda_prof_token_usage.csv",
        "sonnet5_token_usage_cuda_profiled_new8.csv",
    ],
    "cuda/level1-profiled-from-start": ["kimi_token_usage_cuda_from_start.csv.from_10106"],
    "cuda/level1-profiled-from-start-sonnet5": [
        "sonnet5_token_usage_cuda_from_start.csv",
        "sonnet5_token_usage_cuda_from_start_new8.csv",
    ],
    "cute/level1-no-profile-sonnet5": ["sonnet5_token_usage_cute_nopf.csv"],
    "cute/level1-profiled": ["kimi_token_usage.csv"],
    "cute/level1-profiled-sonnet5": [
        "sonnet5_token_usage.csv",
        "sonnet5_token_usage_cute_profiled_new8.csv",
        "sonnet5_token_usage_cute_delayed.csv",
    ],
    "cute/level1-profiled-from-start-sonnet5": [
        "sonnet5_from_start_token_usage.csv",
        "sonnet5_token_usage_cute_from_start_new8.csv",
    ],
    "cute-ptx/level1-nopf-sonnet5": ["sonnet5_token_usage_cute_to_ptx.csv"],
    "cute-ptx/level1-profiled-from-start-sonnet5": [
        "sonnet5_token_usage_cute_to_ptx_from_start.csv.from_120",
    ],
    "ptx/level1-no-profile": ["kimi_token_usage_ptx_nopf_new8.csv"],
    "ptx/level1-no-profile-sonnet5": ["sonnet5_token_usage_ptx_nopf.csv"],
    "ptx/level1-profiled": [
        "kimi_token_usage_ptx_remote.csv",
        "kimi_token_usage_ptx_delayed.csv",
    ],
    "ptx/level1-profiled-sonnet5": [
        "sonnet5_token_usage_ptx_delayed.csv",
        "sonnet5_token_usage_ptx_profiled_new8.csv",
    ],
    "ptx/level1-profiled-from-start": ["kimi_token_usage_ptx_from_start.csv"],
    "ptx/level1-profiled-from-start-sonnet5": ["sonnet5_token_usage_ptx_from_start.csv"],
    "triton/level1-no-profile": ["kimi_token_usage_triton_nopf_new8.csv"],
    "triton/level1-no-profile-sonnet5": [
        "sonnet5_token_usage_triton_nopf_guide_v2.csv",
        "sonnet5_token_usage_triton_nopf.csv",
    ],
    "triton/level1-profiled-sonnet5": ["sonnet5_token_usage_triton_delayed.csv"],
    "triton/level1-profiled-from-start": [
        "kimi_token_usage_triton_from_start.csv.from_10707",
    ],
    "triton/level1-profiled-from-start-sonnet5": [
        "sonnet5_token_usage_triton_from_start.csv",
        "sonnet5_token_usage_triton_from_start_new8.csv",
    ],
}


def _kid(name: str) -> int | None:
    m = re.match(r"^(\d+)_", name or "")
    return int(m.group(1)) if m else None


def _as_int(v) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def _row_matches_model(row: dict, model: str | None) -> bool:
    if not model:
        return True
    server = (row.get("server_type") or "").lower()
    mname = (row.get("model") or "").lower()
    if model == "Sonnet 5":
        return "anthropic" in server or "sonnet" in mname or "claude" in mname
    if model == "Kimi K3":
        return "kimi" in server or "kimi" in mname or "moonshot" in mname
    return True


def _empty_bucket() -> dict:
    return {"input": 0, "output": 0, "total": 0, "calls": 0}


def _add(bucket: dict, inp: int, out: int, tot: int) -> None:
    bucket["input"] += inp
    bucket["output"] += out
    bucket["total"] += tot if tot else inp + out
    bucket["calls"] += 1


def load_csv_tokens(path: Path, model: str | None = None) -> dict[str, dict]:
    """Return {by_kernel: {id: counts}, all: counts, sources: [name]}."""
    by_kernel: dict[str, dict] = {}
    all_counts = _empty_bucket()
    if not path.is_file():
        return {"byKernel": by_kernel, "all": all_counts, "sources": []}

    rows: list[dict] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)

    matched = [r for r in rows if _row_matches_model(r, model)]
    use_rows = matched if matched else rows

    for row in use_rows:
        inp = _as_int(row.get("input_tokens"))
        out = _as_int(row.get("output_tokens"))
        tot = _as_int(row.get("total_tokens"))
        _add(all_counts, inp, out, tot)
        kid = _kid(row.get("kernel") or "")
        if kid is None:
            continue
        key = str(kid)
        if key not in by_kernel:
            by_kernel[key] = _empty_bucket()
        _add(by_kernel[key], inp, out, tot)

    return {"byKernel": by_kernel, "all": all_counts, "sources": [path.name]}


def merge_token_dicts(parts: list[dict]) -> dict:
    by_kernel: dict[str, dict] = {}
    all_counts = _empty_bucket()
    sources: list[str] = []
    for part in parts:
        sources.extend(part.get("sources") or [])
        a = part.get("all") or _empty_bucket()
        all_counts["input"] += a.get("input", 0)
        all_counts["output"] += a.get("output", 0)
        all_counts["total"] += a.get("total", 0)
        all_counts["calls"] += a.get("calls", 0)
        for kid, bucket in (part.get("byKernel") or {}).items():
            if kid not in by_kernel:
                by_kernel[kid] = _empty_bucket()
            by_kernel[kid]["input"] += bucket.get("input", 0)
            by_kernel[kid]["output"] += bucket.get("output", 0)
            by_kernel[kid]["total"] += bucket.get("total", 0)
            by_kernel[kid]["calls"] += bucket.get("calls", 0)
    return {"byKernel": by_kernel, "all": all_counts, "sources": sources}


def tokens_for_experiment(exp_path: str, model: str | None = None, repo: Path | None = None) -> dict | None:
    """Aggregate tokens for one experiment path, or None if unmapped/missing."""
    root = repo or REPO
    files = TOKEN_CSV_BY_PATH.get(exp_path)
    if not files:
        return None
    parts = []
    for name in files:
        p = root / name
        if p.is_file():
            parts.append(load_csv_tokens(p, model=model))
    if not parts:
        return None
    merged = merge_token_dicts(parts)
    if merged["all"]["calls"] == 0:
        return None
    return merged


def build_tokens_by_col(exps: list[dict], repo: Path | None = None) -> dict[str, dict]:
    """Map experiment col → token aggregate (keyed for canvas payload)."""
    out: dict[str, dict] = {}
    for e in exps:
        agg = tokens_for_experiment(e["path"], model=e.get("model"), repo=repo)
        if agg is None:
            continue
        out[e["col"]] = {
            "input": agg["all"]["input"],
            "output": agg["all"]["output"],
            "total": agg["all"]["total"],
            "calls": agg["all"]["calls"],
            "byKernel": agg["byKernel"],
            "sources": agg["sources"],
        }
    return out
