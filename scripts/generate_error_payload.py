#!/usr/bin/env python3
"""Scan saved_nodes and emit JSON payload for the error-report canvas."""

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
    col = f"{backend_label}/{tag}/{model_tag_for(model)}"
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


def empty_kernel_stats() -> dict:
    return {
        "nodes": 0,
        "nodes_pass": 0,
        "nodes_fail": 0,
        "compile_errors": 0,
        "correct_errors": 0,
        "compile_resolved": 0,
        "correct_resolved": 0,
        "codegen_regens": 0,
        "timeouts": 0,
        "resolved": 0,
        "unresolved": 0,
        "max_depth": -1,
        "ever_passed": False,
    }


def classify_msg(msg: str) -> str | None:
    m = msg.lower()
    if "timed out" in m or "timeout" in m:
        return "timeout"
    if "oom" in m or "out of memory" in m:
        return "oom"
    if "runtime error" in m:
        return "runtime"
    return None


def is_pass(error_type) -> bool:
    return error_type in ("PASS", 1, "ErrorType.PASS")


def analyze_node(node: dict) -> dict:
    meta = node.get("metadata") or {}
    history = node.get("history") or []
    passed = is_pass(node.get("error_type"))

    compile_errors = 0
    correct_errors = 0
    compile_resolved = 0
    correct_resolved = 0
    codegen_regens = 0
    timeouts = 0
    resolved = 0
    unresolved = 0

    for entry in history:
        et = entry.get("error_type", "NONE")
        msg = str(entry.get("msg") or "")
        extra = classify_msg(msg)
        if extra == "timeout":
            timeouts += 1

        if et == "COMPILE":
            compile_errors += 1
            if passed:
                resolved += 1
                compile_resolved += 1
            else:
                unresolved += 1
        elif et == "CORRECT":
            correct_errors += 1
            if passed:
                resolved += 1
                correct_resolved += 1
            else:
                unresolved += 1
        elif et in ("NONE", "ErrorType.NONE"):
            codegen_regens += 1

    if meta.get("timeout"):
        timeouts += 1
        if not passed:
            unresolved += 1

    compile_meta = str(meta.get("compile") or "").strip()
    correct_meta = str(meta.get("correct") or "").strip()
    if compile_meta and not passed:
        unresolved += 1
        if classify_msg(compile_meta) == "timeout":
            timeouts += 1
    if correct_meta and not passed:
        unresolved += 1

    if not passed:
        unresolved += 1

    depth = int(node.get("depth", 0))
    return {
        "passed": passed,
        "depth": depth,
        "compile_errors": compile_errors,
        "correct_errors": correct_errors,
        "compile_resolved": compile_resolved,
        "correct_resolved": correct_resolved,
        "codegen_regens": codegen_regens,
        "timeouts": timeouts,
        "resolved": resolved,
        "unresolved": unresolved,
    }


def merge_kernel_stats(acc: dict, part: dict) -> None:
    acc["nodes"] += 1
    if part["passed"]:
        acc["nodes_pass"] += 1
        acc["ever_passed"] = True
    else:
        acc["nodes_fail"] += 1
    acc["compile_errors"] += part["compile_errors"]
    acc["correct_errors"] += part["correct_errors"]
    acc["compile_resolved"] += part["compile_resolved"]
    acc["correct_resolved"] += part["correct_resolved"]
    acc["codegen_regens"] += part["codegen_regens"]
    acc["timeouts"] += part["timeouts"]
    acc["resolved"] += part["resolved"]
    acc["unresolved"] += part["unresolved"]
    acc["max_depth"] = max(acc["max_depth"], part["depth"])


def sum_exp_stats(rows: dict[str, dict]) -> dict:
    totals = empty_kernel_stats()
    for stats in rows.values():
        for key in (
            "nodes",
            "nodes_pass",
            "nodes_fail",
            "compile_errors",
            "correct_errors",
            "compile_resolved",
            "correct_resolved",
            "codegen_regens",
            "timeouts",
            "resolved",
            "unresolved",
        ):
            totals[key] += stats[key]
        totals["ever_passed"] = totals["ever_passed"] or stats["ever_passed"]
        totals["max_depth"] = max(totals["max_depth"], stats["max_depth"])
    return totals


def main() -> None:
    collected: list[tuple[dict, dict[str, dict], dict]] = []
    seen_cols: set[str] = set()

    exp_dirs = sorted(
        {
            bt.parent.parent
            for bt in ROOT.rglob("best_time.txt")
            if re.match(r"^\d+_", bt.parent.name)
        },
        key=str,
    )
    for exp in exp_dirs:
        if exp.name == "kimi-k3":
            continue
        meta = label_experiment(exp)
        col = meta["col"]
        if col in seen_cols:
            col = f"{col}@{meta['leaf']}"
            meta = {**meta, "col": col}
        seen_cols.add(col)

        exp_rows: dict[str, dict] = {}
        for ker_dir in sorted(exp.iterdir()):
            if not ker_dir.is_dir() or not re.match(r"^\d+_", ker_dir.name):
                continue
            k = kid(ker_dir.name)
            if k is None:
                continue
            KERNEL_NAMES[k] = ker_dir.name.replace(".py", "")
            stats = empty_kernel_stats()
            for jp in ker_dir.glob("*.json"):
                try:
                    node = json.loads(jp.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                merge_kernel_stats(stats, analyze_node(node))
            if stats["nodes"] > 0:
                exp_rows[str(k)] = stats

        if not exp_rows:
            continue

        collected.append((meta, exp_rows, sum_exp_stats(exp_rows)))

    available_paths = {meta["path"] for meta, _, _ in collected}
    exclude_paths = paths_superseded_by_minimal(available_paths) | set(paths_in_progress())

    exps: list[dict] = []
    errors: dict[str, dict[str, dict]] = {}
    exp_totals: dict[str, dict] = {}
    for meta, exp_rows, totals in collected:
        if meta["path"] in exclude_paths:
            continue
        exps.append({k: meta[k] for k in ("col", "path", "label", "backend", "method", "model", "leaf")})
        errors[meta["col"]] = exp_rows
        exp_totals[meta["col"]] = totals

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

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "sample19Ids": SAMPLE19,
        "new8Ids": NEW8,
        "all27Ids": ALL27,
        "cohort25Ids": COHORT25,
        "kernelTypes": sorted(set(KERNEL_TYPE.values())),
        "exps": exps,
        "kernels": kernel_meta,
        "errors": errors,
        "expTotals": exp_totals,
        "tokens": tokens,
    }
    out = Path("/tmp/cutegen_error_payload.json")
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    skipped = sorted(paths_in_progress() & available_paths)
    if skipped:
        print(f"excluded in-progress ({len(skipped)}):")
        for p in skipped:
            print(f"  - {p}")
    for e in exps:
        t = exp_totals[e["col"]]
        total_err = t["compile_errors"] + t["correct_errors"]
        rate = (100 * t["resolved"] / total_err) if total_err else 100.0
        print(
            f"  {e['col']:28} nodes={t['nodes']:4} compile={t['compile_errors']:4} "
            f"correct={t['correct_errors']:4} resolved={t['resolved']:4} rate={rate:.0f}%"
        )


if __name__ == "__main__":
    main()
