#!/usr/bin/env python3
"""Export cutegen canvas data in ChatGPT-friendly formats."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "exports" / "chatgpt"
CANVAS_JSON = Path("/tmp/cutegen_canvas_payload.json")
ERROR_JSON = Path("/tmp/cutegen_error_payload.json")


def _regenerate_payloads() -> None:
    subprocess.run(
        [sys.executable, str(REPO / "scripts/generate_canvas_payload.py")],
        cwd=REPO,
        check=True,
    )
    subprocess.run(
        [sys.executable, str(REPO / "scripts/generate_error_payload.py")],
        cwd=REPO,
        check=True,
    )


def _best_curve_point(curve: list) -> tuple[int | None, float | None]:
    best_d = None
    best_v = None
    for d, v in enumerate(curve):
        if v is None:
            continue
        if best_v is None or v > best_v:
            best_v = float(v)
            best_d = d
    return best_d, best_v


def export_speedup_csv(payload: dict, out: Path) -> int:
    depth_cols = [f"d{d}" for d in range(payload["maxDepth"] + 1)]
    cheat_set = {
        (c["col"], c["kernel"]) for c in payload.get("precisionCheats", [])
    }
    kernel_by_id = {k["id"]: k for k in payload["kernels"]}

    rows: list[dict] = []
    for exp in payload["exps"]:
        col = exp["col"]
        curves = payload.get("curves", {}).get(col, {})
        for kid_s, curve in sorted(curves.items(), key=lambda x: int(x[0])):
            kid = int(kid_s)
            meta = kernel_by_id.get(kid, {})
            best_d, best_v = _best_curve_point(curve)
            row = {
                "experiment_col": col,
                "experiment_label": exp["label"],
                "backend": exp["backend"],
                "method": exp["method"],
                "model": exp["model"],
                "save_path": exp["path"],
                "kernel_id": kid,
                "kernel_name": meta.get("name", f"{kid}_"),
                "kernel_type": meta.get("ktype", ""),
                "cohort": meta.get("cohort", ""),
                "precision_io_cheat": (col, kid) in cheat_set,
                "best_depth": best_d if best_d is not None else "",
                "best_speedup": best_v if best_v is not None else "",
            }
            for i, col_name in enumerate(depth_cols):
                v = curve[i] if i < len(curve) else None
                row[col_name] = v if v is not None else ""
            rows.append(row)

    fieldnames = [
        "experiment_col",
        "experiment_label",
        "backend",
        "method",
        "model",
        "save_path",
        "kernel_id",
        "kernel_name",
        "kernel_type",
        "cohort",
        "precision_io_cheat",
        "best_depth",
        "best_speedup",
        *depth_cols,
    ]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def export_token_csv(payload: dict, out: Path) -> int:
    rows: list[dict] = []
    for col, agg in sorted(payload.get("tokens", {}).items()):
        if not isinstance(agg, dict):
            continue
        rows.append(
            {
                "experiment_col": col,
                "input_tokens": agg.get("input", 0),
                "output_tokens": agg.get("output", 0),
                "total_tokens": agg.get("total", 0),
                "api_calls": agg.get("calls", 0),
                "sources": "; ".join(agg.get("sources", [])),
            }
        )
    with out.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "experiment_col",
                "input_tokens",
                "output_tokens",
                "total_tokens",
                "api_calls",
                "sources",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def export_error_csv(payload: dict, out: Path) -> int:
    rows: list[dict] = []
    exp_by_col = {e["col"]: e for e in payload["exps"]}
    for col, totals in sorted(payload.get("expTotals", {}).items()):
        exp = exp_by_col.get(col, {})
        rows.append(
            {
                "experiment_col": col,
                "experiment_label": exp.get("label", ""),
                "backend": exp.get("backend", ""),
                "method": exp.get("method", ""),
                "model": exp.get("model", ""),
                "nodes": totals.get("nodes", 0),
                "compile_errors": totals.get("compile", 0),
                "correctness_errors": totals.get("correct", 0),
                "resolved_errors": totals.get("resolved", 0),
                "resolution_rate_pct": totals.get("rate", 0),
            }
        )
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)
    return len(rows)


def write_readme(out: Path, files: dict[str, str]) -> None:
    lines = [
        "# Cutegen ChatGPT export bundle",
        "",
        "Upload these files to ChatGPT. Start with `cutegen_canvas_payload.json` for full data.",
        "",
        "## Files",
        "",
    ]
    for name, desc in files.items():
        lines.append(f"- `{name}` — {desc}")
    lines.extend(
        [
            "",
            "## Schema (canvas JSON)",
            "",
            "- `exps[]` — experiment configs (backend, profiling method, model, save path)",
            "- `kernels[]` — kernel id, name, type, cohort (sample19 / new8)",
            "- `curves[col][kernel_id]` — speedup vs PyTorch ref at depths 0–10 (`null` = no pass)",
            "- `tokens[col]` — aggregated LLM token usage per experiment",
            "- `precisionCheats[]` — known I/O precision shortcuts; exclude for fair comparisons",
            "",
            "## Suggested prompt",
            "",
            "> Attached: cutegen export bundle. `speedup_curves.csv` has one row per experiment×kernel.",
            "> `precision_io_cheat=true` rows should be excluded from fair speedup rankings.",
            "> [Your question here]",
            "",
        ]
    )
    out.write_text("\n".join(lines))


def main() -> None:
    _regenerate_payloads()
    OUT.mkdir(parents=True, exist_ok=True)

    canvas = json.loads(CANVAS_JSON.read_text())
    errors = json.loads(ERROR_JSON.read_text())

    paths = {
        "cutegen_canvas_payload.json": CANVAS_JSON.read_text(),
        "cutegen_error_payload.json": ERROR_JSON.read_text(),
    }
    for name, text in paths.items():
        (OUT / name).write_text(text)

    n_speed = export_speedup_csv(canvas, OUT / "speedup_curves.csv")
    n_tok = export_token_csv(canvas, OUT / "token_usage_by_experiment.csv")
    n_err = export_error_csv(errors, OUT / "error_summary_by_experiment.csv")

    cheats = canvas.get("precisionCheats", [])
    (OUT / "precision_io_cheats.json").write_text(json.dumps(cheats, indent=2))

    write_readme(
        OUT / "README.md",
        {
            "cutegen_canvas_payload.json": "full structured data (speedups, tokens, metadata)",
            "cutegen_error_payload.json": "full error/debug data per experiment",
            "speedup_curves.csv": f"flattened speedups ({n_speed} rows: experiment × kernel)",
            "token_usage_by_experiment.csv": f"LLM tokens per experiment ({n_tok} rows)",
            "error_summary_by_experiment.csv": f"error counts per experiment ({n_err} rows)",
            "precision_io_cheats.json": "2 known I/O precision cheats to exclude",
            "README.md": "this guide",
        },
    )

    print(f"Wrote ChatGPT bundle to {OUT}/")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name:40} {p.stat().st_size:>10} bytes")


if __name__ == "__main__":
    main()
