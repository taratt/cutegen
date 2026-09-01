#!/usr/bin/env python3
"""Print experiment coverage for the 19+8 x model x backend x profile matrix."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

SAMPLE19 = [1, 4, 9, 21, 22, 33, 40, 49, 53, 54, 55, 58, 59, 80, 83, 88, 99, 102, 103]
NEW8 = [6, 14, 50, 61, 70, 75, 105, 107]
ALL27 = sorted(set(SAMPLE19) | set(NEW8))
SENT = 999999.0


def kid(name: str) -> int | None:
    m = re.match(r"^(\d+)_", name)
    return int(m.group(1)) if m else None


def analyze(exp: Path, cohort: list[int]) -> tuple[int, list[int]]:
    if not exp.is_dir():
        return 0, list(cohort)
    done: list[int] = []
    miss: list[int] = []
    for k in cohort:
        dirs = [d for d in exp.iterdir() if d.is_dir() and kid(d.name) == k]
        if not dirs:
            miss.append(k)
            continue
        bt_path = dirs[0] / "best_time.txt"
        try:
            v = float(bt_path.read_text().strip()) if bt_path.exists() else SENT
        except Exception:
            v = SENT
        n = len(list(dirs[0].glob("*.json")))
        if v < SENT and n > 0:
            done.append(k)
        else:
            miss.append(k)
    return len(done), miss


def classify(name: str) -> tuple[str, str, str | None]:
    n = name.lower()
    if "sonnet" in n:
        model = "sonnet"
    elif "kimi" in n or n in {
        "level1-profiled",
        "level1-no-profile",
        "level1_from_start",
    }:
        model = "kimi"
    else:
        model = "?"
    if "from-start" in n or "from_start" in n:
        profile = "from-start"
    elif "no-profile" in n or "nopf" in n or "minimal" in n:
        profile = "no-profile"
    elif "profiled" in n:
        profile = "delayed"
    else:
        profile = "?"
    ablation = "minimal-d0" if "minimal" in n else None
    return model, profile, ablation


def main() -> None:
    base = Path(sys.argv[1] if len(sys.argv) > 1 else "saved_nodes")
    label = sys.argv[2] if len(sys.argv) > 2 else str(base)
    print(f"SOURCE={label}")
    print(
        "JOBS="
        + subprocess.getoutput(
            'pgrep -af "cutegen.main|run_profiled|resume_interrupted" | grep -v grep || true'
        ).replace("\n", " || ")
    )
    if not base.exists():
        print("NO_SAVED")
        return
    for be in sorted(p.name for p in base.iterdir() if p.is_dir()):
        for exp in sorted((base / be).iterdir()):
            if not exp.is_dir():
                continue
            model, profile, ablation = classify(exp.name)
            if model == "?" or profile == "?":
                continue
            n27, miss27 = analyze(exp, ALL27)
            n8, miss8 = analyze(exp, NEW8)
            n19, miss19 = analyze(exp, SAMPLE19)
            tag = "ABLATION" if ablation else "CELL"
            print(
                json.dumps(
                    {
                        "tag": tag,
                        "backend": be,
                        "model": model,
                        "profile": profile,
                        "name": exp.name,
                        "n27": n27,
                        "n19": n19,
                        "n8": n8,
                        "miss27": miss27,
                        "miss8": miss8,
                        "miss19": miss19,
                        "ablation": ablation,
                    }
                )
            )


if __name__ == "__main__":
    main()
