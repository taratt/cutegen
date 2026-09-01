#!/usr/bin/env python3
import json, re, sys
from pathlib import Path

NEW8 = [6, 14, 50, 61, 70, 75, 105, 107]
SENT = 999999.0


def kid(name: str):
    m = re.match(r"^(\d+)_", name)
    return int(m.group(1)) if m else None


def check(base: Path, ids):
    if not base.is_dir():
        return "NO_TREE"
    parts = []
    correct = 0
    for k in ids:
        dirs = [d for d in base.iterdir() if d.is_dir() and kid(d.name) == k]
        if not dirs:
            parts.append(f"k{k}:MISS")
            continue
        bt_path = dirs[0] / "best_time.txt"
        try:
            bt = float(bt_path.read_text().strip()) if bt_path.exists() else SENT
        except Exception:
            bt = SENT
        depths = []
        for jp in dirs[0].glob("*.json"):
            try:
                depths.append(json.loads(jp.read_text()).get("depth", -1))
            except Exception:
                pass
        mp = max(depths) if depths else -1
        n = len(depths)
        ok = bt < SENT and n > 0
        if ok:
            correct += 1
        parts.append(f"k{k}:d{mp}/n{n}/best={bt}{'*' if ok else ''}")
    return f"correct={correct}/{len(ids)} | " + "; ".join(parts)


def tail_interesting(log: Path, n=20):
    if not log.exists():
        print(f"log missing: {log}")
        return
    print(f"log={log} mtime={log.stat().st_mtime}")
    lines = log.read_text(errors="replace").splitlines()
    keys = (
        "=== ",
        "Saving node",
        "Querying",
        "starting node",
        "DONE",
        "Error running",
        "Non-transient",
    )
    interesting = [l for l in lines if l.startswith(keys) or any(l.startswith(k) for k in keys)]
    # also match contains for starting node
    interesting = [
        l
        for l in lines
        if l.startswith("=== ")
        or l.startswith("Saving node")
        or l.startswith("Querying")
        or "starting node" in l
        or l.strip() == "DONE"
        or l.startswith("Error running")
    ]
    for l in interesting[-n:]:
        print(l[:220])


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    print("ROOT", root)
    # local-style resume
    for name in [
        "sonnet5_new8_api_resume.log",
        "ptx_from_start_api_resume.log",
        "sonnet5_new8_from_start_3backends.log",
        "cute_sonnet5_minimal_d0_new8.log",
    ]:
        p = root / name
        if p.exists():
            print("---", name, "---")
            tail_interesting(p, 18)
    checks = [
        ("ptx delayed", root / "saved_nodes/ptx/level1-profiled-sonnet5", NEW8),
        ("cute delayed", root / "saved_nodes/cute/level1-profiled-sonnet5", NEW8 + [9]),
        ("ptx from-start", root / "saved_nodes/ptx/level1-profiled-from-start-sonnet5", NEW8),
        ("cuda from-start", root / "saved_nodes/cuda/level1-profiled-from-start-sonnet5", NEW8),
        ("cute from-start", root / "saved_nodes/cute/level1-profiled-from-start-sonnet5", NEW8),
        ("triton from-start", root / "saved_nodes/triton/level1-profiled-from-start-sonnet5", NEW8),
        ("cute minimal-d0", root / "saved_nodes/cute/level1-nopf-sonnet5-minimal-d0", NEW8),
    ]
    print("--- TREES ---")
    for label, path, ids in checks:
        print(f"{label}: {check(path, ids)}")


if __name__ == "__main__":
    main()
