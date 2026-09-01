#!/usr/bin/env python3
"""Replay a saved kernel JSON with source overrides; write results elsewhere.

Does not modify the source JSON or saved_nodes experiment trees.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("CUTEGEN_BASE_PATH", str(PROJECT_ROOT))
os.environ.setdefault("KERNEL_BACKEND", "triton")


def _apply_dot_precision(src: str, precision: str | None) -> str:
    """Return src with tl.dot precision override applied."""
    if precision is None:
        return src
    pattern = r"acc\s*=\s*tl\.dot\(\s*a\s*,\s*b\s*,\s*acc\s*\)"
    repl = f'acc = tl.dot(a, b, acc, input_precision="{precision}")'
    new_src, n = re.subn(pattern, repl, src)
    if n == 0:
        raise ValueError(
            f"Could not find tl.dot(a, b, acc) to patch for precision={precision!r}"
        )
    return new_src


def _bench_variant(
    *,
    name: str,
    ref: str,
    src: str,
    metadata: dict,
    build_root: Path,
    num_trials: int,
) -> dict:
    from cutegen.evaluate import (
        check_compile,
        check_correct,
        get_baseline_time,
        get_wallclock_time,
        remove_build_directory,
    )

    variant_meta = dict(metadata)
    variant_meta["kernel_backend"] = variant_meta.get("kernel_backend", "triton")

    build_dir = str(build_root / name)
    result: dict = {"variant": name, "build_directory": build_dir}

    compiled = check_compile(ref, src, variant_meta, build_directory=build_dir)
    result["compile_ok"] = bool(compiled)
    if not compiled:
        result["compile_error"] = variant_meta.get("compile", "")
        remove_build_directory(build_dir)
        return result

    correct = check_correct(
        ref,
        src,
        variant_meta,
        build_directory=build_dir,
        device=0,
    )
    result["correct_ok"] = bool(correct)
    if not correct:
        result["correct_error"] = variant_meta.get("correct", "")
        remove_build_directory(build_dir)
        return result

    ref_time = get_baseline_time(ref, variant_meta, device=0)
    gen_time = get_wallclock_time(
        ref,
        src,
        variant_meta,
        num_trials=num_trials,
        build_directory=build_dir,
        device=0,
    )
    result["ref_time_ms"] = ref_time
    result["gen_time_ms"] = gen_time
    if (
        isinstance(ref_time, dict)
        and isinstance(gen_time, dict)
        and ref_time.get("mean")
        and gen_time.get("mean")
        and gen_time["mean"] > 0
    ):
        result["speedup"] = ref_time["mean"] / gen_time["mean"]
    remove_build_directory(build_dir)
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-json",
        required=True,
        help="Path to a saved node JSON (read-only).",
    )
    p.add_argument(
        "--out-dir",
        default="",
        help="Directory for snapshots + results (default: bench_results/<stamp>).",
    )
    p.add_argument(
        "--variants",
        default="default,ieee,tf32",
        help="Comma-separated variants: default, ieee, tf32.",
    )
    p.add_argument("--num-trials", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source_json = Path(args.source_json).resolve()
    if not source_json.is_file():
        raise SystemExit(f"Missing source JSON: {source_json}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else PROJECT_ROOT / "bench_results" / f"ieee_triton_k1_{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    variants_dir = out_dir / "variants"
    build_root = out_dir / "build"
    variants_dir.mkdir(exist_ok=True)
    build_root.mkdir(exist_ok=True)

    node = json.loads(source_json.read_text())
    ref = node["ref"]
    base_src = node["src"]
    metadata = dict(node.get("metadata") or {})
    metadata["kernel_backend"] = metadata.get("kernel_backend", "triton")

    snapshot = {
        "source_json": str(source_json),
        "uuid": node.get("uuid"),
        "depth": node.get("depth"),
        "original_time_ms": node.get("time"),
        "original_ref_time_ms": node.get("ref_time"),
    }
    (out_dir / "source_snapshot.json").write_text(json.dumps(snapshot, indent=2))
    shutil.copy2(source_json, out_dir / "source_node.json")

    variant_map = {
        "default": None,
        "ieee": "ieee",
        "tf32": "tf32",
    }
    requested = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in requested if v not in variant_map]
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}; valid={list(variant_map)}")

    results = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_json": str(source_json),
        "out_dir": str(out_dir),
        "num_trials": args.num_trials,
        "variants": {},
    }

    for name in requested:
        precision = variant_map[name]
        src = _apply_dot_precision(base_src, precision)
        (variants_dir / f"{name}.py").write_text(src)
        print(f"\n=== benchmarking variant={name} precision={precision!r} ===", flush=True)
        t0 = time.time()
        results["variants"][name] = _bench_variant(
            name=name,
            ref=ref,
            src=src,
            metadata=dict(metadata),
            build_root=build_root,
            num_trials=args.num_trials,
        )
        results["variants"][name]["elapsed_s"] = round(time.time() - t0, 2)
        v = results["variants"][name]
        print(
            f"  compile={v.get('compile_ok')} correct={v.get('correct_ok')} "
            f"speedup={v.get('speedup')}",
            flush=True,
        )

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print("\n=== summary ===")
    for name, v in results["variants"].items():
        if v.get("speedup") is not None:
            print(
                f"  {name:8} speedup={v['speedup']:.3f}x "
                f"gen={v['gen_time_ms']['mean']:.3f}ms "
                f"ref={v['ref_time_ms']['mean']:.3f}ms"
            )
        else:
            print(f"  {name:8} FAILED compile={v.get('compile_ok')} correct={v.get('correct_ok')}")
    print(f"\nWrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
