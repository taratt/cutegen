#!/usr/bin/env python3
import argparse
import importlib
import json
import os
from types import SimpleNamespace
import tempfile
import shutil
import copy
import re

import torch


# -------------------- Node JSON helpers --------------------
def load_node_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def make_node_obj(d: dict):
    """
    Create a minimal object with the attributes your evaluate(node, ...) uses.
    We avoid depending on cutegen.node.Node constructor (which may change).
    """
    node = SimpleNamespace()
    node.uuid = d.get("uuid")
    node.ref = d.get("ref")
    node.src = d.get("src")
    node.depth = int(d.get("depth", 0))
    node.ref_time = d.get("ref_time", None)
    node.time = d.get("time", None)
    node.perf = d.get("perf", None)
    node.prev_src = d.get("prev_src", None)
    node.history = d.get("history", None)

    # evaluate() mutates these
    node.error_type = d.get("error_type", None)

    # metadata must be a dict; evaluate() writes into it
    node.metadata = d.get("metadata", {})
    if node.metadata is None:
        node.metadata = {}

    # evaluate() uses node.save_folder_path for best_time.txt
    node.save_folder_path = d.get("save_folder_path", None)

    return node


# -------------------- Sweep sizes (2D) --------------------
def default_sweep_sizes():
    """
    Curated 2D shapes for activation-style kernels (batch_size, dim).
    Includes the "showoff" (256, 65536) + neighbors and some common hidden sizes.
    """
    batches = [32, 64, 128, 256, 512, 1024, 4096]  # includes token-ish and your big batch
    dims = [4096, 8192, 16384, 32768, 49152, 65536, 98304, 131072]  # common-ish hidden/MLP-ish widths + your dim
    return [(b, d) for b in batches for d in dims]

def parse_sizes_arg(s: str):
    """
    Parse --sizes like: "256x65536,512x65536,128x131072"
    """
    out = []
    s = s.strip()
    if not s:
        return out
    for part in s.split(","):
        part = part.strip().lower()
        m = re.match(r"^(\d+)\s*x\s*(\d+)$", part)
        if not m:
            raise ValueError(f"Bad size '{part}'. Expected like 256x65536.")
        out.append((int(m.group(1)), int(m.group(2))))
    return out


# -------------------- Source patching (override get_inputs + batch/dim) --------------------
def inject_get_inputs_override(src: str, batch: int, dim: int) -> str:
    """
    Append an override block that forces get_inputs() to return CPU tensors
    of shape (batch, dim). This makes the eval module's timing functions
    use exactly these shapes regardless of what the node's original code had.

    We *append* (don’t regex-edit existing code) to avoid breaking anything.
    """
    override = f"""
# ===================== [replay sweep override] =====================
# Force the benchmark input shape, regardless of what the node's code defines.
batch_size = {batch}
dim = {dim}
def get_inputs():
    import torch
    x = torch.rand(batch_size, dim)  # CPU on purpose; eval code moves to CUDA
    return [x]
# ================================================================
"""
    return src + "\n" + override


# -------------------- CSV + heatmap --------------------
def write_csv(rows, csv_path):
    import csv
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def make_heatmap(rows, out_png_path):
    """
    Heatmap of speedup = ref_mean / gen_mean
    Y axis: batch_size
    X axis: dim
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if not rows:
        print("[sweep] No rows to plot.")
        return

    batches = sorted({int(r["batch_size"]) for r in rows})
    dims = sorted({int(r["dim"]) for r in rows})

    b2i = {b: i for i, b in enumerate(batches)}
    d2i = {d: i for i, d in enumerate(dims)}

    mat = np.full((len(batches), len(dims)), np.nan, dtype=np.float64)

    for r in rows:
        b = int(r["batch_size"])
        d = int(r["dim"])
        sp = float(r["speedup"])
        mat[b2i[b], d2i[d]] = sp

    fig = plt.figure(figsize=(1.2 + 0.9 * len(dims), 1.2 + 0.6 * len(batches)))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, aspect="auto", origin="lower")

    ax.set_title("Speedup Heatmap (ref / gen)  —  >1 means custom faster")
    ax.set_xlabel("dim")
    ax.set_ylabel("batch_size")

    ax.set_xticks(np.arange(len(dims)))
    ax.set_xticklabels([str(d) for d in dims], rotation=45, ha="right")

    ax.set_yticks(np.arange(len(batches)))
    ax.set_yticklabels([str(b) for b in batches])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("speedup")

    # mark best
    try:
        best_idx = np.nanargmax(mat)
        bi, di = np.unravel_index(best_idx, mat.shape)
        ax.scatter([di], [bi], s=120, marker="o")
        ax.text(di, bi, f"  best\n  {mat[bi, di]:.2f}×", va="center")
    except Exception:
        pass

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    fig.savefig(out_png_path, dpi=220)
    plt.close(fig)


# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(description="Replay evaluate() + sweep timing over (batch, dim) using YOUR existing eval functions/kernels.")
    ap.add_argument("--path", required=True, help="Path to a single node JSON file")

    ap.add_argument(
        "--eval_module",
        default="cutegen.evaluate",
        help="Python module path that contains evaluate(), get_baseline_time(), get_wallclock_time()",
    )

    # Single-run flags (your original behavior)
    ap.add_argument("--no_time", action="store_true", help="Disable timing inside evaluate() (get_time=False)")
    ap.add_argument("--profile", action="store_true", help="Enable profiling inside evaluate() (get_profile=True). Default is False.")
    ap.add_argument("--no_torch_compile", action="store_true", help="Force torch_compile=False in evaluate() baseline benchmarking.")
    ap.add_argument("--print_times", action="store_true", help="Also call get_baseline_time() and get_wallclock_time() and print both.")

    # Timing knobs (passed to eval module functions)
    ap.add_argument("--num_warmups", type=int, default=5, help="Warmups for timing calls.")
    ap.add_argument("--num_trials", type=int, default=100, help="Trials for timing calls.")
    ap.add_argument("--device", type=int, default=0, help="CUDA device index to use for timing calls.")
    ap.add_argument("--seed", type=int, default=42, help="seed_num for timing calls.")

    # Sweep flags
    ap.add_argument("--sweep", action="store_true", help="Run a (batch, dim) sweep and emit CSV + heatmap.")
    ap.add_argument("--sizes", default="", help="Override sweep sizes: '256x65536,512x65536,128x131072'")
    ap.add_argument("--out_dir", default="./sweep_out", help="Output dir for CSV + heatmap.")
    ap.add_argument("--csv_name", default="sweep.csv", help="CSV filename inside out_dir.")
    ap.add_argument("--heatmap_name", default="speedup_heatmap.png", help="Heatmap filename inside out_dir.")

    args = ap.parse_args()

    node_path = os.path.abspath(args.path)
    d = load_node_json(node_path)
    node = make_node_obj(d)

    if not node.ref or not node.src:
        raise RuntimeError("Node JSON missing required fields: 'ref' and/or 'src'.")

    # import your eval module
    mod = importlib.import_module(args.eval_module)
    if not hasattr(mod, "evaluate"):
        raise RuntimeError(f"Module '{args.eval_module}' does not have evaluate().")
    if not hasattr(mod, "get_baseline_time") or not hasattr(mod, "get_wallclock_time"):
        raise RuntimeError(f"Module '{args.eval_module}' must expose get_baseline_time() and get_wallclock_time().")

    evaluate_fn = getattr(mod, "evaluate")

    # optional: print constants in the eval module
    if hasattr(mod, "EVAL_COLD_CACHE"):
        print(f"[exact] {args.eval_module}.EVAL_COLD_CACHE = {getattr(mod, 'EVAL_COLD_CACHE')}")
    if hasattr(mod, "BENCHMARK_TORCH_COMPILE"):
        print(f"[exact] {args.eval_module}.BENCHMARK_TORCH_COMPILE = {getattr(mod, 'BENCHMARK_TORCH_COMPILE')}")
    if hasattr(mod, "TORCH_COMPILE_MODE"):
        print(f"[exact] {args.eval_module}.TORCH_COMPILE_MODE = {getattr(mod, 'TORCH_COMPILE_MODE')}")

    # ---------- 1) Call evaluate() exactly like your agent does ----------
    print(f"[exact] Calling evaluate() on uuid={node.uuid} depth={node.depth}")
    out = evaluate_fn(
        node,
        get_time=(not args.no_time),
        get_profile=args.profile,
        torch_compile=(False if args.no_torch_compile else getattr(mod, "BENCHMARK_TORCH_COMPILE", False)),
        torch_compile_mode=getattr(mod, "TORCH_COMPILE_MODE", "default"),
    )
    if out is None:
        out = node

    # ---------- 2) Optional: print ref/gen timing using module funcs (single run) ----------
    if args.print_times:
        device = torch.device(f"cuda:{args.device}")
        build_dir = tempfile.mkdtemp(prefix=f"replay_build_{node.uuid}_")
        try:
            print("[exact] ===== Timing (calling eval module functions directly) =====")
            meta = copy.deepcopy(out.metadata) if isinstance(out.metadata, dict) else {}

            ref_stats = mod.get_baseline_time(
                node.ref,
                meta,
                num_warmups=args.num_warmups,
                num_trials=args.num_trials,
                seed_num=args.seed,
                device=device,
                torch_compile=(False if args.no_torch_compile else getattr(mod, "BENCHMARK_TORCH_COMPILE", False)),
                torch_compile_mode=getattr(mod, "TORCH_COMPILE_MODE", "default"),
                get_torch_graph=False,
            )
            gen_stats = mod.get_wallclock_time(
                node.ref,
                node.src,
                meta,
                num_warmups=args.num_warmups,
                num_trials=args.num_trials,
                seed_num=args.seed,
                build_directory=build_dir,
                device=device,
            )

            print(f"[exact] REF  time stats: {ref_stats}")
            print(f"[exact] GEN  time stats: {gen_stats}")
            try:
                speedup = float(ref_stats["mean"]) / float(gen_stats["mean"])
                print(f"[exact] SPEEDUP (ref/gen): {speedup:.3f}x")
            except Exception:
                pass
            print("[exact] ==========================================================")
        finally:
            shutil.rmtree(build_dir, ignore_errors=True)

    print("[exact] ================= RESULT =================")
    print(f"[exact] error_type = {getattr(out, 'error_type', None)}")
    print(f"[exact] compile msg = {out.metadata.get('compile','')[:500]}")
    print(f"[exact] correct msg = {out.metadata.get('correct','')[:500]}")
    print(f"[exact] time = {getattr(out, 'time', None)}")
    print(f"[exact] perf = {getattr(out, 'perf', None)}")
    print("[exact] ========================================")

    # ---------- 3) Sweep: patch sources to force (batch, dim), then call eval timing funcs ----------
    if args.sweep:
        device = torch.device(f"cuda:{args.device}")

        sizes = parse_sizes_arg(args.sizes) if args.sizes.strip() else default_sweep_sizes()
        print(f"[sweep] Running {len(sizes)} sizes on device={device} warmups={args.num_warmups} trials={args.num_trials}")

        out_dir = os.path.abspath(args.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, args.csv_name)
        heatmap_path = os.path.join(out_dir, args.heatmap_name)

        rows = []

        # One base build dir; subdirs per size so builds don't clobber each other.
        base_build = tempfile.mkdtemp(prefix=f"sweep_build_{node.uuid}_")

        try:
            for i, (batch, dim) in enumerate(sizes, 1):
                print(f"[sweep] [{i}/{len(sizes)}] batch={batch} dim={dim}")

                # Force both ref and gen to use the same get_inputs() shape.
                # This ensures "correctness" comparisons are meaningful and timings align.
                ref_src = inject_get_inputs_override(node.ref, batch, dim)
                gen_src = inject_get_inputs_override(node.src, batch, dim)

                # fresh metadata per run (don’t accumulate messages across points)
                meta = {}
                build_dir = os.path.join(base_build, f"b{batch}_d{dim}")
                os.makedirs(build_dir, exist_ok=True)

                # measure
                ref_stats = mod.get_baseline_time(
                    ref_src,
                    meta,
                    num_warmups=args.num_warmups,
                    num_trials=args.num_trials,
                    seed_num=args.seed,
                    device=device,
                    torch_compile=(False if args.no_torch_compile else getattr(mod, "BENCHMARK_TORCH_COMPILE", False)),
                    torch_compile_mode=getattr(mod, "TORCH_COMPILE_MODE", "default"),
                    get_torch_graph=False,
                )
                gen_stats = mod.get_wallclock_time(
                    ref_src,
                    gen_src,
                    meta,
                    num_warmups=args.num_warmups,
                    num_trials=args.num_trials,
                    seed_num=args.seed,
                    build_directory=build_dir,
                    device=device,
                )

                ref_mean = float(ref_stats["mean"])
                gen_mean = float(gen_stats["mean"])
                speedup = ref_mean / gen_mean if gen_mean > 0 else float("inf")

                rows.append({
                    "batch_size": batch,
                    "dim": dim,
                    "ref_mean_ms": ref_mean,
                    "ref_std_ms": float(ref_stats.get("std", 0.0)),
                    "gen_mean_ms": gen_mean,
                    "gen_std_ms": float(gen_stats.get("std", 0.0)),
                    "speedup": speedup,
                })

                print(f"[sweep]    REF={ref_mean:.6g} ms | GEN={gen_mean:.6g} ms | speedup={speedup:.3f}x")

        finally:
            shutil.rmtree(base_build, ignore_errors=True)

        # write CSV + heatmap
        if rows:
            write_csv(rows, csv_path)
            print(f"[sweep] CSV: {csv_path}")
            make_heatmap(rows, heatmap_path)
            print(f"[sweep] Heatmap: {heatmap_path}")

            # best point summary
            best = max(rows, key=lambda r: float(r["speedup"]))
            print("[sweep] BEST:")
            print(f"        batch={best['batch_size']} dim={best['dim']} speedup={best['speedup']:.3f}x "
                  f"(REF={best['ref_mean_ms']:.6g} ms, GEN={best['gen_mean_ms']:.6g} ms)")
        else:
            print("[sweep] No results produced.")


if __name__ == "__main__":
    main()
