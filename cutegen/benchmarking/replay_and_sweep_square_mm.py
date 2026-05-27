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

    node.error_type = d.get("error_type", None)

    node.metadata = d.get("metadata", {})
    if node.metadata is None:
        node.metadata = {}

    node.save_folder_path = d.get("save_folder_path", None)
    return node


# -------------------- Square sweep sizes --------------------
def default_square_sizes():
    """
    Curated square sizes commonly relevant to DL/LLMs:
      - attention score matrices: seq_len x seq_len  (128..2048)
      - hidden-to-hidden projections: d_model x d_model (768..12288)
    """
    attention_like = [128, 256, 512, 1024, 2048]
    hidden_like = [768, 1024, 2048, 3072, 4096, 6144, 8192, 12288]
    out, seen = [], set()
    for n in attention_like + hidden_like:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out

def parse_nsizes_arg(s: str):
    """
    Parse --nsizes like: "256,512,1024,2048"
    """
    s = s.strip()
    if not s:
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if not re.match(r"^\d+$", part):
            raise ValueError(f"Bad N '{part}'. Expected comma-separated ints like 256,512,1024.")
        out.append(int(part))
    return out


# -------------------- Source patching: override get_inputs for square GEMM --------------------
def inject_square_mm_get_inputs_override(ref_src: str, n: int) -> str:
    """
    Append an override that forces A,B = torch.rand(n,n) on CPU.
    (No dtype specified => default torch dtype, matching your ref style.)
    Eval code moves inputs to CUDA.
    """
    override = f"""
# ===================== [replay square mm override] =====================
# Force square matmul inputs (A,B) = ({n},{n}) on CPU; eval code moves to CUDA.
def get_inputs():
    import torch
    A = torch.rand({n}, {n})
    B = torch.rand({n}, {n})
    return [A, B]
# ================================================================
"""
    return ref_src + "\n" + override


# -------------------- CSV + plotting --------------------
def write_csv(rows, csv_path):
    import csv
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def make_speedup_plot(rows, out_png_path, title_suffix=""):
    """
    Nice report plot:
      X: N
      Y: speedup = ref_mean / gen_mean
    """
    import matplotlib.pyplot as plt

    if not rows:
        print("[sweep] No rows to plot.")
        return

    rows_sorted = sorted(rows, key=lambda r: int(r["N"]))
    xs = [int(r["N"]) for r in rows_sorted]
    ys = [float(r["speedup"]) for r in rows_sorted]

    fig = plt.figure(figsize=(9, 5.5))
    ax = fig.add_subplot(111)

    ax.plot(xs, ys, marker="o")
    ax.axhline(1.0, linestyle="--")

    ax.set_title(f"Square GEMM Speedup (ref/gen) {title_suffix}".strip())
    ax.set_xlabel("N (square size)")
    ax.set_ylabel("speedup (ref_mean / gen_mean)  —  >1 means custom faster")

    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs], rotation=45, ha="right")

    # mark best point
    best = max(rows_sorted, key=lambda r: float(r["speedup"]))
    bx, by = int(best["N"]), float(best["speedup"])
    ax.scatter([bx], [by], s=120)
    ax.text(bx, by, f"  best {by:.2f}×", va="center")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    fig.savefig(out_png_path, dpi=220)
    plt.close(fig)


# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(
        description="Replay evaluate() + sweep square matmul sizes using YOUR existing eval module functions/kernels."
    )
    ap.add_argument("--path", required=True, help="Path to a single node JSON file")
    ap.add_argument(
        "--eval_module",
        default="cutegen.evaluate",
        help="Python module path that contains evaluate(), get_baseline_time(), get_wallclock_time()",
    )

    # Optional agent-like evaluate() call
    ap.add_argument("--no_eval", action="store_true", help="Skip calling evaluate() first")
    ap.add_argument("--no_time", action="store_true", help="Disable timing inside evaluate()")
    ap.add_argument("--profile", action="store_true", help="Enable profiling inside evaluate()")
    ap.add_argument("--no_torch_compile", action="store_true", help="Force torch_compile=False in baseline")

    # Timing knobs for eval module time fns
    ap.add_argument("--num_warmups", type=int, default=5)
    ap.add_argument("--num_trials", type=int, default=100)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)

    # Sweep
    ap.add_argument("--sweep", action="store_true", help="Run square size sweep and emit CSV + plot.")
    ap.add_argument("--nsizes", default="", help="Comma-separated N list: '256,512,1024,2048'")
    ap.add_argument("--out_dir", default="./mm_sweep_out")
    ap.add_argument("--csv_name", default="square_mm_sweep.csv")
    ap.add_argument("--plot_name", default="square_mm_speedup.png")

    args = ap.parse_args()

    node_path = os.path.abspath(args.path)
    d = load_node_json(node_path)
    node = make_node_obj(d)

    if not node.ref or not node.src:
        raise RuntimeError("Node JSON missing required fields: 'ref' and/or 'src'.")

    mod = importlib.import_module(args.eval_module)
    if not hasattr(mod, "get_baseline_time") or not hasattr(mod, "get_wallclock_time"):
        raise RuntimeError(f"Module '{args.eval_module}' must expose get_baseline_time() and get_wallclock_time().")

    # Print constants so we know exactly what evaluate module is configured for
    if hasattr(mod, "EVAL_COLD_CACHE"):
        print(f"[exact] {args.eval_module}.EVAL_COLD_CACHE = {getattr(mod, 'EVAL_COLD_CACHE')}")
    if hasattr(mod, "BENCHMARK_TORCH_COMPILE"):
        print(f"[exact] {args.eval_module}.BENCHMARK_TORCH_COMPILE = {getattr(mod, 'BENCHMARK_TORCH_COMPILE')}")
    if hasattr(mod, "TORCH_COMPILE_MODE"):
        print(f"[exact] {args.eval_module}.TORCH_COMPILE_MODE = {getattr(mod, 'TORCH_COMPILE_MODE')}")

    # 1) Optional: call evaluate() once like your agent
    out = node
    if not args.no_eval:
        if not hasattr(mod, "evaluate"):
            raise RuntimeError(f"Module '{args.eval_module}' does not have evaluate().")
        print(f"[exact] Calling evaluate() on uuid={node.uuid} depth={node.depth}")
        out_eval = mod.evaluate(
            node,
            get_time=(not args.no_time),
            get_profile=args.profile,
            torch_compile=(False if args.no_torch_compile else getattr(mod, "BENCHMARK_TORCH_COMPILE", False)),
            torch_compile_mode=getattr(mod, "TORCH_COMPILE_MODE", "default"),
        )
        if out_eval is not None:
            out = out_eval

        print("[exact] ================= RESULT =================")
        print(f"[exact] error_type = {getattr(out, 'error_type', None)}")
        print(f"[exact] compile msg = {out.metadata.get('compile','')[:500]}")
        print(f"[exact] correct msg = {out.metadata.get('correct','')[:500]}")
        print(f"[exact] time = {getattr(out, 'time', None)}")
        print(f"[exact] perf = {getattr(out, 'perf', None)}")
        print("[exact] ========================================")

    # 2) Sweep: override ref get_inputs() to force NxN, then time ref/gen using module functions
    if args.sweep:
        device = torch.device(f"cuda:{args.device}")
        ns = parse_nsizes_arg(args.nsizes) if args.nsizes.strip() else default_square_sizes()

        out_dir = os.path.abspath(args.out_dir)
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, args.csv_name)
        plot_path = os.path.join(out_dir, args.plot_name)

        print(f"[sweep] Running {len(ns)} square sizes on device={device} warmups={args.num_warmups} trials={args.num_trials}")

        rows = []

        sum_ref_mean = 0.0
        sum_gen_mean = 0.0
        sum_speedup = 0.0
        count = 0

        base_build = tempfile.mkdtemp(prefix=f"square_mm_build_{node.uuid}_")

        try:
            for i, n in enumerate(ns, 1):
                print(f"[sweep] [{i}/{len(ns)}] N={n}")

                # Match your semantics: ref controls inputs. Keep gen_src unchanged.
                ref_src = inject_square_mm_get_inputs_override(node.ref, n)
                gen_src = node.src

                # Fresh meta per point (avoid cross-contamination)
                meta = {}

                # Unique build dir per size so builds won't clobber
                build_dir = os.path.join(base_build, f"N{n}")
                os.makedirs(build_dir, exist_ok=True)

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

                # --- NEW: update sweep aggregates ---
                sum_ref_mean += ref_mean
                sum_gen_mean += gen_mean
                sum_speedup += speedup
                count += 1


                rows.append({
                    "N": n,
                    "ref_mean_ms": ref_mean,
                    "ref_std_ms": float(ref_stats.get("std", 0.0)),
                    "gen_mean_ms": gen_mean,
                    "gen_std_ms": float(gen_stats.get("std", 0.0)),
                    "speedup": speedup,
                })

                print(f"[sweep]    REF={ref_mean:.6g} ms | GEN={gen_mean:.6g} ms | speedup={speedup:.3f}x")

        finally:
            shutil.rmtree(base_build, ignore_errors=True)

        if rows:
            write_csv(rows, csv_path)
            print(f"[sweep] CSV: {csv_path}")

            title_suffix = f"(warmups={args.num_warmups}, trials={args.num_trials})"
            make_speedup_plot(rows, plot_path, title_suffix=title_suffix)
            print(f"[sweep] Plot: {plot_path}")

            best = max(rows, key=lambda r: float(r["speedup"]))
            print("[sweep] BEST:")
            print(f"        N={best['N']} speedup={best['speedup']:.3f}x (REF={best['ref_mean_ms']:.6g} ms, GEN={best['gen_mean_ms']:.6g} ms)")
        else:
            print("[sweep] No results produced.")

        # --- NEW: print sweep averages ---
        avg_ref = sum_ref_mean / count
        avg_gen = sum_gen_mean / count
        avg_speedup = sum_speedup / count

        # "speedup of averages" (often more meaningful than avg of ratios)
        speedup_from_avgs = (avg_ref / avg_gen) if avg_gen > 0 else float("inf")


        print("[sweep] AVERAGES (unweighted across N):")
        print(
            f"        ref_mean_ms={avg_ref:.6g} | gen_mean_ms={avg_gen:.6g} | avg_speedup={avg_speedup:.3f}x | speedup_from_avgs={speedup_from_avgs:.3f}x")



if __name__ == "__main__":
    main()
