#!/usr/bin/env python3
import argparse
import importlib
import json
import os
from types import SimpleNamespace
import tempfile
import shutil
import re
import math

import torch
import subprocess, sys


def _safe_write_text(path: str, s: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(s if s is not None else "")


def alias_model_to_modelnew(src: str) -> str:
    """
    eval's Nsight profiler expects GEN_SRC to define ModelNew.
    Reference code usually defines Model (not ModelNew).
    This keeps semantics identical while satisfying the profiler.
    """
    return src + "\n\n# --- alias for Nsight profiler compatibility ---\nModelNew = Model\n"


def run_ncu_profile_via_eval(
    mod,
    ref_src: str,
    gen_src: str,
    meta: dict,
    build_dir: str,
    device: torch.device,
    tag: str,
    num_warmups: int = 40,
    num_iters: int = 1,
    seed: int = 42,
):
    """
    Reuse the EXACT Nsight Compute logic already implemented in cutgen.evaluate (run_nsight_profile()).

    Saves:
      - <build_dir>/ncu_<tag>_stdout.txt
      - <build_dir>/ncu_<tag>_stderr.txt
      - <build_dir>/ncu_<tag>_metrics.json  (if parse succeeded)
      - <build_dir>/ncu_<tag>_cmd.txt
    """
    if not hasattr(mod, "run_nsight_profile"):
        raise RuntimeError(
            f"{mod.__name__} has no run_nsight_profile(). You already implemented it there—export it."
        )

    # Use a fresh per-run metadata dict so ref/gen don't overwrite each other
    prof_meta = {}

    res = mod.run_nsight_profile(
        ref_src,
        gen_src,
        prof_meta,
        build_directory=build_dir,
        device=device,
        num_warmups=num_warmups,
        num_iters=num_iters,
        seed_num=seed,
    )

    _safe_write_text(os.path.join(build_dir, f"ncu_{tag}_cmd.txt"), prof_meta.get("profile_cmd", ""))
    _safe_write_text(os.path.join(build_dir, f"ncu_{tag}_stdout.txt"), prof_meta.get("profile_stdout", ""))
    _safe_write_text(os.path.join(build_dir, f"ncu_{tag}_stderr.txt"), prof_meta.get("profile_stderr", ""))

    metrics = prof_meta.get("nsight_metrics", None)
    if metrics is not None:
        with open(os.path.join(build_dir, f"ncu_{tag}_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    meta[f"ncu_{tag}_returncode"] = prof_meta.get("profile_returncode", None)
    meta[f"ncu_{tag}_error"] = prof_meta.get("profile_error", "")
    meta[f"ncu_{tag}_kernel"] = (metrics or {}).get("kernel", None)

    return res, metrics


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
    node.metadata = d.get("metadata", {}) or {}
    node.save_folder_path = d.get("save_folder_path", None)
    return node


# -------------------- Diagonal multiply shapes (A: (N,), B: (N,M)) --------------------
def default_diag_shapes():
    shapes = [
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (4096, 1024),
        (8192, 1024),
        (16384, 1024),
        (1024, 4096),
        (2048, 8192),
        (4096, 16384),
        (1024, 1025),
        (4096, 4099),
    ]
    out, seen = [], set()
    for t in shapes:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def parse_diag_shapes_arg(s: str):
    s = s.strip()
    if not s:
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^(\d+)\s*[xX]\s*(\d+)$", part)
        if not m:
            raise ValueError(
                f"Bad shape '{part}'. Expected comma-separated NxM like 4096x1024,1024x4096."
            )
        N, M = int(m.group(1)), int(m.group(2))
        out.append((N, M))
    return out


# -------------------- Source patching: override get_inputs for diagmul --------------------
def inject_diag_get_inputs_override(ref_src: str, N: int, M: int) -> str:
    override = f"""
# ===================== [replay diagmul override] =====================
def get_inputs():
    import torch
    A = torch.rand({N})
    B = torch.rand({N}, {M})
    return [A, B]
# ================================================================
"""
    return ref_src + "\n" + override


# -------------------- CSV --------------------
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


# -------------------- Correctness check --------------------
@torch.no_grad()
def check_correctness(ref_src: str, gen_src: str, device: torch.device, seed: int, build_dir: str):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ref_ctx = {}
    exec(ref_src, ref_ctx, ref_ctx)
    if "Model" not in ref_ctx:
        raise RuntimeError("ref_src did not define a Model class.")
    if "get_inputs" not in ref_ctx:
        raise RuntimeError("ref_src did not define get_inputs().")

    gen_ctx = {}
    exec(gen_src, gen_ctx, gen_ctx)
    if "ModelNew" not in gen_ctx:
        raise RuntimeError("gen_src did not define a ModelNew class.")

    ModelRef = ref_ctx["Model"]
    ModelGen = gen_ctx["ModelNew"]

    model_ref = ModelRef().to(device)
    model_gen = ModelGen().to(device)

    A, B = ref_ctx["get_inputs"]()
    A = A.to(device)
    B = B.to(device)

    y_ref = model_ref(A, B)
    y_gen = model_gen(A, B)

    atol = 1e-4
    rtol = 1e-4
    ok = torch.allclose(y_ref, y_gen, atol=atol, rtol=rtol)

    max_abs = (y_ref - y_gen).abs().max().item()
    denom = y_ref.abs().max().item()
    rel = max_abs / (denom + 1e-12)

    return ok, max_abs, rel


# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser(
        description="Sweep diagonal-matmul cases: C = diag(A) @ B where A is (N,) and B is (N,M). Measures correctness + timing."
    )
    ap.add_argument("--path", required=True, help="Path to a single node JSON file")
    ap.add_argument("--eval_module", default="cutgen.evaluate")

    ap.add_argument("--no_eval", action="store_true")
    ap.add_argument("--no_time", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--no_torch_compile", action="store_true")

    ap.add_argument("--num_warmups", type=int, default=5)
    ap.add_argument("--num_trials", type=int, default=100)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--shapes", default="")
    ap.add_argument("--out_dir", default="./diag_sweep_out")
    ap.add_argument("--csv_name", default="diagmul_sweep.csv")

    ap.add_argument("--skip_correctness", action="store_true")

    # Per-shape Nsight (ref + gen) written under out_dir
    ap.add_argument("--profile_each", action="store_true",
                    help="Run Nsight Compute for BOTH ref and gen for every shape in the sweep.")
    ap.add_argument("--ncu_warmups", type=int, default=40)
    ap.add_argument("--ncu_iters", type=int, default=1)

    args = ap.parse_args()

    node_path = os.path.abspath(args.path)
    d = load_node_json(node_path)
    node = make_node_obj(d)

    if not node.ref or not node.src:
        raise RuntimeError("Node JSON missing required fields: 'ref' and/or 'src'.")

    mod = importlib.import_module(args.eval_module)
    if not hasattr(mod, "get_baseline_time") or not hasattr(mod, "get_wallclock_time"):
        raise RuntimeError(
            f"Module '{args.eval_module}' must expose get_baseline_time() and get_wallclock_time()."
        )

    # Optional: call evaluate() once on the original node JSON
    out = node
    if not args.no_eval:
        if not hasattr(mod, "evaluate"):
            raise RuntimeError(f"Module '{args.eval_module}' does not have evaluate().")
        out_eval = mod.evaluate(
            node,
            get_time=(not args.no_time),
            get_profile=args.profile,
            torch_compile=(False if args.no_torch_compile else getattr(mod, "BENCHMARK_TORCH_COMPILE", False)),
            torch_compile_mode=getattr(mod, "TORCH_COMPILE_MODE", "default"),
        )
        if out_eval is not None:
            out = out_eval

    if not args.sweep:
        print("[info] --sweep not provided; nothing to do.")
        return

    device = torch.device(f"cuda:{args.device}")
    shapes = parse_diag_shapes_arg(args.shapes) if args.shapes.strip() else default_diag_shapes()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, args.csv_name)

    rows = []
    sum_ref_mean = 0.0
    sum_gen_mean = 0.0
    sum_speedup = 0.0
    count = 0

    for case_idx, (N, M) in enumerate(shapes, 1):
        shape_str = f"{N}x{M}"
        print(f"[sweep] [{case_idx}/{len(shapes)}] shape={shape_str}")

        ref_src = inject_diag_get_inputs_override(node.ref, N, M)
        gen_src = node.src

        meta = {}

        # Persistent Nsight artifacts under out_dir
        build_dir = os.path.join(out_dir, "ncu_artifacts", shape_str)
        os.makedirs(build_dir, exist_ok=True)

        ok = True
        max_abs = float("nan")
        max_rel = float("nan")
        if not args.skip_correctness:
            try:
                ok, max_abs, max_rel = check_correctness(
                    ref_src, gen_src, device=device, seed=args.seed, build_dir=build_dir
                )
            except Exception as e:
                ok = False
                print(f"[sweep][correctness] FAILED: {e}")

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

        # Nsight profiling (per shape): ref AND gen
        if args.profile_each:
            # IMPORTANT:
            # - ref profiling must pass a GEN_SRC that defines ModelNew (alias Model->ModelNew)
            # - gen profiling uses (ref_src, gen_src) so inputs come from ref_src's get_inputs()
            try:
                run_ncu_profile_via_eval(
                    mod,
                    ref_src=ref_src,
                    gen_src=alias_model_to_modelnew(ref_src),  # <-- FIX
                    meta=meta,
                    build_dir=build_dir,
                    device=device,
                    tag="ref",
                    num_warmups=args.ncu_warmups,
                    num_iters=args.ncu_iters,
                    seed=args.seed,
                )
            except Exception as e:
                print(f"[sweep][ncu][ref] FAILED: {e}")

            try:
                run_ncu_profile_via_eval(
                    mod,
                    ref_src=ref_src,
                    gen_src=gen_src,
                    meta=meta,
                    build_dir=build_dir,
                    device=device,
                    tag="gen",
                    num_warmups=args.ncu_warmups,
                    num_iters=args.ncu_iters,
                    seed=args.seed,
                )
            except Exception as e:
                print(f"[sweep][ncu][gen] FAILED: {e}")

        sum_ref_mean += ref_mean
        sum_gen_mean += gen_mean
        sum_speedup += speedup
        count += 1

        rows.append({
            "shape": shape_str,
            "N": N,
            "M": M,
            "correct": bool(ok),
            "max_abs_err": float(max_abs),
            "max_rel_err": float(max_rel),
            "ref_mean_ms": ref_mean,
            "ref_std_ms": float(ref_stats.get("std", 0.0)),
            "gen_mean_ms": gen_mean,
            "gen_std_ms": float(gen_stats.get("std", 0.0)),
            "speedup": speedup,
            "ncu_ref_returncode": meta.get("ncu_ref_returncode"),
            "ncu_gen_returncode": meta.get("ncu_gen_returncode"),
            "ncu_ref_kernel": meta.get("ncu_ref_kernel"),
            "ncu_gen_kernel": meta.get("ncu_gen_kernel"),
            "ncu_ref_error": meta.get("ncu_ref_error"),
            "ncu_gen_error": meta.get("ncu_gen_error"),
        })

        corr_str = "OK" if ok else "BAD"
        print(f"[sweep]    CORR={corr_str} | REF={ref_mean:.6g} ms | GEN={gen_mean:.6g} ms | speedup={speedup:.3f}x")

    if rows:
        write_csv(rows, csv_path)
        print(f"[sweep] CSV: {csv_path}")

        best = max(rows, key=lambda r: float(r["speedup"]))
        print("[sweep] BEST:", best)

        avg_ref = sum_ref_mean / count
        avg_gen = sum_gen_mean / count
        avg_speedup = sum_speedup / count
        speedup_from_avgs = (avg_ref / avg_gen) if avg_gen > 0 else float("inf")
        print(f"[sweep] avg_speedup={avg_speedup:.3f}x | speedup_from_avgs={speedup_from_avgs:.3f}x")
    else:
        print("[sweep] No results produced.")


if __name__ == "__main__":
    main()