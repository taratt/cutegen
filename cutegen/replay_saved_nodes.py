#!/usr/bin/env python3
import argparse
import importlib
import json
import os
from types import SimpleNamespace
import tempfile
import shutil
import torch

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

def main():
    ap = argparse.ArgumentParser(description="Call your project evaluate(node, ...) on a single saved node json.")
    ap.add_argument("--path", required=True, help="Path to a single node JSON file")
    ap.add_argument(
        "--eval_module",
        default="cutegen.evaluate",  # <-- change this default if your file is named differently
        help="Python module path that contains evaluate() (e.g., cutegen.evaluate or cutegen.evaluator)",
    )
    ap.add_argument(
        "--no_time",
        action="store_true",
        help="Disable timing inside evaluate() (get_time=False)",
    )
    ap.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling inside evaluate() (get_profile=True). Default is False.",
    )
    ap.add_argument(
        "--no_torch_compile",
        action="store_true",
        help="Force torch_compile=False in evaluate() baseline benchmarking.",
    )
    ap.add_argument(
        "--print_times",
        action="store_true",
        help="Also call get_baseline_time() and get_wallclock_time() from the eval module and print both.",
    )
    ap.add_argument(
        "--num_warmups",
        type=int,
        default=5,
        help="Warmups for timing calls.",
    )
    ap.add_argument(
        "--num_trials",
        type=int,
        default=100,
        help="Trials for timing calls.",
    )
    ap.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index to use for timing calls.",
    )

    args = ap.parse_args()

    node_path = os.path.abspath(args.path)
    d = load_node_json(node_path)
    node = make_node_obj(d)

    # sanity
    if not node.ref or not node.src:
        raise RuntimeError("Node JSON missing required fields: 'ref' and/or 'src'.")

    # import your real evaluate()
    mod = importlib.import_module(args.eval_module)
    if not hasattr(mod, "evaluate"):
        raise RuntimeError(f"Module '{args.eval_module}' does not have evaluate().")

    evaluate_fn = getattr(mod, "evaluate")

    # optional: print the actual EVAL_COLD_CACHE constant the module is using
    if hasattr(mod, "EVAL_COLD_CACHE"):
        print(f"[exact] {args.eval_module}.EVAL_COLD_CACHE = {getattr(mod, 'EVAL_COLD_CACHE')}")
    if hasattr(mod, "BENCHMARK_TORCH_COMPILE"):
        print(f"[exact] {args.eval_module}.BENCHMARK_TORCH_COMPILE = {getattr(mod, 'BENCHMARK_TORCH_COMPILE')}")

    # Call evaluate() exactly like your agent does
    print(f"[exact] Calling evaluate() on uuid={node.uuid} depth={node.depth}")
    out = evaluate_fn(
        node,
        get_time=(not args.no_time),
        get_profile=args.profile,
        torch_compile=(False if args.no_torch_compile else getattr(mod, "BENCHMARK_TORCH_COMPILE", False)),
        torch_compile_mode=getattr(mod, "TORCH_COMPILE_MODE", "default"),
    )

    # evaluate() returns node (or None in some failure paths); either way node is mutated
    if out is None:
        out = node
    # -------------------- PRINT BOTH TIMES (REF + GEN) --------------------
    if args.print_times:
        if not hasattr(mod, "get_baseline_time") or not hasattr(mod, "get_wallclock_time"):
            raise RuntimeError(f"Module '{args.eval_module}' does not expose get_baseline_time/get_wallclock_time.")

        device = torch.device(f"cuda:{args.device}")

        # Use a fresh build dir so extension builds don't collide
        build_dir = tempfile.mkdtemp(prefix=f"replay_build_{node.uuid}_")

        try:
            print("[exact] ===== Timing (calling eval module functions directly) =====")

            ref_stats = mod.get_baseline_time(
                node.ref,
                out.metadata,
                num_warmups=args.num_warmups,
                num_trials=args.num_trials,
                seed_num=42,
                device=device,
                torch_compile=(False if args.no_torch_compile else getattr(mod, "BENCHMARK_TORCH_COMPILE", False)),
                torch_compile_mode=getattr(mod, "TORCH_COMPILE_MODE", "default"),
                get_torch_graph=False,
            )

            gen_stats = mod.get_wallclock_time(
                node.ref,
                node.src,
                out.metadata,
                num_warmups=args.num_warmups,
                num_trials=args.num_trials,
                seed_num=42,
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

if __name__ == "__main__":
    main()
