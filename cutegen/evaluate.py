import torch
import torch.nn as nn
import numpy as np
import contextlib
import os
import pathlib
import time
import random
import gc
import torch._dynamo as dynamo
import subprocess
import json
import sys
import tempfile
import multiprocessing
from multiprocessing.connection import Connection
import traceback
from torch.fx import symbolic_trace
import GPUtil
import psutil
import glob
import re
from typing import Dict, Any, List

from cutegen.node import Node, ErrorType
from cutegen.util import set_seed, debug_print, acquire_gpu, release_gpu, remove_build_directory, read_file_with_lock, write_file_with_lock

####### CONSTANTS #######
from cutegen.config import CUTEGEN_BASE_PATH, LOAD_MODEL_BACKOFF_TIME, RUN_MODEL_BACKOFF_TIME, GPU_REQ_SPACE, CPU_REQ_SPACE, BUILD_DIRECTORY_BASE, EVAL_RUN_TIMEOUT, BENCHMARK_TORCH_COMPILE, TORCH_COMPILE_MODE, EVAL_COLD_CACHE, NSIGHT_COMPUTE_BIN, NSIGHT_COMPUTE_SET, USE_PROFILING, KERNEL_BACKEND


import shutil

def tensor_bytes(x: torch.Tensor) -> int:
    return x.numel() * x.element_size()

def sum_tensor_bytes(xs) -> int:
    total = 0
    for x in xs:
        if isinstance(x, torch.Tensor):
            total += tensor_bytes(x)
    return total

def estimate_required_bytes_from_inputs(xs) -> int:
    """
    Conservative but not absurd estimate for forward memory.

    We assume roughly:
      input + output + some overhead

    For elementwise ops this is close to 2x input.
    For many other kernels this is still a safer heuristic than 3x.
    """
    input_bytes = sum_tensor_bytes(xs)
    return (2 * input_bytes) + (512 << 20)   # +512 MiB buffer

def wait_for_gpu_memory(required_bytes: int, device: int):
    """
    Wait until enough GPU memory is available.

    required_bytes is the estimated bytes needed by the upcoming step.
    We also enforce the configured GPU_REQ_SPACE fraction of total VRAM.
    """
    return
    while True:
        free_b, total_b = torch.cuda.mem_get_info(device)
        gpu_free_frac = free_b / total_b

        # Need both:
        #  1) enough absolute free bytes
        #  2) enough fractional free space from config
        #if free_b >= required_bytes and gpu_free_frac >= GPU_REQ_SPACE:
        if free_b >= required_bytes:
            return

        debug_print(
            f"[GPU WAIT] free={free_b / (1024**3):.2f} GiB, "
            f"need={required_bytes / (1024**3):.2f} GiB, "
            f"gpu_free_frac={gpu_free_frac:.3f}, "
            f"gpu_req_frac={GPU_REQ_SPACE}"
        )
        time.sleep(random.randint(1, 5) * RUN_MODEL_BACKOFF_TIME)

def wait_for_resources(device_index: int):
    """
    Wait until CPU usage and GPU free fraction satisfy configured thresholds.
    """
    while True:
        CPU_used = psutil.virtual_memory().percent
        free_b, total_b = torch.cuda.mem_get_info(device_index)
        gpu_free_frac = free_b / total_b

        if CPU_used <= CPU_REQ_SPACE and gpu_free_frac >= GPU_REQ_SPACE:
            return

        debug_print(
            f"[RESOURCE WAIT] CPU_used={CPU_used:.2f}% "
            f"(limit {CPU_REQ_SPACE}), "
            f"GPU_free_frac={gpu_free_frac:.3f} "
            f"(need >= {GPU_REQ_SPACE})"
        )
        time.sleep(random.randint(1, 5) * RUN_MODEL_BACKOFF_TIME)

def clone_to_device(xs, device):
    ys = []
    for x in xs:
        if isinstance(x, torch.Tensor):
            ys.append(x.detach().clone().cuda(device=device))
        else:
            ys.append(x)
    return ys

def cleanup_cuda(device):
    try:
        torch.cuda.synchronize(device=device)
    except Exception:
        pass
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

def validate_tensor_output_contract(ref_output, gen_output, metadata) -> bool:
    """
    Enforce that generated output behaves like the reference output:
    same tensor-ness, device type, dtype, and shape.
    """
    if not isinstance(ref_output, torch.Tensor):
        metadata["correct"] = f"Reference model returned non-tensor output: {type(ref_output)}"
        return False

    if not isinstance(gen_output, torch.Tensor):
        metadata["correct"] = f"Generated model returned non-tensor output: {type(gen_output)}"
        return False

    if ref_output.device.type != gen_output.device.type:
        metadata["correct"] = (
            f"Output device mismatch, expected {ref_output.device}, got {gen_output.device}"
        )
        return False

    if ref_output.dtype != gen_output.dtype:
        metadata["correct"] = (
            f"Output dtype mismatch, expected {ref_output.dtype}, got {gen_output.dtype}"
        )
        return False

    if ref_output.shape != gen_output.shape:
        metadata["correct"] = (
            f"Output shape mismatch, expected {ref_output.shape}, got {gen_output.shape}"
        )
        return False

    return True
def safe_rmtree(path: str):
    """Delete a directory if it exists, ignore errors."""
    try:
        if path and os.path.exists(path):
            shutil.rmtree(path)
            print(f"removed {path}")
    except Exception:
        print(f"failed to remove {path}: {e}")

def load_original_model_and_inputs(ref: str, context: dict, metadata: dict) -> tuple[nn.Module, callable, callable]:
    try:
        compile(ref, "<string>", "exec")
    except SyntaxError as e:
        metadata["compile"] = f"Syntax Error in original code {str(e)}"
        return None
    try:
        exec(ref, context)  # expose to current namespace
    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        time.sleep(LOAD_MODEL_BACKOFF_TIME * random.randint(1,5))
        return load_original_model_and_inputs(ref, context, metadata)
    except Exception as e:
        metadata["compile"] = f"Error in executing original code {str(e)}"
        return None
    # these should be defined in the original model code and present in the context
    get_init_inputs_fn = context.get("get_init_inputs")
    get_inputs_fn = context.get("get_inputs")
    Model = context.get("Model")
    return (Model, get_init_inputs_fn, get_inputs_fn)

def load_custom_model(model_custom_src: str, context: dict, metadata: dict, build_directory: str = None) -> nn.Module:
    assert(build_directory is not None)
    build_path = pathlib.Path(build_directory)
    build_path.mkdir(parents=True, exist_ok=True)

    context["BUILD_DIRECTORY"] = build_directory

    model_custom_src = (
        "import os\nimport gc\n"
        f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_directory}'\n"
    ) + model_custom_src
    model_custom_src = model_custom_src.replace("verbose=False", "verbose=True")
    retval = True

    read_fd, write_fd = os.pipe()
    old_out, old_err = os.dup(1), os.dup(2)

    os.dup2(write_fd, 1)
    os.dup2(write_fd, 2)
    os.close(write_fd)

    try:
        compile(model_custom_src, "<string>", "exec")
        exec(model_custom_src, context)
        validator = context.get("validate_generated_code")
        if KERNEL_BACKEND == "ptx" and callable(validator):
            validator()
        torch.cuda.synchronize()
    except Exception as e:
        metadata["compile"] = (
            f"Generated code compilation/loading error:\n{str(e)}"
        )
        retval = None

    try:
        ModelNew = context.get("ModelNew")
    except Exception as e:
        metadata["compile"] = f"Error in executing generated code {str(e)}"
        retval = None

    os.dup2(old_out, 1)
    os.dup2(old_err, 2)
    os.close(old_out)
    os.close(old_err)

    error = ""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    with os.fdopen(read_fd, "r") as log_file:
        error = log_file.read()

    if retval is not None:
        return ModelNew
    else:
        metadata["compile"] += (
            f"\n Here is the full command line of the program execution: {error}"
            if error != "" else ""
        )
        return None

def _check_compile(ref_src, gen_src, metadata, build_directory=None):
    """
    This function tries to compile the generated source code.
    """

    context = {}
    if ref_src is not None:
        # checking ref src compilability is optimal.
        try:
            Model, get_init_inputs_fn, get_inputs_fn = load_original_model_and_inputs(ref_src, context, metadata)
        except Exception as e:
            return False
    try:
        ModelNew = load_custom_model(gen_src, context, metadata, build_directory)
        if ModelNew is None:
            metadata["compile"] += "\nLoading ModelNew failed: ModelNew is None"
            return False
    except Exception as e:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        return False
    return True

def override_get_inputs_cpu(context: dict):
    """
    Wrap whatever get_inputs() currently is so it returns CPU tensors.
    Works even if generated code's get_inputs() allocates CUDA tensors.
    """
    old_get_inputs = context.get("get_inputs", None)
    if old_get_inputs is None:
        return

    def get_inputs_cpu():
        xs = old_get_inputs()

        out = []
        for x in xs:
            if isinstance(x, torch.Tensor):
                # If generated code made it CUDA, bring it back to CPU
                if x.is_cuda:
                    x = x.detach().cpu()
                else:
                    x = x.detach()
            out.append(x)
        return out

    context["get_inputs"] = get_inputs_cpu


def _check_correct(ref_src, gen_src, metadata,
                   num_trials=10, seed_num=42,
                   build_directory=None, device=None):
    """
    Full-size correctness checker.

    Policy:
    - do NOT slice or chunk inputs
    - if the reference fits full-size, the generated kernel must fit full-size too
    - wait for enough free GPU memory before running
    - enforce output device/dtype/shape contract
    - compare values on CPU only after contract validation
    """
    torch.cuda.set_device(device)
    metadata["hardware"] = torch.cuda.get_device_name(device=device)
    metadata["device"] = str(device)

    ref_context = {}
    gen_context = {}

    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    Model, get_init_inputs_fn, get_inputs_fn = load_original_model_and_inputs(
        ref_src, ref_context, metadata
    )
    ModelNew = load_custom_model(gen_src, gen_context, metadata, build_directory)

    try:
        set_seed(seed_num)
        init_inputs = get_init_inputs_fn()
        init_inputs = [
            x.cuda(device=metadata["device"]) if isinstance(x, torch.Tensor) else x
            for x in init_inputs
        ]

        with torch.inference_mode():
            set_seed(seed_num)
            original_model = Model(*init_inputs)
            if isinstance(original_model, nn.Module):
                original_model = original_model.cuda(device=device)
                original_model.eval()

            set_seed(seed_num)
            custom_model = ModelNew(*init_inputs)
            if isinstance(custom_model, nn.Module):
                custom_model = custom_model.cuda(device=device)
                custom_model.eval()

        del init_inputs
        cleanup_cuda(device)

        pass_count = 0
        torch.manual_seed(seed_num)
        correctness_trial_seeds = [
            torch.randint(0, 2 ** 32 - 1, (1,)).item() for _ in range(num_trials)
        ]

        last_diff_msg = None

        with torch.inference_mode():
            for trial in range(num_trials):
                trial_seed = correctness_trial_seeds[trial]

                debug_print(
                    "REF globals batch_size/dim currently:"
                    + str(get_inputs_fn.__globals__.get("batch_size"))
                    + str(get_inputs_fn.__globals__.get("dim"))
                )

                set_seed(trial_seed)
                inputs_cpu = get_inputs_fn()

                for i, inp in enumerate(inputs_cpu):
                    if isinstance(inp, torch.Tensor):
                        debug_print(
                            f"INPUT[{i}] shape={inp.shape}, device={inp.device}, dtype={inp.dtype}"
                        )

                required_bytes = estimate_required_bytes_from_inputs(inputs_cpu)
                wait_for_gpu_memory(
                    required_bytes,
                    device.index if isinstance(device, torch.device) else int(device)
                )

                # -------- Run generated model first --------
                try:
                    set_seed(trial_seed)
                    inputs_new = clone_to_device(inputs_cpu, device)
                    output_new = custom_model(*inputs_new)
                    torch.cuda.synchronize(device=device)

                    if not isinstance(output_new, torch.Tensor):
                        metadata["correct"] = (
                            f"Generated model returned non-tensor output: {type(output_new)}"
                        )
                        cleanup_cuda(device)
                        return False

                    if not output_new.is_cuda:
                        metadata["correct"] = (
                            f"Generated model returned non-CUDA output: {output_new.device}"
                        )
                        cleanup_cuda(device)
                        return False

                    gen_output_shape = output_new.shape
                    gen_output_dtype = output_new.dtype

                    # IMPORTANT: move generated output to CPU now
                    output_new_cpu = output_new.detach().to(torch.float32).cpu()

                except torch.cuda.OutOfMemoryError as e:
                    metadata["correct"] = f"OOM in generated model correctness run: {e}"
                    cleanup_cuda(device)
                    return False
                except Exception as e:
                    metadata["correct"] = f"Runtime error when checking correctness (generated model): {str(e)}"
                    cleanup_cuda(device)
                    return False
                finally:
                    if "inputs_new" in locals():
                        del inputs_new
                    if "output_new" in locals():
                        del output_new
                    cleanup_cuda(device)

                # -------- Run reference model second --------
                try:
                    set_seed(trial_seed)
                    inputs_ref = clone_to_device(inputs_cpu, device)
                    output = original_model(*inputs_ref)
                    torch.cuda.synchronize(device=device)

                    if not isinstance(output, torch.Tensor):
                        metadata["correct"] = (
                            f"Reference model returned non-tensor output: {type(output)}"
                        )
                        cleanup_cuda(device)
                        return False

                    if not output.is_cuda:
                        metadata["correct"] = (
                            f"Reference model returned non-CUDA output: {output.device}"
                        )
                        cleanup_cuda(device)
                        return False

                    if output.shape != gen_output_shape:
                        metadata["correct"] = (
                            f"Output shape mismatch, expected {output.shape}, got {gen_output_shape}"
                        )
                        cleanup_cuda(device)
                        return False

                    if output.dtype != gen_output_dtype:
                        metadata["correct"] = (
                            f"Output dtype mismatch, expected {output.dtype}, got {gen_output_dtype}"
                        )
                        cleanup_cuda(device)
                        return False

                    # Move reference output to CPU now
                    output_cpu = output.detach().to(torch.float32).cpu()

                except torch.cuda.OutOfMemoryError as e:
                    metadata["correct"] = f"OOM in reference model correctness run: {e}"
                    cleanup_cuda(device)
                    return False
                except Exception as e:
                    metadata["correct"] = f"Runtime error when checking correctness (reference model): {str(e)}"
                    cleanup_cuda(device)
                    return False
                finally:
                    if "inputs_ref" in locals():
                        del inputs_ref
                    if "output" in locals():
                        del output
                    cleanup_cuda(device)

                # -------- Compare on CPU --------
                if not torch.allclose(output_cpu, output_new_cpu, atol=1e-02, rtol=1e-02):
                    diff = torch.abs(output_cpu - output_new_cpu)
                    max_diff = diff.max().item()
                    avg_diff = diff.mean().item()
                    metadata["correct"] = (
                        f"Output value mismatch, max diff: {max_diff}, avg diff: {avg_diff}"
                    )
                    del diff, output_cpu, output_new_cpu, inputs_cpu
                    cleanup_cuda(device)
                    return False
                else:
                    if trial == num_trials - 1:
                        diff = torch.abs(output_cpu - output_new_cpu)
                        max_diff = diff.max().item()
                        avg_diff = diff.mean().item()
                        last_diff_msg = (
                            f"Output value matched, max diff: {max_diff}, avg diff: {avg_diff}"
                        )
                        del diff

                del output_cpu, output_new_cpu, inputs_cpu
                cleanup_cuda(device)
                pass_count += 1

    except Exception as e:
        metadata["correct"] = f"Unexpected error during correctness check: {repr(e)}"
        cleanup_cuda(device)
        return False

    try:
        torch.cuda.synchronize(device=device)
    except Exception as e:
        metadata["correct"] = f"Runtime error when checking correctness: {str(e)}"
        cleanup_cuda(device)
        return False

    metadata["correct"] = (
        f"Passed {pass_count} out of {num_trials} trials: "
        f"{metadata['correct'] if 'correct' in metadata and metadata['correct'] else 'ALL PASSED'}"
    )
    if last_diff_msg is not None:
        metadata["correct"] += f"\n{last_diff_msg}"

    cleanup_cuda(device)
    return pass_count == num_trials

def _get_wallclock_time(ref_src, gen_src, metadata,
                        num_warmups=5, num_trials=100, seed_num=42,
                        build_directory=None, device=None):
    """
    Measure full-size wallclock time of the generated source code.
    Requires the generated kernel to return a CUDA tensor.
    """
    torch.cuda.set_device(device)
    metadata["hardware"] = torch.cuda.get_device_name(device=device)
    metadata["device"] = str(device)

    ref_context = {}
    gen_context = {}
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    Model, get_init_inputs_fn, get_inputs_fn = load_original_model_and_inputs(
        ref_src, ref_context, metadata
    )
    ModelNew = load_custom_model(gen_src, gen_context, metadata, build_directory)

    set_seed(seed_num)
    init_inputs = get_init_inputs_fn()
    init_inputs = [
        x.cuda(device=metadata["device"]) if isinstance(x, torch.Tensor) else x
        for x in init_inputs
    ]

    set_seed(seed_num)
    inputs_cpu = get_inputs_fn()
    required_bytes = estimate_required_bytes_from_inputs(inputs_cpu)
    wait_for_gpu_memory(
        required_bytes,
        device.index if isinstance(device, torch.device) else int(device)
    )

    inputs = [
        x.cuda(device=device) if isinstance(x, torch.Tensor) else x
        for x in inputs_cpu
    ]
    del inputs_cpu
    cleanup_cuda(device)

    elapsed_times = []

    with torch.no_grad():
        set_seed(seed_num)
        custom_model = ModelNew(*init_inputs)
        custom_model = custom_model.cuda(device=device)

        for _ in range(num_warmups):
            out = custom_model(*inputs)
            if not isinstance(out, torch.Tensor):
                raise RuntimeError(f"Generated model returned non-tensor output: {type(out)}")
            if not out.is_cuda:
                raise RuntimeError("Generated model returned non-CUDA output during timing")
            torch.cuda.synchronize(device=device)
            del out

        for _ in range(num_trials):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            out = custom_model(*inputs)
            end_event.record()

            if not isinstance(out, torch.Tensor):
                raise RuntimeError(f"Generated model returned non-tensor output: {type(out)}")
            if not out.is_cuda:
                raise RuntimeError("Generated model returned non-CUDA output during timing")

            torch.cuda.synchronize(device=device)
            elapsed_time_ms = start_event.elapsed_time(end_event)
            elapsed_times.append(elapsed_time_ms)
            del out

    timing_stats = {
        "mean": float(f"{np.mean(elapsed_times):.3g}"),
        "std": float(f"{np.std(elapsed_times):.3g}"),
        "min": float(f"{np.min(elapsed_times):.3g}"),
        "max": float(f"{np.max(elapsed_times):.3g}"),
        "num_trials": len(elapsed_times),
    }

    del inputs, init_inputs, custom_model
    cleanup_cuda(device)
    return timing_stats


def _get_baseline_time(ref_src, metadata,
                           num_warmups=5, num_trials=100, seed_num=42,
                           device=None, torch_compile=False, torch_compile_mode="default", get_torch_graph=False):
    """
    This function gets the wallclock time of the original source code.
    """
    # device: torch.device = torch.cuda.current_device()
    torch.cuda.set_device(device)
    metadata["hardware"] = torch.cuda.get_device_name(device=device)
    metadata["device"] = str(device)

    context = {}
    Model, get_init_inputs_fn, get_inputs_fn = load_original_model_and_inputs(ref_src, context, metadata)

    set_seed(seed_num)
    init_inputs = get_init_inputs_fn()
    init_inputs = [
        x.cuda(device=metadata["device"]) if isinstance(x, torch.Tensor) else x for x in init_inputs
    ]

    set_seed(seed_num)
    inputs = get_inputs_fn()
    inputs = [
        x.cuda(device=device) if isinstance(x, torch.Tensor) else x
        for x in inputs
    ]

    elapsed_times = []

    with torch.no_grad():
        set_seed(seed_num)  # set seed for reproducible weights
        model = Model(*init_inputs)
        model = model.cuda(device=device)
        if torch_compile:
            if get_torch_graph:
                gm = symbolic_trace(model)
                print("\n=== FX Graph Before Compilation ===")
                print(gm.graph)
                # print("\n=== Generated Python Code Before Compilation ===")
                # print(gm.code)
                model = torch.compile(model, backend="inductor", fullgraph=True)
                exported = torch.export.export(model, (inputs,)) # TODO: Fix
            else:
                model = torch.compile(model, mode="max-autotune")
            
        for _ in range(num_warmups):
            model(*inputs)
            torch.cuda.synchronize(device=device)

        for _ in range(num_trials):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            model(*inputs)
            end_event.record()

            torch.cuda.synchronize(device=device)
            elapsed_time_ms = start_event.elapsed_time(end_event)
            elapsed_times.append(elapsed_time_ms)

    # timing stats
    timing_stats = {
        "mean": float(f"{np.mean(elapsed_times):.3g}"),
        "std": float(f"{np.std(elapsed_times):.3g}"),
        "min": float(f"{np.min(elapsed_times):.3g}"),
        "max": float(f"{np.max(elapsed_times):.3g}"),
        "num_trials": len(elapsed_times),
    }
    torch.cuda.synchronize()
    del inputs, init_inputs, model
    cleanup_cuda(device)
    return timing_stats

def _get_baseline_time_2(ref_src, metadata,
                           num_warmups=5, num_trials=100, seed_num=42,
                           device=None, torch_compile=False, torch_compile_mode="default", get_torch_graph=False):
    """
    This function gets the wallclock time of the reference torch code.
    """
    # device: torch.device = torch.cuda.current_device()
    torch.cuda.set_device(device)
    metadata["hardware"] = torch.cuda.get_device_name(device=device)
    metadata["device"] = str(device)
    context = {}
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0" # don't block CUDA calls for better performance
    
     # We already know the generated code compiles and is correct, so we can skip lots of checks
    Model, get_init_inputs_fn, get_inputs_fn = load_original_model_and_inputs(ref_src, context, metadata)

    set_seed(seed_num)
    init_inputs = get_init_inputs_fn()
    init_inputs = [
        x.cuda(device=metadata["device"]) if isinstance(x, torch.Tensor) else x for x in init_inputs
    ]

    model = Model(*init_inputs)
    model = model.cuda(device=device)
    elapsed_times = []

    with torch.no_grad():

        for _ in range(num_warmups):
            inputs = get_inputs_fn()
            inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]
            model(*inputs)
            torch.cuda.synchronize(device=device)

        for _ in range(num_trials):
            inputs = get_inputs_fn()
            inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            model(*inputs)
            end_event.record()

            torch.cuda.synchronize(device=device)
            elapsed_time_ms = start_event.elapsed_time(end_event)
            elapsed_times.append(elapsed_time_ms)

    # timing stats
    timing_stats = {
        "mean": float(f"{np.mean(elapsed_times):.3g}"),
        "std": float(f"{np.std(elapsed_times):.3g}"),
        "min": float(f"{np.min(elapsed_times):.3g}"),
        "max": float(f"{np.max(elapsed_times):.3g}"),
        "num_trials": len(elapsed_times),
    }
    torch.cuda.synchronize()
    del inputs, init_inputs, model
    cleanup_cuda(device)
    return timing_stats

def _get_wallclock_time_2(ref_src, gen_src, metadata,
                          num_warmups=5, num_trials=100, seed_num=42,
                          build_directory=None, device=None):
    """
    Full-size cold-cache timing of the generated source code.
    Requires the generated kernel to return a CUDA tensor.
    """
    torch.cuda.set_device(device)
    metadata["hardware"] = torch.cuda.get_device_name(device=device)
    metadata["device"] = str(device)

    ref_context = {}
    gen_context = {}
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    Model, get_init_inputs_fn, get_inputs_fn = load_original_model_and_inputs(
        ref_src, ref_context, metadata
    )
    ModelNew = load_custom_model(gen_src, gen_context, metadata, build_directory)

    set_seed(seed_num)
    init_inputs = get_init_inputs_fn()
    init_inputs = [
        x.cuda(device=metadata["device"]) if isinstance(x, torch.Tensor) else x
        for x in init_inputs
    ]

    model = ModelNew(*init_inputs)
    model = model.cuda(device=device)

    elapsed_times = []

    with torch.no_grad():
        for _ in range(num_warmups):
            inputs_cpu = get_inputs_fn()
            required_bytes = estimate_required_bytes_from_inputs(inputs_cpu)
            wait_for_gpu_memory(
                required_bytes,
                device.index if isinstance(device, torch.device) else int(device)
            )

            inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in inputs_cpu
            ]
            del inputs_cpu

            out = model(*inputs)
            if not isinstance(out, torch.Tensor):
                raise RuntimeError(f"Generated model returned non-tensor output: {type(out)}")
            if not out.is_cuda:
                raise RuntimeError("Generated model returned non-CUDA output during timing")

            torch.cuda.synchronize(device=device)
            del out, inputs
            cleanup_cuda(device)

        for _ in range(num_trials):
            inputs_cpu = get_inputs_fn()
            required_bytes = estimate_required_bytes_from_inputs(inputs_cpu)
            wait_for_gpu_memory(
                required_bytes,
                device.index if isinstance(device, torch.device) else int(device)
            )

            inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in inputs_cpu
            ]
            del inputs_cpu

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            out = model(*inputs)
            end_event.record()

            if not isinstance(out, torch.Tensor):
                raise RuntimeError(f"Generated model returned non-tensor output: {type(out)}")
            if not out.is_cuda:
                raise RuntimeError("Generated model returned non-CUDA output during timing")

            torch.cuda.synchronize(device=device)
            elapsed_time_ms = start_event.elapsed_time(end_event)
            elapsed_times.append(elapsed_time_ms)

            del out, inputs
            cleanup_cuda(device)

    timing_stats = {
        "mean": float(f"{np.mean(elapsed_times):.3g}"),
        "std": float(f"{np.std(elapsed_times):.3g}"),
        "min": float(f"{np.min(elapsed_times):.3g}"),
        "max": float(f"{np.max(elapsed_times):.3g}"),
        "num_trials": len(elapsed_times),
    }

    del init_inputs, model
    cleanup_cuda(device)
    return timing_stats



def _worker_wrapper(func, args: tuple, kwargs: dict, conn: Connection):
    """
    Runs func(*args, **kwargs) in a child process, captures any exceptions,
    and sends back (success, result_or_exception, metadata) over conn.
    """
    
    # Redirect stderr to capture CUDA assertion messages
    import sys
    from io import StringIO
    stderr_capture = StringIO()
    old_stderr = sys.stderr
    sys.stderr = stderr_capture
    
    metadata = kwargs.get("metadata", {})

    try:
        result = func(*args, **kwargs)
        # assume metadata was one of the kwargs and mutated in‐place
        
        
        # Capture any stderr output (including CUDA assertions)
        stderr_content = stderr_capture.getvalue()
        if stderr_content:
            metadata["_cuda_stderr"] = stderr_content
            
        conn.send((True, result, metadata))
    except Exception as e:
        # capture stack for debugging
        tb = traceback.format_exc()
        metadata["_subproc_error"] = str(e)
        metadata["_subproc_traceback"] = tb
        
        # Capture any stderr output
        stderr_content = stderr_capture.getvalue()
        if stderr_content:
            metadata["_cuda_stderr"] = stderr_content
            
        try:
            conn.send((False, None, metadata))
        except Exception:
            pass
    finally:
        sys.stderr = old_stderr
        stderr_capture.close()
        conn.close()

def run_in_subprocess(func, *args, **kwargs):
    """
    Runs func in a fresh Python process (spawn), returns func's return value
    if success, or False on error or timeout.  Merges updated metadata back
    into your dict on success.
    """
    if "metadata" not in kwargs:
        raise ValueError("you must pass metadata=dict(...)")

    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe()
    p = ctx.Process(
        target=_worker_wrapper,
        args=(func, args, kwargs, child_conn),
    )
    p.start()
    child_conn.close()

    # wait up to RUN_TIMEOUT seconds for a message
    if not parent_conn.poll(EVAL_RUN_TIMEOUT):
        # timed out
        p.terminate()
        p.join()
        debug_print(f"run_in_subprocess: timed out after {EVAL_RUN_TIMEOUT}s")
        kwargs["metadata"]["timeout"] = f"run_in_subprocess: timed out after {EVAL_RUN_TIMEOUT}s"
        return False

    # we got something in time, but need to handle child process crashes
    try:
        success, result, new_meta = parent_conn.recv()
        p.join()

        if not result and "correct" in new_meta and new_meta["correct"] != "":
            kwargs["metadata"]["correct"] += f"\n{new_meta['correct']}"
        if not result and "compile" in new_meta and new_meta["compile"] != "":
            kwargs["metadata"]["compile"] += f"\n{new_meta['compile']}"

        # Check for CUDA stderr output
        if not result and "_cuda_stderr" in new_meta:
            # stderr_content = f"CUDA stderr output: {new_meta['_cuda_stderr']}\n Subprocess error: {new_meta.get('_subproc_error', '')} \n Subprocess traceback: {new_meta.get('_subproc_traceback', '')}"
            stderr_content = new_meta["_cuda_stderr"]
            if "assert" in stderr_content.lower() or "assertion" in stderr_content.lower():
                kwargs["metadata"]["correct"] += f"CUDA assertion failed: {stderr_content}; also check cuTensorMapEncodeTiled! (this is a very likely cause)"
            elif stderr_content.strip():
                kwargs["metadata"]["correct"] += f"\nCUDA stderr output: {stderr_content}"
            del new_meta["_cuda_stderr"]
        
        
        if "compile" in new_meta:
            kwargs["metadata"]["compile"] = new_meta["compile"]

        debug_print(f"run_in_subprocess, result = {result}, success = {success}, new_meta = {new_meta}")
        return result if success else False
    except EOFError:
        # Child process died unexpectedly
        p.join()
        metadata_correct = kwargs["metadata"].get("correct", "")
        if (p.exitcode == 255):
            metadata_correct += "The process we were testing your code in crashed, the most likely error is: CUDA kernel launch failed : invalid argument. The previous code did not have this error."
        elif (p.exitcode == -6):  # SIGABRT - typical for assertion failures
            metadata_correct += "Process terminated with SIGABRT"
        else:
            metadata_correct += f"Process terminated unexpectedly with exit code: {p.exitcode}"
        kwargs["metadata"]["correct"] = metadata_correct
        debug_print(f"run_in_subprocess: child process died unexpectedly (exit code: {p.exitcode})") # TODO: capture runtime error details
        return False
    except Exception as e:
        # Other communication errors
        p.terminate()
        p.join()
        debug_print(f"run_in_subprocess: communication error: {e}")
        kwargs["metadata"]["other"] += f"Communication error: {e}"
        return False

def check_compile(ref_src, gen_src, metadata, build_directory):
    return run_in_subprocess(
        _check_compile,
        ref_src, gen_src,
        metadata=metadata,
        build_directory=build_directory,
    )

def check_correct(ref_src, gen_src, metadata,
                       num_trials=10, seed_num=42,
                       build_directory=None, device=None):
    # return _check_correct(ref_src, gen_src, metadata, num_trials, seed_num, build_directory, device)
    return run_in_subprocess(
        _check_correct,
        ref_src, gen_src,
        metadata=metadata,
        num_trials=num_trials,
        seed_num=seed_num,
        build_directory=build_directory,
        device=device,
    )

def get_wallclock_time(ref_src, gen_src, metadata,
                       num_warmups=5, num_trials=100, seed_num=42,
                       build_directory=None, device=None):
    if EVAL_COLD_CACHE:
        return _get_wallclock_time_2(ref_src, gen_src, metadata, num_warmups, num_trials, seed_num, build_directory, device)
    else:
        return _get_wallclock_time(ref_src, gen_src, metadata, num_warmups, num_trials, seed_num, build_directory, device)


def get_baseline_time(ref_src, metadata,
                       num_warmups=5, num_trials=100, seed_num=42,
                       device=None, torch_compile=False, torch_compile_mode="default", get_torch_graph=False):
    if EVAL_COLD_CACHE:
        return _get_baseline_time_2(ref_src, metadata, num_warmups, num_trials, seed_num, device, torch_compile, torch_compile_mode, get_torch_graph)
    else:
        return _get_baseline_time(ref_src, metadata, num_warmups, num_trials, seed_num, device, torch_compile, torch_compile_mode, get_torch_graph)


import re
from typing import Dict, Any, List

def parse_nsight_text_to_metrics(nsight_text: str) -> Dict[str, Any]:
    """
    Parse Nsight Compute CLI text into a structured dict:
    - 'kernel': name of the kernel (best-effort)
    - 'config': convenient subset of config metrics
    - 'sections': full metric tables by section name
    - 'advice': lists of INF / OPT messages
    """
    result: Dict[str, Any] = {
        "kernel": None,
        "config": {},
        "sections": {},
        "advice": {"INF": [], "OPT": []},
    }
    if not nsight_text:
        return result

    lines = nsight_text.splitlines()
    n = len(lines)
    i = 0
    current_section = None

    # --- Robust kernel name extraction: find "void <name>(" or "void <name><" ---
    for line in lines:
        s = line.strip()
        if s.startswith("void ") and "(" in s:
            after_void = s[len("void "):]
            # End at first '(', '<', or whitespace
            for sep in ("(", "<", " "):
                j = after_void.find(sep)
                if j != -1:
                    after_void = after_void[:j]
            kernel_name = after_void.strip()
            if kernel_name:
                result["kernel"] = kernel_name
                break

    # Driver-API JIT-loaded PTX entries are often printed without a C++ "void"
    # prefix. Support common Nsight Compute text formats while retaining the
    # existing CUDA/C++ parser above.
    if result["kernel"] is None:
        fallback_patterns = (
            r'Profiling\s+"([^"]+)"',
            r"Kernel Name\s*[:=]?\s+([A-Za-z_.$][\w.$:@<>]*)",
            r"^([A-Za-z_.$][\w.$:@<>]*)\s*\(",
        )
        for line in lines:
            stripped = line.strip()
            for pattern in fallback_patterns:
                match = re.search(pattern, stripped)
                if match:
                    result["kernel"] = match.group(1)
                    break
            if result["kernel"] is not None:
                break

    # --- Main pass: sections + tables + advice ---
    while i < n:
        raw_line = lines[i]
        line = raw_line.strip()

        # Skip empty lines
        if not line:
            i += 1
            continue

        # ---------- Section headers ----------
        if line.startswith("Section:"):
            current_section = line[len("Section:"):].strip()
            result["sections"].setdefault(current_section, {})
            i += 1

            # Skip header/separator lines for the table
            while i < n:
                header_line = lines[i].strip()
                if (header_line.startswith("-")
                        or header_line.startswith("Metric Name")):
                    i += 1
                    continue
                break

            # Parse table rows until blank / new section / advice / NVTX/other header
            while i < n:
                row_raw = lines[i]
                row = row_raw.strip()
                if (not row
                    or row.startswith("Section:")
                    or row.startswith("INF")
                    or row.startswith("OPT")
                    or row.startswith("NVTX Push/Pop Stack")):
                    break

                # Example row:
                # "Achieved Occupancy                        %        31.66"
                parts = re.split(r"\s{2,}", row)
                if len(parts) >= 3:
                    metric_name, unit, value_str = parts[0], parts[1], parts[2]
                elif len(parts) == 2:
                    metric_name, unit, value_str = parts[0], "", parts[1]
                else:
                    metric_name, unit, value_str = row, "", ""

                # Try to parse numeric; leave as string if not numeric
                try:
                    value = float(value_str.replace(",", ""))
                except ValueError:
                    value = value_str

                result["sections"][current_section][metric_name] = {
                    "unit": unit,
                    "value": value,
                }

                # Pull out some key config-ish metrics into top-level
                name_lower = metric_name.lower()
                if current_section == "Launch Statistics":
                    if name_lower.startswith("block size"):
                        result["config"]["block_size"] = value
                    elif name_lower.startswith("grid size"):
                        result["config"]["grid_size"] = value
                    elif name_lower.startswith("registers per thread"):
                        result["config"]["registers_per_thread"] = value
                    elif name_lower.startswith("static shared memory per block"):
                        result["config"]["static_smem_kb"] = value
                    elif name_lower.startswith("dynamic shared memory per block"):
                        result["config"]["dynamic_smem_bytes"] = value

                if current_section == "Occupancy":
                    if name_lower.startswith("theoretical occupancy"):
                        result["config"]["theoretical_occupancy_pct"] = value
                    elif name_lower.startswith("achieved occupancy"):
                        result["config"]["achieved_occupancy_pct"] = value

                i += 1

            # We broke out of the table loop; continue outer loop without extra i++
            continue

        # ---------- Advice lines (INF/OPT), with indentation ----------
        if line.startswith("INF") or line.startswith("OPT"):
            key = "INF" if line.startswith("INF") else "OPT"
            # Remove the tag + spaces ("INF   " / "OPT   ")
            # Find first double space after tag to be robust
            msg = line[3:].lstrip()
            msg_lines: List[str] = [msg]
            i += 1
            # Continuation lines are usually heavily indented
            while i < n and lines[i].startswith(" " * 10):
                msg_lines.append(lines[i].strip())
                i += 1
            result["advice"][key].append(" ".join(msg_lines))
            continue

        i += 1

    return result


def _load_profiler_script(
    profiler_script_path: str,
    ref_src: str,
    gen_src: str,
    build_directory: str,
    device_index: int,
    seed_num: int = 42,
    num_warmups: int = 5,
    num_iters: int = 100,
    prebuilt_so_files=None,
):
    """
    Generate profiler.py by reading cutegen/profiler_template.py and replacing placeholders.
    """

    template_path = os.path.join(
        os.path.dirname(__file__),   # directory of current file
        "../scripts/profiler_template.py"       # template file
    )

    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    # Replace placeholders safely
    script = (
        template
        .replace("__BUILD_DIR__", build_directory)
        .replace("__DEVICE_INDEX__", str(device_index))
        .replace("__SEED__", str(seed_num))
        .replace("__NUM_WARMUPS__", str(num_warmups))
        .replace("__NUM_ITERS__", str(num_iters))
        .replace("__REF_SRC_REPR__", repr(ref_src))
        .replace("__GEN_SRC_REPR__", repr(gen_src))
        .replace("__PREBUILT_SO_FILES__", json.dumps(prebuilt_so_files))
    )

    with open(profiler_script_path, "w", encoding="utf-8") as f:
        f.write(script)

def run_nsight_profile(
    ref_src,
    gen_src,
    metadata,
    build_directory=None,
    device=None,
    num_warmups: int = 40,
    num_iters: int = 1,
    seed_num: int = 42,
):
    if NSIGHT_COMPUTE_BIN is None or not os.path.exists(NSIGHT_COMPUTE_BIN):
        metadata["profile_error"] = f"Nsight Compute not found: {NSIGHT_COMPUTE_BIN}"
        return None
    if isinstance(device, torch.device):
        device_index = device.index
    else:
        device_index = int(str(device or 0))

    profile_build_directory = tempfile.mkdtemp(prefix="ncu_build_")
    tmp_dir = tempfile.mkdtemp(prefix="ncu_prof_")
    profiler_script = os.path.join(tmp_dir, "profiler.py")

    metadata["profile_build_directory"] = profile_build_directory
    metadata["profile_tmp_directory"] = tmp_dir
    prebuild_meta = {"compile": "", "correct": ""}
    prebuild_context = {}
    model_new = load_custom_model(gen_src, prebuild_context, prebuild_meta, profile_build_directory)
    if model_new is None:
        metadata["profile_error"] = f"Fresh prebuild failed before ncu: {prebuild_meta.get('compile', '')}"
        metadata["profile_prebuild_meta"] = prebuild_meta
        return None 
    so_files = glob.glob(os.path.join(profile_build_directory, "**", "*.so"), recursive=True)
    if KERNEL_BACKEND != "ptx" and not so_files:
        metadata["profile_error"] = (
            f"No prebuilt .so found under {profile_build_directory}"
        )
        return None

    # Write profiler.py from template
    _load_profiler_script(
        profiler_script_path=profiler_script,
        ref_src=ref_src,
        gen_src=gen_src,
        build_directory=profile_build_directory,
        device_index=device_index,
        seed_num=seed_num,
        num_warmups=num_warmups,
        num_iters=num_iters,
        prebuilt_so_files=so_files,
    )
    

    PYTHON_BIN = f"{CUTEGEN_BASE_PATH}/venv/bin/python3"
    #env = os.environ.copy()
    #venv_bin = os.path.dirname(sys.executable)
    #env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    env = os.environ.copy()

# Force exact same venv
    env["VIRTUAL_ENV"] = f"{CUTEGEN_BASE_PATH}/venv"

# Put venv first in PATH
    env["PATH"] = f"{CUTEGEN_BASE_PATH}/venv/bin:/usr/local/cuda/bin:" + env.get("PATH", "")

# Ensure CUDA is consistent
    env["CUDA_HOME"] = "/usr/local/cuda"
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")

# Keep build directory consistent
    env["TORCH_EXTENSIONS_DIR"] = profile_build_directory
    cmd = [
 #       "sudo",
        "env",
        f"PATH={env['PATH']}",
        f"CUDA_HOME={env['CUDA_HOME']}",
        f"LD_LIBRARY_PATH={env['LD_LIBRARY_PATH']}",
        f"VIRTUAL_ENV={env['VIRTUAL_ENV']}",
        f"TORCH_EXTENSIONS_DIR={env['TORCH_EXTENSIONS_DIR']}",
        NSIGHT_COMPUTE_BIN,
        "--set", "basic",
        "--target-processes", "all",
        "--nvtx",
        "--nvtx-include", "CUTGEN_PROFILE_ITER/",
        PYTHON_BIN,
        profiler_script,
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    metadata["profile_cmd"] = " ".join(cmd)
    metadata["profile_stdout"] = proc.stdout
    metadata["profile_stderr"] = proc.stderr
    print(metadata)
    if proc.returncode == 0:
        metrics = parse_nsight_text_to_metrics(proc.stdout)
        metadata["nsight_metrics"] = metrics

    metadata["profile_returncode"] = proc.returncode
#    safe_rmtree(tmp_dir)
  #      metadata["nsight_metrics"] = metrics
    safe_rmtree(tmp_dir)
    safe_rmtree(profile_build_directory)
    if proc.returncode != 0:
       metadata["profile_error"] = f"Nsight Compute exited {proc.returncode}"
       print("PROFILE FAILED")
       print("CMD:", metadata.get("profile_cmd"))
       print("STDOUT:", metadata.get("profile_stdout"))
       print("STDERR:", metadata.get("profile_stderr"))
       print("BUILD DIR:", metadata.get("profile_build_directory"))
       print("TMP DIR:", metadata.get("profile_tmp_directory"))
       return None
    print(metadata["profile_stdout"])
    return {"returncode": proc.returncode}

def evaluate(node: Node, get_time=True, get_profile=USE_PROFILING, torch_compile=BENCHMARK_TORCH_COMPILE, torch_compile_mode=TORCH_COMPILE_MODE):
    """
    This function evaluates the node.
    """
    if node.depth == 0 and node.ref_time is None:
        wait_for_resources(0)
        node.ref_time = get_baseline_time(
            node.ref,
            node.metadata,
            device=0,
            torch_compile=torch_compile,
            torch_compile_mode=torch_compile_mode
        )
        debug_print(
            f"Node {node.uuid} (root node) ref time: {node.ref_time}, "
            f"benchmarked with torch_compile={torch_compile}, "
            f"torch_compile_mode={torch_compile_mode}"
        )
    time.sleep(2)
    flag_compile = False
    flag_correct = False

    node.error_type = ErrorType.NONE
    node.metadata["compile"] = ""
    node.metadata["correct"] = ""

    debug_print(f"Running node {node.uuid} at depth {node.depth}: process = {os.getpid()}")
    debug_print(f"Src: {node.src}")

    build_directory = BUILD_DIRECTORY_BASE + str(node.uuid) + "/"
    flag_compile = check_compile(node.ref, node.src, node.metadata, build_directory=build_directory)

    debug_print(f"Node {node.uuid} compile: {flag_compile}. Metadata: {node.metadata}")

    if not flag_compile:
        debug_print(f"Node {node.uuid} compile error: {node.metadata['compile']}. Metadata: {node.metadata}")
        node.error_type = ErrorType.COMPILE
        print("before remove:", build_directory, os.path.exists(build_directory))
        remove_build_directory(build_directory)
        print("after remove:", build_directory, os.path.exists(build_directory))
        return

    gpu_lock_fd, device = acquire_gpu()
    device = torch.device(f"cuda:{device}")

    wait_for_resources(device.index)

    flag_correct = check_correct(
        node.ref,
        node.src,
        node.metadata,
        build_directory=build_directory,
        device=device
    )

    release_gpu(device, gpu_lock_fd)

    if not flag_correct:
        debug_print(f"Node {node.uuid} correct error: {node.metadata['correct']}. Metadata: {node.metadata}")
        node.error_type = ErrorType.CORRECT
        print("before remove:", build_directory, os.path.exists(build_directory))
        remove_build_directory(build_directory)
        print("after remove:", build_directory, os.path.exists(build_directory))
        return

    node.error_type = ErrorType.PASS
    debug_print(f"Node {node.uuid} compile and correct. Metadata: {node.metadata}")

    if get_time:
        gpu_lock_fd, device = acquire_gpu()
        device = torch.device(f"cuda:{device}")

        wait_for_resources(device.index)

        time_stats = get_wallclock_time(
            node.ref,
            node.src,
            node.metadata,
            build_directory=build_directory,
            device=device
        )

        release_gpu(device, gpu_lock_fd)
        node.time = time_stats

        best_time = float(read_file_with_lock(f"{node.save_folder_path}/best_time.txt"))
        if best_time > time_stats["mean"]:
            write_file_with_lock(f"{node.save_folder_path}/best_time.txt", str(time_stats["mean"]))

        debug_print(f"Node {node.uuid} time: {time_stats}")

    if get_profile:
        gpu_lock_fd, device = acquire_gpu()
        device = torch.device(f"cuda:{device}")

        wait_for_resources(device.index)

        perf_stats = run_nsight_profile(
            node.ref,
            node.src,
            node.metadata,
            device=device,
        )

        release_gpu(device, gpu_lock_fd)
        node.perf = perf_stats
        debug_print(f"Node {node.uuid} profile: {perf_stats}")

    remove_build_directory(build_directory)
    return node
