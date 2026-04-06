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

import re
from typing import Dict, Any, List

from cutgen.node import Node, ErrorType
from cutgen.util import set_seed, debug_print, acquire_gpu, release_gpu, remove_build_directory, read_file_with_lock, write_file_with_lock

####### CONSTANTS #######
from cutgen.config import LOAD_MODEL_BACKOFF_TIME, RUN_MODEL_BACKOFF_TIME, GPU_REQ_SPACE, CPU_REQ_SPACE, BUILD_DIRECTORY_BASE, EVAL_RUN_TIMEOUT, BENCHMARK_TORCH_COMPILE, TORCH_COMPILE_MODE, EVAL_COLD_CACHE, NSIGHT_COMPUTE_BIN, NSIGHT_COMPUTE_SET, USE_PROFILING


####### EVALUATION HELPER FUNCTIONS #######
import shutil

def safe_rmtree(path: str):
    """Delete a directory if it exists, ignore errors."""
    try:
        if path and os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass

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
    # Add import at the start of the source code
    model_custom_src = (
        "import os\nimport gc\n" f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_directory}'\n"
    ) + model_custom_src + "\ntorch.cuda.synchronize()\ntorch.cuda.empty_cache()\ngc.collect()"
    retval = True
    
    read_fd, write_fd = os.pipe()
    old_out, old_err = os.dup(1), os.dup(2)

    # 2) Point both stdout & stderr at our pipe
    os.dup2(write_fd, 1)
    os.dup2(write_fd, 2)
    os.close(write_fd)
    
    try:
        compile(model_custom_src, "<string>", "exec")
        exec(model_custom_src, context)
        # Force CUDA synchronization to catch any deferred CUDA errors
        torch.cuda.synchronize()
        # DANGER: need to delete refernece from global namespace
    except Exception as e: 
        metadata["compile"] = f"Syntax error in generated code: {str(e)}"
        retval = None
    try:
        ModelNew = context.get("ModelNew")
    except Exception as e:
        metadata["compile"] = f"Error in executing generated code {str(e)}"
        retval = None
    # retval is default unless there was an error. If there was an error, then attach command line output.
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
        metadata["compile"] += f"\n Here is the full command line of the program execution: {error}" if error != "" else ""
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
            metadata["compile"] += f"Loading ModelNew failed: ModelNew is None"
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


# def _check_correct(ref_src, gen_src, metadata,
#                        num_trials=10, seed_num=42,
#                        build_directory=None, device=None):
#     """
#     This function checks if the generated source code is correct.
#     """
#     torch.cuda.set_device(device)
#     metadata["hardware"] = torch.cuda.get_device_name(device=device)
#     metadata["device"] = str(device)
#     context = {}
#     os.environ["TORCH_USE_CUDA_DSA"] = "1"
#     os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Force synchronous execution for better error capture; TODO: don't block CUDA calls for better performance for now
#
#     # We already know the generated code compiles, so we can load the model
#     Model, get_init_inputs_fn, get_inputs_fn = load_original_model_and_inputs(ref_src, context, metadata)
#     debug_print(f"check correct: Loaded original model")
#     ModelNew = load_custom_model(gen_src, context, metadata, build_directory)
#     debug_print(f"check correct: Loaded custom model")
#     # override_get_inputs_cpu(context)
#     # get_inputs_fn = context["get_inputs"]
#     try:
#         set_seed(seed_num)
#         init_inputs = get_init_inputs_fn()
#         init_inputs = [
#             x.cuda(device=metadata["device"]) if isinstance(x, torch.Tensor) else x for x in init_inputs
#         ]
#
#         model = None
#         with torch.no_grad():
#             set_seed(seed_num)  # set seed for reproducible weights
#             model = Model(*init_inputs).cuda(device=device)
#         # debug_print(f"check correct: Initialized original model")
#
#         model_new = None
#         try:
#             with torch.no_grad():
#                 set_seed(seed_num)  # set seed for reproducible weights
#                 model_new = ModelNew(*init_inputs).cuda(device=device)
#         except Exception as e:
#             metadata["correct"] = f"Error initializing custom model: {e}"
#             torch.cuda.empty_cache()
#             torch.cuda.reset_peak_memory_stats()
#             gc.collect()
#             return False
#
#         pass_count = 0
#         torch.manual_seed(seed_num)
#         correctness_trial_seeds = [
#             torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_trials)
#         ]
#
#         with torch.no_grad():
#             for trial in range(num_trials):
#                 # print(f"Checking correctness for trial {trial}")
#                 trial_seed = correctness_trial_seeds[trial]
#
#                 set_seed(trial_seed)
#                 inputs = get_inputs_fn()
#
#                 try:
#                     inputs = [x.cuda(device=device) if isinstance(x, torch.Tensor) else x
#                               for x in inputs]
#                 except torch.cuda.OutOfMemoryError as e:
#                     metadata["correct"] = f"OOM moving inputs to GPU: {e}"
#                     torch.cuda.empty_cache()
#                     gc.collect()
#                     return False
#
#                 # # debug_print(f"check correct: Initialized inputs")
#
#                 set_seed(trial_seed)
#                 # model_new = custom_model.cuda(device=device)
#                 # # debug_print(f"check correct: Moved custom model to device")
#                 #
#                 # set_seed(trial_seed)
#                 # model = original_model.cuda(device=device)
#                 # # debug_print(f"check correct: Moved original model to device")
#
#                 try:
#                     output_new = model_new(*inputs)
#                     torch.cuda.synchronize(device=device)
#                     # debug_print(f"check correct: Synchronized custom model")
#                 except Exception as e:
#                     metadata["correct"] = f"Runtime error when checking correctness: {str(e)}"
#                     torch.cuda.empty_cache()
#                     torch.cuda.reset_peak_memory_stats()
#                     gc.collect()
#                     return False
#                 # cpu_new_output = output_new.detach().float().cpu()
#                 # del output_new
#                 # torch.cuda.empty_cache()
#                 output = model(*inputs)
#                 # debug_print(f"check correct: Computed output for original model")
#                 try:
#                     torch.cuda.synchronize(device=device)
#                     # debug_print(f"check correct: Synchronized original model")
#                 except Exception as e:
#                     metadata["correct"] = f"Runtime error when checking correctness: {str(e)}"
#                     torch.cuda.empty_cache()
#                     torch.cuda.reset_peak_memory_stats()
#                     gc.collect()
#                     return False
#                 # cpu_output = output.detach().float().cpu()
#                 # del output
#                 # torch.cuda.empty_cache()
#
#                 if output.shape != output_new.shape:
#                     metadata["correct"] = f"Output shape mismatch, expected {output.shape}, got {output_new.shape}"
#                     continue
#                 # if cpu_output.shape != cpu_new_output.shape:
#                 #     metadata[
#                 #         "correct"] = f"Output shape mismatch, expected {cpu_output.shape}, got {cpu_new_output.shape}"
#                 #     del inputs, cpu_output, cpu_new_output
#                 #     gc.collect()
#                 #     torch.cuda.empty_cache()
#                 #     continue
#
#                 # if not torch.allclose(cpu_output, cpu_new_output, atol=1e-2, rtol=1e-2):
#                 #     diff = (cpu_output - cpu_new_output).abs()
#                 #     metadata["correct"] = f"Output value mismatch, max diff: {diff.max().item()}, avg diff: {diff.mean().item()}"
#                 #     continue
#
#                 if not torch.allclose(
#                     # output, output_new, atol=2.5e-02, rtol=2.5e-02
#                     output.to(torch.float32), output_new.to(torch.float32), atol=1e-02, rtol=1e-02
#                     # output, output_new, rtol=1e-01, atol=1e+2
#                 ):
#                     max_diff = torch.max(torch.abs(output.to(torch.float32) - output_new.to(torch.float32))).item()
#                     avg_diff = torch.mean(torch.abs(output.to(torch.float32) - output_new.to(torch.float32))).item()
#                     metadata["correct"] = f"Output value mismatch, max diff: {max_diff}, avg diff: {avg_diff}"
#                     continue
#                 else:
#                     # if trial == num_trials - 1:
#                     #     diff = (cpu_output - cpu_new_output).abs()
#                     #     metadata["diff"] = f"Output value matched, max diff: {diff.max().item()}, avg diff: {diff.mean().item()}"
#
#                     if trial == num_trials - 1:
#                         max_diff = torch.max(torch.abs(output.to(torch.float32) - output_new.to(torch.float32))).item()
#                         avg_diff = torch.mean(torch.abs(output.to(torch.float32) - output_new.to(torch.float32))).item()
#                         metadata["diff"] = f"Output value matched, max diff: {max_diff}, avg diff: {avg_diff}"
#                     debug_print(f"check correct: Error checking output value: {str(e)}")
#                 # debug_print(f"check correct: Checked output value")
#                 del inputs
#                 gc.collect()
#                 torch.cuda.empty_cache()
#                 pass_count += 1
#
#     except Exception as e:
#         try:
#             torch.cuda.synchronize(device=device)
#         except Exception as e2:
#             metadata["correct"] = f"Runtime error when checking correctness: {str(e)}; {str(e2)}"
#             torch.cuda.empty_cache()
#             torch.cuda.reset_peak_memory_stats()
#             gc.collect()
#             return False
#         torch.cuda.empty_cache()
#         torch.cuda.reset_peak_memory_stats()
#         gc.collect()
#         return False
#
#     try:
#         torch.cuda.synchronize()
#     except Exception as e:
#         metadata["correct"] = f"Runtime error when checking correctness: {str(e)}"
#         torch.cuda.empty_cache()
#         torch.cuda.reset_peak_memory_stats()
#         gc.collect()
#         return False
#
#     metadata["correct"] = f"Passed {pass_count} out of {num_trials} trials: {metadata['correct'] if 'correct' in metadata else 'ALL PASSED'}"
#     if 'diff' in metadata:
#         metadata["correct"] += f"\n{metadata['diff']}"
#     if 'diff' in metadata:
#         del metadata['diff']
#     if pass_count == num_trials:
#         return True
#     else:
#         return False

def _check_correct(ref_src, gen_src, metadata,
                   num_trials=10, seed_num=42,
                   build_directory=None, device=None):
    """
    This function checks if the generated source code is correct.
    """
    torch.cuda.set_device(device)
    metadata["hardware"] = torch.cuda.get_device_name(device=device)
    metadata["device"] = str(device)
    ref_context = {}
    gen_context = {}
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ[
        "CUDA_LAUNCH_BLOCKING"] = "0"  # Force synchronous execution for better error capture; TODO: don't block CUDA calls for better performance for now

    # We already know the generated code compiles, so we can load the model
    Model, get_init_inputs_fn, get_inputs_fn = load_original_model_and_inputs(ref_src, ref_context, metadata)
    # debug_print(f"check correct: Loaded original model")
    ModelNew = load_custom_model(gen_src, gen_context, metadata, build_directory)
    # debug_print(f"check correct: Loaded custom model")

    try:
        set_seed(seed_num)
        init_inputs = get_init_inputs_fn()
        init_inputs = [
            x.cuda(device=metadata["device"]) if isinstance(x, torch.Tensor) else x for x in init_inputs
        ]

        original_model = None
        with torch.no_grad():
            set_seed(seed_num)  # set seed for reproducible weights
            original_model = Model(*init_inputs)
        # debug_print(f"check correct: Initialized original model")

        custom_model = None
        try:
            with torch.no_grad():
                set_seed(seed_num)  # set seed for reproducible weights
                custom_model = ModelNew(*init_inputs)
        except Exception as e:
            metadata["correct"] = f"Error initializing custom model: {e}"
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()
            return False

        pass_count = 0
        torch.manual_seed(seed_num)
        correctness_trial_seeds = [
            torch.randint(0, 2 ** 32 - 1, (1,)).item() for _ in range(num_trials)
        ]

        with torch.no_grad():
            for trial in range(num_trials):
                # print(f"Checking correctness for trial {trial}")
                trial_seed = correctness_trial_seeds[trial]
                # 👇 ADD THIS DEBUG PRINT HERE
                debug_print(
                    "REF globals batch_size/dim currently:"  +
                    str(get_inputs_fn.__globals__.get("batch_size"))+
                    str(get_inputs_fn.__globals__.get("dim"))
                )

                set_seed(trial_seed)
                inputs = get_inputs_fn()
                inputs = [
                    x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                    for x in inputs
                ]
                # # debug_print(f"check correct: Initialized inputs")
                for i, inp in enumerate(inputs):
                    if isinstance(inp, torch.Tensor):
                        debug_print(f"INPUT[{i}] shape={inp.shape}, device={inp.device}, dtype={inp.dtype}")

                set_seed(trial_seed)
                model_new = custom_model.cuda(device=device)
                # debug_print(f"check correct: Moved custom model to device")

                set_seed(trial_seed)
                model = original_model.cuda(device=device)
                # debug_print(f"check correct: Moved original model to device")

                try:
                    output_new = model_new(*inputs)
                    torch.cuda.synchronize(device=device)
                    # debug_print(f"check correct: Synchronized custom model")
                except Exception as e:
                    metadata["correct"] = f"Runtime error when checking correctness: {str(e)}"
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    gc.collect()
                    return False

                output = model(*inputs)
                # debug_print(f"check correct: Computed output for original model")
                try:
                    torch.cuda.synchronize(device=device)
                    # debug_print(f"check correct: Synchronized original model")
                except Exception as e:
                    metadata["correct"] = f"Runtime error when checking correctness: {str(e)}"
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    gc.collect()
                    return False

                if output.shape != output_new.shape:
                    metadata["correct"] = f"Output shape mismatch, expected {output.shape}, got {output_new.shape}"
                    continue
                # debug_print(f"check correct: Checked output shape: {output.shape} vs. {output_new.shape}")
                if not torch.allclose(
                        # output, output_new, atol=2.5e-02, rtol=2.5e-02
                        output.to(torch.float32), output_new.to(torch.float32), atol=1e-02, rtol=1e-02
                        # output, output_new, rtol=1e-01, atol=1e+2
                ):
                    max_diff = torch.max(torch.abs(output.to(torch.float32) - output_new.to(torch.float32))).item()
                    avg_diff = torch.mean(torch.abs(output.to(torch.float32) - output_new.to(torch.float32))).item()
                    metadata["correct"] = f"Output value mismatch, max diff: {max_diff}, avg diff: {avg_diff}"
                    continue
                else:
                    if trial == num_trials - 1:
                        max_diff = torch.max(torch.abs(output.to(torch.float32) - output_new.to(torch.float32))).item()
                        avg_diff = torch.mean(torch.abs(output.to(torch.float32) - output_new.to(torch.float32))).item()
                        metadata["diff"] = f"Output value matched, max diff: {max_diff}, avg diff: {avg_diff}"
                    # debug_print(f"check correct: Error checking output value: {str(e)}")
                # debug_print(f"check correct: Checked output value")
                pass_count += 1

    except Exception as e:
        try:
            metadata["correct"] = f"Unexpected error during correctness check: {repr(e)}"
            torch.cuda.synchronize(device=device)
        except Exception as e2:
            metadata["correct"] = f"Runtime error when checking correctness: {str(e)}; {str(e2)}"
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()
            return False
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        return False

    try:
        torch.cuda.synchronize()
    except Exception as e:
        metadata["correct"] = f"Runtime error when checking correctness: {str(e)}"
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        return False

    metadata[
        "correct"] = f"Passed {pass_count} out of {num_trials} trials: {metadata['correct'] if 'correct' in metadata else 'ALL PASSED'}"
    if 'diff' in metadata:
        metadata["correct"] += f"\n{metadata['diff']}"
    if 'diff' in metadata:
        del metadata['diff']
    if pass_count == num_trials:
        return True
    else:
        return False

def _get_wallclock_time(ref_src, gen_src, metadata,
                            num_warmups=5, num_trials=100, seed_num=42,
                            build_directory=None, device=None):
    """
    This function gets the wallclock time of the generated source code.
    """
    # device: torch.device = torch.cuda.current_device()
    torch.cuda.set_device(device)
    metadata["hardware"] = torch.cuda.get_device_name(device=device)
    metadata["device"] = str(device)
    ref_context = {}
    gen_context = {}
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0" # don't block CUDA calls for better performance
    
     # We already know the generated code compiles and is correct, so we can skip lots of checks
    Model, get_init_inputs_fn, get_inputs_fn = load_original_model_and_inputs(ref_src, ref_context, metadata)
    ModelNew = load_custom_model(gen_src, gen_context, metadata, build_directory)

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

    custom_model = None
    elapsed_times = []

    with torch.no_grad():
        set_seed(seed_num)  # set seed for reproducible weights
        custom_model = ModelNew(*init_inputs)
        custom_model = custom_model.cuda(device=device)

        for _ in range(num_warmups):
            custom_model(*inputs)
            torch.cuda.synchronize(device=device)

        for _ in range(num_trials):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            custom_model(*inputs)
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
    return timing_stats

def _get_wallclock_time_2(ref_src, gen_src, metadata,
                            num_warmups=5, num_trials=100, seed_num=42,
                            build_directory=None, device=None):
    """
    This function gets the wallclock time of the generated source code.
    """
    # device: torch.device = torch.cuda.current_device()
    torch.cuda.set_device(device)
    metadata["hardware"] = torch.cuda.get_device_name(device=device)
    metadata["device"] = str(device)
    ref_context = {}
    gen_context = {}
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0" # don't block CUDA calls for better performance
    
     # We already know the generated code compiles and is correct, so we can skip lots of checks
    Model, get_init_inputs_fn, get_inputs_fn = load_original_model_and_inputs(ref_src, ref_context, metadata)
    ModelNew = load_custom_model(gen_src, gen_context, metadata, build_directory)

    set_seed(seed_num)
    init_inputs = get_init_inputs_fn()
    init_inputs = [
        x.cuda(device=metadata["device"]) if isinstance(x, torch.Tensor) else x for x in init_inputs
    ]

    model = ModelNew(*init_inputs)
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


def get_profile():
    # TODO: implement
    return None

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
):
    """
    Generate profiler.py by reading cutgen/profiler_template.py and replacing placeholders.
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
        .replace("__REF_SRC__", ref_src.replace('"""', '\\"\\"\\"'))
        .replace("__GEN_SRC__", gen_src.replace('"""', '\\"\\"\\"'))
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

    # resolve device index
    if isinstance(device, torch.device):
        device_index = device.index
    else:
        device_index = int(str(device or 0))

    if build_directory is None:
        build_directory = tempfile.mkdtemp(prefix="ncu_build_")

    tmp_dir = tempfile.mkdtemp(prefix="ncu_prof_", dir=build_directory)
    profiler_script = os.path.join(tmp_dir, "profiler.py")

    # Write profiler.py from template
    _load_profiler_script(
        profiler_script_path=profiler_script,
        ref_src=ref_src,
        gen_src=gen_src,
        build_directory=build_directory,
        device_index=device_index,
        seed_num=seed_num,
        num_warmups=num_warmups,
        num_iters=num_iters,
    )

    cmd = [
        NSIGHT_COMPUTE_BIN,
        "--set", "basic",
        "--target-processes", "all",
        "--nvtx",                    # enable NVTX awareness
        "--nvtx-include", "CUTGEN_PROFILE_ITER/",
        sys.executable,
        profiler_script,
    ]

    env = os.environ.copy()
    venv_bin = os.path.dirname(sys.executable)
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")

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
    if proc.returncode == 0:
        metrics = parse_nsight_text_to_metrics(proc.stdout)
        metadata["nsight_metrics"] = metrics

    metadata["profile_returncode"] = proc.returncode
    safe_rmtree(tmp_dir)

    if proc.returncode != 0:
        metadata["profile_error"] = f"Nsight Compute exited {proc.returncode}"
        return None
    print(metadata["profile_stdout"])
    return {"returncode": proc.returncode}

def evaluate(node: Node, get_time=True, get_profile=USE_PROFILING, torch_compile=BENCHMARK_TORCH_COMPILE, torch_compile_mode=TORCH_COMPILE_MODE):
    """
    This function evaluates the node.
    """
    if node.depth == 0 and node.ref_time is None:
        node.ref_time = get_baseline_time(node.ref, node.metadata, device=0, torch_compile=torch_compile, torch_compile_mode=torch_compile_mode)
        debug_print(f"Node {node.uuid} (root node) ref time: {node.ref_time}, benchmarked with torch_compile={torch_compile}, torch_compile_mode={torch_compile_mode}")

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
    
    # TODO: check these values and implementation
    # GPUfree = 1.0 - GPUtil.getGPUs()[0].memoryUtil # Assumes one GPU.
    CPU_used = psutil.virtual_memory().percent
    # while (GPU_REQ_SPACE >= GPUfree or CPU_used > CPU_REQ_SPACE):
    while (CPU_used > CPU_REQ_SPACE):
        debug_print(f"sleeping for VRAM usage, {CPU_used} on CPU")
        time.sleep(random.randint(1,20) * RUN_MODEL_BACKOFF_TIME)
        CPU_used = psutil.virtual_memory().percent

    if not flag_compile:
        debug_print(f"Node {node.uuid} compile error: {node.metadata['compile']}. Metadata: {node.metadata}")
        node.error_type = ErrorType.COMPILE
        remove_build_directory(build_directory)
        return

    gpu_lock_fd, device = acquire_gpu()
    device = torch.device(f"cuda:{device}")
    flag_correct = check_correct(node.ref, node.src, node.metadata, build_directory=build_directory, device=device)
    release_gpu(device, gpu_lock_fd)

    if not flag_correct:
        debug_print(f"Node {node.uuid} correct error: {node.metadata['correct']}. Metadata: {node.metadata}")
        node.error_type = ErrorType.CORRECT
        remove_build_directory(build_directory)
        return

    # by now, the code is correct and compiled
    node.error_type = ErrorType.PASS
    debug_print(f"Node {node.uuid} compile and correct. Metadata: {node.metadata}")

    if get_time:
        gpu_lock_fd, device = acquire_gpu()
        device = torch.device(f"cuda:{device}")
        time_stats = get_wallclock_time(node.ref, node.src, node.metadata, build_directory=build_directory, device=device)
        release_gpu(device, gpu_lock_fd)
        node.time = time_stats
        best_time = float(read_file_with_lock(f"{node.save_folder_path}/best_time.txt"))
        if best_time > time_stats["mean"]:
            write_file_with_lock(f"{node.save_folder_path}/best_time.txt", str(time_stats["mean"])) # TODO: save best time stats to file
        debug_print(f"Node {node.uuid} time: {time_stats}")

    if get_profile:
        gpu_lock_fd, device = acquire_gpu()
        device = torch.device(f"cuda:{device}")
        perf_stats = run_nsight_profile(
            node.ref,
            node.src,
            node.metadata,
            build_directory=build_directory,
            device=device,
        )

        release_gpu(device, gpu_lock_fd)
        node.perf = perf_stats
        debug_print(f"Node {node.uuid} profile: {perf_stats}")

    remove_build_directory(build_directory)

    return node