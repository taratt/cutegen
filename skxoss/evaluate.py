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

from skxoss.node import Node, ErrorType
from skxoss.util import set_seed, debug_print, acquire_gpu, release_gpu, remove_build_directory, read_file_with_lock, write_file_with_lock

####### CONSTANTS #######
from skxoss.config import LOAD_MODEL_BACKOFF_TIME, RUN_MODEL_BACKOFF_TIME, GPU_REQ_SPACE, CPU_REQ_SPACE, BUILD_DIRECTORY_BASE, EVAL_RUN_TIMEOUT, BENCHMARK_TORCH_COMPILE, TORCH_COMPILE_MODE, EVAL_COLD_CACHE


####### EVALUATION HELPER FUNCTIONS #######

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
    

def _check_correct(ref_src, gen_src, metadata,
                       num_trials=10, seed_num=42,
                       build_directory=None, device=None):
    """
    This function checks if the generated source code is correct.
    """
    torch.cuda.set_device(device)
    metadata["hardware"] = torch.cuda.get_device_name(device=device)
    metadata["device"] = str(device)
    context = {}
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Force synchronous execution for better error capture; TODO: don't block CUDA calls for better performance for now
    
    # We already know the generated code compiles, so we can load the model
    Model, get_init_inputs_fn, get_inputs_fn = load_original_model_and_inputs(ref_src, context, metadata)
    # debug_print(f"check correct: Loaded original model")
    ModelNew = load_custom_model(gen_src, context, metadata, build_directory)
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
            torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_trials)
        ]

        with torch.no_grad():
            for trial in range(num_trials):
                # print(f"Checking correctness for trial {trial}")
                trial_seed = correctness_trial_seeds[trial]

                set_seed(trial_seed)
                inputs = get_inputs_fn()
                inputs = [
                    x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                    for x in inputs
                ]
                # # debug_print(f"check correct: Initialized inputs")

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

    metadata["correct"] = f"Passed {pass_count} out of {num_trials} trials: {metadata['correct'] if 'correct' in metadata else 'ALL PASSED'}"
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
    context = {}
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0" # don't block CUDA calls for better performance
    
     # We already know the generated code compiles and is correct, so we can skip lots of checks
    Model, get_init_inputs_fn, get_inputs_fn = load_original_model_and_inputs(ref_src, context, metadata)
    ModelNew = load_custom_model(gen_src, context, metadata, build_directory)

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
    context = {}
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0" # don't block CUDA calls for better performance
    
     # We already know the generated code compiles and is correct, so we can skip lots of checks
    Model, get_init_inputs_fn, get_inputs_fn = load_original_model_and_inputs(ref_src, context, metadata)
    ModelNew = load_custom_model(gen_src, context, metadata, build_directory)

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


def evaluate(node: Node, get_time=True, get_profile=False, torch_compile=BENCHMARK_TORCH_COMPILE, torch_compile_mode=TORCH_COMPILE_MODE):
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
        perf_stats = get_profile(node.ref, node.src, node.metadata, build_directory=build_directory, device=device)
        release_gpu(device, gpu_lock_fd)
        node.perf = perf_stats

    remove_build_directory(build_directory)

    return node