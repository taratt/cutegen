import os
import fcntl
import torch
import re
import shutil
import json
import hashlib

from skxoss.config import DEBUG_PRINT, SKXOSS_BASE_PATH, GPU_LOCK_FILE

def debug_print(msg: str):
    if DEBUG_PRINT:
        print(msg, flush=True)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def fetch_kernelbench_problem_ref(level, problem_id, local=True):
    PROBLEM_DIR_DISK = f"{SKXOSS_BASE_PATH}/KernelBench"
    if local:
        curr_level_dir = os.path.join(PROBLEM_DIR_DISK, f"level{level}")
        files = [fn for fn in os.listdir(curr_level_dir) if fn.endswith(".py")]
        files.sort(key=lambda fn: int(os.path.splitext(fn)[0].split("_")[0]))
        DATASET = [os.path.join(curr_level_dir, fn) for fn in files]
        idx = problem_id - 1
        assert(idx >= 0 and idx < len(DATASET)), f"Problem ID {problem_id} out of range (0–{len(DATASET)-1})"
        ref_arch_path = DATASET[idx]
        ref_arch_src = read_file(ref_arch_path)
        return ref_arch_src
    else:
        return "" # TODO: fetch from huggingface


def read_file(file_path: str) -> str:
    with open(file_path, "r") as file:
        return file.read()
        
def write_file(file_path: str, content: str):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as file:
        file.write(content)

def read_file_with_lock(file_path: str) -> str:
    with open(file_path, "r") as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_SH)
        try:
            content = file.read()
        finally:
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
        return content

def write_file_with_lock(file_path: str, content: str):
    with open(file_path, "w") as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        try:
            file.write(content)
        finally:
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)

# TODO: make it more robust for multiple GPUs later?
def acquire_gpu():
    gpu_lock_fd = open(GPU_LOCK_FILE, 'a+')
    """Block until we get exclusive access to /tmp/gpu.lock"""
    fcntl.flock(gpu_lock_fd.fileno(), fcntl.LOCK_EX)
    return gpu_lock_fd, 0

def release_gpu(device, gpu_lock_fd):
    """Release the lock on /tmp/gpu.lock"""
    if gpu_lock_fd is None:
        return # This is for double-free case found in 295 -- if we have a compile/reward error and reach the end of counter/while loop, it will double unlock.
    fcntl.flock(gpu_lock_fd.fileno(), fcntl.LOCK_UN)


def extract_first_code(output_string: str, code_language_types: list[str] = ["python", "cpp", "c"]) -> str:
    """
    Extract first code block from model output, specified by code_language_type
    """
    trimmed = output_string.strip()

    # Extracting the first occurrence of content between backticks
    code_match = re.search(r"```(.*?)```", trimmed, re.DOTALL)

    if code_match:
        # Strip leading and trailing whitespace from the extracted code
        code = code_match.group(1).strip()

        # depends on code_language_type: cpp, python, etc.
        # sometimes the block of code is ```cpp ... ``` instead of ``` ... ```
        # in this case strip the cpp out
        for code_type in code_language_types:
            if code.startswith(code_type):
                code = code[len(code_type) :].strip()

        return code

    return None

def remove_build_directory(build_directory: str):
    debug_print(f"build_directory to remove: {build_directory}")
    if os.path.exists(build_directory):
        try:
            shutil.rmtree(build_directory, ignore_errors=True)
            debug_print(f"Removed build directory: {build_directory}")
        except Exception as e:
            debug_print(f"Failed to remove build directory: {build_directory}: {str(e)}")

def load_json(file_path: str):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except Exception as e:
        debug_print(f"Failed to load json file {file_path}: {str(e)}")
        return None

def save_json(file_path: str, data):
    try:
        with open(file_path, "w") as file:
            json.dump(data, file, indent=None)
    except Exception as e:
        debug_print(f"Failed to save json file {file_path}: {str(e)}")

def file_md5(file_path):
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

def get_numbered_lines(lines: list[str]) -> str:
    s = ""
    for i, line in enumerate(lines, 1):
        s += f"{i:4}: {line.rstrip()}\n"
    return s

def src_to_lines(src: str) -> list[str]:
    return src.split("\n")

def lines_to_src(lines: list[str]) -> str:
    return "\n".join(lines)


def extract_src_from_json_trace(json_data, save_path):
    src = json_data["src"]
    write_file(save_path, src)