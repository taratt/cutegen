import os
import fcntl
import torch
import re
import shutil
import json
import hashlib
from typing import Optional, Tuple, Any, List

from cutgen.config import DEBUG_PRINT, SKXOSS_BASE_PATH, GPU_LOCK_FILE

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
def parse_code_and_edit(output_string: str, code_language_types: list[str] = ["python", "cpp", "c"]) -> Tuple[Optional[str], Optional[Any]]:
    """
      Extract the first non-JSON fenced code block and the first JSON fenced block.

      Returns:
          (code_text, edits_obj)
          - code_text: str or None — contents of the first non-JSON code fence.
            If the fence has a language tag (e.g., ```python), it is stripped.
          - edits_obj: parsed JSON (usually list/dict) or None — from the first
            ```json fenced block. If JSON parsing fails, returns None.

      Notes:
          - If `code_language_types` is provided, and the first non-JSON block's
            language tag is present among the list (e.g., ["python","cpp","c"]),
            the tag is simply ignored (content is returned as-is either way).
          - If there is a JSON-looking block without an explicit `json` tag,
            we attempt to parse it as JSON as a fallback.
      """
    if code_language_types is None:
        code_language_types = ["python", "cpp", "c", "cuda", "bash", "text"]

    text = output_string or ""

    # Find all fenced code blocks: ```lang?\n...``` (lang is optional)
    pattern = re.compile(r"```([a-zA-Z0-9_+-]*)\s*\n([\s\S]*?)```", re.MULTILINE)
    blocks = pattern.findall(text)

    first_code_text: Optional[str] = None
    edits_obj: Optional[Any] = None
    fallback_json_candidate: Optional[str] = None

    for lang, body in blocks:
        lang_lower = lang.strip().lower()

        if lang_lower == "json":
            # Try to parse immediately; take the first valid JSON block
            candidate = body.strip()
            try:
                edits_obj = json.loads(candidate)
                # Once JSON is found/parsed, we still continue to find the first non-JSON code
            except Exception:
                # Keep going; maybe another json block parses
                pass
        else:
            # Consider this a candidate for the "code" block (first only)
            if first_code_text is None:
                # If the body itself starts with a language label line (rare), strip it
                # but generally the language is provided in the backticks, so just trim.
                first_code_text = body.strip()

        # As a fallback, keep the first non-tag block that *might* be JSON
        if not lang_lower:
            candidate = body.strip()
            if fallback_json_candidate is None:
                # Quick-and-dirty heuristic to remember a potential JSON block
                if candidate.startswith("{") or candidate.startswith("["):
                    fallback_json_candidate = candidate

    # If we didn't get a JSON block with tag=json, try the fallback candidate
    if edits_obj is None and fallback_json_candidate is not None:
        try:
            edits_obj = json.loads(fallback_json_candidate)
        except Exception:
            pass

    return first_code_text, edits_obj

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