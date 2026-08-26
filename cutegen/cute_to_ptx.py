"""Extract compiler PTX from a passed CuTe build and translate to PtxModule form.

Only used when CUTE_TO_PTX=1 and KERNEL_BACKEND=cute.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from cutegen.backend_runtime import cute_to_ptx_enabled
from cutegen.config import CUTEGEN_BASE_PATH, LLM_CONFIG_CODEGEN
from cutegen.llm_api import create_llm_server_from_config
from cutegen.util import debug_print, extract_first_code, read_file


def _cuobjdump_bin() -> str:
    for candidate in (
        os.environ.get("CUOBJDUMP_BIN"),
        shutil.which("cuobjdump"),
        "/usr/local/cuda/bin/cuobjdump",
    ):
        if candidate and Path(candidate).is_file():
            return candidate
    raise FileNotFoundError("cuobjdump not found; set CUOBJDUMP_BIN or install CUDA toolkit")


def extract_ptx_from_build(build_directory: str) -> str:
    """Dump embedded PTX from the first .so (or .cuda.o) under build_directory."""
    root = Path(build_directory)
    if not root.is_dir():
        raise FileNotFoundError(f"build directory missing: {build_directory}")

    candidates = sorted(root.rglob("*.so")) + sorted(root.rglob("*.cuda.o"))
    if not candidates:
        raise FileNotFoundError(f"no .so/.cuda.o under {build_directory}")

    cuobjdump = _cuobjdump_bin()
    last_err = ""
    for artifact in candidates:
        proc = subprocess.run(
            [cuobjdump, "-ptx", str(artifact)],
            check=False,
            capture_output=True,
            text=True,
        )
        text = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode == 0 and ".entry" in text:
            # Prefer the first PTX module block if cuobjdump wraps headers.
            match = re.search(r"(\.version[\s\S]*)$", text)
            return match.group(1).strip() if match else text.strip()
        last_err = text[-2000:]
    raise RuntimeError(f"cuobjdump -ptx failed for {candidates[0]}: {last_err}")


def maybe_store_extracted_ptx(node, build_directory: str) -> None:
    """Capture PTX into node.metadata before the build tree is deleted."""
    if not cute_to_ptx_enabled():
        return
    if getattr(node, "depth", None) != 0:
        return
    if (getattr(node, "metadata", None) or {}).get("extracted_ptx"):
        return
    try:
        ptx = extract_ptx_from_build(build_directory)
        node.metadata["extracted_ptx"] = ptx
        node.metadata["extracted_ptx_bytes"] = len(ptx)
        debug_print(
            f"Node {node.uuid}: stored extracted PTX ({len(ptx)} chars) for cute→PTX handoff"
        )
    except Exception as exc:
        node.metadata["extracted_ptx_error"] = str(exc)
        debug_print(f"Node {node.uuid}: PTX extraction failed: {exc}")


def translate_cute_src_to_ptx_model(node) -> str:
    """LLM: rewrite passed CuTe ModelNew into PtxModule ModelNew using extracted PTX."""
    if not cute_to_ptx_enabled():
        return ""
    prompt_path = (
        Path(CUTEGEN_BASE_PATH)
        / "cutegen"
        / "prompts"
        / "cute_to_ptx_transition_prompt.txt"
    )
    template = read_file(str(prompt_path))
    extracted = (node.metadata or {}).get("extracted_ptx") or ""
    # Cap enormous dumps so the prompt stays usable.
    max_ptx_chars = int(os.environ.get("CUTE_TO_PTX_MAX_CHARS", "120000"))
    if len(extracted) > max_ptx_chars:
        extracted = (
            extracted[:max_ptx_chars]
            + "\n\n/* … truncated extracted PTX … */\n"
        )
    prompt = (
        template.replace("<REF>", node.ref or "")
        .replace("<CUTE_SRC>", node.src or "")
        .replace("<EXTRACTED_PTX>", extracted)
    )
    import random

    llm = create_llm_server_from_config(random.choice(LLM_CONFIG_CODEGEN))
    response = llm(prompt)
    src = extract_first_code(
        response,
        code_language_types=["python", "ptx", "cpp", "c", ""],
    )
    return src or ""
