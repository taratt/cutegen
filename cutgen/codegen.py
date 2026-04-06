from cutgen.config import LLM_CONFIG_CODEGEN, SKXOSS_BASE_PATH, INITIAL_PROMPT_FILE, OPTIMIZE_PROMPT_FILE, EDIT_PROMPT_FILE, FEEDBACK_MODE, TUNE_PROMPT_FILE, USE_PROFILING, PROFILING_START_DEPTH
from cutgen.llm_api import create_llm_server_from_config
from cutgen.util import parse_code_and_edit, read_file, debug_print
from cutgen.node import Node
from cutgen.code_editor import code_edit_apply_patches
from cutgen.util import extract_first_code, parse_code_and_edit, read_file, get_numbered_lines, src_to_lines, lines_to_src, debug_print
import json
import random
from typing import Dict, Any, List

def build_nsight_addendum_from_metrics(metrics: Dict[str, Any]) -> str:
    if not metrics:
        return ""

    kernel = metrics.get("kernel", "unknown_kernel")
    cfg = metrics.get("config", {})
    sections = metrics.get("sections", {})
    advice = metrics.get("advice", {})

    # Extract key metrics with safe defaults
    block_size = cfg.get("block_size", "unknown")
    grid_size = cfg.get("grid_size", "unknown")
    regs = cfg.get("registers_per_thread", "unknown")
    smem_static = cfg.get("static_smem_kb", "unknown")
    smem_dynamic = cfg.get("dynamic_smem_bytes", "unknown")
    occ_theoretical = cfg.get("theoretical_occupancy_pct", "unknown")
    occ_achieved = cfg.get("achieved_occupancy_pct", "unknown")

    sol = sections.get("GPU Speed Of Light Throughput", {})
    duration = sol.get("Duration", {}).get("value", "unknown")
    mem_throughput = sol.get("Memory Throughput", {}).get("value", "unknown")
    sm_throughput = sol.get("Compute (SM) Throughput", {}).get("value", "unknown")
    dram_throughput = sol.get("DRAM Throughput", {}).get("value", "unknown")

    inf_msgs = advice.get("INF", [])
    opt_msgs = advice.get("OPT", [])

    # Build a concise, LLM-friendly summary
    lines = []
    lines.append(f"Kernel name: {kernel}")
    lines.append(f"Launch config: block_size={block_size}, grid_size={grid_size}")
    lines.append(
        f"Registers per thread={regs}, static_smem_kb={smem_static}, "
        f"dynamic_smem_bytes={smem_dynamic}"
    )
    lines.append(
        f"Duration≈{duration} ms, Compute throughput≈{sm_throughput}%, "
        f"Memory throughput≈{mem_throughput}%, DRAM throughput≈{dram_throughput}%"
    )
    lines.append(
        f"Occupancy: theoretical≈{occ_theoretical}%, achieved≈{occ_achieved}% "
        f"(register-limited if registers per thread is high)."
    )

    if inf_msgs:
        lines.append("Nsight INF advice:")
        for msg in inf_msgs:
            lines.append(f"- {msg}")
    if opt_msgs:
        lines.append("Nsight OPT advice:")
        for msg in opt_msgs:
            lines.append(f"- {msg}")

    addendum = "\n".join(lines)

    # Wrap in a clear block for your prompts
    return (
        "\n\n[NSIGHT_COMPUTE_PROFILE]\n"
        + addendum
        + "\n[/NSIGHT_COMPUTE_PROFILE]\n"
    )

def build_nsight_addendum_for_node(node: Node) -> str:
    metrics = node.metadata.get("prev_nsight_metrics", {})
    return build_nsight_addendum_from_metrics(metrics)

def codegen_initial(node: Node, addendum="", mode="original"):
    if mode == "original":
        return codegen_initial_original(node, addendum)
    else:
        raise ValueError(f"Invalid codegen_initial mode: {mode}")

def codegen_initial_original(node: Node, addendum=""):
    with open(INITIAL_PROMPT_FILE, "r") as file:
        template_prompt = file.read()
    prompt = addendum
    prompt += template_prompt
    prompt += addendum
    prompt = prompt.replace("<REF>", node.ref)
    print(prompt)
    llm_server = create_llm_server_from_config(random.choice(LLM_CONFIG_CODEGEN))
    src = llm_server(prompt)
    src = extract_first_code(src)
    return src

def codegen_optimize_original(node: Node, addendum=""):
    llm_server = create_llm_server_from_config(random.choice(LLM_CONFIG_CODEGEN))
    with open(OPTIMIZE_PROMPT_FILE, "r") as file:
        optimize_prompt = file.read()
    prompt = optimize_prompt.replace("<NODE_PRV_SRC>", node.prev_src)
    src = llm_server(prompt)
    src = parse_code_and_edit(src)
    return src


def codegen_edits(node: Node, addendum=""):
    src_lines = src_to_lines(str(node.prev_src))
    llm_server = create_llm_server_from_config(random.choice(LLM_CONFIG_CODEGEN))
    #if FEEDBACK_MODE == "iterative":
        # if node.depth%2 == 0:
        #     with open(EDIT_PROMPT_FILE, "r") as file:
        #         edit_prompt = file.read()
        # else:
        #     with open(TUNE_PROMPT_FILE, "r") as file:
        #         edit_prompt = file.read()
    with open(EDIT_PROMPT_FILE, "r") as file:
        edit_prompt = file.read()
    prompt = edit_prompt.replace("<NODE_PRV_SRC>", get_numbered_lines(src_lines))
    if USE_PROFILING and node.depth > PROFILING_START_DEPTH:
        prompt += "Here are some profiling data that you should use to decide how to optimize the code:\n"
        prompt += build_nsight_addendum_for_node(node)
    prompt += read_file(SKXOSS_BASE_PATH + "/cutgen/prompts/code_editor_prompt.md")
    prompt += addendum
    llm_server = create_llm_server_from_config(random.choice(LLM_CONFIG_CODEGEN))
    llm_response = llm_server(prompt)
    debug_print(f"node {node.uuid} codegen_edits llm_response: {llm_response}")
    src = ""
    # Apply code editor patches
    code, edits_json = parse_code_and_edit(llm_response, code_language_types=["json", "python", ""])
    edits_json = edits_json if isinstance(edits_json, list) else [edits_json]
    debug_print(edits_json)
    src_lines = code_edit_apply_patches(src_lines, edits_json)
    src = lines_to_src(src_lines)  
    return src

def codegen_optimize(node: Node, addendum="", mode="original"):
    if mode == "original":
        return codegen_optimize_original(node, addendum)
    elif mode == "edits":
        return codegen_edits(node, addendum)
    else:
        raise ValueError(f"Invalid codegen_optimize mode: {mode}")
