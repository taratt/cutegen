from cutegen.config import LLM_CONFIG_CODEGEN, CUTEGEN_BASE_PATH, INITIAL_PROMPT_FILE, OPTIMIZE_PROMPT_FILE, EDIT_PROMPT_FILE, FEEDBACK_MODE, TUNE_PROMPT_FILE, USE_PROFILING, PROFILING_START_DEPTH
from cutegen.llm_api import create_llm_server_from_config
from cutegen.util import parse_code_and_edit, read_file, debug_print
from cutegen.node import Node
from cutegen.code_editor import code_edit_apply_patches
from cutegen.util import extract_first_code, parse_code_and_edit, read_file, get_numbered_lines, src_to_lines, lines_to_src, debug_print
import json
import random
from typing import Dict, Any, List

from typing import Dict, Any, List

def build_nsight_addendum_from_metrics(metrics: Dict[str, Any]) -> str:
    if not metrics:
        return ""

    kernel = metrics.get("kernel", "unknown_kernel")
    cfg = metrics.get("config", {}) or {}
    sections = metrics.get("sections", {}) or {}
    advice = metrics.get("advice", {}) or {}

    block_size = cfg.get("block_size", "unknown")
    grid_size = cfg.get("grid_size", "unknown")
    regs = cfg.get("registers_per_thread", "unknown")
    smem_static = cfg.get("static_smem_kb", "unknown")
    smem_dynamic = cfg.get("dynamic_smem_bytes", "unknown")
    occ_theoretical = cfg.get("theoretical_occupancy_pct", "unknown")
    occ_achieved = cfg.get("achieved_occupancy_pct", "unknown")

    sol = sections.get("GPU Speed Of Light Throughput", {}) or {}

    duration = sol.get("Duration", {}).get("value", "unknown")
    mem_throughput = sol.get("Memory Throughput", {}).get("value", "unknown")
    sm_throughput = sol.get("Compute (SM) Throughput", {}).get("value", "unknown")
    dram_throughput = sol.get("DRAM Throughput", {}).get("value", "unknown")

    inf_msgs = advice.get("INF", []) or []
    opt_msgs = advice.get("OPT", []) or []

    def is_num(x):
        return isinstance(x, (int, float))

    signals: List[str] = []

    if is_num(regs) and is_num(occ_achieved):
        if regs >= 96 and occ_achieved <= 40:
            signals.append("possible_register_pressure")

    if is_num(mem_throughput) and is_num(sm_throughput):
        if mem_throughput >= 70 and sm_throughput <= 50:
            signals.append("likely_memory_bound")

        if sm_throughput >= 70 and mem_throughput <= 50:
            signals.append("likely_compute_bound")

        if mem_throughput <= 40 and sm_throughput <= 40:
            signals.append("possible_structural_inefficiency")

    if is_num(smem_static) and smem_static == 0:
        signals.append("no_static_shared_memory")

    lines: List[str] = []

    lines.append("[NSIGHT_PROFILE]")
    lines.append(f"kernel={kernel}")
    lines.append(f"block_size={block_size}")
    lines.append(f"grid_size={grid_size}")
    lines.append(f"registers_per_thread={regs}")
    lines.append(f"static_smem_kb={smem_static}")
    lines.append(f"dynamic_smem_bytes={smem_dynamic}")
    lines.append(f"duration_ms={duration}")
    lines.append(f"compute_throughput_pct={sm_throughput}")
    lines.append(f"memory_throughput_pct={mem_throughput}")
    lines.append(f"dram_throughput_pct={dram_throughput}")
    lines.append(f"theoretical_occupancy_pct={occ_theoretical}")
    lines.append(f"achieved_occupancy_pct={occ_achieved}")
    lines.append("[/NSIGHT_PROFILE]")

    lines.append("")
    lines.append("[NSIGHT_SIGNALS]")

    if signals:
        for s in signals:
            lines.append(f"- {s}")
    else:
        lines.append("- no_strong_signal")

    lines.append("[/NSIGHT_SIGNALS]")

    lines.append("")
    lines.append("[NSIGHT_USAGE_RULES]")
    lines.append(
        "- Use these signals as supporting evidence, not as the sole driver of the optimization choice."
    )
    lines.append(
        "- For GEMM and convolution kernels, prefer structural optimization guidance unless profiler data gives a very strong bottleneck signal."
    )
    lines.append(
        "- Do not spend the only optimization attempt on minor launch tuning if stronger structural improvements are still missing."
    )
    lines.append("[/NSIGHT_USAGE_RULES]")

    if inf_msgs or opt_msgs:
        lines.append("")
        lines.append("[NSIGHT_TOOL_ADVICE]")

        if inf_msgs:
            lines.append("INF:")
            for msg in inf_msgs[:5]:
                lines.append(f"- {msg}")

        if opt_msgs:
            lines.append("OPT:")
            for msg in opt_msgs[:5]:
                lines.append(f"- {msg}")

        lines.append("[/NSIGHT_TOOL_ADVICE]")

    return "\n" + "\n".join(lines) + "\n"
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
    prompt = prompt.replace(
        "<REFERENCE_TIME>",
        str(getattr(node, "ref_time", "UNKNOWN"))
    )
    prompt = prompt.replace(
        "<PREVIOUS_SOURCE_TIME>",
        str(node.metadata.get("previous_src_time", "UNKNOWN"))
    )
    prompt = prompt.replace(
        "<NODE_REF_SRC>",
        str(node.ref)
    )
    if USE_PROFILING and node.depth > PROFILING_START_DEPTH:
        prompt += "Here are some profiling data that you should use to decide how to optimize the code:\n"
        prompt += build_nsight_addendum_for_node(node)
    prompt += read_file(CUTEGEN_BASE_PATH + "/cutegen/prompts/code_editor_prompt.md")
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
