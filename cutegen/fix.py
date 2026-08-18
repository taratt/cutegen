import random
import json

from cutegen.node import Node
from cutegen.config import (
    CUTEGEN_BASE_PATH,
    LLM_CONFIG_CODEGEN,
    COMPILE_LOG_CHARS,
    DEBUG_GUIDE_FILE,
    KERNEL_BACKEND,
)
from cutegen.llm_api import create_llm_server_from_config
from cutegen.util import extract_first_code, read_file, get_numbered_lines, src_to_lines, lines_to_src, debug_print
from cutegen.code_editor import code_edit_apply_patches

def get_compile_suggestions(node: Node):
    llm_server = create_llm_server_from_config(random.choice(LLM_CONFIG_CODEGEN))
    prompt = f"""The following code is not compiling:
```
{node.src}
```
The error is:
{node.metadata['compile'][0:min(len(node.metadata['compile']), COMPILE_LOG_CHARS)]}
```
The reference code, which you can use to reason about the intended operation, is:
```
{node.ref}
Reason about what is the underlying issue. Generate some useful suggestions for fixing the compile error; here's an doc that might have useful information:
{read_file(DEBUG_GUIDE_FILE)}
"""
    llm_response = llm_server(prompt)
    return llm_response

def get_correctness_suggestions(node: Node):
    llm_server = create_llm_server_from_config(random.choice(LLM_CONFIG_CODEGEN))
    prompt = f"""The following code is not correct (either not producing correct results or having a runtime error):
```
{node.src}
```
The error is (if the error is empty it means it is not generating correct outputs):
{node.metadata['correct'][0:min(len(node.metadata['correct']), COMPILE_LOG_CHARS)]}
```
The reference code, which you can use to reason about the intended operation, is:
```
{node.ref}
Generate some useful suggestions for fixing the correctness error; here's a doc that might have useful information:
{read_file(DEBUG_GUIDE_FILE)}
"""
    llm_response = llm_server(prompt)
    return llm_response

def _fix_compile(node: Node, addendum="", retrieve=False):
    llm_server = create_llm_server_from_config(random.choice(LLM_CONFIG_CODEGEN))
    if retrieve:
        suggestions = get_compile_suggestions(node)
    else:
        suggestions = ""
    backend_name = "PTX and Python driver glue" if KERNEL_BACKEND == "ptx" else KERNEL_BACKEND.upper()
    prompt = f"""The following {backend_name} code is not compiling:
```
{node.src}
```
The error is:
{node.metadata['compile'][0:min(len(node.metadata['compile']), COMPILE_LOG_CHARS)]}
{f"Here are some potentially useful suggestions for fixing the compile error: {suggestions}" if retrieve else ""}
Follow the original structure of the code (the optimized output architecture is named ModelNew with custom {backend_name} kernel(s)).
Generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! This is very important! Output the new code in CODEBLOCKS (wrap in ``` and ```). ONLY fix the compile error, do not change the intended functionality of the code and its optimization techniques!{addendum}
"""
    llm_response = llm_server(prompt)
    src = extract_first_code(llm_response)
    fix = []
    fix.append(suggestions)
    fix.append(llm_response)
    return src, fix

def _fix_compile_edits(node: Node, addendum="", retrieve=False):
    llm_server = create_llm_server_from_config(random.choice(LLM_CONFIG_CODEGEN))
    if retrieve:
        suggestions = get_compile_suggestions(node)
    else:
        suggestions = ""
    src_lines = src_to_lines(str(node.src))
    backend_name = (
        "PTX and Python driver glue"
        if KERNEL_BACKEND == "ptx"
        else KERNEL_BACKEND.upper()
    )

    prompt_content = read_file(CUTEGEN_BASE_PATH + "/cutegen/prompts/code_editor_prompt.md")
    output_instruction = "Output your reasoning for the edits in <reasoning></reasoning> tags. Output the edits in a codeblock ```json and ```."

    prompt = f"""The following {backend_name} code is not compiling:
```
{get_numbered_lines(src_lines)}
```
The error is:
{node.metadata['compile'][0:min(len(node.metadata['compile']), COMPILE_LOG_CHARS)]}
{f"Here are some potentially useful suggestions for fixing the compile error: {suggestions}" if retrieve else ""}
Fix the compile error. DO NOT change the intent by simplifying or replacing the custom {backend_name} implementation.
```
The reference code, which you can use to reason about the intended operation, is:
```
{node.ref}
You will output a list of edits to the code. The semantics of the edits is described below:
{prompt_content}

Do not use pre-optimized libraries like cuBLAS, do not change the intended precision, and do not fall back to PyTorch operators. Preserve ModelNew and its custom {backend_name} kernel launch structure.

{output_instruction}
"""
    llm_response = llm_server(prompt)
    fix = [llm_response]
    try:
        edits = extract_first_code(llm_response, code_language_types=["json", "python", ""])
        edits = json.loads(edits)
        edits = edits if isinstance(edits, list) else [edits]
        #debug_print(edits)
        src_lines = code_edit_apply_patches(src_lines, edits)
        src = lines_to_src(src_lines)
        return src, fix
    except Exception as e:
        debug_print(f"Error parsing edits: {edits} ")
        return node.src, fix

def _fix_correct_edits(node: Node, addendum="", retrieve=False):
    llm_server = create_llm_server_from_config(random.choice(LLM_CONFIG_CODEGEN))
    if retrieve:
        suggestions = get_correctness_suggestions(node)
    else:
        suggestions = ""
    src_lines = src_to_lines(str(node.src))
    prompt_content = read_file(CUTEGEN_BASE_PATH + "/cutegen/prompts/code_editor_prompt.md")
    output_instruction = "Output your reasoning for the edits in <reasoning></reasoning> tags. Output the edits in a codeblock ```json and ```."

    backend_name = "PTX and Python driver glue" if KERNEL_BACKEND == "ptx" else KERNEL_BACKEND.upper()
    prompt = f"""The following {backend_name} code is not correct:
```
{get_numbered_lines(src_lines)}
```
The error is:
{node.metadata['correct'][0:min(len(node.metadata['correct']), COMPILE_LOG_CHARS)]}
The reference code, which you can use to reason about the intended operation, is:
```
{node.ref}
```
{f"Here are some potentially useful suggestions for fixing the correctness error: {suggestions}" if retrieve else ""}
Fix the correctness error. DO NOT change the intent by simplifying or replacing the custom {backend_name} implementation.

You will output a list of edits to the code. The semantics of the edits is described below:
{prompt_content}
Do not use pre-optimized libraries like cuBLAS, do not change the intended precision, and do not fall back to PyTorch operators. Preserve ModelNew and its custom {backend_name} kernel launch structure.

{output_instruction}
"""
    llm_response = llm_server(prompt)
    fix = [llm_response]
    #debug_print(fix)
    try:
        edits = extract_first_code(llm_response, code_language_types=["json", "python", ""])
        edits = json.loads(edits)
      #  debug_print("EDITS: "+edits)
        edits = edits if isinstance(edits, list) else [edits]
        #debug_print(llm_response)
        src_lines = code_edit_apply_patches(src_lines, edits)
        src = lines_to_src(src_lines)
        return src, fix
    except Exception as e:
        debug_print(f"Error parsing edits: {e}")
        return node.src, fix

def _fix_correct(node: Node, addendum="", retrieve=False):
    llm_server = create_llm_server_from_config(random.choice(LLM_CONFIG_CODEGEN))
    if retrieve:
        suggestions = get_correctness_suggestions(node)
    else:
        suggestions = ""
    backend_name = (
        "PTX and Python driver glue"
        if KERNEL_BACKEND == "ptx"
        else KERNEL_BACKEND.upper()
    )
    prompt = f"""The following code is not correct:
```
{node.src}
```
The reference code, which you can use to reason about the intended operation, is:
```
{node.ref}
```
The error is:
{node.metadata['correct'][0:min(len(node.metadata['correct']), COMPILE_LOG_CHARS)]}
{f"Here are some potentially useful suggestions for fixing the correctness error: {suggestions}" if retrieve else ""}
Fix the correctness error. Do not change the intended backend or replace the custom {backend_name} implementation with a fallback.
Generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! This is very important! Output the new code in CODEBLOCKS (wrap in ``` and ```). ONLY fix the correctness error, do not change the intended functionality of the code and its optimization techniques!{addendum}
"""
    llm_response = llm_server(prompt)
    src = extract_first_code(llm_response)
    fix = []
    fix.append(llm_response)
    return src, fix

def fix_compile(node: Node, addendum="", mode="original", retrieve=False):
    if mode == "original":
        return _fix_compile(node, addendum, retrieve)
    elif mode == "edits":
        return _fix_compile_edits(node, addendum, retrieve)
    else:
        raise ValueError(f"Invalid mode: {mode}")

def fix_correct(node: Node, addendum="", mode="original", retrieve=False):
    if mode == "original":
        return _fix_correct(node, addendum, retrieve)
    elif mode == "edits":
        return _fix_correct_edits(node, addendum, retrieve)
    else:
        raise ValueError(f"Invalid mode: {mode}")