from cutegen.node import Node, ErrorType
from cutegen.evaluate import evaluate
from cutegen.validate import validate
from cutegen.codegen import codegen_initial, codegen_optimize

from cutegen.config import MAX_RETRY_ATTEMPTS, CODEGEN_INITIAL_MODE, CODEGEN_OPTIMIZE_MODE
from cutegen.util import debug_print, load_json, read_file_with_lock

def run(node: Node, code_optimize_addendum="", codegen_initial_addendum="", fix_compile_addendum="", fix_correct_addendum=""):
    retry_count = 0
    while retry_count < MAX_RETRY_ATTEMPTS:
        debug_print(f"Node {node.uuid} run iteration {retry_count}")
        if node.error_type == ErrorType.PASS:
            break
        else:
            if retry_count > 0:
                node.update_history(node.src, "Regenerated code")
            # if is root node, generate initial code based on torch ref
            if node.depth == 0:
                if node.src == "":
                    node.src = codegen_initial(node, addendum=codegen_initial_addendum, mode=CODEGEN_INITIAL_MODE)
                    if node.src == "":
                        debug_print(f"Node {node.uuid} codegen_initial failed, retrying")
                        retry_count += 1
                        continue
                else:
                    pass # already supplied initial src, don't regenerate
            else:
                node.src = codegen_optimize(node, addendum=code_optimize_addendum, mode=CODEGEN_OPTIMIZE_MODE)
                if node.src == "":
                    debug_print(f"Node {node.uuid} codegen_optimize failed, retrying")
                    retry_count += 1
                    continue
            validate(node, fix_compile_addendum=fix_compile_addendum, fix_correct_addendum=fix_correct_addendum)
            retry_count += 1