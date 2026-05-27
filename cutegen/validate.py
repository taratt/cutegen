from cutegen.node import Node, ErrorType
from cutegen.evaluate import evaluate

from cutegen.config import MAX_FIX_ATTEMPTS, FIX_COMPILE_MODE, FIX_CORRECT_MODE, FIX_RETRIEVE
from cutegen.fix import fix_compile, fix_correct
from cutegen.util import debug_print

def validate(node: Node, fix_compile_addendum="", fix_correct_addendum=""):
    """
    This function validates the node.
    """
    fix_count = 0
    best_state = (node.src, node.error_type, node.metadata.copy())
    state_rank = {ErrorType.PASS: 3, ErrorType.CORRECT: 2, ErrorType.COMPILE: 1, ErrorType.NONE: 0}
    timeout_count = 0
    while fix_count < MAX_FIX_ATTEMPTS and timeout_count < 3:
        debug_print(f"Node {node.uuid} validate iteration {fix_count}")
        evaluate(node)
        if 'timeout' in node.metadata:
            timeout_count += 1
            continue # timeout, don't fix, try eval again
        
        if state_rank[node.error_type] > state_rank[best_state[1]]:
            best_state = (node.src, node.error_type, node.metadata.copy())
        
        if node.error_type == ErrorType.PASS:
            return True
        else:
            if node.error_type == ErrorType.COMPILE:
                src, fix = fix_compile(node, addendum=fix_compile_addendum, mode=FIX_COMPILE_MODE, retrieve=FIX_RETRIEVE)
                node.update_history(src, fix)
            elif node.error_type == ErrorType.CORRECT:
                src, fix = fix_correct(node, addendum=fix_correct_addendum, mode=FIX_CORRECT_MODE, retrieve=FIX_RETRIEVE)
                node.update_history(src, fix)
            fix_count += 1
    
    node.src, node.error_type, node.metadata = best_state
    return False