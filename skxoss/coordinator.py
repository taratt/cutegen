from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import os

from skxoss.run import run
from skxoss.util import write_file_with_lock
from skxoss.config import SKXOSS_BASE_PATH, MAX_CONCURRENT_PROCESSES, MAX_DEPTH
from skxoss.node import ErrorType, Node

def _run_node_worker(node_data, code_optimize_addendum="", codegen_initial_addendum="", fix_compile_addendum="", fix_correct_addendum=""):
    try:
        node = pickle.loads(node_data)
        run(node, code_optimize_addendum=code_optimize_addendum, codegen_initial_addendum=codegen_initial_addendum, fix_compile_addendum=fix_compile_addendum, fix_correct_addendum=fix_correct_addendum)
        next_node = None
        if (node.error_type == ErrorType.PASS):
            next_node = Node(ref=node.ref, src="", prev_src=node.src, ref_time=node.ref_time, save_folder_path=node.save_folder_path, depth=node.depth + 1)
        return pickle.dumps(node), next_node
        
    except Exception as e:
        print(f"Error running node {getattr(node, 'uuid', 'unknown')}: {e}")
        return None

class Coordinator:
    def __init__(self, starting_nodes):
        self.queue = []
        self.queue.extend(starting_nodes)
        self.max_concurrent_processes = MAX_CONCURRENT_PROCESSES
        self.code_optimize_addendum = ""
        self.codegen_initial_addendum = ""
        self.fix_compile_addendum = ""
        self.fix_correct_addendum = ""

    def run(self):
        with ProcessPoolExecutor(max_workers=self.max_concurrent_processes) as executor:
            future_to_node = {}
            
            while future_to_node or self.queue: # if we have unfinished nodes or nodes to be run
                while self.queue and len(future_to_node) < self.max_concurrent_processes:
                    node = self.queue.pop(0)
                    if not os.path.exists(f"{node.save_folder_path}/best_time.txt"):
                        os.makedirs(node.save_folder_path, exist_ok=True)
                        write_file_with_lock(f"{node.save_folder_path}/best_time.txt", "999999.0")
                    node_data = pickle.dumps(node)
                    future = executor.submit(_run_node_worker, node_data, self.code_optimize_addendum, self.codegen_initial_addendum, self.fix_compile_addendum, self.fix_correct_addendum)
                    future_to_node[future] = node
                
                # Wait for at least one task to complete
                if future_to_node:
                    for future in as_completed(future_to_node):
                        node = future_to_node.pop(future)
                        try:
                            node_data, next_node = future.result()
                            if node_data is not None:
                                node = pickle.loads(node_data)
                                if next_node is not None and node.depth < MAX_DEPTH:
                                    self.queue.append(next_node)
                                node.save_as_json_local(node.save_folder_path)
                            else:
                                print(f"Error processing result for node {getattr(node, 'uuid', 'unknown')}: Node data: {node_data}")
                            
                        except Exception as e:
                            print(f"Error processing result for node {getattr(node, 'uuid', 'unknown')}: {e}")
                            continue
                else:
                    break