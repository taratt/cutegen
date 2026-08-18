from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import os

from cutegen.run import run
from cutegen.util import write_file_with_lock
from cutegen.config import CUTEGEN_BASE_PATH, MAX_CONCURRENT_PROCESSES, MAX_DEPTH, MAX_RETRIES_PER_DEPTH, ALWAYS_OPTIMIZE_BEST
from cutegen.node import ErrorType, Node

def _extract_mean_time(t, default=999999.0):
    """
    Accepts:
        float
        int
        dict with key 'mean'
        None

    Returns:
        float mean time
    """
    if t is None:
        return default

    if isinstance(t, (int, float)):
        return float(t)

    if isinstance(t, dict):
        return float(t.get("mean", default))

    return default

def _run_node_worker(node_data, code_optimize_addendum="", codegen_initial_addendum="", fix_compile_addendum="", fix_correct_addendum=""):
    try:
        node = pickle.loads(node_data)
        node.metadata = getattr(node, "metadata", {}) or {}
        node.metadata.setdefault("last_passed_src", node.prev_src if getattr(node, "prev_src", None) else "")
        node.metadata.setdefault("last_passed_depth", max(0, getattr(node, "depth", 0) - 1))
        node.metadata.setdefault("retries_by_depth", {})

        node.metadata.setdefault("best_passed_src", node.prev_src if getattr(node, "prev_src", None) else "")
        node.metadata.setdefault("best_passed_depth", max(0, getattr(node, "depth", 0) - 1))
        node.metadata.setdefault("best_passed_time", 999999.0)
        node.metadata.setdefault("best_passed_nsight_metrics", {})

        run(node, code_optimize_addendum=code_optimize_addendum, codegen_initial_addendum=codegen_initial_addendum, fix_compile_addendum=fix_compile_addendum, fix_correct_addendum=fix_correct_addendum)
        next_node = None
        if (node.error_type == ErrorType.PASS):
            curr_time = _extract_mean_time(node.time)
            best_time = _extract_mean_time(node.metadata.get("best_passed_time"))
            if curr_time < best_time:
                node.metadata["best_passed_src"] = node.src
                node.metadata["best_passed_depth"] = node.depth
                node.metadata["best_passed_time"] = getattr(node, "time", None)
                node.metadata["best_passed_nsight_metrics"] = node.metadata.get("nsight_metrics", {})

            if ALWAYS_OPTIMIZE_BEST and node.metadata.get("best_passed_src"):
                baseline_src = node.metadata["best_passed_src"]
                baseline_time = node.metadata.get("best_passed_time")
                baseline_nsight = node.metadata.get("best_passed_nsight_metrics", {})
            else:
                baseline_src = node.src
                baseline_time = node.time
                baseline_nsight = node.metadata.get("nsight_metrics", {})

            node.metadata["last_passed_src"] = node.src
            node.metadata["last_passed_depth"] = node.depth
            next_node = Node(ref=node.ref, src="", prev_src=baseline_src, ref_time=node.ref_time, save_folder_path=node.save_folder_path, depth=node.depth + 1)
            next_node.metadata["prev_nsight_metrics"] = baseline_nsight
            next_node.metadata["last_passed_src"] = node.metadata["last_passed_src"]
            next_node.metadata["last_passed_depth"] = node.metadata["last_passed_depth"]
            next_node.metadata["retries_by_depth"] = node.metadata.get("retries_by_depth", {}).copy()
            next_node.metadata["previous_src_time"] = baseline_time
            next_node.metadata["best_passed_src"] = node.metadata.get("best_passed_src", "")
            next_node.metadata["best_passed_depth"] = node.metadata.get("best_passed_depth", max(0, node.depth))
            next_node.metadata["best_passed_time"] = node.metadata.get("best_passed_time", 999999.0)
            next_node.metadata["best_passed_nsight_metrics"] = node.metadata.get("best_passed_nsight_metrics", {})
            next_node.metadata["kernel_backend"] = node.metadata.get("kernel_backend", "unknown")


        else:
            if node.depth > 0:
                retries = node.metadata["retries_by_depth"].get(node.depth, 0)
                if retries < MAX_RETRIES_PER_DEPTH:
                    node.metadata["retries_by_depth"][node.depth] = retries + 1

                    last_src = node.metadata.get("last_passed_src") or node.prev_src or ""
                    last_depth = int(node.metadata.get("last_passed_depth", node.depth - 1))

                    # We want to keep trying to produce code for the "next" depth after last pass.
                    retry_depth = min(node.depth, last_depth + 1)

                    next_node = Node(
                        ref=node.ref,
                        src="",
                        prev_src=last_src,             # <- reset baseline code to last PASS
                        ref_time=node.ref_time,
                        save_folder_path=node.save_folder_path,
                        depth=retry_depth              # <- continue regenerating at this depth
                    )
                    # propagate metadata (including last_passed_*)
                    next_node.metadata["last_passed_src"] = last_src
                    next_node.metadata["last_passed_depth"] = last_depth
                    next_node.metadata["retries_by_depth"] = node.metadata["retries_by_depth"].copy()
                    next_node.metadata["kernel_backend"] = node.metadata.get("kernel_backend", "unknown")

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
