"""PTX integration test through the same Coordinator path as main.py."""

import json
import tempfile
from pathlib import Path

import cutegen.coordinator as coordinator_module
from cutegen.coordinator import Coordinator
from cutegen.node import ErrorType, Node
from scripts.smoke_test_ptx_runtime import PTX_SOURCE
from scripts.test_ptx_evaluator import REFERENCE_SOURCE, generated_source


def main() -> None:
    # A pre-generated root source avoids an LLM request. Preventing depth 1
    # keeps this test focused on one complete runner/evaluator pass.
    coordinator_module.MAX_DEPTH = 0

    with tempfile.TemporaryDirectory(prefix="cutegen_ptx_runner_") as directory:
        save_directory = Path(directory) / "ptx_vector_add"
        node = Node(
            ref=REFERENCE_SOURCE,
            src=generated_source(PTX_SOURCE),
            save_folder_path=str(save_directory),
        )
        node.metadata["kernel_backend"] = "ptx"

        Coordinator([node]).run()

        snapshots = list(save_directory.glob("*.json"))
        if len(snapshots) != 1:
            raise AssertionError(f"Expected one saved node, found {snapshots}")
        saved = json.loads(snapshots[0].read_text())
        if saved["error_type"] != ErrorType.PASS.name:
            raise AssertionError(f"Runner PTX node failed: {saved['metadata']}")
        if saved["time"].get("num_trials") != 100:
            raise AssertionError(f"Runner timing is incomplete: {saved['time']}")

        print("PTX Coordinator runner path: PASS")
        print(f"PTX runner timing: {saved['time']}")


if __name__ == "__main__":
    main()
