"""Triton integration test through the same Coordinator path as main.py."""

import json
import tempfile
from pathlib import Path

import cutegen.coordinator as coordinator_module
from cutegen.coordinator import Coordinator
from cutegen.node import ErrorType, Node
from scripts.test_triton_evaluator import GENERATED_SOURCE, REFERENCE_SOURCE


def main() -> None:
    # Use pre-generated source and stop after one complete runner/evaluator pass.
    coordinator_module.MAX_DEPTH = 0

    with tempfile.TemporaryDirectory(prefix="cutegen_triton_runner_") as directory:
        save_directory = Path(directory) / "triton_vector_add"
        node = Node(
            ref=REFERENCE_SOURCE,
            src=GENERATED_SOURCE,
            save_folder_path=str(save_directory),
        )
        node.metadata["kernel_backend"] = "triton"

        Coordinator([node]).run()

        snapshots = list(save_directory.glob("*.json"))
        if len(snapshots) != 1:
            raise AssertionError(f"Expected one saved node, found {snapshots}")
        saved = json.loads(snapshots[0].read_text())
        if saved["error_type"] != ErrorType.PASS.name:
            raise AssertionError(f"Runner Triton node failed: {saved['metadata']}")
        if saved["time"].get("num_trials") != 100:
            raise AssertionError(f"Runner timing is incomplete: {saved['time']}")

        print("Triton Coordinator runner path: PASS")
        print(f"Triton runner timing: {saved['time']}")


if __name__ == "__main__":
    main()
