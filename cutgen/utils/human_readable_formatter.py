import json
from pathlib import Path
import os


def print_json_pretty(file_path: str):
    """
    Reads a JSON file and prints it in a nicely formatted, human-readable way.

    Args:
        file_path (str): Path to the JSON file.
    """
    path = Path(file_path)
    if not path.exists():
        print(f"Error: File '{file_path}' does not exist.")
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Pretty print with indentation and sorted keys
        pretty_str = json.dumps(data, indent=4, sort_keys=True, ensure_ascii=False)
        print(pretty_str)


    except json.JSONDecodeError as e:
        print(f"Error: File '{file_path}' is not valid JSON.\n{e}")



def print_history(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    history = data.get("history", [])
    for i, entry in enumerate(history, 1):
        print("=" * 40)
        print(f"History Entry {i}")
        print("=" * 40)

        # Pretty-print the src block
        src = entry.get("src", "")
        print("\nSRC:\n")
        print(src)  # prints as normal readable code string

        # Other metadata
        print("\nError Type:", entry.get("error_type"))
        print("Message:\n", entry.get("msg"))

        fixes = entry.get("fix", [])
        if fixes:
            print("\nFix Suggestions:")
            for fix in fixes:
                print(fix)
        print("\n\n")

    print(data["src"])
file_path = "/home/tarasaba/PycharmProjects/cutgen/saved_nodes/cute/attempt_15/19_ReLU.py/2_20260128-213035.json"
print_json_pretty(file_path)
print_history(file_path)