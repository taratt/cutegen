import json
import difflib
from typing import List, Union

def code_edit_apply_patch(lines: List[str], patch: dict):
    action = patch['action']
    line_number = patch['line_number']
    new_code = patch.get('new_code', '')
    num_lines = patch.get('num_lines', 1)  # for delete/replace range

    if isinstance(new_code, str):
        new_code_lines = new_code.splitlines()
    elif isinstance(new_code, list):
        # Flatten list of lines, remove any trailing newlines
        new_code_lines = []
        for line in new_code:
            new_code_lines.extend(line.splitlines())
    else:
        new_code_lines = []  # For delete-only case with no new_code

    if action == 'replace':
        lines[line_number - 1:line_number - 1 + num_lines] = new_code_lines
    elif action == 'delete':
        del lines[line_number - 1:line_number - 1 + num_lines]
    elif action == 'insert':
        for i, line in enumerate(new_code_lines):
            lines.insert(line_number - 1 + i, line)
    else:
        raise ValueError(f"Unknown action: {action}")
    return lines


def code_edit_apply_patches(lines: List[str], patches: List[dict]):
    # Sort patches by line_number descending to avoid offset issues
    patches_sorted = sorted(patches, key=lambda p: p['line_number'], reverse=True)
    for patch in patches_sorted:
        lines = code_edit_apply_patch(lines, patch)
    return lines