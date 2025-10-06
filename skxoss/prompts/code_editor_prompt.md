Generate code changes in JSON format compatible with the CodeEditor class. Each change should be a JSON object with the following structure:

## Patch Format
```json
{
  "action": "replace|delete|insert",
  "line_number": <integer>,
  "new_code": <string or list of strings>,
  "num_lines": <integer>
}
```

## Field Explanations:

1. **action** (required): The type of edit operation
   - `"replace"`: Replace existing lines with new code
   - `"delete"`: Remove lines from the file
   - `"insert"`: Add new lines at a specific position

2. **line_number** (required): The 1-based line number where the operation starts
   - For replace/delete: First line to be affected
   - For insert: New code will be inserted BEFORE this line

3. **new_code** (optional): The code to add/replace
   - Can be a single string or a list of strings
   - Each string represents one line (newlines are added automatically)
   - Not needed for delete operations

4. **num_lines** (optional, default=1): Number of lines to affect
   - Only used for replace/delete operations
   - Specifies how many lines to replace or delete starting from line_number

## Examples:

### Replace a single line:
```json
{
  "action": "replace",
  "line_number": 10,
  "new_code": "    return x * 2"
}
```

### Replace multiple lines:
```json
{
  "action": "replace", 
  "line_number": 5,
  "num_lines": 3,
  "new_code": [
    "def calculate(x, y):",
    "    result = x + y",
    "    return result"
  ]
}
```

### Delete lines:
```json
{
  "action": "delete",
  "line_number": 15,
  "num_lines": 2
}
```

### Insert new code:
```json
{
  "action": "insert",
  "line_number": 8,
  "new_code": [
    "    # New helper function",
    "    def helper():",
    "        pass"
  ]
}
```

## Important Notes:
- Line numbers are 1-based (first line is 1, not 0)
- Newlines are automatically added to each line, don't include them in the strings
- Indents are important, so make sure to preserve them and generate the correct number of spaces or tabs in the new code to be consistent with the original code.
- For multiple changes, provide an array of patch objects

