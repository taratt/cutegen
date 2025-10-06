# Line-Based Code Editor Format

Generate code changes using a line-first format that is easy for LLMs to produce and parse.

## Edit Syntax

All edits must be wrapped in `<EDITS>` and `</EDITS>` tags. Each edit starts with line numbers, followed by the action:

### REPLACE Command
```
<EDITS>
LINE 10 REPLACE:
<new code here>
END
</EDITS>
```

For multiple lines:
```
<EDITS>
LINES 5-7 REPLACE:
<new code here>
END
</EDITS>
```

### DELETE Command  
```
<EDITS>
LINE 15 DELETE
</EDITS>
```

For multiple lines:
```
<EDITS>
LINES 15-16 DELETE
</EDITS>
```

### INSERT Command
```
<EDITS>
LINE 8 INSERT:
<new code here>
END
</EDITS>
```

## Format Rules

1. **Wrapping tags** required: All edits must be inside `<EDITS>...</EDITS>` tags
2. **Line numbers always come first** for easy parsing
3. **Line ranges** use hyphen notation: `LINES 5-10`
4. **Single lines** use: `LINE 10`
5. **Actions** are: REPLACE, DELETE, INSERT
6. **Code blocks** end with `END` on its own line (for REPLACE and INSERT)
7. **Line numbers** are 1-based (first line is 1)
8. **INSERT** adds code BEFORE the specified line

## Examples

### Replace a single line:
```
<EDITS>
LINE 10 REPLACE:
    return x * 2
END
</EDITS>
```

### Replace multiple lines:
```
<EDITS>
LINES 5-7 REPLACE:
def calculate(x, y):
    result = x + y
    return result
END
</EDITS>
```

### Delete a single line:
```
<EDITS>
LINE 15 DELETE
</EDITS>
```

### Delete multiple lines:
```
<EDITS>
LINES 15-16 DELETE
</EDITS>
```

### Insert new code:
```
<EDITS>
LINE 8 INSERT:
    # New helper function
    def helper():
        pass
END
</EDITS>
```

### Multiple edits in one block:
```
<EDITS>
LINE 3 REPLACE:
import numpy as np
END

LINES 10-12 DELETE

LINE 20 INSERT:
    def validate_input(data):
        return data is not None
END
</EDITS>
```

## Important Notes

- **Wrapping tags**: All edits MUST be wrapped in `<EDITS>...</EDITS>` tags
- **Multiple edits**: Can be grouped in a single `<EDITS>` block or separated into multiple blocks
- **Indentation**: Preserve exact indentation in code blocks - spaces and tabs matter
- **Line endings**: Each line in a code block is treated as a separate line
- **Execution order**: Edits within a block are applied in the order they appear
- **Line number shifts**: After each edit, subsequent line numbers refer to the modified file state
- **Empty lines**: Blank lines within code blocks are preserved
- **Parsing**: Line numbers are easily extractable with regex patterns

## Parsing Patterns

Regular expressions for parsing:
- Edit block: `<EDITS>(.*?)</EDITS>` (with DOTALL flag)
- Single line: `^LINE (\d+) (REPLACE|DELETE|INSERT)`
- Line range: `^LINES (\d+)-(\d+) (REPLACE|DELETE)`
- End marker: `^END$`

## Error Handling

Invalid commands will be reported with clear error messages:
- "Missing <EDITS> tags" (edits not wrapped properly)
- "Invalid line number: 0" (must be >= 1)
- "Missing END marker for REPLACE/INSERT"
- "Invalid line range: 7-5" (end before start)
- "Unknown action: MODIFY" (not REPLACE/DELETE/INSERT)
- "Unclosed <EDITS> tag" (missing </EDITS>)

## Advantages Over JSON

1. **Clear boundaries** with `<EDITS>` tags for unambiguous parsing
2. **Line-first design** makes line numbers immediately visible and parsable
3. **Simple parsing** with predictable patterns
4. **No JSON syntax errors** or escaping issues
5. **Clear structure** with line numbers as the primary identifier
6. **Regex-friendly** format for easy extraction of line numbers and edit blocks