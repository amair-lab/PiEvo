import json
import re
from typing import Dict, Any, List


def _sanitize_json_for_llm(json_str: str) -> str:
    """Make LLM-mangled JSON parseable by Python's json.loads.

    Handles all common LLM output issues:
    - Invalid backslash escapes (\\epsilon, \\pi, \\Delta ...) → literal backslash
    - Unescaped newlines, tabs, carriage returns inside string values
    - Trailing commas before } or ]
    - Line (//) and block (/* */) comments
    - Single quotes used instead of double quotes (for keys/strings)
    - Unescaped literal backslashes inside strings
    - BOM and zero-width Unicode characters
    """
    BS = '\\'  # backslash character

    # 1. Strip BOM and zero-width characters
    json_str = json_str.replace('\ufeff', '')
    json_str = json_str.replace('\u200b', '')

    # 2. Remove comments (line and block) — must happen before other processing
    #    but we must NOT remove // or /* that appear inside string values.
    #    Use a simple state machine to skip strings first.
    result = []
    i = 0
    while i < len(json_str):
        c = json_str[i]
        # Skip string literals entirely
        if c == '"':
            result.append(c)
            i += 1
            while i < len(json_str):
                c2 = json_str[i]
                result.append(c2)
                if c2 == BS and i + 1 < len(json_str):
                    i += 1
                    result.append(json_str[i])  # escaped char in string
                elif c2 == '"':
                    break
                i += 1
            i += 1
            continue
        # Line comment
        if c == '/' and i + 1 < len(json_str) and json_str[i + 1] == '/':
            while i < len(json_str) and json_str[i] != '\n':
                i += 1
            continue
        # Block comment
        if c == '/' and i + 1 < len(json_str) and json_str[i + 1] == '*':
            i += 2
            while i + 1 < len(json_str) and not (json_str[i] == '*' and json_str[i + 1] == '/'):
                i += 1
            i += 2  # skip */
            continue
        result.append(c)
        i += 1
    json_str = ''.join(result)

    # 3. Convert single-quoted strings to double-quoted strings
    #    Again, use state machine to avoid converting inside double-quoted strings
    result = []
    i = 0
    in_double_quote = False
    while i < len(json_str):
        c = json_str[i]
        if c == '"':
            in_double_quote = not in_double_quote
            result.append(c)
            i += 1
        elif c == "'" and not in_double_quote:
            result.append('"')
            i += 1
        else:
            result.append(c)
            i += 1
    json_str = ''.join(result)

    # 4. Remove trailing commas before } or ] (handles whitespace/newlines between)
    json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)

    # 5. Fix unescaped control characters inside string values
    #    Walk through strings character by character and escape bad chars
    result = []
    i = 0
    while i < len(json_str):
        c = json_str[i]
        if c == '"':
            # We are at the start of a string — process it character by character
            result.append(c)
            i += 1
            while i < len(json_str):
                c2 = json_str[i]
                if c2 == BS and i + 1 < len(json_str):
                    next_char = json_str[i + 1]
                    valid_escapes = set('"\\/bfnrt')
                    if next_char in valid_escapes:
                        # Valid JSON escape — pass through
                        result.append(c2)
                        result.append(next_char)
                    elif next_char == 'u' and i + 5 < len(json_str):
                        # Unicode escape — validate it
                        u_str = json_str[i:i+6]
                        if re.match(r'\\u[0-9a-fA-F]{4}', u_str):
                            result.append(u_str)
                        else:
                            # Malformed unicode escape — treat as literal
                            result.append(BS + BS)
                            result.append(next_char)
                    else:
                        # Invalid escape (LaTeX \\epsilon, \\pi, etc.) → literal backslash
                        result.append(BS + BS)
                        result.append(next_char)
                    i += 2
                    continue
                elif c2 == '"':
                    # End of string
                    result.append(c2)
                    i += 1
                    break
                elif ord(c2) < 0x20 and c2 != '\t':
                    # Unescaped control characters (newline, CR, etc.) → escaped form
                    if c2 == '\n':
                        result.append(BS + 'n')
                    elif c2 == '\r':
                        result.append(BS + 'r')
                    else:
                        # Other control chars → unicode escape
                        result.append(f'\\u{ord(c2):04x}')
                    i += 1
                else:
                    result.append(c2)
                    i += 1
            continue
        else:
            result.append(c)
            i += 1

    return ''.join(result)


def extract_json_blocks(message_content: str, source_agent: str) -> List[Dict[str, Any]]:
    extracted_objects = []
    if not message_content:
        return extracted_objects

    # JSON block patterns (```json ... ``` and ``` ... ```)
    json_patterns = [
        r"```json\s*\n(.*?)\n```",
        r"```\s*\n(.*?)\n```",
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, message_content, re.DOTALL | re.IGNORECASE)
        for json_str in matches:
            cleaned_json = json_str.strip()
            if not cleaned_json:
                continue
            # Sanitize all LLM JSON issues at once
            cleaned_json = _sanitize_json_for_llm(cleaned_json)
            try:
                parsed_json = json.loads(cleaned_json)
            except json.JSONDecodeError:
                continue  # Skip malformed blocks

            if validate_json_structure(parsed_json, source_agent):
                submission = {
                    "source_agent": source_agent,
                    "json_data": parsed_json,
                }
                if submission not in extracted_objects:
                    extracted_objects.append(submission)

    # Fallback: try to parse the entire message as JSON
    if not extracted_objects:
        try:
            sanitized = _sanitize_json_for_llm(message_content)
            parsed_json = json.loads(sanitized)
            if validate_json_structure(parsed_json, source_agent):
                submission = {
                    "source_agent": source_agent,
                    "json_data": parsed_json,
                }
                if submission not in extracted_objects:
                    extracted_objects.append(submission)
        except (json.JSONDecodeError, TypeError):
            pass

    return extracted_objects


def validate_json_structure(json_data: Dict[str, Any], source_agent: str) -> bool:
    """
    Validates that the JSON data has the expected structure for the given agent type.
    """
    try:
        if source_agent == "principle":
            # Expect keys starting with principle_ containing PRINCIPLE_SUBMISSION
            return any(key.startswith("principle_") for key in json_data.keys())
        elif source_agent == "hypothesis":
            # Expect keys starting with HYPOTHESIS_ containing candidate
            return any(
                key.startswith("HYPOTHESIS_")
                and isinstance(json_data[key], dict)
                and "candidate" in json_data[key]
                and isinstance(json_data[key]["candidate"], (str, dict))
                for key in json_data.keys()
            )
        elif source_agent == "experiment":
            # Expect list of results with candidate and outcome
            if isinstance(json_data, list):
                return all(
                    isinstance(item, dict)
                    and "candidate" in item
                    and "outcome" in item
                    for item in json_data
                )
            # Or single result object
            return (
                    isinstance(json_data, dict)
                    and "candidate" in json_data
                    and "outcome" in json_data
            )
        else:
            # Unknown agent type - accept any valid JSON
            return True
    except Exception as e:
        return False
