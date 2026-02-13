import json
import re
from typing import Dict, Any, List


def extract_json_blocks(message_content: str, source_agent: str) -> List[Dict[str, Any]]:
    extracted_objects = []
    if not message_content:
        return extracted_objects

    # Enhanced JSON pattern to capture both ```json and ``` blocks
    json_patterns = [
        r"```json\s*\n(.*?)\n```",  # Standard ```json blocks
    ]

    for pattern in json_patterns:
        matches = re.findall(
            pattern, message_content, re.DOTALL | re.IGNORECASE
        )
        for json_str in matches:

            # Clean and validate JSON string
            cleaned_json = json_str.strip()
            parsed_json = json.loads(cleaned_json)

            # Validate structure based on source agent
            if validate_json_structure(parsed_json, source_agent):
                submission = {
                    "source_agent": source_agent,
                    "json_data": parsed_json,
                }
                if submission not in extracted_objects:
                    extracted_objects.append(submission)


    # Fallback for raw JSON if no markdown block found
    if not extracted_objects:
        try:
            parsed_json = json.loads(message_content)
            if validate_json_structure(parsed_json, source_agent):
                submission = {
                    "source_agent": source_agent,
                    "json_data": parsed_json,
                }
                if submission not in extracted_objects:
                    extracted_objects.append(submission)
        except (json.JSONDecodeError, TypeError):
            pass  # Not a valid JSON string, ignore

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
