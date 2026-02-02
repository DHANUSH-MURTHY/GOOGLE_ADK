"""
Smart JSON Parser for Agent Evaluation
Handles multiple JSON formats for both agent outputs and golden datasets.
"""

from typing import Any, Dict, List, Optional, Union


def extract_field(obj: Dict, *paths: str, default: Any = None) -> Any:
    """
    Try multiple paths to extract a field value from an object.

    Args:
        obj: Dictionary to search
        *paths: Multiple possible field names to try
        default: Default value if none found

    Returns:
        First found value or default
    """
    for path in paths:
        if '.' in path:
            # Handle nested paths like "conversation.0.user_content"
            value = obj
            for part in path.split('.'):
                if isinstance(value, dict):
                    value = value.get(part)
                elif isinstance(value, list) and part.isdigit():
                    idx = int(part)
                    value = value[idx] if idx < len(value) else None
                else:
                    value = None
                    break
            if value is not None:
                return value
        else:
            # Simple field lookup
            if path in obj:
                return obj[path]
    return default


def extract_text_from_parts(parts_obj: Any) -> str:
    """
    Extract text from ADK parts structure.

    Handles:
    - {"parts": [{"text": "..."}]}
    - [{"text": "..."}]
    - "plain string"
    """
    if isinstance(parts_obj, str):
        return parts_obj

    if isinstance(parts_obj, dict):
        if 'parts' in parts_obj:
            parts = parts_obj['parts']
            if isinstance(parts, list) and len(parts) > 0:
                if isinstance(parts[0], dict) and 'text' in parts[0]:
                    return parts[0]['text']
        if 'text' in parts_obj:
            return parts_obj['text']

    if isinstance(parts_obj, list) and len(parts_obj) > 0:
        if isinstance(parts_obj[0], dict) and 'text' in parts_obj[0]:
            return parts_obj[0]['text']

    return str(parts_obj) if parts_obj else ""


def normalize_agent_record(record: Dict) -> Dict:
    """
    Convert any agent output format to standard schema.

    Standard schema:
    {
        "id": str/int,
        "input": str,
        "output": str,
        "expected": str,
        "tools": list,
        "steps": list,
        "latency": float
    }
    """
    normalized = {
        "id": extract_field(record, "id", "test_id", "eval_id", "case_id", default="unknown"),
        "input": extract_field(record, "input", "prompt", "query", "question", "user_input", "user_query", default=""),
        "output": "",
        "expected": extract_field(record, "expected", "expected_output", "ground_truth", "expected_response", "answer", default=""),
        "tools": extract_field(record, "tools", "tool_calls", "tool_uses", "functions", default=[]),
        "steps": extract_field(record, "steps", "reasoning", "trajectory", default=[]),
        "latency": extract_field(record, "latency", "response_time", "duration", default=0.0)
    }

    # Try to extract from ADK conversation format first (for eval.test.json used as agent output)
    if "conversation" in record and isinstance(record["conversation"], list) and len(record["conversation"]) > 0:
        conv = record["conversation"][0]

        # Extract input from user_content
        if "user_content" in conv:
            normalized["input"] = extract_text_from_parts(conv["user_content"])

        # Extract output from final_response
        if "final_response" in conv:
            final_response = conv["final_response"]
            normalized["output"] = extract_text_from_parts(final_response)

        # Extract tools from intermediate_data
        if "intermediate_data" in conv:
            intermediate = conv["intermediate_data"]
            if isinstance(intermediate, dict):
                tools = extract_field(intermediate, "tool_uses", "tools", "tool_calls", default=[])
                if tools:
                    normalized["tools"] = tools

    # Fallback: Extract output/response text from simple formats
    if not normalized["output"]:
        output = extract_field(
            record,
            "output", "response", "answer", "result", "text", "content"
        )

        if output:
            normalized["output"] = extract_text_from_parts(output)

    # Ensure tools is a list
    if not isinstance(normalized["tools"], list):
        normalized["tools"] = []

    # Ensure steps is a list
    if not isinstance(normalized["steps"], list):
        normalized["steps"] = []

    return normalized


def normalize_test_case(case: Dict) -> Dict:
    """
    Convert any golden dataset format to standard schema.

    Standard schema:
    {
        "id": str/int,
        "prompt": str,
        "expected": str,
        "expected_tools": list
    }
    """
    normalized = {
        "id": extract_field(case, "eval_id", "id", "test_id", "case_id", default="unknown"),
        "prompt": "",
        "expected": "",
        "expected_tools": []
    }

    # Try to extract from ADK conversation format first
    if "conversation" in case and isinstance(case["conversation"], list) and len(case["conversation"]) > 0:
        conv = case["conversation"][0]

        # Extract prompt from user_content
        if "user_content" in conv:
            user_content = conv["user_content"]
            normalized["prompt"] = extract_text_from_parts(user_content)

        # Extract expected from final_response
        if "final_response" in conv:
            final_response = conv["final_response"]
            normalized["expected"] = extract_text_from_parts(final_response)

        # Extract expected tools
        if "intermediate_data" in conv:
            intermediate = conv["intermediate_data"]
            if isinstance(intermediate, dict):
                normalized["expected_tools"] = extract_field(
                    intermediate, "tool_uses", "tools", "tool_calls", default=[]
                )

    # Fallback to simple format
    if not normalized["prompt"]:
        prompt = extract_field(
            case,
            "prompt", "input", "query", "question", "user_input", "user_query", "text"
        )
        if prompt:
            normalized["prompt"] = extract_text_from_parts(prompt)

    if not normalized["expected"]:
        expected = extract_field(
            case,
            "expected", "output", "response", "expected_output", "expected_response", "answer", "ground_truth"
        )
        if expected:
            normalized["expected"] = extract_text_from_parts(expected)

    if not normalized["expected_tools"]:
        normalized["expected_tools"] = extract_field(
            case,
            "expected_tools", "tools", "tool_calls", "tool_uses", default=[]
        )

    # Ensure expected_tools is a list
    if not isinstance(normalized["expected_tools"], list):
        normalized["expected_tools"] = []

    return normalized


def detect_format(data: Any) -> str:
    """
    Detect the format of the JSON data.

    Returns:
        Format description string
    """
    if isinstance(data, dict):
        if "eval_cases" in data:
            return "ADK Evaluation Format (eval_cases)"
        elif "conversation" in data:
            return "ADK Single Case Format"
        elif "output" in data or "response" in data:
            return "Simple Single Record"
        else:
            return "Custom Object Format"
    elif isinstance(data, list):
        if len(data) > 0:
            first = data[0]
            if isinstance(first, dict):
                if "conversation" in first:
                    return "ADK Array Format"
                elif "output" in first or "response" in first:
                    return "Simple Array Format"
                else:
                    return "Custom Array Format"
        return "Empty Array"
    else:
        return "Unknown Format"


def parse_agent_output(data: Any) -> List[Dict]:
    """
    Parse agent output in any reasonable format.

    Args:
        data: Raw JSON data (dict, list, or other)

    Returns:
        List of normalized agent records
    """
    records = []

    # Handle different input formats
    if isinstance(data, dict):
        # Check for ADK eval_cases wrapper
        if "eval_cases" in data:
            items = data["eval_cases"]
        else:
            # Single record
            items = [data]
    elif isinstance(data, list):
        items = data
    else:
        # Unsupported format
        return []

    # Normalize each record
    for item in items:
        if isinstance(item, dict):
            # Check if this eval_case has multiple conversations
            if "conversation" in item and isinstance(item["conversation"], list) and len(item["conversation"]) > 1:
                # Create a separate record for each conversation
                base_id = extract_field(item, "id", "test_id", "eval_id", "case_id", default="unknown")
                for idx, conv in enumerate(item["conversation"]):
                    # Create a temporary record with just this conversation
                    temp_record = item.copy()
                    temp_record["conversation"] = [conv]
                    # Give it a unique ID
                    temp_record["eval_id"] = f"{base_id}_conv{idx+1}"
                    temp_record["invocation_id"] = conv.get("invocation_id", f"{base_id}_conv{idx+1}")

                    normalized = normalize_agent_record(temp_record)
                    records.append(normalized)
            else:
                # Single conversation or no conversation array
                normalized = normalize_agent_record(item)
                records.append(normalized)

    return records


def parse_golden_dataset(data: Any) -> List[Dict]:
    """
    Parse golden dataset in any reasonable format.

    Args:
        data: Raw JSON data (dict, list, or other)

    Returns:
        List of normalized test cases
    """
    cases = []

    # Handle different input formats
    if isinstance(data, dict):
        # Check for ADK eval_cases wrapper
        if "eval_cases" in data:
            items = data["eval_cases"]
        else:
            # Single test case
            items = [data]
    elif isinstance(data, list):
        items = data
    else:
        # Unsupported format
        return []

    # Normalize each case
    for item in items:
        if isinstance(item, dict):
            # Check if this eval_case has multiple conversations
            if "conversation" in item and isinstance(item["conversation"], list) and len(item["conversation"]) > 1:
                # Create a separate test case for each conversation
                base_id = extract_field(item, "id", "test_id", "eval_id", "case_id", default="unknown")
                for idx, conv in enumerate(item["conversation"]):
                    # Create a temporary record with just this conversation
                    temp_case = item.copy()
                    temp_case["conversation"] = [conv]
                    # Give it a unique ID
                    temp_case["eval_id"] = f"{base_id}_conv{idx+1}"
                    temp_case["invocation_id"] = conv.get("invocation_id", f"{base_id}_conv{idx+1}")

                    normalized = normalize_test_case(temp_case)
                    cases.append(normalized)
            else:
                # Single conversation or no conversation array
                normalized = normalize_test_case(item)
                cases.append(normalized)

    return cases


def match_records(agent_records: List[Dict], test_cases: List[Dict]) -> List[tuple]:
    """
    Match agent records with test cases by ID or position.

    Args:
        agent_records: List of normalized agent records
        test_cases: List of normalized test cases

    Returns:
        List of (agent_record, test_case) tuples
    """
    matches = []

    # Create ID lookup for agent records
    agent_by_id = {str(rec["id"]): rec for rec in agent_records}

    # Try to match by ID first, fall back to position
    for i, test_case in enumerate(test_cases):
        test_id = str(test_case["id"])

        # Try ID match
        if test_id in agent_by_id:
            matches.append((agent_by_id[test_id], test_case))
        # Fall back to position match
        elif i < len(agent_records):
            matches.append((agent_records[i], test_case))

    return matches
