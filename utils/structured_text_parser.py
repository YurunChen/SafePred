"""
Structured Text Parser for World Model.

Provides a format-insensitive parser for structured text output (similar to YAML but more lenient).
This format is easier for LLMs to generate correctly and more tolerant of formatting errors.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
from ..utils.logger import get_logger

logger = get_logger("SafePred.StructuredTextParser")


def parse_structured_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse structured text output from LLM and convert to JSON format.
    
    Expected format (flexible, format-insensitive):
    semantic_delta: description text
    risk_affordances:
      new_elements:
        - element1
        - element2
      removed_elements: []
    risk_exposure:
      exposed_risks:
        - risk1
      risk_level: low
    reversibility:
      is_reversible: true
      reversibility_description: description
      return_difficulty: easy
    violated_policy_ids:
      - Policy 1
    risk_explanation: explanation
    url_change: null
    page_type_change: null
    
    Args:
        text: Raw text from LLM containing structured text
    
    Returns:
        Parsed JSON dict or None if parsing fails
    """
    # Extract structured text from markdown code blocks or raw text
    structured_text = _extract_structured_text(text)
    if not structured_text:
        return None
    
    try:
        # Parse structured text
        result = _parse_to_dict(structured_text)
        
        # Validate required fields
        if _validate_world_model_output(result):
            return result
        else:
            logger.warning("Structured text parsed but missing required fields")
            return None
            
    except Exception as e:
        logger.error(f"Structured text parsing failed: {e}")
        return None


def _extract_structured_text(text: str) -> Optional[str]:
    """
    Extract structured text from raw LLM output.
    
    Args:
        text: Raw text from LLM
    
    Returns:
        Extracted structured text or None
    """
    # Try to find structured text in markdown code blocks
    code_block_patterns = [
        r'```(?:yaml|text|plain)?\s*\n([\s\S]*?)\n```',
        r'```(?:yaml|text|plain)?\s*([\s\S]*?)```',
        r'```\s*\n([\s\S]*?)\n```',
        r'```\s*([\s\S]*?)```',
    ]
    
    for pattern in code_block_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            # Check if it looks like structured text (has key-value pairs)
            if re.search(r'^\w+[\s:]+', content, re.MULTILINE):
                return content
    
    # Try to find structured text directly (look for key-value patterns)
    # Find a section that starts with a key (word followed by colon or space)
    lines = text.split('\n')
    start_idx = None
    for i, line in enumerate(lines):
        # Look for lines that look like keys (word followed by colon or space)
        if re.match(r'^\s*\w+[\s:]+', line):
            start_idx = i
            break
    
    if start_idx is not None:
        # Extract from start_idx to end or until we hit a clear separator
        extracted_lines = []
        for i in range(start_idx, len(lines)):
            line = lines[i]
            # Stop if we hit a clear separator (empty line with next line not indented)
            if i > start_idx and not line.strip():
                if i + 1 < len(lines) and not re.match(r'^\s+', lines[i + 1]):
                    break
            extracted_lines.append(line)
        
        if extracted_lines:
            return '\n'.join(extracted_lines).strip()
    
    return None


def _parse_to_dict(text: str) -> Dict[str, Any]:
    """
    Parse structured text to dictionary.
    
    Supports flexible formats:
    - key: value
    - key value
    - key:
        nested_key: value
    - list items: - item, * item, or just item
    
    Args:
        text: Structured text string
    
    Returns:
        Dictionary representation
    """
    lines = text.split('\n')
    result = {}
    stack = [(0, result, None)]  # (indent_level, dict, current_list_key)
    
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        
        # Skip empty lines
        if not line.strip():
            i += 1
            continue
        
        # Calculate indentation
        indent = len(line) - len(line.lstrip())
        line_content = line.strip()
        
        # Check if it's a list item
        list_match = re.match(r'^[-*]\s+(.+)$', line_content)
        if list_match:
            # This is a list item
            value = list_match.group(1).strip()
            parsed_value = _parse_value(value)
            
            # Pop stack until we find the right parent dict
            while len(stack) > 1 and stack[-1][0] >= indent:
                stack.pop()
            
            parent_dict = stack[-1][1]
            current_list_key = stack[-1][2]
            
            # If we have a current list key, add to that list
            if current_list_key and current_list_key in parent_dict:
                if not isinstance(parent_dict[current_list_key], list):
                    parent_dict[current_list_key] = [parent_dict[current_list_key]]
                parent_dict[current_list_key].append(parsed_value)
            else:
                # Try to find the parent key from the previous line
                # Look for a key in the parent dict that was just added (previous line)
                if i > 0:
                    # Find the most recent key that might be the parent
                    # Check previous non-empty lines
                    for j in range(i - 1, -1, -1):
                        prev_line = lines[j].rstrip()
                        if not prev_line.strip():
                            continue
                        prev_indent = len(prev_line) - len(prev_line.lstrip())
                        prev_content = prev_line.strip()
                        
                        # Check if previous line is a key (with or without colon)
                        prev_key_match = re.match(r'^(\w+)\s*:?\s*$', prev_content)
                        if prev_key_match and prev_indent < indent:
                            list_key = prev_key_match.group(1)
                            # Initialize as list if not exists
                            if list_key not in parent_dict:
                                parent_dict[list_key] = []
                            elif not isinstance(parent_dict[list_key], list):
                                parent_dict[list_key] = [parent_dict[list_key]]
                            parent_dict[list_key].append(parsed_value)
                            # Update stack to track this list
                            stack[-1] = (stack[-1][0], stack[-1][1], list_key)
                            break
                    else:
                        # No parent found, skip this item (might be malformed)
                        logger.warning(f"List item found without parent key: {line_content}")
                else:
                    # First line is a list item - skip (malformed)
                    logger.warning(f"List item found at start of text: {line_content}")
            
            i += 1
            continue
        
        # Parse key-value pair
        key, value, _ = _parse_line(line_content)
        
        if key is None:
            i += 1
            continue
        
        # Pop stack until we find the right parent dict
        while len(stack) > 1 and stack[-1][0] >= indent:
            stack.pop()
        
        parent_dict = stack[-1][1]
        current_list_key = None
        
        # Handle key-value pair
        if value is None:
            # This might be a nested structure or a list, check next line
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                next_indent = len(next_line) - len(next_line.lstrip())
                next_content = next_line.strip()
                
                if next_indent > indent:
                    # Check if next line is a list item
                    if re.match(r'^[-*]\s+', next_content):
                        # Next line is a list item, so this key is a list
                        parent_dict[key] = []
                        current_list_key = key
                    else:
                        # Nested structure
                        parent_dict[key] = {}
                        stack.append((indent, parent_dict[key], None))
                else:
                    # Empty value
                    parent_dict[key] = None
            else:
                parent_dict[key] = None
        else:
            # Simple key-value
            parent_dict[key] = _parse_value(value)
        
        # Update stack with current list key
        if current_list_key:
            stack.append((indent, parent_dict, current_list_key))
        
        i += 1
    
    # Clean up any default list keys
    _cleanup_default_lists(result)
    
    return result


def _cleanup_default_lists(obj: Any) -> None:
    """Recursively clean up default list keys."""
    if isinstance(obj, dict):
        if '__list__' in obj and len(obj) == 1:
            # Only has default list, might be an error
            pass
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                _cleanup_default_lists(value)


def _parse_line(line: str) -> Tuple[Optional[str], Optional[str], bool]:
    """
    Parse a single line into key, value, and whether it's a list item.
    
    Note: List items are handled separately in _parse_to_dict.
    
    Returns:
        (key, value, is_list_item) - is_list_item is always False here
    """
    line = line.strip()
    
    # Try key: value format
    colon_match = re.match(r'^(\w+)\s*:\s*(.*)$', line)
    if colon_match:
        key = colon_match.group(1)
        value = colon_match.group(2).strip()
        return key, value if value else None, False
    
    # Try key value format (no colon)
    space_match = re.match(r'^(\w+)\s+(.+)$', line)
    if space_match:
        key = space_match.group(1)
        value = space_match.group(2).strip()
        return key, value, False
    
    # Single word (might be a key with no value)
    if re.match(r'^\w+$', line):
        return line, None, False
    
    return None, None, False


def _parse_value(value_str: str) -> Any:
    """
    Parse a value string to appropriate Python type.
    
    Args:
        value_str: Value string
    
    Returns:
        Parsed value (bool, int, float, None, list, or str)
    """
    if not value_str:
        return None
    
    value_str = value_str.strip()
    
    # Try boolean
    if value_str.lower() in ('true', 'yes', '1'):
        return True
    if value_str.lower() in ('false', 'no', '0'):
        return False
    
    # Try null
    if value_str.lower() in ('null', 'none', 'nil', ''):
        return None
    
    # Try number
    if value_str.isdigit():
        return int(value_str)
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # Try JSON array
    if value_str.startswith('[') and value_str.endswith(']'):
        try:
            import json
            return json.loads(value_str)
        except:
            pass
    
    # Remove quotes if present
    if (value_str.startswith('"') and value_str.endswith('"')) or \
       (value_str.startswith("'") and value_str.endswith("'")):
        value_str = value_str[1:-1]
    
    return value_str


def _validate_world_model_output(result: Dict[str, Any]) -> bool:
    """
    Validate that the parsed output has required fields.
    
    Args:
        result: Parsed dictionary
    
    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        'semantic_delta',
        'risk_affordances',
        'risk_exposure',
        'reversibility',
        'violated_policy_ids',
        'risk_explanation',
    ]
    
    for field in required_fields:
        if field not in result:
            logger.warning(f"Missing required field: {field}")
            return False
    
    # Validate nested structures
    if not isinstance(result.get('risk_affordances'), dict):
        return False
    if not isinstance(result.get('risk_exposure'), dict):
        return False
    if not isinstance(result.get('reversibility'), dict):
        return False
    if not isinstance(result.get('violated_policy_ids'), list):
        return False
    
    return True


def convert_structured_text_to_world_model_format(parsed_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert parsed structured text dictionary to World Model's expected JSON format.
    
    Args:
        parsed_dict: Dictionary parsed from structured text
    
    Returns:
        Dictionary in World Model format
    """
    # Handle nested structures
    risk_affordances = parsed_dict.get('risk_affordances', {})
    if isinstance(risk_affordances, dict):
        # Ensure list fields are lists
        for key in ['new_elements', 'removed_elements', 'risk_relevant']:
            if key in risk_affordances:
                value = risk_affordances[key]
                if isinstance(value, str):
                    # Try to parse as JSON array or split by comma
                    import json
                    try:
                        risk_affordances[key] = json.loads(value)
                    except:
                        risk_affordances[key] = [v.strip() for v in re.split(r'[,\n]', value) if v.strip()]
                elif not isinstance(value, list):
                    risk_affordances[key] = [value] if value else []
            else:
                risk_affordances[key] = []
    
    risk_exposure = parsed_dict.get('risk_exposure', {})
    if isinstance(risk_exposure, dict):
        exposed_risks = risk_exposure.get('exposed_risks', [])
        if isinstance(exposed_risks, str):
            import json
            try:
                risk_exposure['exposed_risks'] = json.loads(exposed_risks)
            except:
                risk_exposure['exposed_risks'] = [v.strip() for v in re.split(r'[,\n]', exposed_risks) if v.strip()]
        elif not isinstance(exposed_risks, list):
            risk_exposure['exposed_risks'] = [exposed_risks] if exposed_risks else []
    
    reversibility = parsed_dict.get('reversibility', {})
    if isinstance(reversibility, dict):
        is_reversible = reversibility.get('is_reversible')
        if isinstance(is_reversible, str):
            reversibility['is_reversible'] = is_reversible.lower() in ('true', '1', 'yes')
    
    violated_policy_ids = parsed_dict.get('violated_policy_ids', [])
    if isinstance(violated_policy_ids, str):
        import json
        try:
            violated_policy_ids = json.loads(violated_policy_ids)
        except:
            violated_policy_ids = [v.strip() for v in re.split(r'[,\n]', violated_policy_ids) if v.strip()]
    elif not isinstance(violated_policy_ids, list):
        violated_policy_ids = [violated_policy_ids] if violated_policy_ids else []
    
    # Convert risk_affordances to element_changes (new format)
    element_changes = {
        'new_elements': risk_affordances.get('new_elements', []),
        'removed_elements': risk_affordances.get('removed_elements', []),
    }
    
    return {
        'semantic_delta': parsed_dict.get('semantic_delta', ''),
        'element_changes': element_changes,  # Convert from risk_affordances
        'long_term_impact': parsed_dict.get('long_term_impact', 'Long-term impact assessment not available in structured text format'),
        'violated_policy_ids': violated_policy_ids,
        'risk_explanation': parsed_dict.get('risk_explanation', ''),
        'optimization_guidance': parsed_dict.get('optimization_guidance'),  # May not be in structured text format
    }

