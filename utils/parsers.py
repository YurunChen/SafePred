"""
Common parsing utilities for SafePred.

Provides reusable parsing functions for JSON, actions, and risk scores.
"""

import json
import re
from typing import Any, Dict, List, Optional


def parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse JSON object from text, handling markdown code blocks, truncated JSON, and format errors.
    
    Args:
        text: Text containing JSON
    
    Returns:
        Parsed JSON dict or None if not found
    """
    # Try to find JSON in markdown code blocks
    code_block_patterns = [
        r'```json\s*\n([\s\S]*?)\n```',
        r'```json\s*([\s\S]*?)```',
        r'```\s*\n([\s\S]*?)\n```',
        r'```\s*([\s\S]*?)```',
    ]
    
    for pattern in code_block_patterns:
        json_match = re.search(pattern, text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                # Try to fix common JSON errors (missing commas, truncated JSON, etc.)
                fixed_json = _try_fix_json_errors(json_str)
                if fixed_json:
                    try:
                        return json.loads(fixed_json)
                    except json.JSONDecodeError:
                        continue
    
    # Try to find JSON object directly (balanced braces)
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(text):
        if char == '{':
            if start_idx == -1:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                json_str = text[start_idx:i+1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Try to fix common JSON errors
                    fixed_json = _try_fix_json_errors(json_str)
                    if fixed_json:
                        try:
                            return json.loads(fixed_json)
                        except json.JSONDecodeError:
                            pass
                    start_idx = -1
                    continue
    
    # If no balanced JSON found, try to extract and fix JSON errors
    # Find the last '{' and try to fix from there
    last_brace_idx = text.rfind('{')
    if last_brace_idx != -1:
        json_str = text[last_brace_idx:]
        fixed_json = _try_fix_json_errors(json_str)
        if fixed_json:
            try:
                return json.loads(fixed_json)
            except json.JSONDecodeError:
                pass
    
    # Try to parse entire text as JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        # Last attempt: try to fix JSON errors
        fixed_json = _try_fix_json_errors(text.strip())
        if fixed_json:
            try:
                return json.loads(fixed_json)
            except json.JSONDecodeError:
                pass
        return None


def _try_fix_json_errors(json_str: str) -> Optional[str]:
    """
    Try to fix common JSON errors including missing commas, truncated JSON, comments, etc.
    
    Args:
        json_str: Potentially malformed JSON string
    
    Returns:
        Fixed JSON string or None if fixing is not possible
    """
    if not json_str or not json_str.strip().startswith('{'):
        return None
    
    fixed = json_str.rstrip()
    
    # Fix 0: Remove single-line comments (// ...)
    # This handles LLM outputs that include comments in JSON
    # Remove comments that are on their own line or at the end of a line
    # Pattern: // ... (until end of line)
    fixed = re.sub(r'//.*?$', '', fixed, flags=re.MULTILINE)
    
    # Fix 1: Add missing commas between key-value pairs
    # This is the most common error: missing comma before a new key
    
    # Pattern 1: String value ending with quote, followed by newline and new key (missing comma)
    # Example: "key": "value"\n  "next_key" -> "key": "value",\n  "next_key"
    # Note: [^"]* doesn't match newlines, so we need to handle multi-line strings
    # Use [\s\S]*? to match any character including newlines, but non-greedy
    fixed = re.sub(r'("\s*:\s*"[\s\S]*?")\s*\n\s*"', r'\1,\n  "', fixed)
    
    # Pattern 2: String value ending with quote, followed by whitespace and new key (missing comma)
    # Example: "key": "value"  "next_key" -> "key": "value", "next_key"
    # This handles the case from the error: "risk_explanation": "..."  "url_change"
    # Use [\s\S]*? to handle multi-line strings
    fixed = re.sub(r'("\s*:\s*"[\s\S]*?")\s+(?=")', r'\1, ', fixed)
    
    # Pattern 3: Non-string values (null, true, false, numbers) followed by new key
    # Example: "key": null\n  "next_key" -> "key": null,\n  "next_key"
    fixed = re.sub(r'("\s*:\s*(null|true|false|\d+\.?\d*))\s*\n\s*"', r'\1,\n  "', fixed)
    fixed = re.sub(r'("\s*:\s*(null|true|false|\d+\.?\d*))\s+(?=")', r'\1, ', fixed)
    
    # Pattern 4: Array values followed by new key
    # Example: "key": [...]\n  "next_key" -> "key": [...],\n  "next_key"
    # Note: This is simplified - doesn't handle nested arrays perfectly
    fixed = re.sub(r'("\s*:\s*\[[^\]]*\])\s*\n\s*"', r'\1,\n  "', fixed)
    fixed = re.sub(r'("\s*:\s*\[[^\]]*\])\s+(?=")', r'\1, ', fixed)
    
    # Pattern 5: Object values followed by new key
    # Example: "key": {...}\n  "next_key" -> "key": {...},\n  "next_key"
    # Note: This is simplified - doesn't handle nested objects perfectly
    fixed = re.sub(r'("\s*:\s*\{[^\}]*\})\s*\n\s*"', r'\1,\n  "', fixed)
    fixed = re.sub(r'("\s*:\s*\{[^\}]*\})\s+(?=")', r'\1, ', fixed)
    
    # Fix 2: Handle truncated JSON (missing closing braces)
    return _try_fix_truncated_json(fixed)


def _try_fix_truncated_json(json_str: str) -> Optional[str]:
    """
    Try to fix truncated JSON by closing unclosed structures.
    
    Args:
        json_str: Potentially truncated JSON string
    
    Returns:
        Fixed JSON string or None if fixing is not possible
    """
    if not json_str or not json_str.strip().startswith('{'):
        return None
    
    # Count braces to see if JSON is incomplete
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    
    # If JSON appears complete, return as-is
    if open_braces == close_braces:
        return None
    
    # Find the last incomplete structure
    # Remove trailing incomplete string/array/object
    fixed = json_str.rstrip()
    
    # Remove incomplete string values (unclosed quotes)
    # Find the last complete key-value pair
    # This is a simple heuristic - look for the last complete "key": value pattern
    last_complete_colon = fixed.rfind(':')
    if last_complete_colon != -1:
        # Check if the value after colon is complete
        value_part = fixed[last_complete_colon + 1:].strip()
        
        # If value starts with quote but doesn't end with quote, it's truncated
        if value_part.startswith('"') and not value_part.endswith('"'):
            # Try to close the string
            # Find where the string might have been cut
            # Remove everything after the last complete structure
            # Simple approach: find last complete key-value pair and close JSON
            pass
    
    # Simple fix: close all unclosed braces and brackets
    brace_diff = open_braces - close_braces
    if brace_diff > 0:
        # Remove trailing incomplete content (incomplete string, array, etc.)
        # Find the last complete key-value pair
        # Look for pattern: "key": value, or "key": value}
        # Remove everything after the last complete structure
        
        # Find last complete key-value pair ending with } or ,
        # Pattern: "key": value followed by } or ,
        pattern = r'"\s*:\s*[^,}]+[,}]'
        matches = list(re.finditer(pattern, fixed))
        if matches:
            # Get position after last complete match
            last_match = matches[-1]
            cut_pos = last_match.end()
            # Keep everything up to the last complete structure
            fixed = fixed[:cut_pos]
            # Close remaining braces
            fixed += '}' * brace_diff
            return fixed
    
    return None


def parse_json_array_from_text(text: str) -> Optional[List[Any]]:
    """
    Parse JSON array from text.
    
    Args:
        text: Text containing JSON array
    
    Returns:
        Parsed JSON array or None if not found
    """
    # Try to extract JSON array
    json_match = re.search(r'\[[\s\S]*?\]', text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    
    # Try parsing from code blocks
    json_obj = parse_json_from_text(text)
    if isinstance(json_obj, list):
        return json_obj
    
    return None


def extract_number_from_text(text: str, min_val: float = 0.0, max_val: float = 1.0) -> Optional[float]:
    """
    Extract a number from text, normalizing to [min_val, max_val] range.
    
    Args:
        text: Text containing a number
        min_val: Minimum value for normalization
        max_val: Maximum value for normalization
    
    Returns:
        Extracted and normalized number or None
    """
    patterns = [
        r'\b(0\.\d+|1\.0|1|0)\b',  # Match 0.0-1.0
        r'\b(\d+\.\d+)\b',  # Match any decimal
        r'\b(\d+)\b',  # Match integer
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            try:
                score = float(matches[0])
                # Normalize to [min_val, max_val] range
                if score > max_val:
                    score = score / 100.0  # Assume percentage
                return max(min_val, min(max_val, score))
            except ValueError:
                continue
    
    return None


def normalize_action(action: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize action dictionary to standard format.
    
    Args:
        action: Action dictionary
    
    Returns:
        Normalized action dictionary
    """
    return {
        "type": action.get("type", "click"),
        "target": action.get("target", action.get("element", "")),
        "value": action.get("value", ""),
    }

