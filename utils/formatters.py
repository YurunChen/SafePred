"""
Formatting utilities for SafePred.

Provides common formatting functions for states, actions, and other objects.
"""

from typing import Any


def format_object_to_string(obj: Any) -> str:
    """
    Convert object to string representation.
    
    Args:
        obj: Object to format (str, dict, list, etc.)
    
    Returns:
        String representation
    """
    if isinstance(obj, str):
        return obj
    elif isinstance(obj, dict):
        return "\n".join(f"{k}: {v}" for k, v in obj.items())
    elif isinstance(obj, (list, tuple)):
        return "\n".join(str(item) for item in obj)
    else:
        return str(obj)

