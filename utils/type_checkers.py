"""
Type checking utilities for SafePred.

Provides unified type validation functions to avoid code duplication.
"""

from typing import Any, List, Dict, Optional
from ..utils.logger import get_logger

logger = get_logger("SafePred.TypeCheckers")


def validate_policies_list(policies: Any, context: str = "") -> List[Dict[str, Any]]:
    """
    Validate that policies is a list of dictionaries.
    
    Args:
        policies: Value to validate
        context: Optional context string for error messages
    
    Returns:
        Validated list of policy dictionaries
    
    Raises:
        TypeError: If policies is not a list or contains non-dict elements
    """
    if policies is None:
        return []
    
    if not isinstance(policies, list):
        error_msg = f"{context}policies must be a list, got type={type(policies)}, value={str(policies)[:200]}"
        logger.error(error_msg)
        raise TypeError(error_msg)
    
    # Validate all elements are dicts
    validated_policies = []
    for i, policy in enumerate(policies):
        if not isinstance(policy, dict):
            error_msg = f"{context}policies[{i}] is not a dict, got type={type(policy)}, value={str(policy)[:200]}"
            logger.error(error_msg)
            raise TypeError(error_msg)
        validated_policies.append(policy)
    
    return validated_policies


def validate_dict(value: Any, name: str, required_keys: Optional[List[str]] = None, context: str = "") -> Dict[str, Any]:
    """
    Validate that a value is a dictionary, optionally checking for required keys.
    
    Args:
        value: Value to validate
        name: Name of the variable for error messages
        required_keys: Optional list of required keys
        context: Optional context string for error messages
    
    Returns:
        Validated dictionary
    
    Raises:
        TypeError: If value is not a dict
        ValueError: If required keys are missing
    """
    if not isinstance(value, dict):
        error_msg = f"{context}{name} must be a dict, got type={type(value)}, value={str(value)[:200]}"
        logger.error(error_msg)
        raise TypeError(error_msg)
    
    if required_keys:
        missing_keys = [key for key in required_keys if key not in value]
        if missing_keys:
            error_msg = f"{context}{name} missing required keys: {missing_keys}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    return value


def validate_list(value: Any, name: str, element_type: Optional[type] = None, context: str = "") -> List[Any]:
    """
    Validate that a value is a list, optionally checking element types.
    
    Args:
        value: Value to validate
        name: Name of the variable for error messages
        element_type: Optional type to validate each element against
        context: Optional context string for error messages
    
    Returns:
        Validated list
    
    Raises:
        TypeError: If value is not a list or elements don't match element_type
    """
    if value is None:
        return []
    
    if not isinstance(value, list):
        error_msg = f"{context}{name} must be a list, got type={type(value)}, value={str(value)[:200]}"
        logger.error(error_msg)
        raise TypeError(error_msg)
    
    if element_type:
        for i, element in enumerate(value):
            if not isinstance(element, element_type):
                error_msg = f"{context}{name}[{i}] is not {element_type.__name__}, got type={type(element)}, value={str(element)[:200]}"
                logger.error(error_msg)
                raise TypeError(error_msg)
    
    return value



