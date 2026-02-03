"""
Policy Formatter Module for SafePred_v3.

Formats policies with reference examples for inclusion in prompts.
"""

from typing import List, Dict, Any, Optional
from ..utils.logger import get_logger
from .type_checkers import validate_policies_list

logger = get_logger("SafePred.PolicyFormatter")


def format_policies_with_references(
    policies: List[Dict[str, Any]],
    include_all_references: bool = True,
    show_references: bool = False  # Default to False: do not show references by default
) -> str:
    """
    Format policies with reference examples for prompt inclusion.
    
    Args:
        policies: List of policy dictionaries
        include_all_references: If False, use summary instead of full references (only used when show_references is True)
        show_references: If False, completely exclude reference examples from output (default: True)
    
    Returns:
        Formatted policy string
    """
    if not policies:
        return ""
    
    # Validate policies type (unified type checking)
    validated_policies = validate_policies_list(policies, context="[PolicyFormatter] ")
    
    formatted_policies = []
    
    for policy in validated_policies:
        
        # Get policy ID (support different field names)
        policy_id = policy.get("policy_id") or policy.get("id") or policy.get("policyId", "N/A")
        
        # Build policy text
        policy_text = f"Policy {policy_id}:\n"
        
        # Add definitions if available
        definitions = policy.get("definitions", [])
        if definitions:
            policy_text += "Definitions:\n"
            for defn in definitions:
                policy_text += f"  - {defn}\n"
        
        # Add scope if available
        scope = policy.get("scope")
        if scope:
            policy_text += f"Scope: {scope}\n"
        
        # Add policy description
        description = policy.get("policy_description") or policy.get("description")
        if description:
            policy_text += f"Description: {description}\n"
        
        # Add risk level
        risk_level = policy.get("risk_level") or policy.get("severity", "medium")
        policy_text += f"Risk Level: {risk_level}\n"
        
        # Add reference examples only if show_references is True
        # NOTE: When show_references=False (current default), this entire block is skipped
        # and no reference examples will be included in the formatted output
        # NOTE: Reference limits are handled by reference_limits config based on risk_level
        # (high/medium/low), which is applied when storing references, not when formatting
        if show_references:
            references = policy.get("reference", [])
            if references:
                policy_text += "\nReference Examples (Historical Violations):\n"
                
                # Use all references (limits are applied when storing references based on risk_level)
                
                for i, ref in enumerate(references, 1):
                    if include_all_references:
                        policy_text += f"  Example {i}:\n{ref}\n"
                    else:
                        # Use summary (first 200 chars)
                        summary = ref[:200] + "..." if len(ref) > 200 else ref
                        policy_text += f"  Example {i}: {summary}\n"
        
        formatted_policies.append(policy_text)
    
    # Combine all policies
    result = "\n\n".join(formatted_policies)
    
    return result


def format_policy_list_simple(policies: List[Dict[str, Any]]) -> str:
    """
    Simple policy formatting without references (for backward compatibility).
    
    Args:
        policies: List of policy dictionaries
    
    Returns:
        Formatted policy string
    """
    if not policies:
        return ""
    
    formatted = []
    for policy in policies:
        policy_id = policy.get("policy_id") or policy.get("id") or policy.get("policyId", "N/A")
        description = policy.get("policy_description") or policy.get("description", "")
        risk_level = policy.get("risk_level") or policy.get("severity", "medium")
        
        formatted.append(f"Policy {policy_id} (Risk: {risk_level}): {description}")
    
    return "\n".join(formatted)

