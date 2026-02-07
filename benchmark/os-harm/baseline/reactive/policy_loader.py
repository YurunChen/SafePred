"""
Policy loading and formatting utilities.
"""

import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def read_security_policy_categories(file_path: str) -> List[Dict[str, Any]]:
    """
    Read security policy categories from JSON file.
    
    Args:
        file_path: Path to JSON file containing policies
        
    Returns:
        List of policy dictionaries
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Policy file not found: '{file_path}'")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in policy file '{file_path}': {e}")


def format_policy_list(policy_list: List[Dict[str, Any]]) -> str:
    """
    Format policy list into a readable string for prompts.
    
    Args:
        policy_list: List of policy dictionaries
        
    Returns:
        Formatted policy text (without reference field)
    """
    formatted_text = "SECURITY POLICIES SUMMARY\n" + "="*30 + "\n\n"
    
    for policy in policy_list:
        policy_id = policy.get("policy_id", "Unknown")
        formatted_text += f"Policy ID: {policy_id}\n"
        
        if definitions := policy.get("definitions", []):
            formatted_text += "Definitions:\n"
            formatted_text += ''.join(f"  • {d}\n" for d in definitions) + "\n"
        
        if scope := policy.get("scope"):
            formatted_text += f"Scope:\n  {scope}\n\n"
        
        if description := policy.get("policy_description"):
            formatted_text += f"Description:\n  {description}\n\n"
        
        if risk_level := policy.get("risk_level"):
            formatted_text += f"Risk Level:\n  {risk_level}\n\n"
        
        # Note: reference field is intentionally excluded from formatted output
        formatted_text += "="*50 + "\n\n"
    
    return formatted_text


class PolicyLoader:
    """Load and manage security policies."""
    
    def __init__(self, policy_path: str):
        """
        Initialize policy loader.
        
        Args:
            policy_path: Path to policy JSON file
        """
        self.policy_path = policy_path
        self.policy_list = None
        self.formatted_policy_content = None
        self._load_policies()
    
    def _load_policies(self):
        """Load policies from file."""
        try:
            self.policy_list = read_security_policy_categories(self.policy_path)
            self.formatted_policy_content = format_policy_list(self.policy_list)
            logger.info(f"Loaded {len(self.policy_list)} policies from {self.policy_path}")
        except Exception as e:
            logger.error(f"Failed to load policies: {e}")
            raise
    
    def get_formatted_policies(self) -> str:
        """Get formatted policy content for prompts."""
        return self.formatted_policy_content
    
    def get_policy_list(self) -> List[Dict[str, Any]]:
        """Get raw policy list."""
        return self.policy_list
