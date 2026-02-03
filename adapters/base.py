"""
Base Benchmark Adapter Interface.

This module defines the abstract interface that all benchmark adapters must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

# Registry for benchmark adapters
_ADAPTER_REGISTRY: Dict[str, type] = {}


class BenchmarkAdapter(ABC):
    """
    Abstract base class for benchmark adapters.
    
    Each benchmark should implement this interface to convert its specific
    state and action formats to SafePred's standard format.
    
    SafePred expects states in this format:
    {
        "axtree_txt": str,  # Accessibility tree or page content as text
        "url": str,         # Current page URL (optional)
        "goal": str,        # Task goal/intent
        "action_history": List[Any],  # Previous actions (optional)
        "chat_messages": List[Dict[str, str]],  # Conversation history (optional)
        ...
    }
    
    SafePred expects actions as strings, e.g.:
    - "click [element_id]"
    - "type [element_id] [text]"
    - "navigate [url]"
    - etc.
    """
    
    @abstractmethod
    def state_to_safepred(
        self,
        raw_state: Any,
        intent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Convert benchmark-specific state to SafePred format.
        
        Args:
            raw_state: Benchmark-specific state representation
            intent: Task intent/instruction (optional)
            metadata: Additional metadata (action_history, etc.) (optional)
        
        Returns:
            State dict in SafePred format
        """
        pass
    
    @abstractmethod
    def action_to_safepred(self, action: Any) -> str:
        """
        Convert benchmark-specific action to SafePred string format.
        
        Args:
            action: Benchmark-specific action representation
        
        Returns:
            Action string in SafePred format (e.g., "click [id]", "type [id] [text]")
        """
        pass
    
    @abstractmethod
    def action_from_safepred(self, action_str: str) -> Any:
        """
        Convert SafePred action string back to benchmark-specific format.
        
        Args:
            action_str: Action string in SafePred format
        
        Returns:
            Benchmark-specific action representation
        """
        pass
    
    def format_policies_for_prompt(self, policies: List[Dict[str, Any]]) -> str:
        """
        Format policies as a string to be included in the agent's prompt.
        
        This is an optional method that can be overridden for custom formatting.
        
        Args:
            policies: List of policy dictionaries
        
        Returns:
            Formatted string containing all policies
        """
        if not policies:
            return ""
        
        policy_lines = [
            "**SAFETY POLICIES (You must comply with these policies):**",
            "",
        ]
        
        for policy in policies:
            policy_id = policy.get("policy_id", "Unknown")
            policy_description = policy.get("policy_description", "No description")
            risk_level = policy.get("risk_level", "medium")
            definitions = policy.get("definitions", [])
            scope = policy.get("scope", "")
            
            policy_lines.append(f"**{policy_id} ({risk_level.upper()})**: {policy_description}")
            if scope:
                policy_lines.append(f"  Scope: {scope}")
            if definitions:
                defs_str = "; ".join(definitions) if isinstance(definitions, list) else str(definitions)
                policy_lines.append(f"  Definitions: {defs_str}")
            policy_lines.append("")
        
        policy_lines.append(
            "**IMPORTANT**: All your actions must comply with the above policies. "
            "Actions that violate these policies will be filtered and you will be asked to regenerate safer alternatives."
        )
        
        return "\n".join(policy_lines)


def register_adapter(benchmark_name: str, adapter_class: type):
    """
    Register a benchmark adapter.
    
    Args:
        benchmark_name: Name of the benchmark (e.g., "visualwebarena", "mind2web")
        adapter_class: Adapter class that implements BenchmarkAdapter
    """
    if not issubclass(adapter_class, BenchmarkAdapter):
        raise TypeError(f"Adapter class must inherit from BenchmarkAdapter")
    _ADAPTER_REGISTRY[benchmark_name.lower()] = adapter_class


def get_adapter(benchmark_name: str) -> type:
    """
    Get registered adapter class for a benchmark.
    
    Args:
        benchmark_name: Name of the benchmark
    
    Returns:
        Adapter class
    
    Raises:
        KeyError: If adapter is not registered
    """
    benchmark_name = benchmark_name.lower()
    if benchmark_name not in _ADAPTER_REGISTRY:
        raise KeyError(
            f"Adapter for benchmark '{benchmark_name}' not found. "
            f"Available adapters: {list(_ADAPTER_REGISTRY.keys())}. "
            f"Please register an adapter using register_adapter() or implement one."
        )
    return _ADAPTER_REGISTRY[benchmark_name]





