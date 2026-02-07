"""
HarmonyGuard Integration for WASP - Direct Import.

This module provides a simple import wrapper for HarmonyGuard's UtilityAgent.
All format conversion is handled by the wrapper.

Uses HarmonyGuard UtilityAgent for safety evaluation.
"""

import logging
from pathlib import Path
import sys

logger = logging.getLogger("HarmonyGuardWrapper")

# Import HarmonyGuard only
HARMONYGUARD_AVAILABLE = False
UtilityAgent = None
HARMONYGUARD_VERSION = "1.0.0"

try:
    # Add HarmonyGuard parent directory to sys.path
    harmonyguard_parent_paths = [
        Path("/data/chenyurun/methods"),
        Path(__file__).parent.parent.parent.parent / "methods",
    ]
    
    harmonyguard_parent_path = None
    harmonyguard_path = None
    
    # Find HarmonyGuard
    for path in harmonyguard_parent_paths:
        harmonyguard_dir = path / "HarmonyGuard"
        if harmonyguard_dir.exists():
            harmonyguard_parent_path = path
            harmonyguard_path = harmonyguard_dir
            break
    
    if not harmonyguard_path:
        raise ImportError("HarmonyGuard not found. Please ensure HarmonyGuard is installed in the methods directory.")
    
    if harmonyguard_parent_path and str(harmonyguard_parent_path) not in sys.path:
        sys.path.insert(0, str(harmonyguard_parent_path))
    
    # Import HarmonyGuard UtilityAgent
    from HarmonyGuard.harmony_agents.utility_agent import UtilityAgent
    HARMONYGUARD_AVAILABLE = True
    logger.info("HarmonyGuard imported successfully")
        
except ImportError as e:
    logger.error(f"HarmonyGuard not available: {e}. Install HarmonyGuard or disable --use_harmonyguard")
    HARMONYGUARD_AVAILABLE = False
    UtilityAgent = None


class HarmonyGuardWrapper:
    """
    Wrapper for HarmonyGuard UtilityAgent to integrate with WASP agent.
    
    This wrapper provides a consistent interface similar to SafePred and Reactive wrappers.
    """
    
    def __init__(self, risk_cat_path=None, config_path=None, policy_txt_path=None):
        """
        Initialize HarmonyGuard wrapper.
        
        Args:
            risk_cat_path: Path to risk category file (optional, will be read from config if not provided)
            config_path: Path to HarmonyGuard config.yaml file (optional, will search for it if not provided)
            policy_txt_path: Path to plain text policy file (used when use_structured_policy=False)
        """
        if not HARMONYGUARD_AVAILABLE:
            raise ImportError("HarmonyGuard is not available. Please install HarmonyGuard first.")
        
        # If config_path not provided, try to find HarmonyGuard/config.yaml
        if not config_path:
            import sys
            from pathlib import Path
            for path_str in sys.path:
                try:
                    potential_config = Path(path_str) / "HarmonyGuard" / "config.yaml"
                    if potential_config.exists():
                        config_path = str(potential_config)
                        logger.info(f"Found HarmonyGuard config.yaml at: {config_path}")
                        break
                except (TypeError, AttributeError):
                    continue
        
        self.utility_agent = UtilityAgent(
            risk_cat_path=risk_cat_path,
            config_path=config_path,
            policy_txt_path=policy_txt_path
        )
        logger.info("HarmonyGuardWrapper initialized successfully")
    
    def evaluate_action_risk(self, state, action, intent, metadata=None, return_usage=False):
        """
        Evaluate action risk using HarmonyGuard's UtilityAgent.
        
        Args:
            state: Current state/trajectory (list of previous states/actions)
            action: Action to evaluate (dict with action_type, etc.)
            intent: Task instruction/intent
            metadata: Additional metadata (optional)
            return_usage: Whether to return token usage information
        
        Returns:
            dict with:
                - is_safe: bool, whether the action is safe
                - threat_risk: str, "safe" or "risky"
                - threat_risk_explanation: str, explanation of the risk
                - task_alignment: str, "aligned" or "misaligned"
                - task_alignment_explanation: str, explanation of alignment
                - optimization_guidance: str, optional guidance for optimization
                - usage: dict, token usage if return_usage=True
        """
        if not self.utility_agent:
            return {
                "is_safe": True,
                "threat_risk": "safe",
                "threat_risk_explanation": "HarmonyGuard not available",
                "task_alignment": "aligned",
                "task_alignment_explanation": "",
                "optimization_guidance": None,
            }
        
        # Extract thought/reasoning from metadata or action
        thought = ""
        if metadata and "current_response" in metadata:
            thought = metadata["current_response"]
        elif isinstance(action, dict) and "reasoning" in action:
            thought = action["reasoning"]
        
        # Format last step message from action
        last_step_message = self._format_action_message(action)
        
        # Call HarmonyGuard's thought_aligner_response
        result = self.utility_agent.thought_aligner_response(
            instruction=intent,
            thought=thought,
            trajectory=state if isinstance(state, list) else [],
            last_step_message=last_step_message,
            return_usage=return_usage
        )
        
        # Extract usage if available
        usage_info = None
        if return_usage and isinstance(result, dict) and "usage" in result:
            usage_info = result.pop("usage")
        
        # Convert HarmonyGuard result to WASP format
        threat_risk = result.get("threat_risk", "safe")
        is_safe = (threat_risk == "safe")
        
        wasp_result = {
            "is_safe": is_safe,
            "threat_risk": threat_risk,
            "threat_risk_explanation": result.get("threat_risk_explanation", ""),
            "task_alignment": result.get("task_alignment", "aligned"),
            "task_alignment_explanation": result.get("task_alignment_explanation", ""),
            "optimization_guidance": result.get("optimization_guidance"),
            "policy_ids": result.get("policy_ids", []),
        }
        
        if return_usage and usage_info:
            wasp_result["usage"] = usage_info
        
        return wasp_result
    
    def _format_action_message(self, action):
        """
        Format action into a message string for HarmonyGuard.
        
        Args:
            action: Action dict
        
        Returns:
            str: Formatted action message
        """
        if not isinstance(action, dict):
            return str(action)
        
        action_type = action.get("action_type", "unknown")
        action_str = f"Action: {action_type}"
        
        # Add action-specific details
        if action_type == "click":
            element_id = action.get("element_id", "")
            if element_id:
                action_str += f" (element_id: {element_id})"
        elif action_type == "type":
            text = action.get("text", "")
            element_id = action.get("element_id", "")
            if element_id:
                action_str += f" (element_id: {element_id}, text: {text[:50]})"
        elif action_type == "navigate":
            url = action.get("url", "")
            if url:
                action_str += f" (url: {url})"
        
        return action_str
