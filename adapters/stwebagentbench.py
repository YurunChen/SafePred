"""
STWebAgentBench Adapter for SafePred Integration.

This adapter converts STWebAgentBench's state and action formats to SafePred's standard format.
"""

import json
from typing import Any, Dict, Optional

from .base import BenchmarkAdapter, register_adapter
from ..utils.logger import get_logger

logger = get_logger("SafePred.Adapter.STWebAgentBench")

# Try to import browsergym utilities, but handle gracefully if not available
try:
    from browsergym.utils.obs import flatten_axtree_to_str
except ImportError:
    logger.warning("browsergym.utils.obs not available, axtree conversion may fail")
    def flatten_axtree_to_str(axtree_object):
        """Fallback function if browsergym is not available."""
        if isinstance(axtree_object, str):
            return axtree_object
        return json.dumps(axtree_object, ensure_ascii=False) if axtree_object else ""


class STWebAgentBenchAdapter(BenchmarkAdapter):
    """
    Adapter for STWebAgentBench benchmark.
    
    Converts STWebAgentBench's observation dict and action strings to SafePred format.
    """
    
    def __init__(self):
        """Initialize STWebAgentBench adapter."""
        # Note: Conversation history is managed by ConversationHistoryManager in SafeAgent
        # This adapter does not maintain its own conversation history
        pass
    
    def state_to_safepred(
        self,
        raw_state: Any,
        intent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Convert STWebAgentBench observation to SafePred state format.
        
        Args:
            raw_state: STWebAgentBench observation dict with keys:
                - goal: Task goal/instruction
                - axtree_object: Accessibility tree object
                - axtree_txt: Accessibility tree text (optional, will be generated if missing)
                - chat_messages: List of chat messages
                - policies: List of policy dicts
                - url: Current page URL
                - etc.
            intent: Task intent/instruction (optional, will use raw_state['goal'] if not provided)
            metadata: Additional metadata (optional)
        
        Returns:
            State dict in SafePred format
        """
        # Ensure metadata is a dict
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            logger.warning(f"Metadata is not a dict (type: {type(metadata)}), using empty dict")
            metadata = {}
        
        # Ensure raw_state is a dict
        if not isinstance(raw_state, dict):
            logger.warning(f"raw_state is not a dict (type: {type(raw_state)}), using empty dict")
            raw_state = {}
        
        # Extract axtree_txt
        axtree_txt = ""
        if "axtree_txt" in raw_state:
            axtree_txt = raw_state.get("axtree_txt", "")
        elif "axtree_object" in raw_state:
            try:
                axtree_txt = flatten_axtree_to_str(raw_state["axtree_object"])
            except Exception as e:
                logger.warning(f"Failed to flatten axtree_object: {e}")
                axtree_txt = ""
        else:
            logger.warning("No axtree_txt or axtree_object found in observation")
        
        # Build SafePred state
        state = {}
        state["axtree_txt"] = axtree_txt
        
        # Extract URL
        url = raw_state.get("url", "")
        if url:
            state["url"] = url
        
        # Extract goal/intent
        goal = intent or raw_state.get("goal", "")
        state["goal"] = goal
        state["intent"] = goal  # SafePred expects both "goal" and "intent"
        
        # Extract chat messages (conversation history)
        chat_messages = raw_state.get("chat_messages", [])
        if chat_messages:
            # Convert to SafePred format: list of dicts with "role" and "message"
            state["chat_messages"] = []
            for msg in chat_messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    message = msg.get("message", "")
                    if message:  # Only add non-empty messages
                        state["chat_messages"].append({
                            "role": role,
                            "message": message
                        })
        
        # Extract policies (if available)
        policies = raw_state.get("policies", [])
        if policies:
            state["policies"] = policies
        
        # Add metadata if available
        if metadata:
            state["metadata"] = metadata
        
        return state
    
    def action_to_safepred(self, action: Any) -> str:
        """
        Convert STWebAgentBench action to SafePred string format.
        
        Args:
            action: STWebAgentBench action, which can be:
                - A string (e.g., 'click("42")', 'fill("input_id", "text")', 'finish("message")')
                - A dict with "raw_prediction" key (preferred if available)
        
        Returns:
            Action string in SafePred format
        """
        # Prefer raw_prediction if available (includes full reasoning)
        if isinstance(action, dict) and "raw_prediction" in action and action["raw_prediction"]:
            return str(action["raw_prediction"]).strip()
        
        # Handle string actions
        if isinstance(action, str):
            return action.strip()
        
        # Handle dict actions (convert to string representation)
        if isinstance(action, dict):
            # Try to extract action type and parameters
            action_type = action.get("action_type", "")
            element_id = action.get("element_id", "")
            text = action.get("text", "")
            url = action.get("url", "")
            
            if action_type == "click" and element_id:
                return f'click("{element_id}")'
            elif action_type == "fill" and element_id and text:
                return f'fill("{element_id}", "{text}")'
            elif action_type == "navigate" and url:
                return f'goto_url("{url}")'
            elif action_type == "finish":
                message = action.get("message", "")
                if message:
                    return f'finish("{message}")'
                else:
                    return "finish()"
            else:
                # Fallback: JSON representation
                return json.dumps(action, ensure_ascii=False)
        
        # Fallback: convert to string
        return str(action)
    
    def action_from_safepred(self, action_str: str) -> Any:
        """
        Convert SafePred action string back to STWebAgentBench format.
        
        Note: STWebAgentBench actions are already strings, so this is mostly a pass-through.
        However, we can parse and validate the action format.
        
        Args:
            action_str: Action string in SafePred format (e.g., 'click("42")', 'finish("message")')
        
        Returns:
            Action string (STWebAgentBench uses string actions directly)
        """
        # STWebAgentBench actions are already strings, so return as-is
        # But we can validate the format
        if not isinstance(action_str, str):
            logger.warning(f"action_str is not a string (type: {type(action_str)}), converting")
            action_str = str(action_str)
        
        # Validate common action patterns
        action_str = action_str.strip()
        
        # Common action patterns:
        # - click("id") or click('id')
        # - fill("id", "text") or fill('id', 'text')
        # - goto_url("url")
        # - finish("message") or finish()
        # - send_msg_to_user("message")
        
        # Return the action string as-is (STWebAgentBench expects string actions)
        return action_str


# Register the adapter
register_adapter("stwebagentbench", STWebAgentBenchAdapter)

