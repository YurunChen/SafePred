"""
VisualWebArena Adapter for SafePred Integration.

This adapter converts VisualWebArena's state and action formats to SafePred's standard format.
"""

import json
import re
from typing import Any, Dict, List, Optional
import numpy as np

from .base import BenchmarkAdapter, register_adapter
from ..utils.logger import get_logger

logger = get_logger("SafePred.Adapter.VisualWebArena")


class VisualWebArenaAdapter(BenchmarkAdapter):
    """
    Adapter for VisualWebArena benchmark.
    
    Converts VisualWebArena's trajectory-based state representation and
    ActionTypes-based actions to SafePred format.
    """
    
    def __init__(self):
        """Initialize VisualWebArena adapter."""
        self.conversation_history: List[Dict[str, str]] = []
    
    def state_to_safepred(
        self,
        raw_state: Any,
        intent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Convert VisualWebArena trajectory to SafePred state format.
        
        Args:
            raw_state: Can be:
                - A trajectory list (List[Dict]) from VisualWebArena
                - A single observation dict
            intent: Task intent/instruction
            metadata: Additional metadata with keys:
                - action_history: List of previous actions
                - prompt_injection: Optional prompt injection text
        
        Returns:
            State dict in SafePred format
        """
        # Ensure metadata is a dict
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            logger.error(f"[Adapter] Metadata is not a dict (type: {type(metadata)}), value: {metadata}")
            logger.warning(f"Metadata is not a dict (type: {type(metadata)}), using empty dict")
            metadata = {}
        
        # Safely get values from metadata
        if not isinstance(metadata, dict):
            raise TypeError(f"metadata must be a dict, got {type(metadata)}: {metadata}")
        prompt_injection = metadata.get("prompt_injection")
        
        # Note: action_history is not used by SafePred_v5 for conversation history
        # SafePred_v5 uses ConversationHistoryManager internally
        # action_history may still be in metadata for web agent's use (e.g., prompt_constructor)
        # We don't process it here since it's not needed for SafePred_v5
        
        # Handle trajectory format (list of state-action pairs)
        if isinstance(raw_state, list) and len(raw_state) > 0:
            # Get the most recent state - filter out non-dict elements
            last_state_info = None
            for item in reversed(raw_state):
                if isinstance(item, dict):
                    last_state_info = item
                    break
            
            # If no dict found, use empty dict
            if last_state_info is None or not isinstance(last_state_info, dict):
                logger.warning(f"No valid dict found in raw_state, using empty dict. Last item type: {type(raw_state[-1]) if raw_state else 'N/A'}")
                last_state_info = {}
            
            # Get observation and info, ensuring they are dicts
            observation_raw = last_state_info.get("observation", {}) if isinstance(last_state_info, dict) else {}
            if isinstance(observation_raw, dict):
                observation = observation_raw
            elif isinstance(observation_raw, str):
                # If observation is a string (e.g., accessibility tree text), wrap it
                logger.debug(f"Observation is string, wrapping in dict with 'text' key")
                observation = {"text": observation_raw}
            else:
                logger.warning(f"Unexpected observation type {type(observation_raw)}, using empty dict")
                observation = {}
            
            # Get info safely
            info_raw = last_state_info.get("info", {}) if isinstance(last_state_info, dict) else {}
            if isinstance(info_raw, dict):
                info = info_raw
            else:
                logger.warning(f"Info is not a dict (type: {type(info_raw)}), using empty dict")
                info = {}
            
            # Reset conversation history if this is a new task
            if len(raw_state) <= 2:
                if self.conversation_history:
                    last_user_msg = None
                    for msg in reversed(self.conversation_history):
                        # Ensure msg is a dict before calling .get()
                        if not isinstance(msg, dict):
                            logger.error(f"[Adapter] conversation_history contains non-dict element: type={type(msg)}, value={msg}")
                            continue
                        if msg.get("role") == "user":
                            last_user_msg = msg.get("message", "")
                            break
                    
                    current_intent = intent
                    if prompt_injection:
                        current_intent = f"{intent}\n\n[User Message]: {prompt_injection}"
                    
                    if last_user_msg != current_intent:
                        self.conversation_history = []
        else:
            # Single observation dict
            observation = raw_state if isinstance(raw_state, dict) else {}
            info = {}
        
        # Build SafePred state
        state = {}
        
        # Extract axtree_txt
        if "text" in observation:
            state["axtree_txt"] = observation["text"]
        elif "accessibility_tree" in observation:
            axtree = observation["accessibility_tree"]
            state["axtree_txt"] = axtree if isinstance(axtree, str) else str(axtree)
        elif "html" in observation:
            state["axtree_txt"] = observation["html"]
        else:
            state["axtree_txt"] = ""
        
        # Extract URL
        # Ensure observation and info are dicts before calling .get()
        if not isinstance(observation, dict):
            logger.warning(f"Observation is not a dict (type: {type(observation)}), cannot extract URL")
            observation = {}
        if not isinstance(info, dict):
            logger.warning(f"Info is not a dict (type: {type(info)}), cannot extract URL")
            info = {}
        
        url = info.get("url", "") or observation.get("url", "")
        if url:
            state["url"] = url
        
        # Add intent/goal
        if prompt_injection:
            full_intent = f"{intent}\n\n[User Message]: {prompt_injection}"
            state["intent"] = full_intent
            state["prompt_injection"] = prompt_injection
        else:
            state["intent"] = intent or ""
        
        # Map intent to goal (StatePreprocessor expects "goal")
        state["goal"] = state["intent"]
        
        # Note: SafePred_v5 uses ConversationHistoryManager for conversation history
        # We don't set action_history or chat_messages here since they're managed internally
        # action_history in metadata is available for web agent's use (e.g., prompt_constructor)
        
        # Add metadata if available
        if "observation_metadata" in info:
            state["metadata"] = info["observation_metadata"]
        
        return state
    
    def action_to_safepred(self, action: Any) -> str:
        """
        Convert VisualWebArena action to SafePred string format.
        
        Args:
            action: VisualWebArena action dict with keys:
                - action_type: ActionTypes enum
                - element_id: Element identifier
                - text: Text input (for type actions)
                - url: URL (for navigate actions)
                - raw_prediction: Full LLM response (preferred if available)
        
        Returns:
            Action string in SafePred format
        """
        # Prefer raw_prediction if available (includes full reasoning)
        if isinstance(action, dict) and "raw_prediction" in action and action["raw_prediction"]:
            return str(action["raw_prediction"]).strip()
        
        # Ensure action is a dict before calling .get()
        if action is None:
            error_msg = "[Adapter] action_to_safepred: action is None"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if not isinstance(action, dict):
            # Add more detailed logging to understand the issue
            action_repr = repr(action)
            action_str_repr = str(action)
            error_msg = f"[Adapter] action_to_safepred: action is not a dict, type={type(action)}, repr={action_repr}, str={action_str_repr}"
            logger.error(error_msg)
            logger.error(f"  - type: {type(action)}")
            logger.error(f"  - repr: {action_repr}")
            logger.error(f"  - str: {action_str_repr}")
            logger.error(f"  - is None: {action is None}")
            logger.error(f"  - equals 'None': {action == 'None'}")
            raise TypeError(error_msg)
        
        # Parse from action object
        action_type = action.get("action_type", "")
        element_id = action.get("element_id", "")
        
        # Convert action_type to string
        if hasattr(action_type, 'name'):
            action_type_name = action_type.name.lower()
        else:
            action_type_str = str(action_type).lower()
            if 'click' in action_type_str:
                action_type_name = "click"
            elif 'type' in action_type_str:
                action_type_name = "type"
            elif 'scroll' in action_type_str:
                action_type_name = "scroll"
            elif 'stop' in action_type_str or 'none' in action_type_str:
                action_type_name = "stop"
            elif 'go_back' in action_type_str:
                action_type_name = "go_back"
            elif 'navigate' in action_type_str or 'goto_url' in action_type_str:
                action_type_name = "navigate"
            else:
                action_type_name = str(action_type).lower()
        
        # Format action string
        if action_type_name == "click":
            return f"click [{element_id}]"
        elif action_type_name == "type":
            text = action.get("text", "")
            if isinstance(text, (list, np.ndarray)):
                text_str = "[text input]"
            else:
                text_str = str(text) if text else ""
            return f"type [{element_id}] {text_str}"
        elif action_type_name in ["navigate", "goto_url"]:
            url = action.get("url", "")
            return f"navigate {url}"
        elif action_type_name == "scroll":
            direction = action.get("direction", "down")
            return f"scroll {direction}"
        elif action_type_name == "go_back":
            return "go_back"
        elif action_type_name in ["stop", "none"]:
            return "stop"
        else:
            # Fallback: JSON representation
            safe_action = {}
            for key, value in action.items():
                if isinstance(value, (np.ndarray, np.generic)):
                    safe_action[key] = value.tolist() if hasattr(value, 'tolist') else str(value)
                elif hasattr(value, 'name'):  # Enum
                    safe_action[key] = value.name
                else:
                    try:
                        json.dumps(value)
                        safe_action[key] = value
                    except (TypeError, ValueError):
                        safe_action[key] = str(value)
            return json.dumps(safe_action, ensure_ascii=False)
    
    def action_from_safepred(self, action_str: str) -> Any:
        """
        Convert SafePred action string back to VisualWebArena format.
        
        Note: This is a simplified implementation. In practice, you may need
        to parse the action string and reconstruct the action dict.
        
        Args:
            action_str: Action string in SafePred format
        
        Returns:
            VisualWebArena action dict (simplified)
        """
        # Parse action string (simplified - may need more sophisticated parsing)
        if action_str.startswith("click"):
            match = re.search(r'\[(\d+)\]', action_str)
            element_id = match.group(1) if match else ""
            return {"action_type": "click", "element_id": element_id}
        elif action_str.startswith("type"):
            match = re.search(r'\[(\d+)\]\s*(.+)', action_str)
            if match:
                element_id = match.group(1)
                text = match.group(2).strip()
                return {"action_type": "type", "element_id": element_id, "text": text}
        elif action_str.startswith("navigate"):
            url = action_str.replace("navigate", "").strip()
            return {"action_type": "navigate", "url": url}
        elif action_str == "go_back":
            return {"action_type": "go_back"}
        elif action_str == "stop":
            return {"action_type": "stop"}
        
        # Fallback: return as-is
        return {"action_type": "unknown", "raw": action_str}
    
    def _build_conversation_history(
        self,
        intent: Optional[str],
        prompt_injection: Optional[str],
        action_history: List[Any],
    ) -> List[Dict[str, str]]:
        """Build conversation history from intent and action history."""
        chat_messages = []
        
        # Build user message
        if intent:
            user_message = intent
            if prompt_injection:
                user_message = f"{intent}\n\n[User Message]: {prompt_injection}"
            
            if not self.conversation_history:
                chat_messages.append({"role": "user", "message": user_message})
            else:
                # Check if intent changed
                last_user_msg = None
                for msg in reversed(self.conversation_history):
                    # Ensure msg is a dict before calling .get()
                    if not isinstance(msg, dict):
                        logger.error(f"[Adapter] _build_conversation_history: conversation_history contains non-dict element: type={type(msg)}, value={msg}")
                        continue
                    if msg.get("role") == "user":
                        last_user_msg = msg.get("message", "")
                        break
                
                if last_user_msg != user_message:
                    chat_messages.append({"role": "user", "message": user_message})
        
        # Build assistant messages from action history
        # Filter out non-dict elements from conversation_history
        existing_actions = set()
        for msg in self.conversation_history:
            if not isinstance(msg, dict):
                logger.error(f"[Adapter] _build_conversation_history: conversation_history contains non-dict element: type={type(msg)}, value={msg}")
                continue
            if msg.get("role") == "assistant":
                existing_actions.add(msg.get("message", ""))
        
        for action in action_history:
            # Validate action - no fallback, raise error if invalid
            if action is None:
                error_msg = "[Adapter] _build_conversation_history: action is None in action_history"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            if not isinstance(action, dict):
                error_msg = f"[Adapter] _build_conversation_history: action is not a dict, type={type(action)}, value={repr(action)}"
                logger.error(error_msg)
                raise TypeError(error_msg)
            
            # Handle dict actions
            if "raw_prediction" in action and action["raw_prediction"]:
                action_str = str(action["raw_prediction"]).strip()
            else:
                action_str = self.action_to_safepred(action)
            
            if action_str and action_str not in existing_actions and action_str != "None":
                chat_messages.append({"role": "assistant", "message": action_str})
                existing_actions.add(action_str)
        
        # Update internal history
        self.conversation_history.extend(chat_messages)
        
        return self.conversation_history.copy()


# Register the adapter
register_adapter("visualwebarena", VisualWebArenaAdapter)



