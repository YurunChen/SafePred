"""
OSWorld Adapter for SafePred Integration.

This adapter converts OSWorld's state and action formats to SafePred's standard format.
OSWorld is a desktop automation benchmark that uses accessibility trees and pyautogui actions.
"""

import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from .base import BenchmarkAdapter, register_adapter
from ..utils.logger import get_logger

logger = get_logger("SafePred.Adapter.OSWorld")


def _get_state_summary(linearized_accessibility_tree: str, max_chars: int = 200) -> str:
    """
    Generate a summary of the state for debugging purposes.
    
    Args:
        linearized_accessibility_tree: The linearized accessibility tree string
        max_chars: Maximum number of characters to include in summary
        
    Returns:
        A summary string with key information
    """
    if not linearized_accessibility_tree:
        return "empty"
    
    # Get first few lines
    lines = linearized_accessibility_tree.split('\n')
    first_lines = '\n'.join(lines[:3]) if len(lines) > 3 else linearized_accessibility_tree[:max_chars]
    summary = f"first_lines={first_lines[:max_chars]}"
    
    # Check for key UI elements that might indicate state
    key_elements = []
    if "push-button\tOK" in linearized_accessibility_tree:
        key_elements.append("OK_button")
    if "push-button\tCancel" in linearized_accessibility_tree:
        key_elements.append("Cancel_button")
    if "push-button\tReset" in linearized_accessibility_tree:
        key_elements.append("Reset_button")
    if "menu-item\tParagraph" in linearized_accessibility_tree:
        key_elements.append("Paragraph_menu")
    if "menu\tFormat" in linearized_accessibility_tree:
        key_elements.append("Format_menu")
    
    if key_elements:
        summary += f", key_elements={','.join(key_elements)}"
    
    return summary


class OSWorldAdapter(BenchmarkAdapter):
    """
    Adapter for OSWorld benchmark.
    
    Converts OSWorld's observation dict (with screenshot and accessibility_tree)
    and pyautogui action strings to SafePred format.
    """
    
    def __init__(self):
        """Initialize OSWorld adapter."""
        self.conversation_history: List[Dict[str, str]] = []
    
    def state_to_safepred(
        self,
        raw_state: Any,
        intent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Convert OSWorld observation to SafePred state format.
        
        Args:
            raw_state: OSWorld observation dict with keys:
                - accessibility_tree: XML string or ElementTree object
                - screenshot: PIL Image or numpy array (optional)
                - instruction: Task instruction (optional)
            intent: Task intent/instruction (optional, will use raw_state['instruction'] if not provided)
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
        
        # Extract accessibility tree text
        # IMPORTANT: Use the SAME accessibility tree that mm_agent sees to maintain state space consistency
        # Priority: Use linearized_accessibility_tree (same as mm_agent uses, may be truncated)
        # Fallback: Use raw accessibility_tree if linearized version is not available
        axtree_txt = ""
        
        # Check if linearized_accessibility_tree is available (same as mm_agent sees)
        if "linearized_accessibility_tree" in raw_state:
            axtree_txt = raw_state["linearized_accessibility_tree"]
        elif "accessibility_tree" in raw_state:
            axtree_obj = raw_state["accessibility_tree"]
            if isinstance(axtree_obj, str):
                # Check if it's already linearized (table format) or raw XML
                if axtree_obj.startswith("tag\tname\ttext\tclass"):
                    # Already linearized format
                    axtree_txt = axtree_obj
                    logger.debug("[OSWorldAdapter] Using accessibility_tree (already linearized)")
                else:
                    # Raw XML format - use as-is
                    axtree_txt = axtree_obj
                    logger.debug("[OSWorldAdapter] Using accessibility_tree (XML format)")
            elif isinstance(axtree_obj, ET.Element):
                try:
                    axtree_txt = ET.tostring(axtree_obj, encoding='unicode')
                    logger.debug("[OSWorldAdapter] Converted accessibility_tree Element to string")
                except Exception as e:
                    logger.warning(f"Failed to convert accessibility_tree Element to string: {e}")
                    axtree_txt = ""
            else:
                logger.warning(f"Unexpected accessibility_tree type: {type(axtree_obj)}")
        else:
            logger.warning("No accessibility_tree found in observation")
        
        # Build SafePred state
        state = {}
        state["axtree_txt"] = axtree_txt
        
        # OSWorld doesn't have URLs, use placeholder
        state["url"] = "desktop"
        
        # Extract goal/intent
        goal = intent or raw_state.get("instruction", "")
        state["goal"] = goal
        state["intent"] = goal  # SafePred expects both "goal" and "intent"
        
        # Extract chat messages from metadata or build from intent
        chat_messages = []
        if metadata and "chat_messages" in metadata:
            chat_messages = metadata["chat_messages"]
        elif goal:
            # Build initial chat message from intent
            chat_messages = [
                {"role": "user", "message": goal}
            ]
        
        state["chat_messages"] = chat_messages
        
        # Set page type to desktop
        state["page_type"] = "desktop"
        
        # Extract action history from metadata if available
        if metadata and "action_history" in metadata:
            state["action_history"] = metadata["action_history"]
        
        # Extract key elements from accessibility tree if needed
        state["key_elements"] = []
        
        return state
    
    def action_to_safepred(self, action: Any) -> str:
        """
        Convert OSWorld action to SafePred string format.
        
        OSWorld actions can be:
        - pyautogui code strings: "pyautogui.click(100, 200)"
        - Special codes: "WAIT", "DONE", "FAIL"
        - Action dicts: {"action_type": "CLICK", "x": 100, "y": 200}
        
        Args:
            action: OSWorld action (string, dict, or other format)
        
        Returns:
            Action string in SafePred format
        """
        # Handle string actions (most common in OSWorld)
        if isinstance(action, str):
            action_str = action.strip()
            
            # Handle special codes
            if action_str in ["WAIT", "DONE", "FAIL"]:
                return action_str.lower()
            
            # Handle pyautogui code
            # Extract meaningful parts from pyautogui code
            # Examples:
            # - "pyautogui.click(100, 200)" -> "click (100, 200)"
            # - "pyautogui.typewrite('hello')" -> "type 'hello'"
            # - "pyautogui.press('enter')" -> "press enter"
            
            # Try to parse pyautogui code
            # First, extract all pyautogui calls to check if this is multi-step
            # Improved regex to capture full function calls including button='right' and other kwargs
            pyautogui_call_matches = re.findall(r'pyautogui\.\w+\([^)]*(?:\([^)]*\)[^)]*)*\)', action_str)
            # Fallback to simpler pattern if the above doesn't work
            if not pyautogui_call_matches:
                pyautogui_call_matches = re.findall(r'pyautogui\.\w+\([^)]*\)', action_str)
            
            if len(pyautogui_call_matches) > 1:
                # Multi-step action: parse each call individually and preserve ALL details
                parsed_actions = []
                
                # Define patterns for parsing individual pyautogui calls with full parameter preservation
                pyautogui_patterns = [
                    # click with button='right' or button='left' and coordinates
                    (r'pyautogui\.click\(x\s*=\s*(\d+),\s*y\s*=\s*(\d+),\s*button\s*=\s*[\'"]?(\w+)[\'"]?\)', 
                     lambda m: f"click({m.group(1)}, {m.group(2)}, button={m.group(3)})"),
                    (r'pyautogui\.click\((\d+),\s*(\d+),\s*button\s*=\s*[\'"]?(\w+)[\'"]?\)', 
                     lambda m: f"click({m.group(1)}, {m.group(2)}, button={m.group(3)})"),
                    # click with button='right' or button='left' (no coordinates)
                    (r'pyautogui\.click\(button\s*=\s*[\'"]?(\w+)[\'"]?\)', 
                     lambda m: f"click(button={m.group(1)})"),
                    # click with keyword args: click(x=100, y=70)
                    (r'pyautogui\.click\(x\s*=\s*(\d+),\s*y\s*=\s*(\d+)\)', 
                     lambda m: f"click({m.group(1)}, {m.group(2)})"),
                    # click with positional args: click(100, 70)
                    (r'pyautogui\.click\((\d+),\s*(\d+)\)', 
                     lambda m: f"click({m.group(1)}, {m.group(2)})"),
                    # click with no args (uses current mouse position)
                    (r'pyautogui\.click\(\)', 
                     lambda m: "click()"),
                    # moveTo with keyword args and duration
                    (r'pyautogui\.moveTo\(x\s*=\s*(\d+),\s*y\s*=\s*(\d+),\s*duration\s*=\s*([\d.]+)\)', 
                     lambda m: f"moveTo({m.group(1)}, {m.group(2)}, duration={m.group(3)})"),
                    # moveTo with positional args and duration
                    (r'pyautogui\.moveTo\((\d+),\s*(\d+),\s*duration\s*=\s*([\d.]+)\)', 
                     lambda m: f"moveTo({m.group(1)}, {m.group(2)}, duration={m.group(3)})"),
                    # moveTo with keyword args
                    (r'pyautogui\.moveTo\(x\s*=\s*(\d+),\s*y\s*=\s*(\d+)\)', 
                     lambda m: f"moveTo({m.group(1)}, {m.group(2)})"),
                    # moveTo with positional args
                    (r'pyautogui\.moveTo\((\d+),\s*(\d+)\)', 
                     lambda m: f"moveTo({m.group(1)}, {m.group(2)})"),
                    # drag with keyword args
                    (r'pyautogui\.drag\(x\s*=\s*(\d+),\s*y\s*=\s*(\d+)\)', 
                     lambda m: f"drag({m.group(1)}, {m.group(2)})"),
                    # drag with positional args
                    (r'pyautogui\.drag\((\d+),\s*(\d+)\)', 
                     lambda m: f"drag({m.group(1)}, {m.group(2)})"),
                    # typewrite with string and interval
                    (r'pyautogui\.typewrite\([\'"]?([^\'"]+)[\'"]?,\s*interval\s*=\s*([\d.]+)\)', 
                     lambda m: f"typewrite('{m.group(1)}', interval={m.group(2)})"),
                    # typewrite with string
                    (r'pyautogui\.typewrite\([\'"]?([^\'"]+)[\'"]?\)', 
                     lambda m: f"typewrite('{m.group(1)}')"),
                    # write with string
                    (r'pyautogui\.write\([\'"]?([^\'"]+)[\'"]?\)', 
                     lambda m: f"write('{m.group(1)}')"),
                    # press with key
                    (r'pyautogui\.press\([\'"]?([^\'"]+)[\'"]?\)', 
                     lambda m: f"press('{m.group(1)}')"),
                    # scroll
                    (r'pyautogui\.scroll\((-?\d+)\)', 
                     lambda m: f"scroll({m.group(1)})"),
                    # hotkey with multiple keys
                    (r'pyautogui\.hotkey\(([^)]+)\)', 
                     lambda m: f"hotkey({m.group(1)})"),
                ]
                
                # Parse each call individually
                for call in pyautogui_call_matches:
                    parsed = None
                    for pattern, formatter in pyautogui_patterns:
                        match = re.search(pattern, call)
                        if match:
                            parsed = formatter(match)
                            break
                    
                    if parsed:
                        parsed_actions.append(parsed)
                    else:
                        # Fallback: extract function name and try to preserve basic info
                        func_match = re.search(r'pyautogui\.(\w+)\(', call)
                        if func_match:
                            func_name = func_match.group(1)
                            # Try to extract coordinates if present
                            coord_match = re.search(r'\((\d+),\s*(\d+)', call)
                            if coord_match:
                                parsed_actions.append(f"{func_name}({coord_match.group(1)}, {coord_match.group(2)})")
                            else:
                                parsed_actions.append(func_name)
                
                if parsed_actions:
                    # Return detailed multi-step description with all parameters preserved
                    return f"multi-step: {'; '.join(parsed_actions)}"
                else:
                    # Fallback to simple summary if parsing failed
                    func_names = re.findall(r'pyautogui\.(\w+)\(', action_str)
                    return f"multi-step: {', '.join(func_names)}"
            
            # Single action: try to match patterns
            # Support both positional and keyword arguments for click/moveTo/drag
            # Preserve ALL parameter information including button type, coordinates, duration, etc.
            pyautogui_patterns = [
                # click with button='right' or button='left' and coordinates
                (r'pyautogui\.click\(x\s*=\s*(\d+),\s*y\s*=\s*(\d+),\s*button\s*=\s*[\'"]?(\w+)[\'"]?\)', 
                 lambda m: f"click({m.group(1)}, {m.group(2)}, button={m.group(3)})"),
                (r'pyautogui\.click\((\d+),\s*(\d+),\s*button\s*=\s*[\'"]?(\w+)[\'"]?\)', 
                 lambda m: f"click({m.group(1)}, {m.group(2)}, button={m.group(3)})"),
                # click with button='right' or button='left' (no coordinates)
                (r'pyautogui\.click\(button\s*=\s*[\'"]?(\w+)[\'"]?\)', 
                 lambda m: f"click(button={m.group(1)})"),
                # click with keyword args: click(x=100, y=70)
                (r'pyautogui\.click\(x\s*=\s*(\d+),\s*y\s*=\s*(\d+)\)', 
                 lambda m: f"click({m.group(1)}, {m.group(2)})"),
                # click with positional args: click(100, 70)
                (r'pyautogui\.click\((\d+),\s*(\d+)\)', 
                 lambda m: f"click({m.group(1)}, {m.group(2)})"),
                # click with no args
                (r'pyautogui\.click\(\)', 
                 lambda m: "click()"),
                # moveTo with keyword args and duration
                (r'pyautogui\.moveTo\(x\s*=\s*(\d+),\s*y\s*=\s*(\d+),\s*duration\s*=\s*([\d.]+)\)', 
                 lambda m: f"moveTo({m.group(1)}, {m.group(2)}, duration={m.group(3)})"),
                # moveTo with positional args and duration
                (r'pyautogui\.moveTo\((\d+),\s*(\d+),\s*duration\s*=\s*([\d.]+)\)', 
                 lambda m: f"moveTo({m.group(1)}, {m.group(2)}, duration={m.group(3)})"),
                # moveTo with keyword args
                (r'pyautogui\.moveTo\(x\s*=\s*(\d+),\s*y\s*=\s*(\d+)\)', 
                 lambda m: f"moveTo({m.group(1)}, {m.group(2)})"),
                # moveTo with positional args
                (r'pyautogui\.moveTo\((\d+),\s*(\d+)\)', 
                 lambda m: f"moveTo({m.group(1)}, {m.group(2)})"),
                # typewrite with string and interval
                (r'pyautogui\.typewrite\([\'"]?([^\'"]+)[\'"]?,\s*interval\s*=\s*([\d.]+)\)', 
                 lambda m: f"typewrite('{m.group(1)}', interval={m.group(2)})"),
                # typewrite with string
                (r'pyautogui\.typewrite\([\'"]?([^\'"]+)[\'"]?\)', 
                 lambda m: f"typewrite('{m.group(1)}')"),
                # press
                (r'pyautogui\.press\([\'"]?([^\'"]+)[\'"]?\)', 
                 lambda m: f"press('{m.group(1)}')"),
                # write
                (r'pyautogui\.write\([\'"]?([^\'"]+)[\'"]?\)', 
                 lambda m: f"write('{m.group(1)}')"),
                # scroll
                (r'pyautogui\.scroll\((-?\d+)\)', 
                 lambda m: f"scroll({m.group(1)})"),
                # drag with positional args
                (r'pyautogui\.drag\((\d+),\s*(\d+)\)', 
                 lambda m: f"drag({m.group(1)}, {m.group(2)})"),
                # drag with keyword args
                (r'pyautogui\.drag\(x\s*=\s*(\d+),\s*y\s*=\s*(\d+)\)', 
                 lambda m: f"drag({m.group(1)}, {m.group(2)})"),
                # hotkey
                (r'pyautogui\.hotkey\(([^)]+)\)', 
                 lambda m: f"hotkey({m.group(1)})"),
            ]
            
            for pattern, formatter in pyautogui_patterns:
                match = re.search(pattern, action_str)
                if match:
                    return formatter(match)
            
            # If no pattern matches, return the action string as-is (may be multi-line code)
            # Truncate if too long
            if len(action_str) > 200:
                return action_str[:200] + "..."
            return action_str
        
        # Handle dict actions (structured format)
        elif isinstance(action, dict):
            action_type = action.get("action_type", "")
            params = {k: v for k, v in action.items() if k != "action_type"}
            
            # Format as "action_type(param1=value1, param2=value2)"
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            if param_str:
                return f"{action_type}({param_str})"
            else:
                return action_type
        
        # Fallback: convert to string
        else:
            action_str = str(action)
            if len(action_str) > 200:
                return action_str[:200] + "..."
            return action_str
    
    def action_from_safepred(self, action_str: str) -> Any:
        """
        Convert SafePred action string back to OSWorld format.
        
        Note: This is a simplified implementation. In practice, OSWorld typically
        uses pyautogui code strings, so we may need to reconstruct them.
        
        Args:
            action_str: Action string in SafePred format
        
        Returns:
            OSWorld action (string format, typically pyautogui code)
        """
        # Handle special codes
        if action_str.lower() in ["wait", "done", "fail"]:
            return action_str.upper()
        
        # Try to parse and convert back to pyautogui format
        # This is a simplified conversion - may need more sophisticated parsing
        
        # Pattern: "click (100, 200)" -> "pyautogui.click(100, 200)"
        click_match = re.match(r'click\s*\((\d+),\s*(\d+)\)', action_str)
        if click_match:
            return f"pyautogui.click({click_match.group(1)}, {click_match.group(2)})"
        
        # Pattern: "type 'text'" -> "pyautogui.typewrite('text')"
        type_match = re.match(r"type\s+['\"]([^'\"]+)['\"]", action_str)
        if type_match:
            return f"pyautogui.typewrite('{type_match.group(1)}')"
        
        # Pattern: "press key" -> "pyautogui.press('key')"
        press_match = re.match(r'press\s+([^\s]+)', action_str)
        if press_match:
            return f"pyautogui.press('{press_match.group(1)}')"
        
        # If no pattern matches, return as-is (may already be pyautogui code)
        return action_str


# Register the adapter
register_adapter("osworld", OSWorldAdapter)
register_adapter("os-harm", OSWorldAdapter)  # Also register as "os-harm" for compatibility
