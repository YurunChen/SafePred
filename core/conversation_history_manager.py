"""
Conversation History Manager for SafePred_v3.

Manages conversation history with filtering and truncation capabilities.
"""

from typing import List, Dict, Optional, Any
from ..utils.logger import get_logger
from ..utils.text_cleaner import clean_template_fields

logger = get_logger("SafePred.ConversationHistoryManager")


class ConversationHistoryManager:
    """
    Manages conversation history for SafePred_v3.
    
    Records responses from web agent and builds conversation history
    for World Model and Safety Classifier inputs.
    """
    
    def __init__(self, max_history_length: Optional[int] = 20, show_full_response: bool = True):
        """
        Initialize conversation history manager.
        
        Args:
            max_history_length: Maximum number of messages to keep in conversation history.
                              When exceeded, older messages are removed (FIFO).
                              Set to None or 0 to disable limiting (keep all messages).
                              Default: 20.
            show_full_response: Whether to show full response or only action in conversation history.
                              If True, saves the full agent response (including reasoning).
                              If False, extracts and saves only the action string (e.g., "click [42]").
                              Default: True.
        """
        # Main conversation history: list of {"role": "user"/"assistant", "message": "..."}
        self.conversation_history: List[Dict[str, str]] = []
        
        # Current task intent (user message)
        self.current_intent: Optional[str] = None
        
        # Current task ID (for task switching)
        self.current_task_id: Optional[str] = None
        
        # Maximum history length (None or 0 = unlimited)
        self.max_history_length: Optional[int] = max_history_length if max_history_length and max_history_length > 0 else None
        
        # Whether to show full response or only action
        self.show_full_response: bool = show_full_response
        
        # Track actual step count (number of executed actions, regardless of history truncation)
        # This ensures Current Step number is accurate even when history is truncated
        self.total_executed_steps: int = 0
    
    def _clean_template_fields(self, response: str) -> str:
        """
        Remove template fields (OBSERVATION:, OBJECTIVE:, URL:, PREVIOUS ACTION:)
        if LLM echoes the input prompt format at the beginning of the response.
        
        This filters out cases where LLM echoes the prompt template structure,
        keeping only the actual reasoning and action.
        
        Args:
            response: Raw LLM response that may contain template fields
        
        Returns:
            Cleaned response without template fields
        """
        # Use unified cleaning function
        return clean_template_fields(response)
    
    def _extract_action_from_response(self, response: str, action_string: Optional[str] = None) -> str:
        """
        Extract action string from full response.
        
        Extracts the action command from code blocks (e.g., "type [2288] [We are working on it]")
        using the same method as WASP's _extract_action (action_splitter pattern).
        
        Args:
            response: Full agent response
            action_string: Optional action string to use as fallback (not used, kept for compatibility)
        
        Returns:
            Extracted action string from code blocks (e.g., "type [2288] [We are working on it]")
        """
        import re
        
        # Reference: WASP's _extract_action method uses action_splitter pattern
        # Pattern: ```((.|\n)*?)```  (matches ALL content between ```, exactly like WASP)
        action_splitter = "```"
        pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
        match = re.search(pattern, response, re.DOTALL)
        
        if match:
            # Return the captured content (group 1 is the full content between ```)
            action_candidate = match.group(1).strip()
            if action_candidate:
                return action_candidate
        
        # Fallback: use action_string if provided
        if action_string:
            return str(action_string)
        
        # Last resort: return original response (may already be an action string)
        logger.warning(f"[ConversationHistoryManager] Could not extract action from code blocks, returning original response: {response[:100]}")
        return response
    
    def set_task(self, task_id: Optional[str] = None, intent: str = "") -> None:
        """
        Set the current task (by task ID) and intent.
        
        Conversation history is reset when task_id changes (not when intent changes).
        This ensures different task instances have separate conversation histories,
        even if they share the same intent description.
        
        Args:
            task_id: Task identifier (used to detect task switching). If None, falls back to intent-based switching.
            intent: Task intent/instruction (user message)
        """
        # Normalize task_id and intent
        task_id = task_id.strip() if task_id and isinstance(task_id, str) else None
        intent = intent.strip() if intent else ""
        
        # Check if task has changed
        # Priority: task_id (if provided) > intent (fallback)
        task_changed = False
        if task_id:
            # Use task_id for task switching if provided
            if self.current_task_id != task_id:
                task_changed = True
        elif intent:
            # Fall back to intent-based switching if task_id not provided
            if self.current_intent != intent:
                task_changed = True
        else:
            # No task_id or intent provided
            logger.warning("[ConversationHistoryManager] No task_id or intent provided, skipping task setup")
            return
        
        # Reset history if task changed
        if task_changed:
            # New task: reset conversation history
            self.conversation_history = []
            self.current_task_id = task_id
            self.current_intent = intent
            self.total_executed_steps = 0  # Reset step count for new task
            
            # Add user message if intent is provided
            if intent:
                self.conversation_history.append({
                    "role": "user",
                    "message": intent
                })
                logger.debug(
                    f"[ConversationHistoryManager] New task detected (task_id={task_id}), reset history. "
                    f"Intent: {intent}"
                )
            else:
                logger.debug(
                    f"[ConversationHistoryManager] New task detected (task_id={task_id}), reset history (no intent)"
                )
        else:
            # Same task: just update intent if it changed
            if intent and self.current_intent != intent:
                # Update first user message if it exists
                if self.conversation_history and self.conversation_history[0].get("role") == "user":
                    self.conversation_history[0]["message"] = intent
                elif intent:
                    # No user message yet, add it
                    self.conversation_history.insert(0, {
                        "role": "user",
                        "message": intent
                    })
                self.current_intent = intent
                logger.debug(f"[ConversationHistoryManager] Updated intent for current task: {intent}")
    
    def add_executed_response(self, response: str, action_string: Optional[str] = None) -> None:
        """
        Add an executed action's response to the main conversation history.
        
        This should only be called for actions that passed risk evaluation and were actually executed.
        The response will be added to the main conversation history and will persist across the task.
        
        Args:
            response: Agent's full response (raw_prediction from LLM) that was executed
            action_string: Optional action string (e.g., "click [42]") for extraction fallback
        """
        if not response or response.strip() == "":
            logger.warning(f"[ConversationHistoryManager] Empty response for executed action, skipping")
            return
        
        # Clean template fields (OBSERVATION:, OBJECTIVE:, etc.) from response
        cleaned_response = self._clean_template_fields(response.strip())
        
        if not cleaned_response:
            logger.warning(f"[ConversationHistoryManager] Response for executed action became empty after cleaning, skipping")
            return
        
        # Decide what to save based on configuration
        if self.show_full_response:
            # Save full response (including reasoning)
            message_to_save = cleaned_response
        else:
            # Extract and save only action string
            message_to_save = self._extract_action_from_response(cleaned_response, action_string)
            logger.debug(f"[ConversationHistoryManager] Extracted action for executed response: {message_to_save[:50]}")
        
        # Add to main conversation history (persistent, accumulates across task)
        self.conversation_history.append({
            "role": "assistant",
            "message": message_to_save.strip()
        })
        
        # Increment total executed steps count
        self.total_executed_steps += 1
        
        logger.debug(f"[ConversationHistoryManager] Added executed response to main history (length: {len(message_to_save)} chars, total messages: {len(self.conversation_history)}, total executed steps: {self.total_executed_steps})")
        
        # Note: Full response content is logged by agent.py, avoiding duplicate logging here
    
    def _truncate_history(self, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Truncate conversation history to max_history_length, keeping the most recent assistant messages.
        
        Always preserves the first user message (task intent) which does NOT count toward max_history_length.
        Only assistant messages count toward the limit.
        
        Args:
            history: List of conversation messages
        
        Returns:
            Truncated history: first user message (task intent) + most recent max_history_length assistant messages
        """
        if not self.max_history_length:
            return history
        
        # Find the first user message (task intent)
        first_user_idx = None
        for i, msg in enumerate(history):
            if isinstance(msg, dict) and msg.get("role") == "user":
                first_user_idx = i
                break
        
        # Separate user message and assistant messages
        if first_user_idx is not None:
            # Get first user message (task intent)
            first_user_msg = history[first_user_idx]
            
            # Get all assistant messages (excluding the first user message)
            assistant_messages = [msg for i, msg in enumerate(history) if i != first_user_idx and msg.get("role") == "assistant"]
            
            # Check if we need to truncate
            if len(assistant_messages) <= self.max_history_length:
                # No truncation needed, return first user + all assistant messages
                if first_user_idx == 0:
                    return history[:1] + assistant_messages
                else:
                    return [first_user_msg] + assistant_messages
            else:
                # Truncate to keep only the most recent max_history_length assistant messages
                recent_assistant_messages = assistant_messages[-self.max_history_length:]
                # Always include first user message (task intent), which is NOT counted in max_length
                if first_user_idx == 0:
                    return history[:1] + recent_assistant_messages
                else:
                    return [first_user_msg] + recent_assistant_messages
        else:
            # No user message found, just return most recent assistant messages (up to max_length)
            assistant_messages = [msg for msg in history if msg.get("role") == "assistant"]
            if len(assistant_messages) <= self.max_history_length:
                return assistant_messages
            else:
                return assistant_messages[-self.max_history_length:]
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get conversation history for the current task.
        
        Returns:
            List of conversation messages in format [{"role": "user"/"assistant", "message": "..."}]
            History is truncated to max_history_length if configured
        """
        # Truncate history if max_length is configured
        return self._truncate_history(self.conversation_history)
    
    def get_total_executed_steps(self) -> int:
        """
        Get the total number of executed steps (actions) for the current task.
        
        This count is independent of conversation history truncation and represents
        the actual number of actions that have been executed.
        
        Returns:
            Total number of executed steps (0 if no actions executed yet)
        """
        return self.total_executed_steps
    
    def reset(self) -> None:
        """Reset conversation history (e.g., for a new task)."""
        self.conversation_history = []
        self.current_intent = None
        self.current_task_id = None
        self.total_executed_steps = 0
        logger.debug("[ConversationHistoryManager] Reset conversation history")

