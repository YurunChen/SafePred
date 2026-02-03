"""
Trajectory Storage Module for SafePred.

Provides functionality to save, load, and export trajectory data for:
1. World model experience replay
2. World model training data
3. Trajectory analysis and debugging
"""

import json
import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from ..utils.logger import get_logger
from ..utils.type_checkers import validate_dict

logger = get_logger("SafePred.TrajectoryStorage")


@dataclass
class TrajectoryEntry:
    """
    Single trajectory entry (state-action-next_state transition).
    
    Attributes:
        state: Current state representation (optional in compact mode)
        action: Action taken
        next_state: Resulting state after action (optional in compact mode)
        state_id: Unique identifier for current state
        next_state_id: Unique identifier for next state
        actual_delta: Actual state changes computed from real execution (for experience replay)
        predicted_delta: World Model predicted delta (optional, for comparison)
        risk_score: Risk score of the transition
        risk_explanation: Risk evaluation explanation
        action_success: Whether action execution was successful
        reward: Reward received (if available)
        metadata: Additional metadata (timestamp, task_id, etc.)
    """
    state: Optional[Any] = None
    action: Any = None
    next_state: Optional[Any] = None
    state_id: str = ""
    next_state_id: str = ""
    actual_delta: Optional[Dict[str, Any]] = None
    predicted_delta: Optional[Dict[str, Any]] = None
    risk_score: float = 0.0
    risk_explanation: Optional[str] = None
    action_success: bool = True
    reward: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "state_id": self.state_id,
            "action": self._serialize(self.action),
            "next_state_id": self.next_state_id,
            "risk_score": self.risk_score,
            "action_success": self.action_success,
            "reward": self.reward,
            "metadata": self._serialize(self.metadata),
        }
        
        if self.actual_delta is not None:
            result["actual_delta"] = self._serialize(self.actual_delta)
        
        # Only include predicted_delta if available
        if self.predicted_delta is not None:
            result["predicted_delta"] = self._serialize(self.predicted_delta)
        
        # Only include risk_explanation if available
        if self.risk_explanation is not None:
            result["risk_explanation"] = self._serialize(self.risk_explanation)
        
        # Only include full state/next_state if provided (for backward compatibility or debugging)
        if self.state is not None:
            result["state"] = self._serialize(self.state)
        if self.next_state is not None:
            result["next_state"] = self._serialize(self.next_state)
        
        return result
    
    @staticmethod
    def _serialize(obj: Any) -> Any:
        """Serialize object to JSON-compatible format."""
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, dict):
            return {k: TrajectoryEntry._serialize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [TrajectoryEntry._serialize(item) for item in obj]
        else:
            # Handle numpy arrays and other non-serializable types
            try:
                import numpy as np
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
            except ImportError:
                pass
            # Fallback: convert to string for other types
            return str(obj)


class TrajectoryStorage:
    """
    Storage manager for trajectory data.
    
    Separates experience data (for replay) and training data (for model training).
    Experience data: Raw trajectory entries for experience replay.
    Training data: ShareGPT format conversations for supervised learning.
    
    Usage:
        storage = TrajectoryStorage(
            experience_dir="trajectories/experience/",
            training_dir="trajectories/training/"
        )
        storage.save_entry(state, action, next_state, state_id, next_state_id)
    """
    
    def __init__(
        self,
        experience_dir: Optional[Union[str, Path]] = None,
        training_dir: Optional[Union[str, Path]] = None,
        auto_save: bool = True,
        max_entries_in_memory: int = 1000,
        experience_format: str = "jsonl",  # "jsonl" or "json"
        system_prompt: Optional[str] = None,
        include_policies_in_training: bool = True,  # Whether to include policies in training data
        policies: Optional[List[Dict[str, Any]]] = None,  # Policies to include in training data (if not in obs)
        web_agent_model_name: Optional[str] = None,  # Web agent model name for organizing experience data
        world_model_name: Optional[str] = None,  # World model name for organizing experience data by model
    ):
        """
        Initialize trajectory storage.
        
        Args:
            experience_dir: Directory to save experience data (default: ./trajectories/experience/)
            training_dir: Directory to save training data in ShareGPT format (default: ./trajectories/training/)
            auto_save: Whether to automatically save entries to disk
            max_entries_in_memory: Maximum entries to keep in memory before flushing
            experience_format: Format for experience data ("jsonl" or "json")
            system_prompt: System prompt for ShareGPT format training data
            web_agent_model_name: Web agent model name for organizing experience data
            world_model_name: World model name for organizing experience data by model (creates subdirectory)
            save_inaccurate_for_analysis: Whether to save inaccurate predictions for analysis (Scheme 2)
            risk_consistency_threshold: Threshold for risk consistency (0.0-1.0)
        """
        # Store model names for organizing experience data
        self.web_agent_model_name = web_agent_model_name
        self.world_model_name = world_model_name
        
        # Experience storage (for replay)
        # Default to SafePred/trajectories/ directory
        if experience_dir:
            base_experience_dir = Path(experience_dir)
        else:
            # Get SafePred root (trajectory_storage.py is in SafePred/core/)
            safepred_root = Path(__file__).parent.parent
            base_experience_dir = safepred_root / "trajectories" / "experience"
        
        # Create subdirectory based on web_agent_model_name and world_model_name
        # Format: {web_agent_model_name}_{world_model_name}
        # Sanitize model names for filesystem (remove invalid characters)
        safe_web_agent_name = self._sanitize_model_name(self.web_agent_model_name) or "unknown_web_agent"
        safe_world_model_name = self._sanitize_model_name(self.world_model_name) or "unknown_world_model"
        # Combine into directory name: web_agent_model_name_world_model_name
        combined_name = f"{safe_web_agent_name}_{safe_world_model_name}"
        self.experience_dir = base_experience_dir / combined_name
        
        self.experience_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for accurate and inaccurate experience data (Scheme 2)
        self.experience_dir_accurate = self.experience_dir / "accurate"
        self.experience_dir_inaccurate = self.experience_dir / "inaccurate"
        self.experience_dir_accurate.mkdir(parents=True, exist_ok=True)
        self.experience_dir_inaccurate.mkdir(parents=True, exist_ok=True)
        
        # Training storage (ShareGPT format)
        # Default to SafePred/trajectories/training/ directory
        if training_dir:
            base_training_dir = Path(training_dir)
        else:
            # Get SafePred root (trajectory_storage.py is in SafePred/core/)
            safepred_root = Path(__file__).parent.parent
            base_training_dir = safepred_root / "trajectories" / "training"
        
        # Create subdirectory based on web_agent_model_name and world_model_name
        # Format: {web_agent_model_name}_{world_model_name} (same as experience data)
        # Sanitize model names for filesystem (remove invalid characters)
        safe_web_agent_name = self._sanitize_model_name(self.web_agent_model_name) or "unknown_web_agent"
        safe_world_model_name = self._sanitize_model_name(self.world_model_name) or "unknown_world_model"
        # Combine into directory name: web_agent_model_name_world_model_name
        combined_name = f"{safe_web_agent_name}_{safe_world_model_name}"
        base_training_subdir = base_training_dir / combined_name
        
        # Create two separate directories for training data:
        # 1. ground_truth: Actual state changes (for ground truth training data)
        # 2. prediction: World Model predictions (for prediction training data)
        self.training_dir_ground_truth = base_training_subdir / "ground_truth"
        self.training_dir_prediction = base_training_subdir / "prediction"
        self.training_dir_ground_truth.mkdir(parents=True, exist_ok=True)
        self.training_dir_prediction.mkdir(parents=True, exist_ok=True)
        
        # Keep backward compatibility: training_dir points to base subdirectory
        self.training_dir = base_training_subdir
        
        
        self.auto_save = auto_save
        self.max_entries_in_memory = max_entries_in_memory
        self.experience_format = experience_format
        self.system_prompt = system_prompt or "You are a World Model that predicts state changes and evaluates risks based on the current state and action. Your task is to predict the state delta (changes) that will occur after executing an action, and assess the safety risk of that action according to the provided policies."
        self.include_policies_in_training = include_policies_in_training
        self.policies = policies or []  # Policies to use if not in observation
        
        # In-memory storage
        self.experience_entries: List[TrajectoryEntry] = []  # For experience replay
        self.training_conversations_ground_truth: List[Dict[str, Any]] = []  # For ground truth training data
        self.training_conversations_prediction: List[Dict[str, Any]] = []  # For prediction training data
        self.entry_count = 0
        
        # Statistics
        self.stats = {
            "total_entries": 0,
            "successful_actions": 0,
            "failed_actions": 0,
            "total_reward": 0.0,
            "avg_risk_score": 0.0,
        }
        
        # Log initialization as single JSON
        import json
        init_info = {
            "component": "TrajectoryStorage",
            "status": "Initialized",
            "experience_dir": str(self.experience_dir),
            "training_dir": str(self.training_dir),
            "training_dir_ground_truth": str(self.training_dir_ground_truth),
            "training_dir_prediction": str(self.training_dir_prediction),
            "auto_save": auto_save,
            "max_entries_in_memory": max_entries_in_memory,
            "experience_format": experience_format,
            "include_policies_in_training": include_policies_in_training,
            "web_agent_model_name": web_agent_model_name,
            "world_model_name": world_model_name,
        }
        logger.info(
            f"[TrajectoryStorage] Initialization\n{json.dumps(init_info, indent=2, ensure_ascii=False)}"
        )
    
    @staticmethod
    def _sanitize_model_name(model_name: Optional[str]) -> str:
        """
        Sanitize model name for use in filesystem paths.
        
        Args:
            model_name: Original model name (can be None)
            
        Returns:
            Sanitized model name safe for filesystem use
        """
        if model_name is None:
            return "unknown"
        import re
        # Remove or replace invalid filesystem characters
        # Replace slashes, colons, and other special chars with underscores
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', model_name)
        # Replace multiple underscores with single underscore
        safe_name = re.sub(r'_+', '_', safe_name)
        # Remove leading/trailing underscores and dots
        safe_name = safe_name.strip('_.')
        # Limit length to avoid filesystem issues
        if len(safe_name) > 100:
            safe_name = safe_name[:100]
        return safe_name or "unknown"
    
    def save_entry(
        self,
        state: Any,
        action: Any,
        next_state: Any,
        state_id: str,
        next_state_id: str,
        risk_score: float = 0.0,
        action_success: bool = True,
        reward: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        actual_delta: Optional[Dict[str, Any]] = None,
        predicted_delta: Optional[Dict[str, Any]] = None,
        risk_explanation: Optional[str] = None,
        raw_obs: Optional[Dict[str, Any]] = None,
        raw_next_obs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save a single trajectory entry.
        
        Args:
            state: Current state (optional in compact mode)
            action: Action taken
            next_state: Resulting state (optional in compact mode)
            state_id: Unique identifier for current state
            next_state_id: Unique identifier for next state
            risk_score: Risk score of the transition
            action_success: Whether action execution was successful
            reward: Reward received (if available)
            metadata: Additional metadata
            actual_delta: Actual state changes computed from real execution (for experience replay)
            predicted_delta: World Model predicted delta (optional, for comparison)
            risk_explanation: Risk evaluation explanation
        """
        entry = TrajectoryEntry(
            state=state,  # Save compact state (not None)
            action=action,
            next_state=next_state,  # Save compact next_state (not None)
            state_id=state_id,
            next_state_id=next_state_id,
            actual_delta=actual_delta,
            predicted_delta=predicted_delta,
            risk_score=risk_score,
            risk_explanation=risk_explanation,
            action_success=action_success,
            reward=reward,
            metadata=metadata or {},
        )
        
        # Add timestamp if not present
        if "timestamp" not in entry.metadata:
            entry.metadata["timestamp"] = datetime.now().isoformat()
        
        # Check risk consistency from metadata for classification
        risk_consistency_info = metadata.get("risk_consistency") if metadata else None
        is_consistent = False
        validation_failed = False
        if risk_consistency_info:
            is_consistent = risk_consistency_info.get("is_consistent", False)
            validation_failed = risk_consistency_info.get("validation_failed", False)
        
        # Add consistency label to entry metadata
        if not validation_failed and risk_consistency_info:
            entry.metadata["prediction_accuracy"] = "accurate" if is_consistent else "inaccurate"
            entry.metadata["risk_consistency_score"] = risk_consistency_info.get("overall_consistency")
        elif validation_failed:
            entry.metadata["prediction_accuracy"] = "validation_failed"
        
        # Save to experience storage (for replay) - will be classified on flush
        # Store consistency flag in metadata for later classification (dataclass doesn't support dynamic attributes)
        if not validation_failed and risk_consistency_info:
            entry.metadata["_is_consistent"] = is_consistent
        entry.metadata["_validation_failed"] = validation_failed
        self.experience_entries.append(entry)
        
        # Save to training storage (ShareGPT format) - only for successful actions
        if action_success:
            # Extract task_id and step_number from metadata for tracking
            task_id = metadata.get("task_id") if metadata else None
            step_number = metadata.get("step_number") if metadata else None
            
            # Create base metadata for both training data types
            base_metadata = {
                "timestamp": metadata.get("timestamp") if metadata else datetime.now().isoformat(),
            }
            if task_id is not None:
                base_metadata["task_id"] = task_id
            if step_number is not None:
                base_metadata["step_number"] = step_number
            
            # 1. Save ground truth training data (using actual_delta)
            if actual_delta is not None:
                conversation_ground_truth = self._create_conversation_entry(
                    state,
                    action,
                    next_state,
                    actual_delta=actual_delta,
                    risk_score=risk_score,
                    risk_explanation=risk_explanation
                )
                
                # Add metadata with task_id and step_number
                if "metadata" not in conversation_ground_truth:
                    conversation_ground_truth["metadata"] = {}
                conversation_ground_truth["metadata"].update(base_metadata)
                conversation_ground_truth["metadata"]["data_type"] = "ground_truth"
                
                self.training_conversations_ground_truth.append(conversation_ground_truth)
            
            # 2. Save prediction training data (using predicted_delta)
            if predicted_delta is not None:
                conversation_prediction = self._create_conversation_entry(
                    state,
                    action,
                    next_state,
                    actual_delta=predicted_delta,  # Use predicted_delta as the output
                    risk_score=risk_score,
                    risk_explanation=risk_explanation
                )
                
                # Add metadata with task_id and step_number
                if "metadata" not in conversation_prediction:
                    conversation_prediction["metadata"] = {}
                conversation_prediction["metadata"].update(base_metadata)
                conversation_prediction["metadata"]["data_type"] = "prediction"
                
                self.training_conversations_prediction.append(conversation_prediction)
        
        self.entry_count += 1
        
        # Update statistics
        self.stats["total_entries"] += 1
        if action_success:
            self.stats["successful_actions"] += 1
        else:
            self.stats["failed_actions"] += 1
        
        if reward is not None:
            self.stats["total_reward"] += reward
        
        # Update average risk score
        total_risk = self.stats["avg_risk_score"] * (self.stats["total_entries"] - 1) + risk_score
        self.stats["avg_risk_score"] = total_risk / self.stats["total_entries"]
        
        # Auto-save if enabled and memory limit reached
        # Check any of the lists (experience, ground_truth, or prediction) to trigger flush
        total_training_items = (
            len(self.training_conversations_ground_truth) + 
            len(self.training_conversations_prediction)
        )
        if self.auto_save and (
            len(self.experience_entries) >= self.max_entries_in_memory or
            total_training_items >= self.max_entries_in_memory
        ):
            self.flush()
        
        logger.debug(f"[TrajectoryStorage] Saved entry {self.entry_count} (success={action_success}, risk={risk_score:.3f})")
    
    def save_filtered_action(
        self,
        state: Any,
        action: Any,
        state_id: str,
        risk_score: float,
        risk_explanation: Optional[str],
        predicted_delta: Optional[Dict[str, Any]],
        violated_policy_ids: List[str],
    ) -> None:
        """
        Save a filtered (high-risk) action to training data.
        
        This method saves actions that were filtered due to high risk (risk_score >= threshold).
        These actions are important for training the world model to recognize high-risk scenarios.
        
        For filtered actions, we don't have a next_state (action wasn't executed), so we use
        predicted_delta as the ground truth for training.
        
        Args:
            state: Current state (compact format)
            action: Action that was filtered
            state_id: Unique identifier for current state
            risk_score: Risk score of the action
            risk_explanation: Risk evaluation explanation
            predicted_delta: World Model's predicted delta (used as ground truth)
            violated_policy_ids: List of violated policy IDs
        """
        # For filtered actions, we use predicted_delta as the ground truth
        # (since there's no actual_delta from execution)
        # Convert predicted_delta to the format expected by _format_delta_as_json
        # predicted_delta already contains all necessary fields (semantic_delta, risk_affordances, etc.)
        
        # Save to training storage (ShareGPT format for risk prediction)
        # Even though action_success=False, we still save to training data for high-risk scenario training
        conversation = self._create_conversation_entry(
            state,
            action,
            next_state=None,  # No next_state for filtered actions
            actual_delta=predicted_delta,  # Use predicted_delta as ground truth
            risk_score=risk_score,
            risk_explanation=risk_explanation
        )
        # Save to prediction training data (filtered actions use predicted_delta as ground truth)
        if "metadata" not in conversation:
            conversation["metadata"] = {}
        conversation["metadata"]["data_type"] = "prediction"
        self.training_conversations_prediction.append(conversation)
        
        # Update statistics
        self.stats["total_entries"] += 1
        self.stats["filtered_actions"] = self.stats.get("filtered_actions", 0) + 1
        
        # Update average risk score
        total_risk = self.stats["avg_risk_score"] * (self.stats["total_entries"] - 1) + risk_score
        self.stats["avg_risk_score"] = total_risk / self.stats["total_entries"]
        
        # Auto-save if enabled and memory limit reached
        if self.auto_save and len(self.training_conversations_prediction) >= self.max_entries_in_memory:
            self.flush()
        
        logger.debug(
            f"[TrajectoryStorage] Saved filtered action to training data "
            f"(risk={risk_score:.3f}, violated_policies={violated_policy_ids})"
        )
    
    def _create_conversation_entry(
        self,
        state: Any,
        action: Any,
        next_state: Any,
        raw_obs: Optional[Dict[str, Any]] = None,
        raw_next_obs: Optional[Dict[str, Any]] = None,
        actual_delta: Optional[Dict[str, Any]] = None,
        risk_score: float = 0.0,
        risk_explanation: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a ShareGPT format conversation entry for risk prediction training.
        
        Uses the same prompt format as world model for consistency.
        Format: Input = world_model prompt (state + action), Output = JSON delta + risk info
        
        Args:
            state: Current state (formatted string or dict)
            action: Action taken
            next_state: Resulting state (formatted string or dict)
            raw_obs: Optional raw observation dictionary for current state
            raw_next_obs: Optional raw observation dictionary for next state
            actual_delta: Actual state changes (used as ground truth for predicted_delta)
            risk_score: Risk score of the transition
            risk_explanation: Risk evaluation explanation
        
        Returns:
            Dictionary with "conversations" and "system" keys
        """
        # Use compact state format (must have key_elements)
        validate_dict(state, "state", required_keys=["key_elements"], context="[TrajectoryStorage] ")
        
        action_str = self._serialize_action(action)
        
        # Generate prompt using world model format
        # Get policies list (not string) to match world_model_state_delta signature
        policies_list = None
        if self.include_policies_in_training:
            policies = state.get('policies', [])
            if not policies and self.policies:
                policies = self.policies
            
            if policies:
                # Ensure policies is a list of dicts
                if isinstance(policies, list):
                    # Filter to only dict elements
                    policies_list = [p for p in policies if isinstance(p, dict)]
                else:
                    policies_list = []
        
        # Use world model prompt format for input
        input_prompt = self._create_world_model_prompt(state, action_str, policies_list)
        
        # Format output as JSON (matching world model output format)
        output_json = self._format_delta_as_json(actual_delta, risk_score, risk_explanation)
        
        conversations = [
            {
                "from": "human",
                "value": input_prompt
            },
            {
                "from": "gpt",
                "value": output_json
            }
        ]
        
        return {
            "conversations": conversations,
            "system": self.system_prompt or "You are a World Model for autonomous web agents. Your task has two sequential phases: Phase 1: State Transition Prediction, Phase 2: Risk Assessment."
        }
    
    def _serialize_compact_state(self, state: Dict[str, Any]) -> str:
        """
        Serialize compact state dictionary to string format for training data.
        
        Compact state format:
        {
            "goal": str,
            "policies": List[Dict],
            "url": str,
            "page_type": str,
            "key_elements": List[Dict],
            "chat_history": List[Dict]
        }
        
        Args:
            state: Compact state dictionary
        
        Returns:
            Formatted string representation
        """
        parts = []
        
        # Goal
        if state.get("goal"):
            parts.append(f"Goal: {state['goal']}")
        
        # Policies
        if self.include_policies_in_training:
            policies = state.get('policies', [])
            # If not in state, use provided policies
            if not policies and self.policies:
                policies = self.policies
            
            if policies:
                policies_text = "\n".join([
                    f"Policy {p.get('policy_id', p.get('id', 'N/A'))} ({p.get('priority', p.get('risk_level', p.get('severity', 'Unknown')))}): {p.get('policy_template', p.get('policy_description', p.get('description', str(p))))}"
                    for p in policies
                    if isinstance(p, dict)
                ])
                if policies_text:
                    parts.append(f"Policies:\n{policies_text}")
        
        # URL and Page Type
        if state.get("url"):
            parts.append(f"URL: {state['url']}")
        if state.get("page_type"):
            parts.append(f"Page Type: {state['page_type']}")
        
        # Key Elements (compact representation instead of full Accessibility Tree)
        if state.get("key_elements"):
            elements_lines = []
            for elem in state["key_elements"]:
                elem_str = f"[{elem.get('bid', 'N/A')}] {elem.get('type', 'unknown')}"
                # Use label, or fallback to description (placeholder/aria-label/name/title)
                display_label = elem.get("label") or elem.get("description") or ""
                if display_label:
                    elem_str += f" '{display_label}'"
                if elem.get("critical"):
                    elem_str += " [CRITICAL]"
                if elem.get("disabled"):
                    elem_str += " [DISABLED]"
                if elem.get("required"):
                    elem_str += " [REQUIRED]"
                if elem.get("value"):
                    elem_str += f" value='{elem['value']}'"
                if elem.get("url"):
                    elem_str += f" url='{elem['url']}'"
                # Add additional attributes for better identification (if label is empty and description is also empty)
                if not display_label:
                    # Show all available attributes (not just one) for better identification
                    attr_parts = []
                    if elem.get("placeholder"):
                        attr_parts.append(f"placeholder='{elem['placeholder']}'")
                    if elem.get("aria_label"):
                        attr_parts.append(f"aria-label='{elem['aria_label']}'")
                    if elem.get("name"):
                        attr_parts.append(f"name='{elem['name']}'")
                    if elem.get("title"):
                        attr_parts.append(f"title='{elem['title']}'")
                    if attr_parts:
                        elem_str += " " + " ".join(attr_parts)
                elements_lines.append(elem_str)
            
            if elements_lines:
                parts.append(f"Key Elements:\n" + "\n".join(elements_lines))
        
        # Chat History (simplified, already limited in compact state)
        if state.get("chat_history"):
            chat = state.get('chat_history', [])
            if chat:
                chat_str = '\n'.join([
                    f"{msg.get('role', 'unknown')}: {msg.get('message', '')}"
                    for msg in chat
                ])
                parts.append(f"Chat History:\n{chat_str}")
        
        return "\n\n".join(parts)
    
    def _serialize_state(self, state: Any) -> str:
        """Serialize state to string format for training data, including full observation and chat history."""
        if isinstance(state, str):
            # If state is already a formatted string, return as is
            return state
        elif isinstance(state, dict):
            # Format state with all observation information in English
            parts = []
            
            # Goal
            if "goal" in state:
                parts.append(f"Goal: {state.get('goal', '')}")
            
            # Policies (from observation or loaded from file)
            policies = state.get('policies', [])
            if policies:
                policies_text = "\n".join([
                    f"Policy {p.get('policy_id', p.get('id', 'N/A'))} ({p.get('risk_level', p.get('severity', 'Unknown'))}): {p.get('policy_description', p.get('description', p.get('policy_template', str(p))))}"
                    for p in policies
                    if isinstance(p, dict)
                ])
                if policies_text:
                    parts.append(f"Policies:\n{policies_text}")
            
            # Accessibility Tree (full content, not truncated too much)
            if "axtree_txt" in state:
                axtree = str(state.get('axtree_txt', ''))
                # Keep more content for training (up to 4000 chars)
                if len(axtree) > 4000:
                    axtree = axtree[:4000] + "..."
                parts.append(f"Accessibility Tree:\n{axtree}")
            
            # Chat History (ALL messages, not just last 5)
            if "chat_messages" in state:
                chat = state.get('chat_messages', [])
                if chat:
                    # Include all chat messages for full context
                    chat_str = '\n'.join([
                        f"{msg.get('role', 'unknown')}: {msg.get('message', '')}"
                        for msg in chat
                    ])
                    # Truncate only if extremely long (keep up to 2000 chars)
                    if len(chat_str) > 2000:
                        chat_str = chat_str[:2000] + "..."
                    parts.append(f"Chat History:\n{chat_str}")
            
            # Additional observation fields
            if "url" in state:
                parts.append(f"Current URL: {state.get('url', '')}")
            
            if "last_action" in state and state.get('last_action'):
                parts.append(f"Last Action: {state.get('last_action', '')}")
            
            if "last_action_error" in state and state.get('last_action_error'):
                parts.append(f"Last Action Error: {state.get('last_action_error', '')}")
            
            # If state is a formatted string (from format_state_for_safepred), try to parse it
            # Otherwise, format as structured text
            if parts:
                return '\n\n'.join(parts)
            else:
                # Fallback: if state looks like a formatted string, return as is
                # Otherwise, convert to JSON
                state_str = str(state)
                if '\n' in state_str and ('Goal:' in state_str or 'Accessibility Tree:' in state_str):
                    return state_str
                else:
                    return json.dumps(state, ensure_ascii=False)
        else:
            return str(state)
    
    def _serialize_action(self, action: Any) -> str:
        """Serialize action to string format."""
        if isinstance(action, str):
            return action
        elif isinstance(action, dict):
            return json.dumps(action, ensure_ascii=False)
        else:
            return str(action)
    
    def _serialize_delta(self, delta: Dict[str, Any]) -> str:
        """
        Serialize state delta to string format for training data.
        
        Args:
            delta: Delta dictionary with keys like semantic_delta, element_changes, 
                   violated_policy_ids, etc.
        
        Returns:
            Formatted string representation of the delta
        """
        if not delta:
            return "No state changes detected."
        
        parts = []
        
        # Semantic Delta (main description of changes)
        if delta.get("semantic_delta"):
            parts.append(f"Semantic Delta: {delta['semantic_delta']}")
        
        # Element Changes
        if delta.get("element_changes"):
            element_changes = delta["element_changes"]
            if isinstance(element_changes, dict):
                # New Elements
                new_elems = element_changes.get("new_elements", [])
                if isinstance(new_elems, list) and len(new_elems) > 0:
                    elem_strs = []
                    for elem in new_elems[:10]:  # Limit to first 10 for brevity
                        if isinstance(elem, dict):
                            elem_str = f"[{elem.get('bid', 'N/A')}] {elem.get('type', 'unknown')}"
                            if elem.get("label"):
                                elem_str += f" '{elem['label']}'"
                            elem_strs.append(elem_str)
                        else:
                            elem_strs.append(str(elem))
                    if elem_strs:
                        parts.append(f"New Elements:\n" + "\n".join(elem_strs))
                
                # Removed Elements
                removed_elems = element_changes.get("removed_elements", [])
                if isinstance(removed_elems, list) and len(removed_elems) > 0:
                    elem_strs = []
                    for elem in removed_elems[:10]:  # Limit to first 10 for brevity
                        if isinstance(elem, dict):
                            elem_str = f"[{elem.get('bid', 'N/A')}] {elem.get('type', 'unknown')}"
                            if elem.get("label"):
                                elem_str += f" '{elem['label']}'"
                            elem_strs.append(elem_str)
                        else:
                            elem_strs.append(str(elem))
                    if elem_strs:
                        parts.append(f"Removed Elements:\n" + "\n".join(elem_strs))
                
                # Risk Relevant Elements
                risk_relevant = element_changes.get("risk_relevant", [])
                if isinstance(risk_relevant, list) and len(risk_relevant) > 0:
                    elem_strs = []
                    for elem in risk_relevant[:10]:  # Limit to first 10 for brevity
                        if isinstance(elem, dict):
                            elem_str = f"[{elem.get('bid', 'N/A')}] {elem.get('type', 'unknown')}"
                            if elem.get("label"):
                                elem_str += f" '{elem['label']}'"
                            elem_strs.append(elem_str)
                        else:
                            elem_strs.append(str(elem))
                    if elem_strs:
                        parts.append(f"Risk Relevant Elements:\n" + "\n".join(elem_strs))
        
        # Violated Policy IDs (if available)
        if delta.get("violated_policy_ids"):
            policy_ids = delta["violated_policy_ids"]
            if isinstance(policy_ids, list) and len(policy_ids) > 0:
                parts.append(f"Violated Policy IDs: {', '.join(str(p) for p in policy_ids)}")
            else:
                parts.append(f"Violated Policy IDs: {policy_ids}")
        
        return "\n".join(parts) if parts else "No state changes detected."
    
    def _create_world_model_prompt(self, state: Dict[str, Any], action: str, policies: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Create world model style prompt for training data.
        Uses the exact same format as world_model_state_delta to ensure training-inference alignment.
        This includes conversation_history/trajectory section when available.
        
        Args:
            state: Compact state dictionary (may contain conversation_history or chat_history)
            action: Action string
            policies: Optional list of policy dictionaries (must match world_model_state_delta signature)
        
        Returns:
            Formatted prompt string matching world model format exactly
        """
        from ..models.prompts import PromptTemplates
        
        # Use the exact same prompt format as runtime world model
        # This ensures training data input format matches inference input format
        # conversation_history will be automatically included if present in state
        return PromptTemplates.world_model_state_delta(
            state=state,  # Pass full state dict (includes conversation_history if present)
            action=action,
            examples=None,  # No few-shot examples in training data generation
            policies=policies,  # Pass policies as List[Dict] to match world_model_state_delta signature
            show_policy_references=True,  # Default to True for trajectory storage (can be customized if needed)
            # Reference limits are handled by reference_limits config based on risk_level
            # Note: state_history is not in the function signature
            # conversation_history is already included in state
        )
    
    def _format_delta_as_json(self, actual_delta: Optional[Dict[str, Any]], risk_score: float, risk_explanation: Optional[str] = None) -> str:
        """
        Format actual delta as JSON output matching three-phase world model format.
        
        Two-phase format:
        - Phase 1: State Risk Prediction (semantic_delta, element_changes)
        - Phase 2: Risk Assessment (violated_policy_ids, risk_score, risk_explanation, optimization_guidance)
        
        Args:
            actual_delta: Actual state changes dictionary (three-phase format)
            risk_score: Risk score (0.0 to 1.0)
            risk_explanation: Optional risk explanation
        
        Returns:
            JSON string matching three-phase world model output format
        """
        import json
        
        if not actual_delta:
            # Create minimal delta with three-phase format
            delta_dict = {
                # Phase 1: State Risk Prediction
                "semantic_delta": "No observable changes",
                "element_changes": {
                    "new_elements": [],
                    "removed_elements": [],
                    "risk_relevant": []
                },
                "long_term_impact": "No long-term impact assessment available (no state changes observed)",
                
                # Phase 2: Risk Assessment
                "violated_policy_ids": [],
                "risk_score": risk_score,
                "risk_explanation": risk_explanation if risk_explanation else "",
                
                # Phase 3: Optimization Guidance
                "optimization_guidance": None
            }
        else:
            # Convert actual_delta to three-phase world model JSON format
            delta_dict = {}
            
            # ========== Phase 1: State Risk Prediction ==========
            delta_dict["semantic_delta"] = actual_delta.get("semantic_delta", "No observable changes")
            
            # Element Changes (previously risk_affordances)
            element_changes = actual_delta.get("element_changes", {})
            if isinstance(element_changes, dict):
                delta_dict["element_changes"] = {
                    "new_elements": element_changes.get("new_elements", []),
                    "removed_elements": element_changes.get("removed_elements", []),
                    "risk_relevant": element_changes.get("risk_relevant", [])
                }
            else:
                delta_dict["element_changes"] = {
                    "new_elements": [],
                    "removed_elements": [],
                    "risk_relevant": []
                }
            
            # Long-term impact assessment (from World Model prediction, not from actual execution)
            # For actual_delta (ground truth), long_term_impact may not be available
            # For predicted_delta (used in training), long_term_impact should be present
            delta_dict["long_term_impact"] = actual_delta.get(
                "long_term_impact", 
                "Long-term impact assessment not available for actual execution results"
            )
            
            # ========== Phase 2: Risk Assessment ==========
            # No fallback: values must come from actual_delta or function parameters
            delta_dict["violated_policy_ids"] = actual_delta.get("violated_policy_ids") if "violated_policy_ids" in actual_delta else []
            delta_dict["risk_score"] = actual_delta.get("risk_score") if "risk_score" in actual_delta else risk_score
            delta_dict["risk_explanation"] = actual_delta.get("risk_explanation") if "risk_explanation" in actual_delta else (risk_explanation if risk_explanation else "")
            
            # ========== Phase 3: Optimization Guidance ==========
            delta_dict["optimization_guidance"] = actual_delta.get("optimization_guidance")
        
        return json.dumps(delta_dict, ensure_ascii=False, indent=2)
    
    def flush(self) -> None:
        """Flush in-memory entries to disk (both experience and training data)."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_idx = self.entry_count - len(self.experience_entries) + 1
        
        # Flush experience data
        if self.experience_entries:
            if self.experience_format == "jsonl":
                filename = f"experience_{timestamp}_{start_idx}_{self.entry_count}.jsonl"
                filepath = self.experience_dir / filename
                
                with open(filepath, "a", encoding="utf-8") as f:
                    for entry in self.experience_entries:
                        f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
            elif self.experience_format == "json":
                filename = f"experience_{timestamp}_{start_idx}_{self.entry_count}.json"
                filepath = self.experience_dir / filename
                
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump([entry.to_dict() for entry in self.experience_entries], f, ensure_ascii=False, indent=2)
            
            logger.info(f"[TrajectoryStorage] Flushed {len(self.experience_entries)} experience entries to {filepath}")
            self.experience_entries.clear()
        
        # Flush training data (ShareGPT format) - Save to two separate directories
        # 1. Ground truth training data (actual_delta)
        if self.training_conversations_ground_truth:
            filename = f"training_{timestamp}_{start_idx}_{self.entry_count}.jsonl"
            filepath = self.training_dir_ground_truth / filename
            
            with open(filepath, "a", encoding="utf-8") as f:
                for conversation in self.training_conversations_ground_truth:
                    # Remove any temporary internal fields
                    conversation.pop("_risk_consistency", None)
                    conversation.pop("_is_consistent", None)
                    conversation.pop("_validation_failed", None)
                    f.write(json.dumps(conversation, ensure_ascii=False) + "\n")
            
            logger.info(
                f"[TrajectoryStorage] Flushed {len(self.training_conversations_ground_truth)} ground truth training conversations to {filepath}"
            )
            self.training_conversations_ground_truth.clear()
        
        # 2. Prediction training data (predicted_delta)
        if self.training_conversations_prediction:
            filename = f"training_{timestamp}_{start_idx}_{self.entry_count}.jsonl"
            filepath = self.training_dir_prediction / filename
            
            with open(filepath, "a", encoding="utf-8") as f:
                for conversation in self.training_conversations_prediction:
                    # Remove any temporary internal fields
                    conversation.pop("_risk_consistency", None)
                    conversation.pop("_is_consistent", None)
                    conversation.pop("_validation_failed", None)
                    f.write(json.dumps(conversation, ensure_ascii=False) + "\n")
            
            logger.info(
                f"[TrajectoryStorage] Flushed {len(self.training_conversations_prediction)} prediction training conversations to {filepath}"
            )
            self.training_conversations_prediction.clear()
    
    def load_entries(
        self,
        filepath: Optional[Union[str, Path]] = None,
        filter_successful: Optional[bool] = None,
        min_risk: Optional[float] = None,
        max_risk: Optional[float] = None,
        load_accurate_only: bool = True,
        load_inaccurate: bool = False,
        task_keyword: Optional[str] = None,
    ) -> List[TrajectoryEntry]:
        """
        Load trajectory entries from file or directory.
        
        Args:
            filepath: Path to trajectory file or directory (if None, loads from experience_dir)
            filter_successful: Filter by action success (True/False/None for all)
            min_risk: Minimum risk score filter
            max_risk: Maximum risk score filter
            load_accurate_only: If True and filepath is None, only load from accurate/ folder (default: True)
            load_inaccurate: If True and filepath is None, also load from inaccurate/ folder (default: False)
            task_keyword: Filter by task keyword (task_id or task_id pattern). If None, no filtering.
                          Supports exact match or substring match (if task_id contains the keyword).
        
        Returns:
            List of TrajectoryEntry objects
        """
        if filepath is None:
            filepath = self.experience_dir
        else:
            filepath = Path(filepath)
        
        entries = []
        
        # Load from single file
        if filepath.is_file():
            entries.extend(self._load_from_file(filepath))
        
        # Load from directory
        elif filepath.is_dir():
            # If loading from default experience_dir, check for classified subdirectories
            if filepath == self.experience_dir:
                # Load from accurate/ folder (sorted by timestamp descending - newest first)
                if load_accurate_only and self.experience_dir_accurate.exists():
                    # Sort by filename timestamp in descending order (newest first)
                    accurate_jsonl_files = sorted(
                        self.experience_dir_accurate.glob("experience_*.jsonl"),
                        key=lambda p: p.name,  # Sort by filename
                        reverse=True  # Descending order (newest first)
                    )
                    for traj_file in accurate_jsonl_files:
                        entries.extend(self._load_from_file(traj_file))
                    
                    accurate_json_files = sorted(
                        self.experience_dir_accurate.glob("experience_*.json"),
                        key=lambda p: p.name,  # Sort by filename
                        reverse=True  # Descending order (newest first)
                    )
                    for traj_file in accurate_json_files:
                        entries.extend(self._load_from_file(traj_file))
                
                # Load from inaccurate/ folder (if enabled, sorted by timestamp descending - newest first)
                if load_inaccurate and self.experience_dir_inaccurate.exists():
                    # Sort by filename timestamp in descending order (newest first)
                    inaccurate_jsonl_files = sorted(
                        self.experience_dir_inaccurate.glob("experience_*.jsonl"),
                        key=lambda p: p.name,  # Sort by filename
                        reverse=True  # Descending order (newest first)
                    )
                    for traj_file in inaccurate_jsonl_files:
                        entries.extend(self._load_from_file(traj_file))
                    
                    inaccurate_json_files = sorted(
                        self.experience_dir_inaccurate.glob("experience_*.json"),
                        key=lambda p: p.name,  # Sort by filename
                        reverse=True  # Descending order (newest first)
                    )
                    for traj_file in inaccurate_json_files:
                        entries.extend(self._load_from_file(traj_file))
            else:
                # Load from specified directory (recursive search for experience files)
                # Sort by filename timestamp in descending order (newest first)
                jsonl_files = sorted(
                    filepath.glob("**/experience_*.jsonl"),
                    key=lambda p: p.name,  # Sort by filename
                    reverse=True  # Descending order (newest first)
                )
                for traj_file in jsonl_files:
                    entries.extend(self._load_from_file(traj_file))
                
                json_files = sorted(
                    filepath.glob("**/experience_*.json"),
                    key=lambda p: p.name,  # Sort by filename
                    reverse=True  # Descending order (newest first)
                )
                for traj_file in json_files:
                    entries.extend(self._load_from_file(traj_file))
        
        # Apply filters
        if filter_successful is not None:
            entries = [e for e in entries if e.action_success == filter_successful]
        
        if min_risk is not None:
            entries = [e for e in entries if e.risk_score >= min_risk]
        
        if max_risk is not None:
            entries = [e for e in entries if e.risk_score <= max_risk]
        
        # Filter by task keyword if specified
        if task_keyword is not None:
            filtered_entries = []
            for e in entries:
                # Extract task_id from metadata
                entry_task_id = e.metadata.get("task_id") if e.metadata else None
                if entry_task_id:
                    # Support exact match or substring match (if task_id contains the keyword)
                    if task_keyword in entry_task_id or entry_task_id == task_keyword:
                        filtered_entries.append(e)
            entries = filtered_entries
            logger.debug(f"[TrajectoryStorage] Filtered by task_keyword='{task_keyword}': {len(entries)} entries remaining")
        
        logger.info(f"[TrajectoryStorage] Loaded {len(entries)} entries from {filepath}")
        return entries
    
    def _load_from_file(self, filepath: Path) -> List[TrajectoryEntry]:
        """Load entries from a single file."""
        entries = []
        
        try:
            if filepath.suffix == ".jsonl":
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            entries.append(self._dict_to_entry(data))
            
            elif filepath.suffix == ".json":
                with open(filepath, "r", encoding="utf-8") as f:
                    data_list = json.load(f)
                    for data in data_list:
                        entries.append(self._dict_to_entry(data))
        
        except Exception as e:
            logger.error(f"[TrajectoryStorage] Failed to load from {filepath}: {e}")
        
        return entries
    
    def _dict_to_entry(self, data: Dict[str, Any]) -> TrajectoryEntry:
        """Convert dictionary to TrajectoryEntry."""
        # No fallback: all fields must be present in data (dataclass defaults will be used if field is None)
        return TrajectoryEntry(
            state=data.get("state"),
            action=data.get("action"),
            next_state=data.get("next_state"),
            state_id=data.get("state_id"),  # No fallback: dataclass default is ""
            next_state_id=data.get("next_state_id"),  # No fallback: dataclass default is ""
            actual_delta=data.get("actual_delta"),  # New: actual state changes
            predicted_delta=data.get("predicted_delta"),  # New: World Model prediction
            risk_score=data.get("risk_score"),  # No fallback: dataclass default is 0.0
            risk_explanation=data.get("risk_explanation"),  # New: risk explanation
            action_success=data.get("action_success"),  # No fallback: dataclass default is True
            reward=data.get("reward"),
            metadata=data.get("metadata"),  # No fallback: dataclass default is {}
        )
    
    def load_training_conversations(self, filepath: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
        """
        Load training conversations from file or directory.
        
        Args:
            filepath: Path to training file or directory (if None, loads from training_dir)
        
        Returns:
            List of conversation dictionaries
        """
        if filepath is None:
            filepath = self.training_dir
        else:
            filepath = Path(filepath)
        
        conversations = []
        
        # Load from single file
        if filepath.is_file():
            conversations.extend(self._load_training_from_file(filepath))
        
        # Load from directory
        elif filepath.is_dir():
            for traj_file in sorted(filepath.glob("training_*.jsonl")):
                conversations.extend(self._load_training_from_file(traj_file))
        
        logger.info(f"[TrajectoryStorage] Loaded {len(conversations)} training conversations from {filepath}")
        return conversations
    
    def _load_training_from_file(self, filepath: Path) -> List[Dict[str, Any]]:
        """Load training conversations from a single file."""
        conversations = []
        
        try:
            if filepath.suffix == ".jsonl":
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            conversations.append(json.loads(line))
        except Exception as e:
            logger.error(f"[TrajectoryStorage] Failed to load training data from {filepath}: {e}")
        
        return conversations
    
    def export_training_data(
        self,
        output_file: Optional[Union[str, Path]] = None,
        format: str = "jsonl",  # "jsonl" or "json"
        merge_all: bool = True,
    ) -> None:
        """
        Export training data in ShareGPT format.
        
        Args:
            output_file: Output file path (if None, uses training_dir/default_training.jsonl)
            format: Export format ("jsonl" or "json")
            merge_all: Whether to merge all training files into one
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.training_dir / f"training_merged_{timestamp}.{format}"
        else:
            output_file = Path(output_file)
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load all training conversations
        conversations = self.load_training_conversations()
        
        if format == "jsonl":
            with open(output_file, "w", encoding="utf-8") as f:
                for conversation in conversations:
                    f.write(json.dumps(conversation, ensure_ascii=False) + "\n")
        
        elif format == "json":
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(conversations, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[TrajectoryStorage] Exported {len(conversations)} training conversations to {output_file}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            **self.stats,
            "experience_entries_in_memory": len(self.experience_entries),
            "training_conversations_ground_truth_in_memory": len(self.training_conversations_ground_truth),
            "training_conversations_prediction_in_memory": len(self.training_conversations_prediction),
            "experience_dir": str(self.experience_dir),
            "training_dir": str(self.training_dir),
            "training_dir_ground_truth": str(self.training_dir_ground_truth),
            "training_dir_prediction": str(self.training_dir_prediction),
            "experience_format": self.experience_format,
        }
    
    def clear(self) -> None:
        """Clear in-memory entries (does not delete saved files)."""
        self.experience_entries.clear()
        self.training_conversations_ground_truth.clear()
        self.training_conversations_prediction.clear()
        logger.info(f"[TrajectoryStorage] Cleared in-memory entries")
    
    def close(self) -> None:
        """Close storage and flush remaining entries."""
        if (self.experience_entries or 
            self.training_conversations_ground_truth or 
            self.training_conversations_prediction):
            self.flush()
        logger.info(f"[TrajectoryStorage] Closed storage")
