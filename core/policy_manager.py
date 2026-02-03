"""
Policy Manager Module for SafePred_v3.

Manages dynamic policy updates with reference examples.
Supports FIFO reference management and similarity detection.
"""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from difflib import SequenceMatcher
import threading

from ..utils.logger import get_logger

logger = get_logger("SafePred.PolicyManager")


class PolicyManager:
    """
    Manages policy loading, updating, and reference management.
    
    Features:
    - Dynamic policy updates with reference examples
    - FIFO reference management (prevents unlimited growth)
    - Similarity detection (prevents duplicate references)
    - Thread-safe operations
    """
    
    def __init__(
        self,
        policy_file_path: str,
        enable_cache: bool = True,
        update_mode: str = "async",
        similarity_threshold: float = 0.85,
        reference_limits: Optional[Dict[str, int]] = None
    ):
        """
        Initialize PolicyManager.
        
        Args:
            policy_file_path: Path to policy JSON file
            enable_cache: Whether to cache policies in memory
            update_mode: "sync" or "async" (async recommended)
            similarity_threshold: Threshold for duplicate detection (0.0-1.0)
            reference_limits: Dict mapping risk_level to max references (e.g., {"high": 10, "medium": 7, "low": 5})
                             If None, uses default values
        """
        self.policy_file_path = Path(policy_file_path)
        self.enable_cache = enable_cache
        self.update_mode = update_mode
        self.similarity_threshold = similarity_threshold
        self._lock = threading.Lock()
        
        # Set reference limits based on risk level
        if reference_limits is None:
            reference_limits = {"high": 10, "medium": 7, "low": 5}
        self.reference_limits = reference_limits
        
        # Load policies
        self.policies = self._load_policies()
        
        logger.info(f"PolicyManager initialized with {len(self.policies)} policies, reference_limits={reference_limits}")
    
    def _load_policies(self) -> List[Dict[str, Any]]:
        """Load policies from JSON file."""
        if not self.policy_file_path.exists():
            logger.warning(f"Policy file not found: {self.policy_file_path}")
            return []
        
        try:
            with open(self.policy_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, dict) and "policies" in data:
                policies = data["policies"]
            elif isinstance(data, list):
                policies = data
            else:
                logger.error(f"Invalid policy file format: {self.policy_file_path}")
                return []
            
            # Ensure each policy has a reference field
            for policy in policies:
                if "reference" not in policy:
                    policy["reference"] = []
                # Ensure reference is a list
                if not isinstance(policy.get("reference"), list):
                    policy["reference"] = []
            
            logger.info(f"Loaded {len(policies)} policies from {self.policy_file_path}")
            return policies
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in policy file: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading policies: {e}")
            return []
    
    def get_policy(self, policy_id: str) -> Optional[Dict[str, Any]]:
        """Get policy by ID."""
        for policy in self.policies:
            # Support different ID field names
            pid = policy.get("policy_id") or policy.get("id") or policy.get("policyId")
            if pid == policy_id:
                return policy
        return None
    
    def get_all_policies(self) -> List[Dict[str, Any]]:
        """Get all policies."""
        return self.policies.copy()
    
    def update_policy_references(
        self,
        policy_ids: List[str],
        violation_context: Dict[str, Any]
    ) -> bool:
        """
        Update policy references with violation context.
        
        Args:
            policy_ids: List of violated policy IDs
            violation_context: Dict containing:
                - task: Task objective
                - thought: Agent reasoning (optional)
                - action: Action taken (optional)
                - violation_description: Description of violation
        
        Returns:
            True if successful, False otherwise
        """
        if not policy_ids:
            return False
        
        # Build reference string
        reference = self._build_reference(violation_context)
        
        updated = False
        for policy_id in policy_ids:
            policy = self.get_policy(policy_id)
            if not policy:
                logger.warning(f"Policy {policy_id} not found, skipping update")
                continue
            
            # Check for duplicates
            if self._is_duplicate(policy, reference):
                logger.debug(f"Duplicate reference detected for policy {policy_id}, skipping")
                continue
            
            # Update reference using FIFO
            references = policy.get("reference", [])
            limit = self._get_reference_limit(policy)
            
            # Remove oldest if at limit
            if len(references) >= limit:
                references.pop(0)
            
            references.append(reference)
            policy["reference"] = references
            updated = True
            
            logger.info(f"Updated policy {policy_id} with new reference (total: {len(references)})")
        
        if updated:
            if self.update_mode == "sync":
                self._save_policies()
            else:
                # Async mode: save in background thread
                threading.Thread(target=self._save_policies, daemon=True).start()
        
        return updated
    
    def _build_reference(self, violation_context: Dict[str, Any]) -> str:
        """Build reference string from violation context (optimized to avoid redundancy)."""
        parts = []
        
        # Task (required)
        if "task" in violation_context:
            parts.append(f"Task: {violation_context['task']}")
        
        # Thought (save full response without truncation)
        thought = violation_context.get("thought", "")
        if thought:
            # Save complete thought/reasoning without truncation
            # This preserves the full agent response including all reasoning steps
            thought_str = str(thought)
            parts.append(f"Thought: {thought_str}")
        
        # Action (simplified - only key fields)
        action_desc = None
        if "action" in violation_context:
            action = violation_context["action"]
            # If action is a dict with simplified fields
            if isinstance(action, dict):
                action_type = action.get("action_type", "")
                element_id = action.get("element_id", "")
                
                # Format action_type
                action_type_str = str(action_type)
                if "." in action_type_str:
                    action_type_str = action_type_str.split(".")[-1].rstrip(">")
                
                # Build concise action description
                if element_id:
                    action_desc = f"{action_type_str} [{element_id}]"
                else:
                    action_desc = action_type_str
            else:
                # If action is a string (e.g., pyautogui code), use it directly
                # Don't truncate - keep full action for reference
                action_str = str(action)
                action_desc = action_str
        
        if action_desc:
            parts.append(f"Action: {action_desc}")
        
        # Violation Description (REQUIRED - must not be empty)
        violation_desc = violation_context.get("violation_description", "")
        if not violation_desc or not violation_desc.strip():
            # CRITICAL: violation_description is required for reference examples
            error_msg = (
                f"[PolicyManager] CRITICAL: violation_description is missing or empty in violation_context. "
                f"This is required for saving policy reference examples. "
                f"Available keys in context: {list(violation_context.keys())}, "
                f"Context content: {str(violation_context)[:500]}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        parts.append(f"Violation Description: {violation_desc}")
        
        return "\n".join(parts)
    
    def _get_reference_limit(self, policy: Dict[str, Any]) -> int:
        """Get reference limit based on risk level."""
        risk_level = policy.get("risk_level", "medium")
        if isinstance(risk_level, str):
            risk_level = risk_level.lower()
        
        # Get limit from config, default to "medium" if risk_level not found
        return self.reference_limits.get(risk_level, self.reference_limits.get("medium", 7))
    
    def _is_duplicate(self, policy: Dict[str, Any], new_reference: str) -> bool:
        """Check if new reference is too similar to existing ones."""
        existing_refs = policy.get("reference", [])
        for ref in existing_refs:
            similarity = SequenceMatcher(None, ref, new_reference).ratio()
            if similarity >= self.similarity_threshold:
                return True
        return False
    
    def _save_policies(self) -> None:
        """Save policies to file (thread-safe)."""
        with self._lock:
            try:
                # Save to temporary file first (atomic write)
                temp_path = self.policy_file_path.with_suffix(
                    self.policy_file_path.suffix + ".tmp"
                )
                
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(self.policies, f, indent=2, ensure_ascii=False)
                
                # Atomic rename
                temp_path.replace(self.policy_file_path)
                
                logger.debug(f"Policies saved to {self.policy_file_path}")
                
            except Exception as e:
                logger.error(f"Error saving policies: {e}")
                raise
    
    def clear_policy_references(self, policy_id: str) -> bool:
        """Clear all references for a policy (for testing/reset)."""
        policy = self.get_policy(policy_id)
        if not policy:
            return False
        
        policy["reference"] = []
        self._save_policies()
        logger.info(f"Cleared references for policy {policy_id}")
        return True

