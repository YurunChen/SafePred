"""
Optional policy reference updater.
Extracted from HarmonyGuard's PolicyAgent_Update.
"""

import json
import logging
from typing import List, Dict, Any
from difflib import SequenceMatcher

from .policy_loader import read_security_policy_categories

logger = logging.getLogger(__name__)


def is_similar(text1: str, text2: str, threshold: float = 0.85) -> bool:
    """Check if two texts are similar."""
    return SequenceMatcher(None, text1.strip(), text2.strip()).ratio() >= threshold


def get_reference_limit(policy: Dict[str, Any]) -> int:
    """Get reference limit based on risk level."""
    risk_level = policy.get("risk_level", "")
    if "High" in str(risk_level):
        return 10
    if "low" in str(risk_level).lower():
        return 5
    return 7


class PolicyUpdater:
    """Update policy references when violations are detected."""
    
    def __init__(self, policy_path: str):
        """
        Initialize policy updater.
        
        Args:
            policy_path: Path to policy JSON file
        """
        self.policy_path = policy_path
    
    def update_references(
        self,
        policy_ids: List[str],
        reference_example: str
    ) -> Dict[str, Any]:
        """
        Update policy references.
        
        Args:
            policy_ids: List of policy IDs to update
            reference_example: Reference example to add
            
        Returns:
            Update result dictionary
        """
        update_results = {
            "total_policies_loaded": 0,
            "policies_updated": 0,
            "policies_skipped": 0,
            "policies_not_found": 0,
            "details": []
        }
        
        try:
            policies = read_security_policy_categories(self.policy_path)
            update_results["total_policies_loaded"] = len(policies)
        except Exception as e:
            logger.error(f"Failed to load policy file: {e}")
            return update_results
        
        if not policy_ids:
            return update_results
        
        for policy_id in policy_ids:
            policy = next(
                (p for p in policies if str(p.get("policy_id")) == str(policy_id)),
                None
            )
            
            if not policy:
                update_results["policies_not_found"] += 1
                update_results["details"].append({
                    "policy_id": policy_id,
                    "status": "not_found",
                    "message": f"Policy ID {policy_id} not found"
                })
                continue
            
            references = policy.get("reference", [])
            limit = get_reference_limit(policy)
            
            # Check if reference is too similar to existing ones
            if any(is_similar(reference_example, existing) for existing in references):
                update_results["policies_skipped"] += 1
                update_results["details"].append({
                    "policy_id": policy_id,
                    "status": "skipped",
                    "message": "Reference too similar to existing ones"
                })
                continue
            
            # Remove oldest reference if limit reached
            if len(references) >= limit:
                references.pop(0)
            
            # Add new reference
            references.append(reference_example)
            policy["reference"] = references
            update_results["policies_updated"] += 1
            update_results["details"].append({
                "policy_id": policy_id,
                "status": "updated",
                "message": f"Added new reference, total references: {len(references)}"
            })
        
        # Save updated policies
        if update_results["policies_updated"] > 0:
            try:
                with open(self.policy_path, "w", encoding="utf-8") as f:
                    json.dump(policies, f, indent=2, ensure_ascii=False)
                logger.info(f"Updated {update_results['policies_updated']} policy references")
            except Exception as e:
                logger.error(f"Failed to write policy file: {e}")
        
        return update_results
