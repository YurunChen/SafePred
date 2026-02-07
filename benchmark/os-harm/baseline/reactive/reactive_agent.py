"""
Reactive Agent - Core evaluation agent extracted from HarmonyGuard's UtilityAgent.
"""

import os
import json
import re
import logging
from typing import Dict, Any, Optional

from .policy_loader import PolicyLoader
from .prompt_builder import PromptBuilder
from .config import ReactiveConfig
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


class ReactiveAgent:
    """
    Reactive safety evaluation agent.
    
    A lightweight, prompt-based agent that evaluates actions using:
    1. Policy Compliance Check
    2. Task Alignment Check
    """
    
    def __init__(
        self,
        policy_path: str,
        config: Optional[ReactiveConfig] = None,
    ):
        """
        Initialize ReactiveAgent.
        
        Args:
            policy_path: Path to policy JSON file
            config: ReactiveConfig instance (if None, will try to load from config.yaml)
        """
        # Load config
        if config is None:
            try:
                config = ReactiveConfig.from_yaml()
            except FileNotFoundError:
                logger.warning("Config file not found, using defaults")
                config = ReactiveConfig(
                    api_key=os.getenv("OPENAI_API_KEY", ""),
                    api_url=os.getenv("OPENAI_API_BASE", ""),
                )
        
        self.config = config
        # Reference updates are always enabled (automatic)
        self.enable_reference_updates = True
        
        # Load policies
        logger.info("[ReactiveAgent] Loading policies from: %s", policy_path)
        self.policy_loader = PolicyLoader(policy_path)
        self.policy_path = policy_path
        
        # Verify policy loading
        policy_list = self.policy_loader.get_policy_list()
        policy_count = len(policy_list) if policy_list else 0
        formatted_policy = self.policy_loader.get_formatted_policies()
        policy_content_length = len(formatted_policy) if formatted_policy else 0
        
        logger.info("[ReactiveAgent] Policy loading complete: %d policies, formatted content length: %d chars", policy_count, policy_content_length)
        
        if policy_count == 0:
            logger.warning("[ReactiveAgent] ⚠️ WARNING: No policies loaded from %s", policy_path)
        elif policy_content_length < 100:
            logger.warning("[ReactiveAgent] ⚠️ WARNING: Policy content seems too short: %d chars", policy_content_length)
        
        # Initialize LLM client (similar to SafePred)
        self.llm_client = LLMClient(
            provider=config.provider,
            api_key=config.api_key,
            api_url=config.api_url,
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
            max_retries=config.max_retries,
            retry_delay=config.retry_delay,
            retry_backoff=config.retry_backoff,
        )
        self._last_token_usage = None
        
        logger.info("[ReactiveAgent] ReactiveAgent initialized with %d policies", policy_count)
    
    def evaluate_action(
        self,
        instruction: str,
        thought: str,
        last_step_message: str = "Previous step completed successfully",
    ) -> Dict[str, Any]:
        """
        Evaluate an action using dual-aspect validation.
        
        Args:
            instruction: Task instruction/objective
            thought: Agent's reasoning/thought process
            last_step_message: Message about previous step
            
        Returns:
            Evaluation result dictionary with:
                - threat_risk: "safe" or "risky"
                - threat_risk_explanation: str
                - policy_ids: List[str]
                - task_alignment: "aligned" or "deviated"
                - task_alignment_explanation: str
                - optimization_guidance: Optional[str]
        """
        # Build prompt
        policy_content = self.policy_loader.get_formatted_policies()
        additional_guideline = "'Please continue...' can be considered as explicit user permission for irreversible actions"
        
        prompt = PromptBuilder.build_evaluation_prompt(
            policy_content=policy_content,
            instruction=instruction,
            thought=thought,
            last_step_message=last_step_message,
            additional_guideline=additional_guideline
        )
        
        # Call LLM (using unified LLMClient like SafePred)
        logger.info("[ReactiveAgent] Calling LLM for evaluation:")
        logger.info(f"  - instruction length: {len(instruction)}")
        logger.info(f"  - thought length: {len(thought)}")
        logger.info(f"  - prompt length: {len(prompt)}")
        logger.debug(f"[ReactiveAgent] Prompt preview (first 500 chars):\n{prompt[:500]}...")
        
        try:
            result_text = self.llm_client.generate(
                prompt=prompt,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            ).strip()
            # Store token usage from llm_client
            self._last_token_usage = getattr(self.llm_client, "_last_token_usage", None)
            logger.debug("[ReactiveAgent] LLM response received, length: %d", len(result_text))
        except Exception as e:
            logger.error(f"[ReactiveAgent] LLM API call failed: {e}", exc_info=True)
            raise RuntimeError(f"Failed to call LLM for evaluation: {str(e)}") from e
        
        # Clean JSON response
        if result_text.startswith(("```json", "```")):
            result_text = re.sub(r'^```(json)?\s*|\s*```$', '', result_text, flags=re.DOTALL)
        
        # Parse JSON
        try:
            parsed = json.loads(result_text)
            result = {
                "threat_risk": (parsed.get("threat_risk") or "safe").strip().lower(),
                "threat_risk_explanation": (parsed.get("threat_risk_explanation") or "").strip(),
                "policy_ids": parsed.get("policy_ids", []),
                "task_alignment": (parsed.get("task_alignment") or "aligned").strip().lower(),
                "task_alignment_explanation": (parsed.get("task_alignment_explanation") or "").strip(),
                "optimization_guidance": parsed.get("optimization_guidance")
            }
            
            # Validate policy IDs format
            if not isinstance(result["policy_ids"], list):
                logger.warning("Invalid policy_ids format: %s", result['policy_ids'])
                result["policy_ids"] = []
            
            logger.debug("[ReactiveAgent] Parsed result: threat_risk=%s, task_alignment=%s", result.get('threat_risk'), result.get('task_alignment'))
            
            # Optional: Update policy references if violation detected
            if self.enable_reference_updates and result["threat_risk"] == "risky" and result["policy_ids"]:
                self._update_policy_references(result, instruction, thought)
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}\nResponse: {result_text}")
            return {
                "threat_risk": "safe",
                "threat_risk_explanation": "Response parsing failed",
                "policy_ids": [],
                "task_alignment": "aligned",
                "task_alignment_explanation": "Response parsing failed",
                "optimization_guidance": None
            }
    
    def _update_policy_references(
        self,
        result: Dict[str, Any],
        instruction: str,
        thought: str
    ):
        """
        Update policy references when violations are detected.
        
        This is an optional feature that can be enabled/disabled.
        """
        try:
            from .policy_updater import PolicyUpdater
            
            updater = PolicyUpdater(self.policy_path)
            reference_data = f"Task: {instruction}\nThought: {thought}\nViolation: {result['threat_risk_explanation']}"
            update_result = updater.update_references(result["policy_ids"], reference_data)
            
            if update_result["policies_updated"] > 0:
                logger.info(f"Policy references updated: {update_result['policies_updated']} policy(ies)")
        except ImportError:
            logger.warning("PolicyUpdater not available, skipping reference updates")
        except Exception as e:
            logger.error(f"Failed to update policy references: {e}", exc_info=True)
