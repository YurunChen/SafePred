"""
HarmonyGuard Wrapper for OS-Harm Integration.

This module integrates HarmonyGuard's UtilityAgent for safety evaluation.
It provides a compatible interface with SafePred's SafetyWrapper.
"""

import os
import sys
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable

logger = logging.getLogger(__name__)

# Try to import HarmonyGuard components
HARMONYGUARD_AVAILABLE = None
UtilityAgent = None

try:
    # Add HarmonyGuard to path
    harmonyguard_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "baseline",
        "HarmonyGuard"
    )
    if os.path.exists(harmonyguard_path):
        sys.path.insert(0, harmonyguard_path)
        from harmony_agents.utility_agent import UtilityAgent
        HARMONYGUARD_AVAILABLE = True
        logger.info("HarmonyGuard UtilityAgent imported successfully")
    else:
        HARMONYGUARD_AVAILABLE = False
        logger.warning(f"HarmonyGuard path not found: {harmonyguard_path}")
except ImportError as e:
    HARMONYGUARD_AVAILABLE = False
    logger.warning(f"Failed to import HarmonyGuard UtilityAgent: {e}")


class HarmonyGuardWrapper:
    """
    Wrapper for HarmonyGuard UtilityAgent integration with OS-Harm.
    
    Provides a compatible interface with SafePred's SafetyWrapper.
    Uses HarmonyGuard's UtilityAgent for dual-aspect validation:
    1. Policy Compliance Check
    2. Alignment Check
    """
    
    def __init__(
        self,
        enabled: bool = True,
        risk_cat_path: Optional[str] = None,
        harmonyguard_config_path: Optional[str] = None,
        max_regeneration_attempts: int = 2,
    ):
        """
        Initialize HarmonyGuard Wrapper.
        
        Args:
            enabled: Whether to enable safety checks
            risk_cat_path: Path to HarmonyGuard risk category/policy file.
                          If None, will try to read from HarmonyGuard config.yaml
            harmonyguard_config_path: Path to HarmonyGuard config.yaml file.
                                    If None, uses default config in HarmonyGuard directory
            max_regeneration_attempts: Maximum number of regeneration attempts when all actions are filtered
        """
        self.enabled = enabled and HARMONYGUARD_AVAILABLE
        self.max_regeneration_attempts = max_regeneration_attempts
        self.utility_agent = None
        self.planning_enabled = False  # HarmonyGuard doesn't support planning feature
        self._step_utility_agent_token_usages = []
        self._last_step_utility_agent_token_usage = None
        
        if not HARMONYGUARD_AVAILABLE:
            logger.warning("HarmonyGuard not available. Safety checks disabled.")
            return
        
        if not self.enabled:
            logger.info("HarmonyGuard safety checks disabled by configuration.")
            return
        
        try:
            # Determine config path
            if harmonyguard_config_path and os.path.exists(harmonyguard_config_path):
                config_path = harmonyguard_config_path
            else:
                # Try to find config in HarmonyGuard directory
                default_config_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "baseline",
                    "HarmonyGuard",
                    "config.yaml"
                )
                if os.path.exists(default_config_path):
                    config_path = default_config_path
                else:
                    config_path = None  # UtilityAgent will use default
            
            # Change to HarmonyGuard directory if config exists (for relative paths)
            original_cwd = os.getcwd()
            if config_path:
                harmonyguard_dir = os.path.dirname(config_path)
                os.chdir(harmonyguard_dir)
                logger.info(f"[HarmonyGuardWrapper] Changed working directory to: {harmonyguard_dir}")
            
            try:
                # Initialize UtilityAgent
                # If risk_cat_path is provided, use it; otherwise UtilityAgent will read from config
                logger.info(f"[HarmonyGuardWrapper] Initializing UtilityAgent with risk_cat_path={risk_cat_path}")
                self.utility_agent = UtilityAgent(risk_cat_path=risk_cat_path)
                logger.info(f"[HarmonyGuardWrapper] ✅ HarmonyGuard initialized successfully (risk_cat_path={risk_cat_path}, utility_agent={self.utility_agent is not None})")
            except Exception as e:
                logger.error(f"[HarmonyGuardWrapper] ❌ Failed to initialize UtilityAgent: {e}", exc_info=True)
                self.enabled = False
                self.utility_agent = None
                raise RuntimeError(f"Failed to initialize HarmonyGuard UtilityAgent: {str(e)}") from e
            finally:
                # Restore original working directory
                os.chdir(original_cwd)
            
        except Exception as e:
            logger.error(f"[HarmonyGuardWrapper] ❌ Failed to initialize HarmonyGuard wrapper: {e}", exc_info=True)
            self.enabled = False
            self.utility_agent = None
            raise RuntimeError(f"Failed to initialize HarmonyGuard safety wrapper: {str(e)}") from e
    
    def _aggregate_step_utility_agent_token_usage(self) -> Optional[Dict[str, Any]]:
        """Sum token usage from all UtilityAgent calls in this filter_actions step."""
        usages = getattr(self, "_step_utility_agent_token_usages", []) or []
        if not usages:
            return None
        model = usages[0].get("model", "harmonyguard_utility_agent") if usages else "harmonyguard_utility_agent"
        return {
            "model": model,
            "prompt_tokens": sum(u.get("prompt_tokens", 0) or 0 for u in usages),
            "completion_tokens": sum(u.get("completion_tokens", 0) or 0 for u in usages),
            "total_tokens": sum(u.get("total_tokens", 0) or 0 for u in usages),
        }

    def evaluate_action_risk(
        self,
        obs: Dict[str, Any],
        action: Any,
        instruction: str,
        candidate_actions: Optional[List[Any]] = None,
        plan_text: Optional[str] = None,  # Not used by HarmonyGuard
        current_response: Optional[str] = None,  # Full agent response including reasoning
    ) -> Dict[str, Any]:
        """
        Evaluate risk of an action using HarmonyGuard UtilityAgent.
        
        Args:
            obs: Current observation (not used by HarmonyGuard, but kept for compatibility)
            action: Action to evaluate
            instruction: Current task instruction
            candidate_actions: Optional list of candidate actions (not used, HarmonyGuard evaluates one at a time)
            plan_text: Optional plan text (not used by HarmonyGuard)
            current_response: Full agent response including reasoning (used as "thought")
        
        Returns:
            Dict with:
                - 'risk_score': float (0.0 if safe, 1.0 if risky)
                - 'risk_explanation': str
                - 'is_safe': bool
                - 'safe_action': Optional[Any] (original action if safe, None if unsafe)
                - 'requires_regeneration': bool
                - 'task_alignment': str ("aligned" or "deviated")
                - 'task_alignment_explanation': str
                - 'optimization_guidance': str (guidance for regeneration if needed)
        """
        if not self.enabled or not self.utility_agent:
            return {
                'risk_score': 0.0,
                'risk_explanation': 'HarmonyGuard safety checks disabled',
                'is_safe': True,
                'safe_action': action,
                'requires_regeneration': False,
                'task_alignment': 'aligned',
                'task_alignment_explanation': 'Safety checks disabled',
                'optimization_guidance': None,
            }
        
        try:
            # Extract thought from current_response
            thought = current_response if current_response else str(action)
            logger.debug(f"[HarmonyGuard] evaluate_action_risk: action={str(action)[:100]}..., current_response={'provided' if current_response else 'None'}, thought_length={len(thought) if thought else 0}")
            
            # Build trajectory (simplified - HarmonyGuard may need more context)
            # For now, use empty trajectory as HarmonyGuard's thought_aligner_response
            # accepts trajectory but we don't have full trajectory history here
            trajectory = []
            last_step_message = "Previous step completed successfully"
            
            logger.info(f"[HarmonyGuard] Calling utility_agent.thought_aligner_response with instruction length={len(instruction)}, thought length={len(thought)}")
            
            # Call HarmonyGuard's thought_aligner_response
            try:
                result = self.utility_agent.thought_aligner_response(
                    instruction=instruction,
                    thought=thought,
                    trajectory=trajectory,
                    last_step_message=last_step_message
                )
                logger.debug(f"[HarmonyGuard] utility_agent.thought_aligner_response returned: threat_risk={result.get('threat_risk')}, task_alignment={result.get('task_alignment')}")
            except Exception as e:
                logger.error(f"[HarmonyGuard] Error calling utility_agent.thought_aligner_response: {e}", exc_info=True)
                raise RuntimeError(f"UtilityAgent.thought_aligner_response failed: {str(e)}") from e
            
            # Log Utility Agent evaluation result in JSON format
            evaluation_result = {
                "instruction": instruction[:100] + "..." if len(instruction) > 100 else instruction,
                "threat_risk": result.get('threat_risk', 'unknown'),
                "threat_risk_explanation": result.get('threat_risk_explanation', ''),
                "task_alignment": result.get('task_alignment', 'unknown'),
                "task_alignment_explanation": result.get('task_alignment_explanation', ''),
                "policy_ids": result.get('policy_ids', []),
                "optimization_guidance": result.get('optimization_guidance', 'None')
            }
            logger.info("=" * 80)
            logger.info("[HarmonyGuard] Utility Agent Evaluation Result:")
            logger.info(json.dumps(evaluation_result, indent=2, ensure_ascii=False))
            logger.info("=" * 80)
            
            # Convert HarmonyGuard result to SafePred-compatible format
            threat_risk = result.get('threat_risk', 'safe').strip().lower()
            task_alignment = result.get('task_alignment', 'aligned').strip().lower()
            
            # Determine if action is safe
            # Action is unsafe if threat_risk is "risky" OR task_alignment is "deviated"
            is_safe = (threat_risk == 'safe') and (task_alignment == 'aligned')
            
            # Convert to risk_score (0.0 = safe, 1.0 = risky)
            risk_score = 0.0 if is_safe else 1.0
            
            # Determine if regeneration is required
            requires_regeneration = not is_safe
            
            # Build optimization guidance
            optimization_guidance = result.get('optimization_guidance')
            if not optimization_guidance:
                # Build guidance from threat_risk and task_alignment explanations
                guidance_parts = []
                if threat_risk == 'risky':
                    guidance_parts.append(f"Threat risk: {result.get('threat_risk_explanation', '')}")
                if task_alignment == 'deviated':
                    guidance_parts.append(f"Task alignment: {result.get('task_alignment_explanation', '')}")
                optimization_guidance = "\n".join(guidance_parts) if guidance_parts else None
            
            return {
                'risk_score': risk_score,
                'risk_explanation': result.get('threat_risk_explanation', ''),
                'is_safe': is_safe,
                'safe_action': action if is_safe else None,
                'requires_regeneration': requires_regeneration,
                'risk_guidance': optimization_guidance or '',
                'task_alignment': task_alignment,
                'task_alignment_explanation': result.get('task_alignment_explanation', ''),
                'optimization_guidance': optimization_guidance,
                'policy_ids': result.get('policy_ids', []),  # Include policy IDs for compatibility
            }
            
        except Exception as e:
            logger.error(f"[HarmonyGuard] ❌ Error evaluating action risk with HarmonyGuard: {e}", exc_info=True)
            # Re-raise the exception instead of returning safe by default
            # This ensures errors are visible and not silently ignored
            raise RuntimeError(f"HarmonyGuard evaluation failed: {str(e)}") from e
    
    def filter_actions(
        self,
        obs: Dict[str, Any],
        actions: List[Any],
        instruction: str,
        action_generator: Optional[Callable] = None,
        max_regeneration_attempts: Optional[int] = None,
        plan_text: Optional[str] = None,  # Not used by HarmonyGuard
        current_response: Optional[str] = None,  # Full agent response including reasoning
        plan_update_callback: Optional[Callable] = None,  # Not used by HarmonyGuard
    ) -> Tuple[List[Any], Dict[str, Any], Optional[str], Optional[str]]:
        """
        Filter actions based on HarmonyGuard's dual-aspect validation.
        
        Process:
        1. Evaluate each action using HarmonyGuard (policy compliance + task alignment)
        2. Filter unsafe actions (risky OR deviated)
        3. If all actions filtered, generate guidance and call action_generator
        
        Args:
            obs: Current observation
            actions: List of actions to filter
            instruction: Current task instruction
            action_generator: Callback function(state, risk_guidance, num_samples) -> (actions, response)
                            Required for regeneration when all actions are filtered
            max_regeneration_attempts: Maximum number of regeneration attempts (default: uses self.max_regeneration_attempts)
            plan_text: Optional plan text (not used by HarmonyGuard)
            current_response: Full agent response including reasoning
            plan_update_callback: Optional callback (not used by HarmonyGuard)
        
        Returns:
            Tuple of (filtered_actions, safety_info, risk_guidance, regeneration_response)
            - filtered_actions: List of safe actions (empty if all filtered and no regeneration)
            - safety_info: Risk scores and explanations for each action
            - risk_guidance: Guidance for regeneration if all actions filtered (None otherwise)
            - regeneration_response: Full response from regeneration if occurred (None otherwise)
        """
        # Check if HarmonyGuard is enabled and utility_agent is initialized
        logger.info(f"[HarmonyGuard] filter_actions entry: enabled={self.enabled}, utility_agent={self.utility_agent is not None}")
        if not self.enabled:
            logger.warning("[HarmonyGuard] ❌ filter_actions called but HarmonyGuard is disabled (self.enabled=False)")
            self._last_step_utility_agent_token_usage = None
            return actions, {}, None, None
        if not self.utility_agent:
            logger.warning("[HarmonyGuard] ❌ filter_actions called but utility_agent is None")
            self._last_step_utility_agent_token_usage = None
            return actions, {}, None, None

        self._step_utility_agent_token_usages = []
        self._last_step_utility_agent_token_usage = None
        
        logger.info(f"[HarmonyGuard] ✅ filter_actions called with {len(actions)} action(s), instruction: {instruction[:100]}...")
        
        # Use instance max_regeneration_attempts if not provided
        if max_regeneration_attempts is None:
            max_regeneration_attempts = self.max_regeneration_attempts
        
        # Evaluate each action and filter unsafe ones
        filtered_actions = []
        safety_info = {}
        filtered_explanations = []
        
        for i, action in enumerate(actions):
            logger.info(f"[HarmonyGuard] Evaluating action {i+1}/{len(actions)}: {str(action)[:100]}...")
            # Evaluate risk using HarmonyGuard
            try:
                risk_result = self.evaluate_action_risk(
                    obs=obs,
                    action=action,
                    instruction=instruction,
                    candidate_actions=[action],
                    plan_text=plan_text,
                    current_response=current_response,
                )
            except Exception as e:
                logger.error(f"[HarmonyGuard] Error in evaluate_action_risk for action {i+1}: {e}", exc_info=True)
                raise RuntimeError(f"evaluate_action_risk failed for action {i+1}: {str(e)}") from e

            u = getattr(self.utility_agent, "_last_token_usage", None)
            if u and isinstance(u, dict):
                self._step_utility_agent_token_usages.append(u)
            
            risk_score = risk_result['risk_score']
            is_safe = risk_result['is_safe']
            task_alignment = risk_result.get('task_alignment', 'aligned')
            
            # Store safety info
            safety_info[f'action_{i}'] = {
                'risk_score': risk_score,
                'risk_explanation': risk_result['risk_explanation'],
                'is_safe': is_safe,
                'task_alignment': task_alignment,
                'task_alignment_explanation': risk_result.get('task_alignment_explanation', ''),
            }
            
            # Filter unsafe actions
            if is_safe:
                filtered_actions.append(action)
                logger.debug(f"[HarmonyGuard] ✅ Action {i} passed safety check: risk_score={risk_score:.3f}, task_alignment={task_alignment}")
            else:
                logger.info(f"[HarmonyGuard] ❌ Action {i} filtered: risk_score={risk_score:.3f}, task_alignment={task_alignment}")
                logger.info(f"  Action: {str(action)[:200]}...")
                logger.info(f"  Threat Risk Explanation: {risk_result.get('risk_explanation', '')[:200]}...")
                logger.info(f"  Task Alignment Explanation: {risk_result.get('task_alignment_explanation', '')[:200]}...")
                # Store filtered action info for potential regeneration
                filtered_explanations.append({
                    'action': str(action),
                    'risk_score': risk_score,
                    'explanation': risk_result['risk_explanation'],
                    'task_alignment': task_alignment,
                    'task_alignment_explanation': risk_result.get('task_alignment_explanation', ''),
                    'optimization_guidance': risk_result.get('optimization_guidance', ''),
                })

        self._last_step_utility_agent_token_usage = self._aggregate_step_utility_agent_token_usage()
        
        # If all actions filtered and action_generator provided, try regeneration
        risk_guidance = None
        if not filtered_actions and actions and action_generator is not None:
            logger.info("=" * 80)
            logger.info("[HarmonyGuard] ⚠️ All actions filtered, starting regeneration process...")
            logger.info(f"  Total actions filtered: {len(actions)}")
            logger.info("=" * 80)
            # Build risk_guidance from filtered explanations
            def build_risk_guidance(current_filtered_explanations):
                """Build risk_guidance from current filtered explanations."""
                all_risk_explanations = []
                all_task_alignment_explanations = []
                all_optimization_guidance = []
                
                for exp in current_filtered_explanations:
                    if exp.get('explanation'):
                        all_risk_explanations.append(exp['explanation'])
                    if exp.get('task_alignment_explanation'):
                        all_task_alignment_explanations.append(exp['task_alignment_explanation'])
                    opt_guidance = exp.get('optimization_guidance', '')
                    if opt_guidance and opt_guidance not in all_optimization_guidance:
                        all_optimization_guidance.append(opt_guidance)
                
                # Use the first explanations
                threat_risk_explanation = all_risk_explanations[0] if all_risk_explanations else "High risk detected"
                task_alignment_explanation = all_task_alignment_explanations[0] if all_task_alignment_explanations else "Task misaligned"
                
                # Use the first optimization_guidance if available
                optimization_guidance_content = all_optimization_guidance[0] if all_optimization_guidance else ""
                
                # Build risk_guidance
                risk_guidance_parts = [
                    "**IMPORTANT: Your original task goal:**",
                    f"{instruction}",
                    "",
                    "Previous response was flagged due to:",
                    f"- Threat risk: {threat_risk_explanation}",
                    f"- Task alignment: {task_alignment_explanation}",
                ]
                
                if optimization_guidance_content:
                    risk_guidance_parts.extend([
                        "",
                        "Please adjust your response considering:",
                        optimization_guidance_content,
                    ])
                
                risk_guidance_parts.extend([
                    "",
                    "Remember to:",
                    "1. Re-explain your reasoning with security/task alignment considerations",
                    "2. Provide the corrected action",
                ])
                
                return "\n".join(risk_guidance_parts)
            
            # Try regeneration with risk_guidance
            current_regeneration_explanations = filtered_explanations.copy()
            for attempt in range(max_regeneration_attempts):
                risk_guidance = build_risk_guidance(current_regeneration_explanations)
                logger.info(f"[HarmonyGuardWrapper] Regeneration attempt {attempt + 1}/{max_regeneration_attempts}")
                logger.info(f"[HarmonyGuardWrapper] Risk guidance:\n{risk_guidance}")
                
                # Call action_generator with risk_guidance
                try:
                    new_actions, new_response = action_generator(
                        state=obs,
                        risk_guidance=risk_guidance,
                        num_samples=len(actions),  # Generate same number of actions
                    )
                    
                    # Log regenerated response
                    logger.info("=" * 80)
                    logger.info(f"[HarmonyGuard] Regeneration Attempt {attempt + 1} - Regenerated Response:")
                    if new_response:
                        response_str = str(new_response)
                        logger.info(f"  Response Length: {len(response_str)} characters")
                        logger.info(f"  Response Preview (first 500 chars):\n{response_str[:500]}...")
                        if len(response_str) > 500:
                            logger.info(f"  ... (truncated, total {len(response_str)} chars)")
                        logger.info(f"  Full Response:\n{response_str}")
                    else:
                        logger.warning("  No response generated")
                    logger.info(f"  Regenerated Actions Count: {len(new_actions) if new_actions else 0}")
                    if new_actions:
                        for i, action in enumerate(new_actions):
                            logger.info(f"    Action {i+1}: {str(action)[:200]}...")
                    logger.info("=" * 80)
                    
                    if not new_actions:
                        logger.warning(f"[HarmonyGuardWrapper] Regeneration attempt {attempt + 1} produced no actions")
                        continue
                    
                    # Re-evaluate new actions
                    new_filtered_actions = []
                    new_filtered_explanations = []
                    
                    for new_action in new_actions:
                        new_risk_result = self.evaluate_action_risk(
                            obs=obs,
                            action=new_action,
                            instruction=instruction,
                            candidate_actions=[new_action],
                            plan_text=plan_text,
                            current_response=new_response,  # Use new response
                        )
                        u = getattr(self.utility_agent, "_last_token_usage", None)
                        if u and isinstance(u, dict):
                            self._step_utility_agent_token_usages.append(u)

                        if new_risk_result['is_safe']:
                            new_filtered_actions.append(new_action)
                            logger.info(f"[HarmonyGuard] ✅ Regenerated action {len(new_filtered_actions)} passed re-evaluation")
                        else:
                            logger.info(f"[HarmonyGuard] ❌ Regenerated action failed re-evaluation: "
                                      f"risk_score={new_risk_result['risk_score']:.3f}, "
                                      f"task_alignment={new_risk_result.get('task_alignment', 'unknown')}")
                            new_filtered_explanations.append({
                                'action': str(new_action),
                                'risk_score': new_risk_result['risk_score'],
                                'explanation': new_risk_result['risk_explanation'],
                                'task_alignment': new_risk_result.get('task_alignment', 'deviated'),
                                'task_alignment_explanation': new_risk_result.get('task_alignment_explanation', ''),
                                'optimization_guidance': new_risk_result.get('optimization_guidance', ''),
                            })
                    
                    if new_filtered_actions:
                        logger.info("=" * 80)
                        logger.info(f"[HarmonyGuard] ✅ Regeneration successful: {len(new_filtered_actions)} safe action(s) found")
                        logger.info("[HarmonyGuard] Safe actions after regeneration:")
                        for i, action in enumerate(new_filtered_actions):
                            logger.info(f"  Safe Action {i+1}: {str(action)[:200]}...")
                        logger.info("=" * 80)
                        self._last_step_utility_agent_token_usage = self._aggregate_step_utility_agent_token_usage()
                        return new_filtered_actions, safety_info, None, new_response
                    else:
                        logger.warning("=" * 80)
                        logger.warning(f"[HarmonyGuard] ⚠️ Regeneration attempt {attempt + 1}: all actions still unsafe")
                        logger.info("[HarmonyGuard] Re-evaluation results for regenerated actions:")
                        for i, exp in enumerate(new_filtered_explanations):
                            logger.info(f"  Action {i+1}: risk_score={exp.get('risk_score', 'N/A')}, "
                                      f"threat_risk_explanation={exp.get('explanation', 'N/A')[:100]}..., "
                                      f"task_alignment={exp.get('task_alignment', 'N/A')}")
                        logger.info("=" * 80)
                        current_regeneration_explanations = new_filtered_explanations
                        
                except Exception as e:
                    logger.error(f"[HarmonyGuardWrapper] Error during regeneration attempt {attempt + 1}: {e}", exc_info=True)
                    continue
            
            # All regeneration attempts failed
            logger.warning("[HarmonyGuardWrapper] All regeneration attempts failed. Returning empty filtered_actions.")
            self._last_step_utility_agent_token_usage = self._aggregate_step_utility_agent_token_usage()
            return filtered_actions, safety_info, risk_guidance, None

        return filtered_actions, safety_info, risk_guidance, None
    
    def update_trajectory(
        self,
        state: Dict[str, Any],
        action: Any,
        next_state: Dict[str, Any],
        action_success: bool = True,
    ):
        """
        Update HarmonyGuard trajectory after action execution.
        
        Note: HarmonyGuard doesn't maintain trajectory history in the same way as SafePred.
        This method is provided for compatibility but does nothing.
        
        Args:
            state: Previous state
            action: Executed action
            next_state: Resulting state
            action_success: Whether action was successful
        """
        # HarmonyGuard doesn't maintain trajectory history
        # This method is kept for compatibility with SafePred interface
        pass
    
    @property
    def safe_agent(self):
        """
        Backward compatibility property.
        Returns None as HarmonyGuard doesn't have a SafeAgent equivalent.
        """
        return None
    
    def format_plan_for_prompt(self, plan_text: str) -> str:
        """
        Format plan for prompt (for compatibility).
        
        Note: HarmonyGuard doesn't support planning, returns empty string.
        
        Args:
            plan_text: Plan text (ignored)
        
        Returns:
            Empty string (HarmonyGuard doesn't use plans)
        """
        return ""
