"""
Reactive Safety Wrapper for OS-Harm mmagent Integration.

Provides a compatible interface with SafePred's SafetyWrapper.
"""

import os
import json
import logging
import sys
from typing import Any, Dict, List, Optional, Tuple, Callable

# Ensure logger is configured to output to stdout
logger = logging.getLogger(__name__)
# Add console handler if not already present
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Try to import reactive components
REACTIVE_AVAILABLE = None
ReactiveAgent = None

try:
    # Import reactive components directly (relative import)
    from .reactive_agent import ReactiveAgent
    REACTIVE_AVAILABLE = True
    logger.info("Reactive safety system imported successfully")
except ImportError as e:
    REACTIVE_AVAILABLE = False
    logger.warning(f"Failed to import Reactive safety system: {e}")


class ReactiveWrapper:
    """
    Wrapper for Reactive safety system integration with OS-Harm.
    
    Provides a compatible interface with SafePred's SafetyWrapper.
    Uses ReactiveAgent for prompt-based dual-aspect validation:
    1. Policy Compliance Check
    2. Alignment Check
    """
    
    def __init__(
        self,
        enabled: bool = True,
        policy_path: Optional[str] = None,
        config_path: Optional[str] = None,
        max_regeneration_attempts: int = 2,
    ):
        """
        Initialize Reactive Wrapper.
        
        Args:
            enabled: Whether to enable safety checks
            policy_path: Path to policy JSON file (required, passed from run.py)
            config_path: Path to reactive config.yaml file
            max_regeneration_attempts: Maximum number of regeneration attempts
        """
        logger.info(f"[ReactiveWrapper] __init__ called: enabled={enabled}, REACTIVE_AVAILABLE={REACTIVE_AVAILABLE}, policy_path={policy_path}, config_path={config_path}")
        
        self.enabled = enabled and REACTIVE_AVAILABLE
        self.max_regeneration_attempts = max_regeneration_attempts
        self.reactive_agent = None
        self.planning_enabled = False  # Reactive doesn't support planning feature
        self._step_reactive_token_usages = []
        self._last_step_reactive_token_usage = None
        
        if not REACTIVE_AVAILABLE:
            logger.warning("[ReactiveWrapper] ❌ Reactive safety system not available. Safety checks disabled.")
            return
        
        if not self.enabled:
            logger.info("[ReactiveWrapper] ⚠️ Reactive safety checks disabled by configuration.")
            return
        
        if not policy_path:
            logger.error("[ReactiveWrapper] ❌ policy_path is required for ReactiveWrapper")
            raise ValueError("policy_path is required for ReactiveWrapper")
        
        try:
            logger.info(f"[ReactiveWrapper] Loading config from {config_path if config_path else 'default location'}")
            # Load config if provided
            config = None
            if config_path and os.path.exists(config_path):
                from .config import ReactiveConfig
                config = ReactiveConfig.from_yaml(config_path)
                logger.info(f"[ReactiveWrapper] ✅ Loaded config from {config_path}")
            else:
                logger.warning(f"[ReactiveWrapper] ⚠️ Config file not found at {config_path}, using defaults")
            
            logger.info(f"[ReactiveWrapper] Initializing ReactiveAgent with policy_path={policy_path}")
            # Initialize ReactiveAgent (reference updates are always enabled)
            self.reactive_agent = ReactiveAgent(
                policy_path=policy_path,
                config=config,
            )
            logger.info(f"[ReactiveWrapper] ✅ Reactive safety system initialized successfully (policy_path={policy_path}, enabled={self.enabled})")
            
        except Exception as e:
            logger.error(f"[ReactiveWrapper] ❌ Failed to initialize Reactive wrapper: {e}", exc_info=True)
            self.enabled = False
            self.reactive_agent = None
            raise RuntimeError(f"Failed to initialize Reactive safety wrapper: {str(e)}") from e

    def _aggregate_step_reactive_token_usage(self) -> Optional[Dict[str, Any]]:
        """Sum token usage from all ReactiveAgent LLM calls in this filter_actions step."""
        usages = getattr(self, "_step_reactive_token_usages", []) or []
        if not usages:
            return None
        model = usages[0].get("model", "reactive_agent") if usages else "reactive_agent"
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
        plan_text: Optional[str] = None,  # Not used by Reactive
        current_response: Optional[str] = None,  # Full agent response including reasoning
    ) -> Dict[str, Any]:
        """
        Evaluate risk of an action using ReactiveAgent.
        
        Args:
            obs: Current observation (not used by Reactive, but kept for compatibility)
            action: Action to evaluate
            instruction: Current task instruction
            candidate_actions: Optional list of candidate actions (not used)
            plan_text: Optional plan text (not used by Reactive)
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
        logger.info("[Reactive] evaluate_action_risk called: enabled=%s, reactive_agent=%s", self.enabled, self.reactive_agent is not None)
        if not self.enabled or not self.reactive_agent:
            return {
                'risk_score': 0.0,
                'risk_explanation': 'Reactive safety checks disabled',
                'is_safe': True,
                'safe_action': action,
                'requires_regeneration': False,
                'task_alignment': 'aligned',
                'task_alignment_explanation': 'Safety checks disabled',
                'optimization_guidance': None,
            }
        
        try:
            # Extract thought from current_response
            # current_response might be a string, dict, or list (depending on model type)
            if current_response:
                if isinstance(current_response, str):
                    thought = current_response
                elif isinstance(current_response, list):
                    # Handle multi-modal response format (e.g., computer-use-preview)
                    thought_parts = []
                    for item in current_response:
                        if isinstance(item, dict):
                            if item.get('type') == 'reasoning':
                                thought_parts.append(item.get('text', ''))
                            elif item.get('type') == 'text':
                                thought_parts.append(item.get('text', ''))
                            elif 'text' in item:
                                thought_parts.append(item['text'])
                        elif isinstance(item, str):
                            thought_parts.append(item)
                    thought = '\n'.join(thought_parts) if thought_parts else str(action)
                elif isinstance(current_response, dict):
                    # Handle dict format
                    thought = current_response.get('text', current_response.get('reasoning', str(current_response)))
                else:
                    thought = str(current_response)
            else:
                thought = str(action)
            
            logger.debug("[Reactive] evaluate_action_risk: action=%s, thought_length=%d", str(action)[:100], len(thought))
            
            last_step_message = "Previous step completed successfully"
            
            logger.info("[Reactive] Calling reactive_agent.evaluate_action with instruction length=%d, thought length=%d", len(instruction), len(thought))
            
            # Call ReactiveAgent's evaluate_action
            try:
                result = self.reactive_agent.evaluate_action(
                    instruction=instruction,
                    thought=thought,
                    last_step_message=last_step_message
                )
                logger.debug(f"[Reactive] reactive_agent.evaluate_action returned: threat_risk={result.get('threat_risk')}, task_alignment={result.get('task_alignment')}")
            except Exception as e:
                logger.error(f"[Reactive] Error calling reactive_agent.evaluate_action: {e}", exc_info=True)
                raise RuntimeError(f"ReactiveAgent.evaluate_action failed: {str(e)}") from e
            
            # Log evaluation result in JSON format
            evaluation_result = {
                "instruction": instruction[:100] + "..." if len(instruction) > 100 else instruction,
                "threat_risk": result.get('threat_risk', 'unknown'),
                "threat_risk_explanation": result.get('threat_risk_explanation', ''),
                "task_alignment": result.get('task_alignment', 'unknown'),
                "task_alignment_explanation": result.get('task_alignment_explanation', ''),
                "policy_ids": result.get('policy_ids', []),
                "optimization_guidance": result.get('optimization_guidance', 'None')
            }
            # Print evaluation result for each step (use sys.stdout to avoid log redirection)
            sys.stdout.write("=" * 80 + "\n")
            sys.stdout.write("[Reactive] Step Evaluation Result:\n")
            sys.stdout.write(json.dumps(evaluation_result, indent=2, ensure_ascii=False) + "\n")
            sys.stdout.write("=" * 80 + "\n")
            sys.stdout.flush()
            logger.info("[Reactive] Evaluation Result: %s", json.dumps(evaluation_result, ensure_ascii=False))
            
            # Convert Reactive result to SafePred-compatible format
            threat_risk = (result.get('threat_risk') or 'safe').strip().lower()
            task_alignment = (result.get('task_alignment') or 'aligned').strip().lower()
            
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
            logger.error(f"[Reactive] ❌ Error evaluating action risk: {e}", exc_info=True)
            # Re-raise the exception instead of returning safe by default
            raise RuntimeError(f"Reactive evaluation failed: {str(e)}") from e
    
    def filter_actions(
        self,
        obs: Dict[str, Any],
        actions: List[Any],
        instruction: str,
        action_generator: Optional[Callable] = None,
        max_regeneration_attempts: Optional[int] = None,
        plan_text: Optional[str] = None,  # Not used by Reactive
        current_response: Optional[str] = None,  # Full agent response including reasoning
        plan_update_callback: Optional[Callable] = None,  # Not used by Reactive
    ) -> Tuple[List[Any], Dict[str, Any], Optional[str], Optional[str]]:
        """
        Filter actions based on Reactive's dual-aspect validation.
        
        Process:
        1. Evaluate each action using ReactiveAgent (policy compliance + task alignment)
        2. Filter unsafe actions (risky OR deviated)
        3. If all actions filtered, generate guidance and call action_generator
        
        Args:
            obs: Current observation
            actions: List of actions to filter
            instruction: Current task instruction
            action_generator: Callback function(state, risk_guidance, num_samples) -> (actions, response)
            max_regeneration_attempts: Maximum number of regeneration attempts
            plan_text: Optional plan text (not used by Reactive)
            current_response: Full agent response including reasoning
            plan_update_callback: Optional callback (not used by Reactive)
        
        Returns:
            Tuple of (filtered_actions, safety_info, risk_guidance, regeneration_response)
            - filtered_actions: List of safe actions
            - safety_info: Risk scores and explanations for each action
            - risk_guidance: Guidance for regeneration if all actions filtered (None otherwise)
            - regeneration_response: Full response from regeneration if occurred (None otherwise)
        """
        # Check if Reactive is enabled and agent is initialized
        if not self.enabled:
            logger.warning("[Reactive] ❌ filter_actions called but Reactive is disabled (self.enabled=False)")
            self._last_step_reactive_token_usage = None
            return actions, {}, None, None
        if not self.reactive_agent:
            logger.warning("[Reactive] ❌ filter_actions called but reactive_agent is None")
            self._last_step_reactive_token_usage = None
            return actions, {}, None, None

        self._step_reactive_token_usages = []
        self._last_step_reactive_token_usage = None
        
        # Print policy info only on first call (track with a flag)
        # Use sys.stdout.write() to avoid log redirection
        if not hasattr(self, '_policy_printed'):
            policy_list = self.reactive_agent.policy_loader.get_policy_list()
            policy_count = len(policy_list) if policy_list else 0
            formatted_policy = self.reactive_agent.policy_loader.get_formatted_policies()
            policy_length = len(formatted_policy) if formatted_policy else 0
            sys.stdout.write("=" * 80 + "\n")
            sys.stdout.write(f"[Reactive] Policy loaded: {policy_count} policies, content length: {policy_length} chars\n")
            if policy_list and len(policy_list) > 0:
                policy_ids = [p.get('policy_id', 'Unknown') for p in policy_list[:3]]
                sys.stdout.write(f"[Reactive] Sample policy IDs: {policy_ids}\n")
            sys.stdout.write("=" * 80 + "\n")
            sys.stdout.flush()
            self._policy_printed = True
        
        logger.info(f"[Reactive] filter_actions called with {len(actions)} action(s), instruction: {instruction[:100]}...")
        
        # Use instance max_regeneration_attempts if not provided
        if max_regeneration_attempts is None:
            max_regeneration_attempts = self.max_regeneration_attempts
        
        # Evaluate each action and filter unsafe ones
        filtered_actions = []
        safety_info = {}
        filtered_explanations = []
        
        for i, action in enumerate(actions):
            logger.debug(f"[Reactive] Evaluating action {i+1}/{len(actions)}: {str(action)[:100]}...")
            # Evaluate risk using ReactiveAgent
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
                logger.error(f"[Reactive] Error in evaluate_action_risk for action {i+1}: {e}", exc_info=True)
                raise RuntimeError(f"evaluate_action_risk failed for action {i+1}: {str(e)}") from e

            u = getattr(self.reactive_agent, "_last_token_usage", None)
            if u and isinstance(u, dict):
                self._step_reactive_token_usages.append(u)
            
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
                logger.debug(f"[Reactive] ✅ Action {i+1} passed: risk_score={risk_score:.3f}, task_alignment={task_alignment}")
            else:
                logger.info(f"[Reactive] ❌ Action {i+1} filtered: risk_score={risk_score:.3f}, task_alignment={task_alignment}")
                logger.info(f"  Threat Risk: {risk_result.get('risk_explanation', '')[:200]}...")
                logger.info(f"  Task Alignment: {risk_result.get('task_alignment_explanation', '')[:200]}...")
                # Store filtered action info for potential regeneration
                filtered_explanations.append({
                    'action': str(action),
                    'risk_score': risk_score,
                    'explanation': risk_result['risk_explanation'],
                    'task_alignment': task_alignment,
                    'task_alignment_explanation': risk_result.get('task_alignment_explanation', ''),
                    'optimization_guidance': risk_result.get('optimization_guidance', ''),
                })

        self._last_step_reactive_token_usage = self._aggregate_step_reactive_token_usage()
        
        # If all actions filtered and action_generator provided, try regeneration
        risk_guidance = None
        if not filtered_actions and actions and action_generator is not None:
            logger.info("=" * 80)
            logger.info("[Reactive] ⚠️ All actions filtered, starting regeneration process...")
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
                logger.info(f"[ReactiveWrapper] Regeneration attempt {attempt + 1}/{max_regeneration_attempts}")
                logger.info(f"[ReactiveWrapper] Risk guidance:\n{risk_guidance}")
                
                # Call action_generator with risk_guidance
                try:
                    new_actions, new_response = action_generator(
                        state=obs,
                        risk_guidance=risk_guidance,
                        num_samples=len(actions),  # Generate same number of actions
                    )
                    
                    # Log regenerated response
                    logger.info("=" * 80)
                    logger.info(f"[Reactive] Regeneration Attempt {attempt + 1} - Regenerated Response:")
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
                        logger.warning(f"[ReactiveWrapper] Regeneration attempt {attempt + 1} produced no actions")
                        continue
                    
                    # Re-evaluate new actions
                    new_filtered_actions = []
                    new_filtered_explanations = []
                    
                    for new_action in new_actions:
                        try:
                            new_risk_result = self.evaluate_action_risk(
                                obs=obs,
                                action=new_action,
                                instruction=instruction,
                                candidate_actions=[new_action],
                                plan_text=plan_text,
                                current_response=new_response,  # Use new response
                            )
                        except Exception as e:
                            logger.error(f"[Reactive] Error in evaluate_action_risk for regenerated action: {e}", exc_info=True)
                            raise RuntimeError(f"evaluate_action_risk failed for regenerated action: {str(e)}") from e

                        u = getattr(self.reactive_agent, "_last_token_usage", None)
                        if u and isinstance(u, dict):
                            self._step_reactive_token_usages.append(u)
                        
                        if new_risk_result['is_safe']:
                            new_filtered_actions.append(new_action)
                            logger.info(f"[Reactive] ✅ Regenerated action {len(new_filtered_actions)} passed re-evaluation")
                        else:
                            logger.info(f"[Reactive] ❌ Regenerated action failed re-evaluation: "
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
                        logger.info(f"[Reactive] ✅ Regeneration successful: {len(new_filtered_actions)} safe action(s) found")
                        logger.info("[Reactive] Safe actions after regeneration:")
                        for i, action in enumerate(new_filtered_actions):
                            logger.info(f"  Safe Action {i+1}: {str(action)[:200]}...")
                        logger.info("=" * 80)
                        self._last_step_reactive_token_usage = self._aggregate_step_reactive_token_usage()
                        return new_filtered_actions, safety_info, None, new_response
                    else:
                        logger.warning("=" * 80)
                        logger.warning(f"[Reactive] ⚠️ Regeneration attempt {attempt + 1}: all actions still unsafe")
                        logger.info("[Reactive] Re-evaluation results for regenerated actions:")
                        for i, exp in enumerate(new_filtered_explanations):
                            logger.info(f"  Action {i+1}: risk_score={exp.get('risk_score', 'N/A')}, "
                                      f"threat_risk_explanation={exp.get('explanation', 'N/A')[:100]}..., "
                                      f"task_alignment={exp.get('task_alignment', 'N/A')}")
                        logger.info("=" * 80)
                        current_regeneration_explanations = new_filtered_explanations
                        
                except Exception as e:
                    logger.error(f"[ReactiveWrapper] Error during regeneration attempt {attempt + 1}: {e}", exc_info=True)
                    continue
            
            # All regeneration attempts failed
            logger.warning("[ReactiveWrapper] All regeneration attempts failed. Returning empty filtered_actions.")
            self._last_step_reactive_token_usage = self._aggregate_step_reactive_token_usage()
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
        Update Reactive trajectory after action execution.
        
        Note: Reactive doesn't maintain trajectory history.
        This method is provided for compatibility but does nothing.
        """
        # Reactive doesn't maintain trajectory history
        pass
    
    @property
    def safe_agent(self):
        """
        Backward compatibility property.
        Returns None as Reactive doesn't have a SafeAgent equivalent.
        """
        return None
    
    def format_plan_for_prompt(self, plan_text: str, progress_step: int = 1) -> str:
        """
        Format plan for prompt (for compatibility).
        
        Note: Reactive doesn't support planning, returns empty string.
        """
        return ""
