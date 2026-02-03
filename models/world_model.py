"""
World Model Module for Safety-TS-LMA.

Simulates state transitions for internal tree search without executing
actions in the real environment. Supports multiple simulation strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict
import copy
import json
import re
from ..utils.logger import get_logger
from ..utils.parsers import parse_json_from_text, parse_json_array_from_text, normalize_action
from ..utils.formatters import format_object_to_string
from ..utils.xml_parser import parse_xml_output, convert_xml_to_world_model_format
from ..utils.structured_text_parser import parse_structured_text, convert_structured_text_to_world_model_format

logger = get_logger("SafePred.WorldModel")


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


class BaseWorldModel(ABC):
    """
    Abstract base class for world models.
    
    World models simulate state transitions: s' = simulate(s, a)
    without actually executing actions in the real environment.
    """
    
    @abstractmethod
    def simulate(self, state: Any, action: Any) -> Any:
        """
        Simulate the next state given current state and action.
        
        Args:
            state: Current state representation
            action: Action to simulate
        
        Returns:
            Predicted next state
        """
        pass
    
    @abstractmethod
    def get_valid_actions(self, state: Any) -> List[Any]:
        """
        Get valid actions for a given state.
        
        Args:
            state: Current state representation
        
        Returns:
            List of valid actions
        """
        pass
    
    def simulate_batch(self, states: List[Any], actions: List[Any]) -> List[Any]:
        """
        Simulate state transitions for a batch of state-action pairs.
        
        Args:
            states: List of current states
            actions: List of actions
        
        Returns:
            List of predicted next states
        """
        return [self.simulate(s, a) for s, a in zip(states, actions)]


class LLMBasedWorldModel(BaseWorldModel):
    """
    LLM-based world model using language models to predict state transitions.
    
    Uses LLMs (e.g., Qwen, GPT) to generate predictions about next states
    given current state and action.
    """
    
    def __init__(
        self,
        model: Optional[Any] = None,
        model_name: Optional[str] = None,
        device: str = "cuda",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        provider: Optional[str] = None,
        timeout: Optional[int] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        use_few_shot: bool = False,
        use_chain_of_thought: bool = False,
        require_json_output: bool = True,
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
        use_state_delta: bool = True,
        prediction_steps: int = 1,
    ):
        """
        Initialize LLM-based world model.
        
        Args:
            model: Pre-loaded model object
            model_name: Name/path of the model to load
            device: Device to run model on
            temperature: Temperature for generation (overrides config if provided)
            max_tokens: Maximum tokens for generation (overrides config if provided)
            api_key: LLM API key (overrides config if provided)
            api_url: LLM API URL (overrides config if provided)
            provider: LLM provider type (overrides config if provided)
            timeout: Request timeout in seconds (overrides config if provided)
            llm_config: Dictionary with LLM configuration (from SafetyConfig.get_llm_config())
            use_few_shot: Whether to use few-shot learning in prompts (default: False)
            use_chain_of_thought: Whether to use chain-of-thought reasoning (default: False)
            require_json_output: Whether to require JSON format output (default: True)
            few_shot_examples: List of example dicts with keys: 'state', 'action', 'next_state'
            use_state_delta: Whether to use state delta mode - predicts only changes instead of full state (default: True)
            prediction_steps: Number of steps to predict ahead (1 = single-step, >1 = multi-step) (default: 1)
        """
        self.model = model
        self.device = device
        self._model_loaded = model is not None
        
        # Load LLM configuration from llm_config or individual parameters
        if llm_config:
            self.api_key = api_key or llm_config.get("api_key")
            self.api_url = api_url or llm_config.get("api_url")
            self.model_name = model_name or llm_config.get("model_name")
            # Provider from llm_config takes precedence, then explicit parameter, then default
            self.provider = llm_config.get("provider") or provider or "openai"
            self.temperature = temperature if temperature is not None else llm_config.get("temperature", 0.7)
            self.max_tokens = max_tokens if max_tokens is not None else llm_config.get("max_tokens", 512)
            self.timeout = timeout if timeout is not None else llm_config.get("timeout", 30)
        else:
            # Fallback to individual parameters or defaults
            self.api_key = api_key
            self.api_url = api_url
            self.model_name = model_name
            self.provider = provider or "openai"
            self.temperature = temperature if temperature is not None else 0.7
            self.max_tokens = max_tokens if max_tokens is not None else 512
            self.timeout = timeout if timeout is not None else 30
        
        # Prompt optimization settings: directly use passed parameters (not from config)
        self.use_few_shot = use_few_shot
        self.use_chain_of_thought = use_chain_of_thought
        self.require_json_output = require_json_output
        self.few_shot_examples = few_shot_examples or []
        
        # State delta mode: predict only changes instead of full state (more efficient)
        self.use_state_delta = use_state_delta if use_state_delta is not None else True  # Default to state delta mode for efficiency
        
        # Initialize token usage tracking
        self._last_token_usage = None  # Store token usage from last LLM API call
        
        # Multi-step prediction configuration
        if llm_config:
            self.prediction_steps = llm_config.get("prediction_steps", prediction_steps)
            self.log_prompt = llm_config.get("log_prompt", False)  # Read from config
            # Policy reference configuration
            self.show_policy_references = llm_config.get("show_policy_references", False)  # Default: False (do not show references)
            # Risk score calculation configuration
            self.risk_score_violation_penalty = llm_config.get("risk_score_violation_penalty", 0.5)
            # Prediction type configuration
            self.enable_short_term_prediction = llm_config.get("enable_short_term_prediction", True)  # Default: True
            self.enable_long_term_prediction = llm_config.get("enable_long_term_prediction", True)  # Default: True
        else:
            self.prediction_steps = prediction_steps
            self.log_prompt = False  # Default to False
            self.show_policy_references = False  # Default: False (do not show references)
            # Default risk score values
            self.risk_score_violation_penalty = 0.5
            # Default prediction type values
            self.enable_short_term_prediction = True  # Default: True
            self.enable_long_term_prediction = True  # Default: True
        
        # Log prediction type configuration
        logger.info(f"[World Model] Prediction type configuration: short_term={'enabled' if self.enable_short_term_prediction else 'disabled'}, long_term={'enabled' if self.enable_long_term_prediction else 'disabled'}")
        
        if model is None and self.model_name:
            self._load_model()
    
    def _load_model(self) -> None:
        """
        Load the LLM model configuration.
        
        Supports API-based models (openai, qwen, custom).
        Configuration is loaded from config.yaml via SafetyConfig.
        """
        # API-based model (OpenAI, Qwen, custom)
        # Model will be called via API, no need to load locally
        # API credentials must be set, otherwise will raise error when used
        if not self.api_key or not self.api_url:
            self._use_api = False
        else:
            self._use_api = True
        self._model_loaded = True
    
    def _create_prompt(self, state: Any, action: Any, policies: Optional[List[Dict[str, Any]]] = None, plan_text: Optional[str] = None) -> str:
        """
        Create a prompt for LLM to predict next state.
        
        Uses unified prompt templates from prompts module.
        Prefers state delta mode for efficiency (predicts only changes).
        
        Args:
            state: Current state (already contains complete chat_history)
            action: Action to simulate
            policies: Optional list of policy dictionaries for risk evaluation
            plan_text: Optional execution plan text (only used when violation detected for optimization_guidance generation, not for progress tracking)
        
        Returns:
            Formatted prompt string
        """
        from .prompts import PromptTemplates
        
        action_str = self._to_string(action)
        
        # Use state delta mode by default (more efficient)
        if self.use_state_delta:
            # Pass few-shot examples if available (for experience replay)
            examples = self.few_shot_examples if (self.use_few_shot and self.few_shot_examples) else None
            # Pass compact state dict directly (not string) to world_model_state_delta
            # Current State already contains complete chat_history
            return PromptTemplates.world_model_state_delta(
                state=state,  # Pass compact state dict directly
                action=action_str,
                examples=examples,
                policies=policies,
                show_policy_references=self.show_policy_references,  # Read from config
                plan_text=plan_text,  # Pass plan_text only when violation detected (for optimization_guidance generation)
                enable_short_term_prediction=self.enable_short_term_prediction,  # Read from config
                enable_long_term_prediction=self.enable_long_term_prediction  # Read from config
            )
        
        # Fallback to other modes if state delta is disabled
        # Convert compact state to string for legacy prompts
        # Use full axtree_txt if available, otherwise fall back to key_elements
        if isinstance(state, dict):
            if 'axtree_txt' in state and state.get('axtree_txt'):
                state_str = PromptTemplates._format_state_with_full_tree(state)
            elif 'key_elements' in state:
                state_str = PromptTemplates._format_compact_state(state)
            else:
                state_str = self._to_string(state)
        else:
            state_str = self._to_string(state)
        
        if self.use_few_shot and self.few_shot_examples:
            return PromptTemplates.world_model_few_shot(
                state=state_str,
                action=action_str,
                examples=self.few_shot_examples
            )
        elif self.use_chain_of_thought:
            return PromptTemplates.world_model_chain_of_thought(
                state=state_str,
                action=action_str,
                require_json=self.require_json_output
            )
        elif self.require_json_output:
            return PromptTemplates.world_model_json(
                state=state_str,
                action=action_str
            )
        else:
            return PromptTemplates.world_model_basic(
                state=state_str,
                action=action_str
            )
    
    
    def _to_string(self, obj: Any) -> str:
        """Convert object to string representation."""
        return format_object_to_string(obj)
    
    def _parse_llm_output(self, generated_text: str, original_state: Any, action: Any) -> Any:
        """
        Parse LLM output to extract next state.
        
        Supports:
        - State delta mode (predicts only changes, then synthesizes full state)
        - Full state prediction mode (legacy)
        - JSON format parsing
        - Error handling
        
        Args:
            generated_text: Raw text from LLM
            original_state: Original state before action
            action: Action that was executed
        
        Returns:
            Parsed next state (synthesized from delta or full prediction)
        """
        # Try state delta mode first (more efficient)
        if self.use_state_delta:
            parsed_delta = self._parse_json_output(generated_text)
            # Ensure parsed_delta is a dict before processing - raise error if not
            if not parsed_delta:
                error_msg = f"[WorldModel] Failed to parse JSON output. Raw text: {generated_text[:500]}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            if not isinstance(parsed_delta, dict):
                error_msg = f"[WorldModel] parsed_delta is not a dict: type={type(parsed_delta)}, value={str(parsed_delta)[:200]}"
                logger.error(error_msg)
                raise TypeError(error_msg)
            
            # Add default values for disabled prediction types
            # Note: We set these to None (not empty string) so that subsequent code using
            # `if delta.get('semantic_delta'):` will correctly skip disabled predictions
            if not getattr(self, 'enable_short_term_prediction', True):
                parsed_delta.setdefault("semantic_delta", None)
                logger.debug("[World Model] Short-term prediction disabled, field set to None")
            if not getattr(self, 'enable_long_term_prediction', True):
                parsed_delta.setdefault("long_term_impact", None)
                logger.debug("[World Model] Long-term prediction disabled, field set to None")
            
            if not self._is_state_delta_format(parsed_delta):
                error_msg = f"[WorldModel] parsed_delta is not in state_delta format. Keys: {list(parsed_delta.keys())}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Calculate risk score based on violated policies (rule-based)
            parsed_delta = self._calculate_risk_score_by_rules(parsed_delta)
            
            # Ensure parsed_delta is still a dict after processing - raise error if not
            if not isinstance(parsed_delta, dict):
                error_msg = f"[WorldModel] parsed_delta is not a dict after _calculate_risk_score_by_rules: type={type(parsed_delta)}"
                logger.error(error_msg)
                raise TypeError(error_msg)
            
            # Store delta for risk evaluator to access
            if not isinstance(parsed_delta, dict):
                logger.warning(f"[WorldModel] parsed_delta is NOT a dict before storing! value={str(parsed_delta)[:500]}")
            self._last_predicted_delta = parsed_delta
            
            # Apply delta to current state to synthesize next state
            synthesized_state = self._apply_state_delta(original_state, parsed_delta, action)
            return synthesized_state
        
        # Fallback to full state prediction mode
        if self.require_json_output:
            parsed_state = self._parse_json_output(generated_text)
            if parsed_state:
                # Merge with original state structure
                next_state = copy.deepcopy(original_state)
                if isinstance(next_state, dict):
                    next_state.update(parsed_state)
                    next_state["action_history"] = next_state.get("action_history", []) + [action]
                    next_state["step"] = next_state.get("step", 0) + 1
                    next_state["simulated"] = True
                    next_state["llm_prediction"] = generated_text
                    next_state["parsed_from_json"] = True
                return next_state
        
        # If JSON parsing failed, raise error with full debug output
        logger.error("=" * 80)
        logger.error("[World Model] JSON Parsing Failed - Full LLM Output:")
        logger.error("=" * 80)
        logger.error(f"Raw LLM Output (length: {len(generated_text)}):")
        logger.error(generated_text)
        logger.error("=" * 80)
        error_msg = f"Failed to parse JSON from LLM output: {generated_text[:200]}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    def _is_state_delta_format(self, parsed: Dict[str, Any]) -> bool:
        """
        Check if parsed JSON is in state delta format.
        
        Args:
            parsed: Parsed JSON dictionary
        
        Returns:
            True if it's state delta format, False otherwise
        """
        if not isinstance(parsed, dict):
            return False
        
        # Check required top-level fields
        # Note: risk_score is calculated by _calculate_risk_score_by_rules, not required from LLM output
        required_fields = [
            "element_changes",
            "violated_policy_ids",
            "risk_explanation",
            "optimization_guidance"
        ]
        
        # Add optional fields based on configuration
        if getattr(self, 'enable_short_term_prediction', True):
            required_fields.append("semantic_delta")
        if getattr(self, 'enable_long_term_prediction', True):
            required_fields.append("long_term_impact")
        
        if not all(field in parsed for field in required_fields):
            return False
        
        # Validate nested field types - raise error if wrong type
        if not isinstance(parsed.get("element_changes"), dict):
            error_msg = f"[WorldModel] element_changes is not a dict: type={type(parsed.get('element_changes'))}, value={str(parsed.get('element_changes'))[:200]}"
            logger.error(error_msg)
            raise TypeError(error_msg)
        
        return True
    
    def _calculate_risk_score_by_rules(self, parsed_delta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate risk score based on violated policies using rules.
        
        Note: Goal alignment is now a policy (automatically added to all policy files),
        so goal deviation violations will be included in violated_policy_ids.
        
        Rules:
        - Has violated_policy_ids → risk_score = violation_penalty (default 0.5)
        - No violated_policy_ids → risk_score = 0.0
        
        Args:
            parsed_delta: Parsed delta dictionary from LLM
        
        Returns:
            Updated parsed_delta with calculated risk_score
        """
        violated_policy_ids = parsed_delta.get('violated_policy_ids', [])
        has_violations = violated_policy_ids and len(violated_policy_ids) > 0
        
        # Rule-based risk score calculation
        # Note: Goal alignment violations are now included in violated_policy_ids
        # (goal alignment is automatically added as a policy to all policy files)
        # Risk score is calculated based on policy violations only
        risk_score = 0.0
        if has_violations:
            risk_score = self.risk_score_violation_penalty
        
        # Clamp risk_score to [0.0, 1.0]
        risk_score = max(0.0, min(1.0, risk_score))
        
        # Update parsed_delta with calculated risk score
        # Note: risk_level is NOT calculated here - it comes from policy definitions
        # risk_level in policies is used for reference storage limits
        # risk_explanation is generated by World Model's LLM and should be preserved as-is
        parsed_delta['risk_score'] = risk_score
        
        return parsed_delta
    
    def get_risk_score(self) -> Optional[float]:
        """
        Get the risk score from the last prediction.
        
        Returns:
            Risk score (0.0-1.0) if available, None otherwise
        """
        if hasattr(self, '_last_predicted_delta') and self._last_predicted_delta:
            return self._last_predicted_delta.get('risk_score')
        return None
    
    def get_risk_explanation(self) -> Optional[str]:
        """
        Get the risk explanation from the last prediction.
        
        Returns:
            Risk explanation string if available, None otherwise
        """
        if hasattr(self, '_last_predicted_delta') and self._last_predicted_delta:
            return self._last_predicted_delta.get('risk_explanation')
        return None
    
    def get_multi_step_risk_details(self) -> Optional[Dict[str, Any]]:
        """
        Get detailed risk information for multi-step predictions.
        
        Returns:
            Dictionary with multi-step risk details if available, None otherwise.
            Contains:
            - weighted_risk: Overall weighted risk score
            - num_steps: Number of prediction steps
            - step_risks: List of risk scores for each step
            - step_explanations: List of risk explanations for each step
        """
        if hasattr(self, '_last_predicted_delta') and self._last_predicted_delta:
            delta = self._last_predicted_delta
            if delta.get('multi_step', False):
                return {
                    'weighted_risk': delta.get('risk_score'),
                    'num_steps': delta.get('num_steps'),
                    'step_risks': delta.get('step_risks', []),
                    'step_explanations': delta.get('step_explanations', []),
                }
        return None
    
    def _calculate_weighted_risk(self, step_risks: List[float]) -> float:
        """
        Calculate weighted risk score from multiple step risks.
        
        Uses exponential decay weights: step 0 = 1.0, step 1 = 0.5, step 2 = 0.25, etc.
        
        Args:
            step_risks: List of risk scores for each step
        
        Returns:
            Weighted risk score (0.0-1.0)
        """
        if not step_risks:
            error_msg = "[WorldModel] Cannot calculate weighted risk: step_risks is empty"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        if len(step_risks) == 1:
            return step_risks[0]
        
        # Calculate weights: exponential decay
        weights = []
        total_weight = 0.0
        for i in range(len(step_risks)):
            weight = 1.0 / (2 ** i)  # 1.0, 0.5, 0.25, 0.125, ...
            weights.append(weight)
            total_weight += weight
        
        # Calculate weighted average
        weighted_sum = sum(step_risks[i] * weights[i] for i in range(len(step_risks)))
        weighted_risk = weighted_sum / total_weight
        
        return weighted_risk
    
    def _combine_risk_explanations(
        self, 
        step_explanations: List[Optional[str]], 
        step_risks: List[float],
        weighted_risk: float
    ) -> str:
        """
        Combine risk explanations from multiple steps into a single explanation.
        
        Args:
            step_explanations: List of risk explanations for each step
            step_risks: List of risk scores for each step
            weighted_risk: Calculated weighted risk score
        
        Returns:
            Combined risk explanation string
        """
        if not step_explanations:
            return f"Weighted risk score: {weighted_risk:.3f}"
        
        if len(step_explanations) == 1:
            base_explanation = step_explanations[0] or "Risk assessed"
            return f"{base_explanation} (risk: {step_risks[0]:.3f})"
        
        # Build combined explanation
        parts = []
        parts.append(f"Multi-step weighted risk: {weighted_risk:.3f}")
        parts.append("Step risks:")
        
        for i, (explanation, risk) in enumerate(zip(step_explanations, step_risks)):
            step_num = i + 1
            weight = 1.0 / (2 ** i)
            exp_text = explanation or f"Step {step_num}"
            parts.append(f"  Step {step_num} (weight {weight:.3f}): {exp_text} (risk: {risk:.3f})")
        
        return " | ".join(parts)
    
    def _apply_state_delta(self, current_state: Any, delta: Dict[str, Any], action: Any) -> str:
        """
        Apply state delta to current state to synthesize next state.
        
        Strategy:
        1. Keep Goal and Policies unchanged (they don't change due to actions)
        2. Update URL if changed
        3. Update Accessibility Tree based on semantic_delta and element_changes
        4. Update Chat History (add action)
        5. Add metadata
        
        Args:
            current_state: Current state (string representation)
            delta: State delta dictionary from LLM
            action: Action that was executed
        
        Returns:
            Synthesized next state as string
        """
        # Parse current state if it's a string
        if isinstance(current_state, str):
            # Extract components from current state string
            lines = current_state.split('\n')
            goal_line = None
            policies_section = []
            axtree_section = []
            chat_section = []
            
            in_policies = False
            in_axtree = False
            in_chat = False
            
            for line in lines:
                if line.startswith('Goal:'):
                    goal_line = line
                    in_policies = False
                    in_axtree = False
                    in_chat = False
                elif line.startswith('Policies:'):
                    in_policies = True
                    in_axtree = False
                    in_chat = False
                    policies_section.append(line)
                elif line.startswith('Accessibility Tree:'):
                    in_policies = False
                    in_axtree = True
                    in_chat = False
                    axtree_section.append(line)
                elif line.startswith('Chat History:'):
                    in_policies = False
                    in_axtree = False
                    in_chat = True
                    chat_section.append(line)
                else:
                    if in_policies:
                        policies_section.append(line)
                    elif in_axtree:
                        axtree_section.append(line)
                    elif in_chat:
                        chat_section.append(line)
            
            # Build synthesized state
            synthesized_lines = []
            
            # 1. Keep Goal unchanged
            if goal_line:
                synthesized_lines.append(goal_line)
            
            # 2. Keep Policies unchanged
            if policies_section:
                synthesized_lines.extend(policies_section)
            
            # 3. Update Accessibility Tree with changes
            if axtree_section:
                # Keep original axtree header
                synthesized_lines.append(axtree_section[0] if axtree_section else 'Accessibility Tree:')
                # Add semantic delta as annotation
                if delta.get('semantic_delta'):
                    synthesized_lines.append(f"[State Changes] {delta['semantic_delta']}")
                # Add risk-relevant element changes
                risk_aff = delta.get('element_changes', {})
                if risk_aff.get('new_elements'):
                    synthesized_lines.append(f"[New Elements] {', '.join(risk_aff['new_elements'])}")
                # Keep original axtree content (truncated if too long)
                if len(axtree_section) > 1:
                    # Keep first part of original tree, add changes annotation
                    synthesized_lines.extend(axtree_section[1:min(50, len(axtree_section))])
            
            # 4. Update Chat History (add action)
            action_str = str(action) if action else "None"
            if chat_section:
                synthesized_lines.extend(chat_section)
                synthesized_lines.append(f"assistant: {action_str}")
            else:
                synthesized_lines.append(f"Chat History: assistant: {action_str}")
            
            # 5. Add metadata
            synthesized_lines.append("")
            synthesized_lines.append("[World Model Prediction Metadata]")
            
            synthesized_state = '\n'.join(synthesized_lines)
            return synthesized_state
        else:
            # If current_state is not a string, convert delta to string representation
            # This is a fallback for non-string states
            delta_str = json.dumps(delta, indent=2, ensure_ascii=False)
            return f"{current_state}\n\n[State Changes]\n{delta_str}"
    
    def _parse_json_output(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract and parse output from LLM (supports JSON, XML, and structured text formats).
        
        Parsing priority:
        1. JSON (primary format, with error fixing)
        2. XML (format-insensitive fallback)
        3. Structured Text (format-insensitive fallback)
        
        Args:
            text: Raw text from LLM
        
        Returns:
            Parsed dict or None if parsing fails
        """
        # Try JSON format first (primary format, with robust error fixing)
        parsed = parse_json_from_text(text)
        if isinstance(parsed, dict):
            return parsed
        
        # Try XML format (format-insensitive fallback)
        xml_result = parse_xml_output(text)
        if xml_result:
            # Convert XML format to World Model's expected format
            return convert_xml_to_world_model_format(xml_result)
        
        # Try Structured Text format (format-insensitive fallback)
        structured_result = parse_structured_text(text)
        if structured_result:
            # Convert structured text format to World Model's expected format
            return convert_structured_text_to_world_model_format(structured_result)
        
        return None
    
    def simulate(self, state: Any, action: Any, policies: Optional[List[Dict[str, Any]]] = None, steps: Optional[int] = None, plan_text: Optional[str] = None) -> Any:
        """
        Simulate next state(s) using LLM prediction.
        
        Args:
            state: Current state (already contains complete chat_history)
            action: Action to simulate
            policies: Optional list of policy dictionaries for risk evaluation (will be formatted with references)
            steps: Number of steps to predict (overrides self.prediction_steps if provided)
            plan_text: Optional execution plan text for progress tracking
        
        Returns:
            Predicted next state (single-step) or list of predicted states (multi-step)
        """
        if not self._model_loaded:
            self._load_model()
        
        # Determine number of steps to predict
        num_steps = steps if steps is not None else getattr(self, 'prediction_steps', 1)
        
        # Single-step prediction (default)
        if num_steps == 1:
            # Create prompt from state, action, and state history
            # Policies are passed as List[Dict[str, Any]] and will be formatted in the prompt
            if policies:
                from ..utils.logger import get_logger
                logger = get_logger("SafePred.WorldModel")
            # Use plan_text parameter if provided, otherwise try to get from state
            # This allows plan_text to be passed explicitly or extracted from state
            # Note: plan_text is only used when violation is detected (to generate optimization_guidance)
            # If plan_text is None, world model will not track plan progress (more efficient)
            effective_plan_text = plan_text
            if effective_plan_text is None and isinstance(state, dict):
                effective_plan_text = state.get('plan_text')
            
            # Only include plan_text in prompt if explicitly provided (for violation-based plan updates)
            # This avoids unnecessary plan tracking overhead for safe actions
            prompt = self._create_prompt(state, action, policies=policies, plan_text=effective_plan_text)
            
            # Check API availability
            if not hasattr(self, '_use_api') or not self._use_api or not self.api_key or not self.api_url:
                error_msg = f"LLM API not configured. Please set api_key and api_url for provider '{self.provider}'."
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Use API-based LLM (via unified LLMClient)
            next_state = self._simulate_api(prompt, state, action)
            return next_state
        
        # Multi-step prediction
        else:
            predicted_states = []
            step_risks = []  # Store risk scores for each step
            step_explanations = []  # Store risk explanations for each step
            
            current_state = state
            current_action = action
            
            for step_idx in range(num_steps):
                # Create prompt from current state and action
                prompt = self._create_prompt(current_state, current_action, policies=policies)
                
                # Check API availability
                if not hasattr(self, '_use_api') or not self._use_api or not self.api_key or not self.api_url:
                    error_msg = f"LLM API not configured. Please set api_key and api_url for provider '{self.provider}'."
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # Predict next state (this updates _last_predicted_delta)
                next_state = self._simulate_api(prompt, current_state, current_action)
                predicted_states.append(next_state)
                
                # Immediately save risk information for this step before next iteration overwrites it
                # Get risk score and explanation from _last_predicted_delta (set by _simulate_api)
                step_delta = getattr(self, '_last_predicted_delta', None)
                if step_delta is None:
                    error_msg = f"[WorldModel] _last_predicted_delta is None for step {step_idx + 1} in multi-step prediction"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                if not isinstance(step_delta, dict):
                    error_msg = f"[WorldModel] step_delta is not a dict: type={type(step_delta)}, value={str(step_delta)[:200]}"
                    logger.error(error_msg)
                    raise TypeError(error_msg)
                
                step_risk = step_delta.get('risk_score')
                step_explanation = step_delta.get('risk_explanation')
                
                # Store risk information for this step
                # World Model must return risk_score for each step, raise error if None
                if step_risk is None:
                    error_msg = f"[WorldModel] Risk score not available for step {step_idx + 1} in multi-step prediction"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                step_risks.append(step_risk)
                step_explanations.append(step_explanation or f"Step {step_idx + 1} risk")
                
                # For subsequent steps, we need to generate a plausible next action
                # This is a simplified approach - in practice, you might want to use
                # get_valid_actions() or another method to generate next actions
                if step_idx < num_steps - 1:
                    # Use the predicted state as the new current state
                    current_state = next_state
                    # Generate a plausible next action (simplified - you might want to improve this)
                    # For now, we'll use a placeholder action
                    current_action = {"type": "continue", "step": step_idx + 1}
            
            # Calculate weighted risk score
            # Weight decreases with step depth: step 0 = 1.0, step 1 = 0.5, step 2 = 0.25, etc.
            weighted_risk = self._calculate_weighted_risk(step_risks)
            
            # Log multi-step prediction summary
            logger.debug(
                f"[WorldModel] Multi-step prediction ({num_steps} steps): "
                f"step_risks={step_risks}, weighted_risk={weighted_risk:.3f}"
            )
            
            # Combine risk explanations
            combined_explanation = self._combine_risk_explanations(step_explanations, step_risks, weighted_risk)
            
            # Store multi-step prediction results
            self._last_predicted_delta = {
                'risk_score': weighted_risk,
                'risk_explanation': combined_explanation,
                'multi_step': True,
                'num_steps': num_steps,
                'step_risks': step_risks,
                'step_explanations': step_explanations,
            }
            
            # Return list of predicted states
            return predicted_states
    
    def _simulate_api(self, prompt: str, state: Any, action: Any) -> Any:
        """
        Simulate next state using LLM API with retry on format errors.
        
        Uses unified LLMClient for all providers.
        Automatically retries API calls if JSON parsing fails due to format issues.
        
        Args:
            prompt: Formatted prompt
            state: Current state
            action: Action to simulate
        
        Returns:
            Predicted next state
        """
        # Use unified LLM client
        from ..utils.llm_client import LLMClient
        import time
        
        # Create or reuse LLM client
        if not hasattr(self, '_llm_client'):
            self._llm_client = LLMClient(
                api_key=self.api_key,
                api_url=self.api_url,
                model_name=self.model_name,
                provider=self.provider,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
        
        # Retry configuration for format errors
        max_format_retries = getattr(self, 'max_format_retries', 2)  # Default: 2 retries (3 total attempts)
        format_retry_delay = getattr(self, 'format_retry_delay', 0.5)  # Default: 0.5 seconds
        
        retries = 0
        last_error = None
        
        while retries <= max_format_retries:
            try:
                # Generate text using unified client
                generated_text = self._llm_client.generate(prompt)
                
                # Store generated_text for error handling
                self._last_generated_text = generated_text
                
                # Store token usage from LLM client
                if hasattr(self._llm_client, '_last_token_usage') and self._llm_client._last_token_usage:
                    self._last_token_usage = self._llm_client._last_token_usage.copy()
                
                # Check if generated_text is empty
                if not generated_text or len(generated_text.strip()) == 0:
                    error_msg = "LLM returned empty response"
                    logger.error("=" * 80)
                    logger.error("[World Model] Empty Response Error:")
                    logger.error("=" * 80)
                    logger.error(f"Model: {self.model_name}")
                    logger.error(f"Prompt length: {len(prompt)}")
                    logger.error(f"Prompt preview: {prompt[:500]}...")
                    logger.error(f"Generated text length: {len(generated_text) if generated_text else 0}")
                    logger.error(f"Generated text: {repr(generated_text)}")
                    logger.error("=" * 80)
                    raise ValueError(error_msg)
                
                # Parse generated text to get next state
                next_state = self._parse_llm_output(generated_text, state, action)
                
                # Log World Model prediction result (consolidated log)
                # Note: Use _last_predicted_delta
                if self.use_state_delta:
                    if hasattr(self, '_last_predicted_delta') and self._last_predicted_delta:
                        parsed_delta = self._last_predicted_delta
                        # Ensure parsed_delta is a dict - raise error if not
                        if not isinstance(parsed_delta, dict):
                            error_msg = f"[WorldModel] _last_predicted_delta is not a dict: type={type(parsed_delta)}, value={str(parsed_delta)[:200]}"
                            logger.error(error_msg)
                            raise TypeError(error_msg)
                        # State delta mode: log complete delta information
                        # Include all fields from parsed_delta (complete world model output)
                        # Extract reasoning part from raw_response (text before JSON code block)
                        reasoning_text = ""
                        if generated_text:
                            # Try to extract reasoning part (text before JSON code block)
                            # Match JSON code blocks: ```json ... ``` or ``` ... ```
                            json_block_patterns = [
                                r'```json\s*\n',
                                r'```json\s*',
                                r'```\s*\n',
                                r'```\s*'
                            ]
                            json_start_idx = len(generated_text)
                            for pattern in json_block_patterns:
                                match = re.search(pattern, generated_text, re.IGNORECASE)
                                if match and match.start() < json_start_idx:
                                    json_start_idx = match.start()
                            
                            if json_start_idx < len(generated_text):
                                reasoning_text = generated_text[:json_start_idx].strip()
                            else:
                                # No JSON block found, use entire text as reasoning
                                reasoning_text = generated_text.strip()
                        
                        # Extract task goal from state if available
                        task_goal = None
                        if isinstance(state, dict):
                            task_goal = state.get("goal", None)
                        
                        world_model_log_json = {
                            "component": "World Model",
                            "mode": "state_delta",
                            "action": str(action) if action else None,
                            "task_goal": task_goal,  # Add task goal to log
                            "reasoning": reasoning_text,  # Only include reasoning part
                            **parsed_delta  # Include all fields from parsed_delta
                        }
                        logger.info(
                            f"[World Model] Prediction Result\n{json.dumps(world_model_log_json, indent=2, ensure_ascii=False)}"
                        )
                    else:
                        # Fallback: log raw response if delta parsing failed
                        # Extract task goal from state if available
                        task_goal = None
                        if isinstance(state, dict):
                            task_goal = state.get("goal", None)
                        
                        world_model_log_json = {
                            "component": "World Model",
                            "mode": "state_delta",
                            "action": str(action) if action else None,
                            "task_goal": task_goal,  # Add task goal to log
                            "raw_response": generated_text[:500] + "..." if len(generated_text) > 500 else generated_text,
                            "raw_response_length": len(generated_text),
                        }
                        logger.info(
                            f"[World Model] Prediction Result\n{json.dumps(world_model_log_json, indent=2, ensure_ascii=False)}"
                        )
                else:
                    # Full state mode: log parsed state preview
                    # Extract task goal from state if available
                    task_goal = None
                    if isinstance(state, dict):
                        task_goal = state.get("goal", None)
                    
                    parsed_state_str = str(next_state)
                    world_model_log_json = {
                        "component": "World Model",
                        "mode": "full_state",
                        "action": str(action) if action else None,
                        "task_goal": task_goal,  # Add task goal to log
                        "predicted_state_preview": parsed_state_str[:500] + "..." if len(parsed_state_str) > 500 else parsed_state_str,
                        "predicted_state_length": len(parsed_state_str),
                    }
                    logger.info(
                        f"[World Model] Prediction Result\n{json.dumps(world_model_log_json, indent=2, ensure_ascii=False)}"
                    )
                
                return next_state
                
            except ValueError as e:
                # Check if this is a JSON parsing error (format issue)
                error_str = str(e)
                is_format_error = "Failed to parse JSON" in error_str or "Failed to parse" in error_str
                
                if is_format_error and retries < max_format_retries:
                    # Format error - retry API call
                    retries += 1
                    last_error = e
                    logger.warning(
                        f"[World Model] JSON format error detected (attempt {retries}/{max_format_retries + 1}), "
                        f"retrying API call after {format_retry_delay}s delay..."
                    )
                    time.sleep(format_retry_delay)
                    continue  # Retry the API call
                else:
                    # Not a format error, or max retries exceeded - raise immediately
                    error_msg = f"LLM simulation failed: {e}"
                    logger.error(f"{error_msg} | Prompt: {prompt[:200]}... | State: {state} | Action: {action}")
                    # Log the generated text if available (for debugging)
                    if hasattr(self, '_last_generated_text') and self._last_generated_text:
                        logger.error("=" * 80)
                        logger.error("[World Model] Error - Full LLM Output:")
                        logger.error("=" * 80)
                        logger.error(f"Raw LLM Output (length: {len(self._last_generated_text)}):")
                        logger.error(self._last_generated_text)
                        logger.error("=" * 80)
                    raise RuntimeError(error_msg) from e
                    
            except Exception as e:
                # Other types of errors (not format-related) - raise immediately without retry
                error_msg = f"LLM simulation failed: {e}"
                logger.error(f"{error_msg} | Prompt: {prompt[:200]}... | State: {state} | Action: {action}")
                # Log the generated text if available (for debugging)
                if hasattr(self, '_last_generated_text') and self._last_generated_text:
                    # Always log the full output for debugging (JSON parsing errors are already logged in _parse_llm_output)
                    # But also log here for other types of errors
                    if not (hasattr(e, '__cause__') and isinstance(e.__cause__, ValueError) and "Failed to parse JSON" in str(e.__cause__)):
                        # Not a JSON parsing error (or not logged yet), log the output
                        logger.error("=" * 80)
                        logger.error("[World Model] Error - Full LLM Output:")
                        logger.error("=" * 80)
                        logger.error(f"Raw LLM Output (length: {len(self._last_generated_text)}):")
                        logger.error(self._last_generated_text)
                        logger.error("=" * 80)
                raise RuntimeError(error_msg) from e
        
        # If we exhausted all retries, raise the last error
        if last_error:
            error_msg = f"LLM simulation failed after {max_format_retries + 1} attempts (format errors): {last_error}"
            logger.error(f"{error_msg} | Prompt: {prompt[:200]}... | State: {state} | Action: {action}")
            if hasattr(self, '_last_generated_text') and self._last_generated_text:
                logger.error("=" * 80)
                logger.error("[World Model] Final Error After Retries - Full LLM Output:")
                logger.error("=" * 80)
                logger.error(f"Raw LLM Output (length: {len(self._last_generated_text)}):")
                logger.error(self._last_generated_text)
                logger.error("=" * 80)
            raise RuntimeError(error_msg) from last_error
    
    
    def get_valid_actions(self, state: Any) -> List[Any]:
        """
        Get valid actions using LLM to analyze state.
        
        Args:
            state: Current state
        
        Returns:
            List of valid actions
        
        Raises:
            ValueError: If API credentials are not configured
            RuntimeError: If LLM API call fails
        """
        if not self._model_loaded:
            self._load_model()
        
        # Check API availability
        if not hasattr(self, '_use_api') or not self._use_api or not self.api_key or not self.api_url:
            error_msg = f"LLM API not configured. Please set api_key and api_url for provider '{self.provider}'."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            from .prompts import PromptTemplates
            
            # Create or reuse LLM client
            if not hasattr(self, '_llm_client'):
                from ..utils.llm_client import LLMClient
                self._llm_client = LLMClient(
                    api_key=self.api_key,
                    api_url=self.api_url,
                    model_name=self.model_name,
                    provider=self.provider,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                )
            
            # Create prompt for action generation
            state_str = self._to_string(state)
            prompt = PromptTemplates.action_generation(state_str)
            
            # Call LLM
            response = self._llm_client.generate(prompt, temperature=0.7, max_tokens=500)
            
            # Parse actions from response
            actions = self._parse_actions(response)
            if not actions:
                error_msg = "Failed to parse actions from LLM response. No valid actions found."
                logger.error(f"{error_msg} | Response: {response[:200]}")
                raise ValueError(error_msg)
            
            return actions
                    
        except Exception as e:
            error_msg = f"LLM action generation failed: {e}"
            logger.error(f"{error_msg} | State: {state}")
            raise RuntimeError(error_msg) from e
    
    def _parse_actions(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse actions from LLM response.
        
        Args:
            text: LLM response text
        
        Returns:
            List of action dictionaries
        
        Raises:
            ValueError: If actions cannot be parsed from response
        """
        # Try to parse JSON array
        actions = parse_json_array_from_text(text)
        
        if actions:
            # Validate and normalize actions
            valid_actions = []
            for action in actions:
                if isinstance(action, dict) and "type" in action:
                    valid_actions.append(normalize_action(action))
            
            if valid_actions:
                return valid_actions
            else:
                error_msg = "No valid actions found in parsed JSON array."
                logger.error(f"{error_msg} | Response: {text[:200]}")
                raise ValueError(error_msg)
        
        # If parsing fails, raise error
        error_msg = "Failed to parse actions from LLM response. No valid JSON array found."
        logger.error(f"{error_msg} | Response: {text[:200]}")
        raise ValueError(error_msg)
    
    def simulate_batch(self, states: List[Any], actions: List[Any]) -> List[Any]:
        """
        Simulate state transitions for a batch (optimized for LLM inference).
        
        Args:
            states: List of current states
            actions: List of actions
        
        Returns:
            List of predicted next states
        """
        # Could batch LLM calls for efficiency
        # For now, use sequential simulation
        return super().simulate_batch(states, actions)


# Default world model factory
def WorldModel(
    llm_config: Optional[Dict[str, Any]] = None,
    use_few_shot: bool = False,
    use_chain_of_thought: bool = False,
    require_json_output: bool = True,
    few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> BaseWorldModel:
    """
    Factory function to create world models.
    
    Args:
        llm_config: Optional LLM configuration dictionary (from SafetyConfig.get_llm_config())
        use_few_shot: Whether to use few-shot learning (default: False)
        use_chain_of_thought: Whether to use chain-of-thought reasoning (default: False)
        require_json_output: Whether to require JSON format output (default: True)
        few_shot_examples: List of example dicts with keys: 'state', 'action', 'next_state'
        **kwargs: Additional arguments passed to model constructor
    
    Returns:
        World model instance (LLMBasedWorldModel)
    
    Example:
        # Using config from SafetyConfig
        config = SafetyConfig.from_yaml("config.yaml")
        llm_config = config.get_llm_config("world_model")
        world_model = WorldModel(llm_config=llm_config)
        
        # With prompt optimizations
        examples = [
            {"state": {...}, "action": {...}, "next_state": {...}},
            {"state": {...}, "action": {...}, "next_state": {...}}
        ]
        world_model = WorldModel(
            llm_config=llm_config,
            use_few_shot=True,
            use_chain_of_thought=True,
            require_json_output=True,
            few_shot_examples=examples
        )
    """
    if llm_config:
        kwargs["llm_config"] = llm_config
    # Add prompt optimization parameters
    kwargs["use_few_shot"] = use_few_shot
    kwargs["use_chain_of_thought"] = use_chain_of_thought
    kwargs["require_json_output"] = require_json_output
    kwargs["few_shot_examples"] = few_shot_examples
    return LLMBasedWorldModel(**kwargs)

