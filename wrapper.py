"""
Universal SafePred Wrapper for Easy Integration.

This module provides a simple, benchmark-agnostic wrapper for integrating
SafePred with any benchmark. The wrapper automatically handles format conversion
using benchmark adapters.
"""

import json
import re
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path

from .agent.agent import SafeAgent
from .config.config import SafetyConfig
from .adapters.base import BenchmarkAdapter, get_adapter
from .utils.logger import get_logger
from .utils.parsers import parse_json_array_from_text
from .core.policy_manager import PolicyManager
from .core.plan_monitor import PlanMonitor
from .utils.llm_client import LLMClient

logger = get_logger("SafePred.Wrapper")


class SafePredWrapper:
    """
    Universal wrapper for SafePred integration with any benchmark.
    
    This wrapper automatically handles format conversion between benchmark-specific
    formats and SafePred's standard format using adapters.
    
    Usage:
        # Initialize wrapper
        wrapper = SafePredWrapper(
            benchmark="visualwebarena",
            config_path="config/config.yaml",
            policy_path="policies/my_policies.json"
        )
        
        # Evaluate action risk
        result = wrapper.evaluate_action_risk(
            state=benchmark_state,
            action=benchmark_action,
            candidate_actions=[action1, action2, ...],
            intent="Task description",
            metadata={"action_history": [...]}
        )
    """
    
    def __init__(
        self,
        benchmark: str,
        config_path: Optional[str] = None,
        policy_path: Optional[str] = None,
        web_agent_model_name: Optional[str] = None,
        use_planning: bool = False,
        web_agent_llm_config: Optional[Dict[str, Any]] = None,
        web_agent_prompt_template: Optional[str] = None,
        web_agent_prompt_constructor: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize SafePred wrapper.
        
        Args:
            benchmark: Benchmark name (e.g., "visualwebarena", "mind2web")
            config_path: Path to SafePred config YAML file (optional)
            policy_path: Path to policy JSON file (optional)
            web_agent_model_name: Name of the web agent model for logging
            use_planning: Whether to enable plan monitoring
            web_agent_llm_config: Optional LLM configuration dict for action generation.
                                 If None, will try to use action_agent_llm config from config.yaml.
                                 Format: {
                                     "provider": "openai" | "qwen" | "custom",
                                     "api_key": "...",
                                     "api_url": "...",
                                     "model_name": "...",
                                     "temperature": 0.7,
                                     "max_tokens": 512,
                                     ...
                                 }
            web_agent_prompt_template: Optional custom prompt template for action generation.
                                      If None, uses default template.
            **kwargs: Additional arguments passed to SafeAgent
        """
        self.benchmark = benchmark.lower()
        
        # Get benchmark adapter
        try:
            adapter_class = get_adapter(self.benchmark)
            self.adapter = adapter_class()
            logger.info(f"Using adapter for benchmark: {self.benchmark}")
        except KeyError:
            raise ValueError(
                f"Adapter for benchmark '{benchmark}' not found. "
                f"Please implement a BenchmarkAdapter for this benchmark. "
                f"See SafePred_v3/adapters/base.py for the interface."
            )
        
        # Store benchmark name for logging
        self.benchmark = self.benchmark
        
        # Load SafePred configuration
        if config_path and Path(config_path).exists():
            # Load .env from SafePred repo root (parent of config dir) so api_key/api_url are available when from_yaml reads env (e.g. when running from WASP subprocess)
            try:
                from dotenv import load_dotenv
                repo_root = Path(config_path).resolve().parent.parent
                env_file = repo_root / ".env"
                if env_file.exists():
                    load_dotenv(env_file)
            except Exception as e:
                logger.debug(f"Optional .env load from repo root: {e}")
            logger.info(f"Loading SafePred config from {config_path}")
            self.config = SafetyConfig.from_yaml(config_path)
        else:
            if config_path:
                logger.warning(f"SafePred config path not found or not a file: {config_path}. Using default config (planning.enable=False, etc.).")
            else:
                logger.info("Using default SafePred config")
            self.config = SafetyConfig()
        
        # Load policies (only from initialization parameter)
        policies = self._load_policies(policy_path)
        self.policies = policies
        self.policy_path = policy_path
        
        self.use_planning = use_planning or getattr(self.config, 'planning_enable', False)
        # Expose for WASP/benchmarks: risk_guidance_enable controls whether to use optimization_guidance when updating plan
        self.enable_risk_guidance = getattr(self.config, 'risk_guidance_enable', True)
        self.plan_monitor = None
        if self.use_planning:
            # Plan monitor will be initialized after SafeAgent is created
            logger.info("[SafePredWrapper] Planning feature enabled")
        
        # Initialize PolicyManager if policy path is provided
        self.policy_manager = None
        if policy_path and Path(policy_path).exists():
            try:
                # Get reference limits from config (default: high=10, medium=7, low=5)
                reference_limits = getattr(self.config, 'reference_limits', None)
                if reference_limits is None:
                    reference_limits = {"high": 10, "medium": 7, "low": 5}
                
                # Get similarity threshold from config or use default
                similarity_threshold = getattr(self.config, 'similarity_threshold', 0.85)
                
                # Default: enable dynamic update with sync mode for immediate file writes
                self.policy_manager = PolicyManager(
                    policy_file_path=policy_path,
                    enable_cache=True,
                    update_mode="sync",  # Sync mode to ensure immediate file writes
                    similarity_threshold=similarity_threshold,  # From config.yaml or default 0.85
                    reference_limits=reference_limits  # Pass reference limits from config
                )
                logger.info(f"PolicyManager initialized with dynamic update enabled (sync mode for immediate saves, reference_limits={reference_limits})")
            except Exception as e:
                logger.warning(f"Failed to initialize PolicyManager: {e}, continuing without dynamic updates")
        
        # Initialize SafeAgent
        root_threshold = getattr(self.config, 'root_risk_threshold', 0.7)
        child_threshold = getattr(self.config, 'child_risk_threshold', 0.8)
        logger.info(f"Initializing SafeAgent with root_risk_threshold={root_threshold}, child_risk_threshold={child_threshold}")
        if policies:
            logger.info(f"[SafePredWrapper] Loaded {len(policies)} policies from {policy_path if policy_path else 'default location'}")
            logger.info(f"[SafePredWrapper] Policies will be passed to SafeAgent and then to WorldModel.simulate()")
            # Log first policy as example
            if len(policies) > 0:
                first_policy = policies[0]
                logger.debug(f"[SafePredWrapper] Example policy: ID={first_policy.get('policy_id', 'N/A')}, "
                           f"Description={first_policy.get('policy_description', 'N/A')[:50]}...")
        else:
            logger.warning("No policies loaded - SafePred will run without policy-based risk evaluation")
        
        self.safe_agent = SafeAgent(
            config=self.config,
            benchmark=self.benchmark,
            web_agent_model_name=web_agent_model_name,
            policies=policies,  # Pass policies list (not path) to SafeAgent
            **kwargs
        )
        
        # Verify policies are stored in SafeAgent
        if policies and hasattr(self.safe_agent, 'policies'):
            logger.info(f"[SafePredWrapper] Verified: SafeAgent has {len(self.safe_agent.policies)} policies stored "
                       f"(will be passed to WorldModel.simulate())")
        
        if self.use_planning and self.safe_agent:
            # Get LLM config from config for PlanMonitor (api_key/api_url from .env by provider)
            world_llm_config = self.config.get_llm_config("world_model")
            if not world_llm_config:
                raise ValueError("world_model LLM config is required for PlanMonitor. Please configure it in config.yaml")
            if not world_llm_config.get("api_key") or not world_llm_config.get("api_url"):
                provider = world_llm_config.get("provider", "openai")
                env_hint = f"Set {provider.upper()}_API_KEY and {provider.upper()}_API_URL in .env at SafePred repo root"
                if config_path:
                    env_path = Path(config_path).resolve().parent.parent / ".env"
                    env_hint += f" (e.g. {env_path})"
                env_hint += " or in config.yaml world_model_llm section."
                raise ValueError(
                    f"PlanMonitor requires world_model LLM api_key and api_url. {env_hint}"
                )
            self.plan_monitor = PlanMonitor(
                llm_config=world_llm_config,
            )
            logger.info("[SafePredWrapper] PlanMonitor initialized")
        
        # Store web agent configuration for action_generator
        self.web_agent_model_name = web_agent_model_name
        self.web_agent_prompt_template = web_agent_prompt_template
        
        # Get web_agent_llm_config: use provided config, or extract from SafetyConfig
        if web_agent_llm_config is None:
            # Try to get from config.yaml (action_agent_llm section)
            try:
                web_agent_llm_config = self.config.get_llm_config("action_agent")
                if web_agent_llm_config and web_agent_llm_config.get("api_key") and web_agent_llm_config.get("model_name"):
                    logger.info(f"Using action_agent_llm config from config.yaml: {web_agent_llm_config.get('model_name')}")
                else:
                    # Not configured - will raise error if max_depth > 1 and action_generator is needed
                    web_agent_llm_config = None
            except ValueError as e:
                # Configuration is missing or invalid - raise error immediately
                error_msg = (
                    f"action_agent_llm configuration is required but not properly configured: {e}. "
                    "Please configure action_agent_llm section in config.yaml with model_name and api_key."
                )
                logger.error(error_msg)
                raise ValueError(error_msg) from e
        
        self.web_agent_llm_config = web_agent_llm_config
        
        # Initialize web agent LLM client - required for action_generator when max_depth > 1
        self.web_agent_llm_client = None
        if web_agent_llm_config:
            try:
                self.web_agent_llm_client = LLMClient.from_config(web_agent_llm_config)
                logger.info(f"Initialized web agent LLM client: {web_agent_llm_config.get('model_name', 'unknown')}")
            except Exception as e:
                error_msg = f"Failed to initialize web agent LLM client: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
        
        # Print world model LLM config information
        if self.safe_agent and hasattr(self.safe_agent, 'world_model'):
            try:
                world_llm_config = self.config.get_llm_config("world_model")
                if world_llm_config and world_llm_config.get("model_name"):
                    logger.info(f"Using world_model_llm config from config.yaml: {world_llm_config.get('model_name')}")
                    # Try to get the actual model name from world_model if available
                    world_model_name = getattr(self.safe_agent.world_model, 'model_name', None)
                    if world_model_name:
                        logger.info(f"Initialized world model LLM client: {world_model_name}")
                else:
                    logger.warning("world_model_llm config not found or incomplete in config.yaml")
            except Exception as e:
                logger.warning(f"Failed to get world_model LLM config info: {e}")
        
        logger.info("SafePred wrapper initialized successfully")
    
    def format_policies_for_prompt(self) -> str:
        """Format policies as a string to be included in the agent's prompt."""
        return self.adapter.format_policies_for_prompt(self.policies)
    
    def _create_default_action_generator(
        self, 
        web_agent_prompt: Optional[List[Dict[str, str]]] = None,  # Backward compatibility
        action_context: Optional[Dict[str, Any]] = None,
        prompt_builder: Optional[Callable[[Any, Dict[str, Any]], List[Dict[str, str]]]] = None,
    ) -> Callable[[Any, str, int], List[str]]:
        """
        Create default action_generator using web agent LLM configuration.
        
        This method supports multiple ways to provide context for action generation:
        1. prompt_builder (recommended): A callback function that builds prompt from predicted state and context
        2. action_context: Abstract context information (plan, intent, history, etc.)
        3. web_agent_prompt (backward compatibility): Full prompt from web agent
        
        Args:
            web_agent_prompt: Optional web agent's full prompt (list of message dicts). Deprecated.
            action_context: Optional abstract context dict with keys:
                           - plan_text: Plan text
                           - intent: Task intent
                           - conversation_history: Optional conversation history
                           - policies_text: Optional policies text
            prompt_builder: Optional callback function(predicted_state, context) -> List[Dict[str, str]]
                          Should return a prompt (list of message dicts) based on predicted state and context.
                          Context dict contains: plan_text, intent, conversation_history, policies_text, etc.
        
        Returns:
            action_generator callback function(state, risk_guidance, num_samples) -> List[str]
        
        Raises:
            ValueError: If web_agent_llm_config was not provided during initialization
        """
        # Get LLM client for web agent - must be provided
        if self.web_agent_llm_client is None:
            error_msg = (
                "[Wrapper] web_agent_llm_config is required for action_generator when max_depth > 1. "
                "Please provide web_agent_llm_config in SafePredWrapper.__init__() "
                "or set action_agent_llm in config.yaml."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        llm_client = self.web_agent_llm_client
        
        # Store context for use in action_generator
        # Priority: prompt_builder > action_context > web_agent_prompt
        use_prompt_builder = prompt_builder is not None
        use_action_context = action_context is not None and not use_prompt_builder
        use_web_agent_prompt = web_agent_prompt is not None and not use_prompt_builder and not use_action_context
        
        # Log which context method is being used
        if use_prompt_builder:
            logger.info(f"[Wrapper] [Action Agent] Context source: PROMPT_BUILDER (callback function)")
        elif use_action_context:
            logger.info(f"[Wrapper] [Action Agent] Context source: ACTION_CONTEXT (dict with plan, intent, etc.)")
        elif use_web_agent_prompt:
            logger.info(f"[Wrapper] [Action Agent] Context source: WEB_AGENT_PROMPT (full prompt from web agent, {len(web_agent_prompt)} messages)")
        else:
            logger.warning(f"[Wrapper] [Action Agent] Context source: NONE (will use simple prompt template)")
        
        # Store base_prompt for use in closure (deep copy to avoid modification)
        base_prompt = None
        if use_web_agent_prompt and web_agent_prompt is not None:
            import copy
            base_prompt = copy.deepcopy(web_agent_prompt)
        
        # Get prompt template (only used if web_agent_prompt is None)
        prompt_template = self.web_agent_prompt_template
        if prompt_template is None:
            # Use default prompt template
            prompt_template = self._get_default_action_prompt_template()
        
        def default_action_generator(
            safepred_state: Any, 
            risk_guidance: str, 
            num_samples: int,
            simulated_path: Optional[List[str]] = None  # Path of actions executed in simulation
        ) -> List[str]:
            """
            Default action generator that uses web agent LLM to generate actions based on predicted state.
            
            This function is called for each depth expansion in tree search:
            - Root node (depth 0): Not called (root actions are provided directly)
            - Depth 1: Called once per root path, with predicted state after root action
                      simulated_path = [root_action]
            - Depth 2: Called once per depth-1 path, with predicted state after depth-1 action
                      simulated_path = [root_action, depth1_action]
            - etc.
            
            Key logic:
            1. Each call receives a predicted_state from World Model (state after executing simulated_path)
            2. We start from base_prompt (original web agent prompt with initial conversation history)
            3. We add simulated_path actions as assistant messages to reflect the execution path
            4. We add a new user message with the predicted state (observation after simulated_path)
            5. Action agent generates next actions based on this predicted state
            
            Conversation history structure:
            base_prompt -> ... -> user -> assistant(action1) -> assistant(action2) -> ... -> assistant(actionN) -> user(predicted_state)
            
            This ensures:
            1. Context (plan, policies, initial history) remains consistent from base_prompt
            2. Conversation history correctly reflects the simulated execution path
            3. Predicted state represents the observation after executing all actions in simulated_path
            4. Action agent generates next actions based on the correct predicted state
            
            Args:
                safepred_state: SafePred compact state format (predicted state from World Model)
                               This is the state AFTER executing all actions in simulated_path
                risk_guidance: Guidance for generating safer actions (from risk assessment)
                num_samples: Number of actions to generate (N_child)
                simulated_path: Optional list of action strings executed in the simulation path from root to current node.
                               - For depth 1: [root_action]
                               - For depth 2: [root_action, depth1_action]
                               - If None or empty, no actions are added (root node case)
            
            Returns:
                List of action strings in SafePred format
            """
            try:
                # Log action generation start
                depth = len(simulated_path) if simulated_path else 0
                logger.info(f"[Wrapper] [Action Agent] Generating {num_samples} action(s) at depth {depth + 1}")
                
                # If web_agent_prompt is provided, use it (replacing the last user message with predicted state)
                # Otherwise, use simple prompt template
                if base_prompt is not None:
                    # IMPORTANT: Always start from the original base_prompt (not a modified version)
                    # This ensures that for each depth expansion, we use the same base context
                    # and only update the current observation (last user message)
                    import copy
                    prompt = copy.deepcopy(base_prompt)
                    
                    # Update conversation history to match simulated path if provided
                    # This ensures that the prompt's conversation history is consistent with the simulation path
                    # 
                    # IMPORTANT: WASP's prompt structure:
                    # - base_prompt contains: [system(intro), system(examples), system(policies), system(plan), user(current_obs)]
                    # - WASP does NOT maintain conversation history (user-assistant pairs) in the prompt
                    # - Each call to construct() creates a fresh prompt from intro and examples
                    # 
                    # For simulated paths, we need to ADD conversation history to reflect the execution:
                    # - simulated_path contains all actions from root to current node (e.g., [root_action, depth1_action])
                    # - For depth 1: simulated_path = [root_action] (one action executed)
                    # - For depth 2: simulated_path = [root_action, depth1_action] (two actions executed)
                    # 
                    # Conversation history structure after adding simulated path:
                    # base_prompt -> ... -> user(initial_obs) -> assistant(action1) -> user(obs_after_action1) -> assistant(action2) -> ... -> user(predicted_state)
                    if simulated_path and len(simulated_path) > 0:
                        # Find the last user message index in base_prompt
                        # This is the initial observation before any simulated actions
                        last_user_idx = -1
                        for i in range(len(prompt) - 1, -1, -1):
                            if prompt[i].get("role") == "user":
                                last_user_idx = i
                                break
                        
                        if last_user_idx >= 0:
                            # Insert simulated actions as assistant messages after the last user message
                            # For each action, we add: assistant(action) -> user(observation_after_action)
                            # But since we only have the final predicted state, we add all actions first, then the final user message
                            insert_position = last_user_idx + 1
                            
                            # Add each action in simulated_path as an assistant message
                            # Each action represents a step executed in the simulation path from root to current node
                            # 
                            # IMPORTANT: Format consistency with web agent's prompt
                            # - Web agent's response format: "Let's think step-by-step. [reasoning] In summary, the next action I will perform is ```action```"
                            # - In simulated path, we only have action strings (e.g., "click [123]")
                            # - We use action strings directly as assistant messages to reflect the execution path
                            # - This is consistent because the action string is the key information extracted from web agent's response
                            for idx, action_str in enumerate(simulated_path):
                                # Insert assistant message with the action
                                # Format: {"role": "assistant", "content": action_str}
                                # This matches the format that would be extracted from web agent's response
                                prompt.insert(insert_position + idx, {"role": "assistant", "content": action_str})
                            
                        else:
                            logger.warning(f"[Wrapper] No user message found in base_prompt, cannot add simulated path actions")
                    
                    # Convert predicted state to the format expected by the prompt
                    # This is the state AFTER executing all actions in simulated_path (predicted by World Model)
                    # 
                    # IMPORTANT: Format consistency with web agent's prompt template
                    # Web agent's user message format (from template):
                    #   "OBSERVATION:\n{observation}\nURL: {url}\nOBJECTIVE: {objective}\nPREVIOUS ACTION: {previous_action}"
                    # 
                    # We format the predicted state to match this structure as closely as possible
                    # while indicating it's a predicted state (not real observation)
                    state_str = self._format_state_for_action_prompt(safepred_state)
                    
                    # Extract information from safepred_state to match web agent's format
                    observation_part = state_str  # Already formatted state information
                    url_part = safepred_state.get("url", "Unknown URL") if isinstance(safepred_state, dict) else "Unknown URL"
                    objective_part = safepred_state.get("goal", "Complete the task") if isinstance(safepred_state, dict) else "Complete the task"
                    previous_action_part = simulated_path[-1] if simulated_path and len(simulated_path) > 0 else "None"
                    
                    # Build user message in format similar to web agent's template
                    # We add "(predicted by World Model)" prefix to indicate this is a simulated state
                    # Format matches web agent's template exactly
                    # IMPORTANT: Do NOT add extra generation instructions here, because base_prompt already contains:
                    # - intro with format requirements (rule 4: "Generate the action in the correct format...")
                    # - examples showing the correct format
                    # We should let LLM naturally follow the format from base_prompt, just like web agent does
                    predicted_state_message = f"OBSERVATION (predicted by World Model):\n{observation_part}\nURL: {url_part}\nOBJECTIVE: {objective_part}\nPREVIOUS ACTION: {previous_action_part}"
                    
                    # Append new user message with predicted state (after all simulated actions)
                    # This represents the observation after executing the simulated path
                    prompt.append({"role": "user", "content": predicted_state_message})
                    
                    # Log full prompt only when enabled (avoid privacy leak in logs)
                    prompt_str = "\n".join([
                        f"Message {i+1} ({msg.get('role', 'unknown')}):\n{msg.get('content', '')}"
                        for i, msg in enumerate(prompt)
                    ])
                    if getattr(self.config, "action_agent_log_prompt", False):
                        logger.info("=" * 80)
                        logger.info(f"[Wrapper] [Action Agent] Full Prompt:")
                        logger.info("=" * 80)
                        logger.info(prompt_str)
                        logger.info("=" * 80)
                    else:
                        logger.info(f"[Wrapper] [Action Agent] Prompt length: {len(prompt_str)} chars, preview: {prompt_str[:200]}...")
                    
                    # For chat format prompt, use generate with messages parameter
                    # LLMClient.generate() now supports messages-only calls (prompt is optional)
                    response = llm_client.generate(
                        messages=prompt,  # Pass messages list directly
                        temperature=0.7,
                        max_tokens=512
                    )
                else:
                    # Use simple prompt template (fallback)
                    state_str = self._format_state_for_action_prompt(safepred_state)
                    
                    # Build prompt
                    prompt = prompt_template.format(
                        state=state_str,
                        num_samples=num_samples,
                        goal=safepred_state.get("goal", "Complete the task"),
                        policies=self.format_policies_for_prompt() if self.policies else "",
                        risk_guidance=risk_guidance if risk_guidance else ""
                    )
                    
                    # Log full prompt only when enabled (avoid privacy leak in logs)
                    if getattr(self.config, "action_agent_log_prompt", False):
                        logger.info("=" * 80)
                        logger.info(f"[Wrapper] [Action Agent] Full Prompt:")
                        logger.info("=" * 80)
                        logger.info(prompt)
                        logger.info("=" * 80)
                    else:
                        logger.info(f"[Wrapper] [Action Agent] Prompt length: {len(prompt)} chars, preview: {prompt[:200]}...")
                    
                    response = llm_client.generate(
                        prompt=prompt,
                        temperature=0.7,  # Use moderate temperature for diversity
                        max_tokens=512
                    )
                
                # Parse actions from response
                actions = self._parse_actions_from_response(response, num_samples)
                
                if not actions:
                    error_msg = f"Failed to parse actions from LLM response. Expected {num_samples} actions."
                    logger.error(f"{error_msg} | Response: {response[:200]}")
                    raise ValueError(error_msg)
                
                logger.info(f"[Wrapper] [Action Agent] Generated {len(actions)} actions")
                
                return actions
                
            except Exception as e:
                error_msg = f"Default action_generator failed: {e}"
                logger.error(error_msg, exc_info=True)
                raise RuntimeError(error_msg) from e
        
        return default_action_generator
    
    def _convert_to_compact_state(self, state: Any, intent: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Convert raw state -> SafePred state -> compact state for World Model."""
        effective_intent = intent or (metadata.get("intent", "") if metadata else "")
        safepred_state = self.adapter.state_to_safepred(
            raw_state=state,
            intent=effective_intent,
            metadata=metadata
        )
        compact_state = self.safe_agent._preprocess_state(safepred_state)
        
        return compact_state
    
    def _convert_path_to_safepred(self, path: Any, action_str: str) -> Optional[List[str]]:
        """Convert path to list of SafePred-format action strings."""
        if not path:
            return None
        if isinstance(path, list) and len(path) > 0:
            return [self.adapter.action_to_safepred(a) if not isinstance(a, str) else a for a in path]
        return [action_str] if action_str else None
    
    def _determine_plan_update(
        self,
        requires_regeneration: bool,
        should_check_plan: bool,
        optimization_guidance: str,
        risk_guidance: str,
    ) -> Dict[str, Any]:
        """Determine whether to update plan and return update info (should_update_plan, update_reason, optimization_guidance)."""
        # Plan update should use optimization_guidance (from PlanMonitor or World Model), not risk_guidance (for action regeneration)
        # Action regeneration doesn't require plan update - risk_guidance is for action generation, not plan generation
        # Plan can be updated when:
        # 1. Plan misaligned (should_check_plan=True and optimization_guidance from PlanMonitor)
        # 2. Policy violation detected (optimization_guidance from World Model)
        should_update_plan = bool(optimization_guidance)  # Update if any optimization_guidance is available
        
        if not should_update_plan:
            return {
                'should_update_plan': False,
                'update_reason': "",
                'optimization_guidance': "",
            }
        else:
            # Determine update reason based on source
            if should_check_plan:
                # Plan misaligned - optimization_guidance from PlanMonitor
                update_reason = "Path feasible but plan misaligned"
            else:
                # Policy violation - optimization_guidance from World Model
                update_reason = "Policy violation detected, plan needs correction"
            
            return {
                'should_update_plan': True,
                'update_reason': update_reason,
                'optimization_guidance': optimization_guidance,
            }
    
    def _format_state_for_action_prompt(self, state: Any) -> str:
        """Format state for action generation prompt."""
        if isinstance(state, dict):
            # Format compact state
            parts = []
            if "axtree_txt" in state:
                parts.append(f"Page Content:\n{state['axtree_txt'][:2000]}")  # Limit length
            elif "key_elements" in state:
                # Use key_elements if axtree_txt is not available
                elements_str = "\n".join([f"- {elem.get('label', '')} (type: {elem.get('type', 'unknown')})" for elem in state.get('key_elements', [])[:50]])
                parts.append(f"Available Elements:\n{elements_str}")
            if "url" in state and state.get("url"):
                parts.append(f"Current URL: {state['url']}")
            if "page_type" in state:
                parts.append(f"Page Type: {state['page_type']}")
            if "goal" in state:
                parts.append(f"Task Goal: {state['goal']}")
            return "\n\n".join(parts) if parts else "No state information available"
        else:
            return str(state)[:2000]  # Limit length
    
    def _get_default_action_prompt_template(self) -> str:
        """Get default prompt template for action generation."""
        return """You are a web agent that generates diverse actions to accomplish a task.

Task Goal: {goal}

Current State (predicted by World Model):
{state}

{risk_guidance}

{policies}

Generate exactly {num_samples} diverse actions that could be taken from the current predicted state to progress toward the task goal. 
Actions should be diverse (different approaches, different elements) to explore multiple paths.
If risk_guidance is provided, prioritize safer actions that avoid the mentioned risks.

Output format: Return a JSON array of action strings, each in SafePred format (e.g., "click [element_id]", "type [element_id] [text]", "navigate [url]").

Example output:
["click [123]", "type [456] search query", "navigate https://example.com"]

Actions:"""
    
    def _clean_action_string(self, action: str) -> str:
        """
        Clean action string by removing markdown code block markers and extra whitespace.
        
        Args:
            action: Raw action string
        
        Returns:
            Cleaned action string
        """
        if not action:
            return action
        
        # Remove markdown code block markers (``` at start or end)
        action = action.strip()
        action = re.sub(r'^```+\s*', '', action)  # Remove leading ```
        action = re.sub(r'\s*```+$', '', action)  # Remove trailing ```
        action = action.strip()
        
        return action
    
    def _parse_actions_from_response(self, response: str, expected_count: int) -> List[str]:
        """
        Parse actions from LLM response using the same logic as web agent.
        
        Web agent uses the format: "In summary, the next action I will perform is ```action```"
        We extract actions from ```action``` patterns, matching web agent's extract_action logic.
        
        Args:
            response: LLM response text
            expected_count: Expected number of actions
        
        Returns:
            List of action strings
        """
        # Primary method: Extract actions using web agent's format
        # Pattern: "In summary, the next action I will perform is ```action```"
        # Action splitter is ``` (same as web agent)
        action_splitter = "```"
        # Pattern to match: ```action``` (action can contain newlines)
        pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
        matches = re.findall(pattern, response)
        
        if matches:
            actions = []
            for match in matches:
                action = match[0].strip()  # match[0] is the captured group
                if action:
                    cleaned_action = self._clean_action_string(action)
                    actions.append(cleaned_action)
            
            if actions:
                return actions[:expected_count]  # Limit to expected count
        
        # Fallback 1: Try to parse JSON array
        actions_json = parse_json_array_from_text(response)
        if actions_json:
            # Extract action strings
            actions = []
            for item in actions_json:
                if isinstance(item, str):
                    cleaned_action = self._clean_action_string(item)
                    actions.append(cleaned_action)
                elif isinstance(item, dict):
                    # Try to convert dict to action string
                    action_str = self.adapter.action_to_safepred(item)
                    cleaned_action = self._clean_action_string(action_str)
                    actions.append(cleaned_action)
            
            if actions:
                return actions[:expected_count]  # Limit to expected count
        
        # Fallback 2: Try to extract actions from text using regex patterns
        # Look for action patterns like "click [id]", "type [id] [text]", etc.
        action_patterns = [
            r'click\s+\[[^\]]+\]',
            r'type\s+\[[^\]]+\]\s+[^\n]+',
            r'stop\s+\[[^\]]+\]',  # Add stop action pattern
            r'navigate\s+[^\s\n]+',
            r'submit\s+\[[^\]]+\]',
            r'press\s+\[[^\]]+\]',
        ]
        
        found_actions = []
        for pattern in action_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            found_actions.extend(matches)
        
        if found_actions:
            # Clean all found actions
            cleaned_actions = [self._clean_action_string(action) for action in found_actions]
            return cleaned_actions[:expected_count]
        
        # Fallback 3: Check if response indicates task completion
        # If LLM says task is complete, generate stop action
        completion_keywords = [
            "i have achieved the objective",
            "task is complete",
            "task has been completed",
            "already been posted",
            "already completed",
        ]
        response_lower = response.lower()
        if any(keyword in response_lower for keyword in completion_keywords):
            # Try to extract answer from response if available
            # Look for patterns like "answer is X" or "the answer is X"
            answer_match = re.search(r'(?:answer|result|value)\s+(?:is|:)\s*([^\n.]+)', response_lower)
            if answer_match:
                answer = answer_match.group(1).strip()
                return [f"stop [{answer}]"][:expected_count]
            else:
                # Return generic stop action
                return ["stop [task completed]"][:expected_count]
        
        # Last resort: split by lines and try to extract actions
        lines = response.strip().split('\n')
        actions = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('//'):
                # Try to extract action from line
                if any(keyword in line.lower() for keyword in ['click', 'type', 'navigate', 'submit', 'press', 'stop']):
                    cleaned_line = self._clean_action_string(line)
                    actions.append(cleaned_line)
        
        return actions[:expected_count] if actions else []
    
    def _load_policies(self, policy_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load policies from JSON file."""
        if not policy_path:
            # Try default location
            default_path = Path(__file__).parent / "policies" / "my_policies.json"
            if default_path.exists():
                policy_path = str(default_path)
            else:
                return []
        
        policy_path = Path(policy_path)
        if not policy_path.exists():
            logger.warning(f"Policy file not found: {policy_path}")
            return []
        
        try:
            with open(policy_path, 'r', encoding='utf-8') as f:
                policy_data = json.load(f)
            
            # Support both array format and dict format ({"policies": [...]})
            policies_list = None
            if isinstance(policy_data, list):
                policies_list = policy_data
            elif isinstance(policy_data, dict):
                # Try to extract policies from dict
                if "policies" in policy_data:
                    policies_list = policy_data["policies"]
                else:
                    logger.error(f"Invalid policy format: dict must contain 'policies' key. Got keys: {list(policy_data.keys())}")
                    return []
            else:
                logger.error(f"Invalid policy format: expected array or dict with 'policies' key, got {type(policy_data)}")
                return []
            
            # Validate format
            if not isinstance(policies_list, list):
                logger.error(f"Invalid policy format: 'policies' must be an array, got {type(policies_list)}")
                return []
            
            validated_policies = []
            for policy in policies_list:
                if not isinstance(policy, dict):
                    logger.warning(f"Policy is not a dict, skipping: {type(policy)}")
                    continue
                
                # Normalize field names to SafePred_v8 format
                # Support both SafePred format (id, description, severity) and SafePred_v8 format (policy_id, policy_description, risk_level)
                normalized_policy = {}
                
                # Normalize policy_id
                if "policy_id" in policy:
                    normalized_policy["policy_id"] = policy["policy_id"]
                elif "id" in policy:
                    normalized_policy["policy_id"] = policy["id"]
                else:
                    logger.warning(f"Policy missing policy_id/id, skipping: {policy}")
                    continue
                
                # Normalize policy_description
                if "policy_description" in policy:
                    normalized_policy["policy_description"] = policy["policy_description"]
                elif "description" in policy:
                    normalized_policy["policy_description"] = policy["description"]
                else:
                    logger.warning(f"Policy {normalized_policy.get('policy_id')} missing policy_description/description, skipping")
                    continue
                
                # Normalize risk_level
                if "risk_level" in policy:
                    normalized_policy["risk_level"] = policy["risk_level"]
                elif "severity" in policy:
                    # Map severity to risk_level
                    severity = policy["severity"].lower()
                    if severity in ["high", "medium", "low"]:
                        normalized_policy["risk_level"] = severity
                    else:
                        normalized_policy["risk_level"] = "medium"
                else:
                    normalized_policy["risk_level"] = "medium"
                
                # Copy other fields if present
                if "definitions" in policy:
                    normalized_policy["definitions"] = policy["definitions"]
                if "scope" in policy:
                    normalized_policy["scope"] = policy["scope"]
                if "risk_patterns" in policy:
                    normalized_policy["risk_patterns"] = policy["risk_patterns"]
                if "name" in policy:
                    normalized_policy["name"] = policy["name"]
                # Ensure reference field exists (copy if present, otherwise initialize as empty list)
                if "reference" in policy:
                    normalized_policy["reference"] = policy["reference"]
                else:
                    normalized_policy["reference"] = []
                
                validated_policies.append(normalized_policy)
            
            logger.info(f"Loaded {len(validated_policies)} policies from {policy_path}")
            return validated_policies
        except Exception as e:
            logger.warning(f"Failed to load policies from {policy_path}: {e}")
            return []
    
    def format_policies_for_prompt(self) -> str:
        """Format policies as a string to be included in the agent's prompt."""
        return self.adapter.format_policies_for_prompt(self.policies)
    
    def evaluate_action_risk(
        self,
        state: Any,
        action: Any,
        candidate_actions: Optional[List[Any]] = None,
        intent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        action_generator: Optional[Callable[[Any, str, int], List[str]]] = None,
        web_agent_prompt: Optional[List[Dict[str, str]]] = None,
        action_context: Optional[Dict[str, Any]] = None,
        prompt_builder: Optional[Callable[[Any, Dict[str, Any]], List[Dict[str, str]]]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate risk of an action using SafePred.
        
        Args:
            state: Benchmark-specific state representation
            action: Benchmark-specific action to evaluate
            candidate_actions: Optional list of candidate actions to evaluate
            intent: Task intent/instruction (optional, can be in metadata)
            metadata: Additional metadata with keys:
                - intent: Task intent (if not provided separately)
                - action_history: List of previous actions
                - prompt_injection: Optional prompt injection text
                - Other benchmark-specific metadata
            action_generator: Optional function(state, risk_guidance, num_samples) -> List[str]
                             Used for tree search depth expansion when max_depth > 1.
                             If not provided and max_depth > 1, a simple generator based on candidate_actions will be used.
        
        Returns:
            Dict containing:
                - 'risk_score': Risk score (0.0-1.0)
                - 'is_safe': Whether action is safe (risk < threshold)
                - 'risk_explanation': Explanation of risk
                - 'requires_regeneration': Whether action requires regeneration
                - 'risk_guidance': Guidance for regenerating safer action
        """
        metadata = metadata or {}
        
        # Get intent from metadata if not provided separately
        if intent is None:
            intent = metadata.get("intent", "")
        
        # Set task (by task_id) and intent in conversation history manager
        # Task switching is based on task_id if available, otherwise falls back to intent
        if hasattr(self.safe_agent, 'conversation_history_manager'):
            task_id = metadata.get("task_id") or metadata.get("task_index")  # Support both task_id and task_index
            task_intent = intent or metadata.get("intent", "")
            self.safe_agent.conversation_history_manager.set_task(task_id=task_id, intent=task_intent)
        
        # Convert state to SafePred format
        
        try:
            safepred_state = self.adapter.state_to_safepred(
                raw_state=state,
                intent=intent,
                metadata=metadata
            )
            
            # Add current action reasoning to state from action_context or metadata
            # This allows world model prompt to show full reasoning in Current Step
            # Priority: action_context > metadata
            current_response = None
            if action_context and isinstance(action_context, dict) and "current_response" in action_context:
                current_response = action_context["current_response"]
            elif metadata and isinstance(metadata, dict) and "current_response" in metadata:
                current_response = metadata["current_response"]
            
            if current_response and current_response.strip():
                safepred_state["current_action_reasoning"] = current_response
            else:
                # current_response should always be provided for initial actions
                logger.warning(f"[Wrapper] No current_response found in action_context or metadata. This may indicate a bug.")
            
            # Add plan_text to state for World Model to use (for plan progress tracking)
            # plan_text is added in evaluate_action_risk_with_plan method, but we also preserve it here if it exists
            if isinstance(state, dict) and 'plan_text' in state:
                safepred_state['plan_text'] = state['plan_text']
        except Exception as e:
            import traceback
            logger.error(f"Error converting state to SafePred format: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Log type/size only to avoid privacy leak (state/observation may contain user content)
            logger.error(f"State type: {type(state).__name__}, len: {len(state) if isinstance(state, (list, dict)) else 'N/A'}")
            if isinstance(state, list) and len(state) > 0:
                logger.error(f"First element type: {type(state[0]).__name__}, keys: {list(state[0].keys()) if isinstance(state[0], dict) else 'N/A'}")
            logger.error(f"Metadata type: {type(metadata).__name__}, keys: {list(metadata.keys()) if isinstance(metadata, dict) else 'N/A'}")
            raise
        
        # Convert actions to SafePred format
        # Validate action - no fallback, raise error if invalid
        if action is None:
            error_msg = "[Wrapper] evaluate_action_risk: action is None"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if candidate_actions is None:
            candidate_actions = [action]
        
        # Validate all candidate actions - no fallback, raise error if any is invalid
        # Note: We don't check the type of actions here because the adapter can handle
        # different types (strings, dicts, etc.) depending on the benchmark format
        for i, a in enumerate(candidate_actions):
            if a is None:
                error_msg = f"[Wrapper] evaluate_action_risk: candidate_actions[{i}] is None"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Convert all candidate actions to SafePred format using adapter
        # The adapter handles type conversion (string, dict, etc.) based on benchmark format
        candidate_action_strings = [
            self.adapter.action_to_safepred(a) for a in candidate_actions
        ]
        
        # Evaluate using SafeAgent
        try:
            # Check if tree search is enabled (max_depth > 1) and create action_generator if needed
            config_max_depth = getattr(self.config, 'tree_search_max_depth', 1)
            effective_action_generator = action_generator
            if config_max_depth > 1:
                if effective_action_generator is None:
                    # Use default action_generator (requires LLM configuration)
                    llm_model = self.web_agent_llm_config.get('model_name', 'unknown') if self.web_agent_llm_config else 'unknown'
                    logger.info(f"[Wrapper] [Action Agent] Type: DEFAULT_ACTION_GENERATOR | LLM: {llm_model} | Config: from config.yaml (action_agent_llm)")
                    # Pass context to action_generator (priority: prompt_builder > action_context > web_agent_prompt)
                    effective_action_generator = self._create_default_action_generator(
                        web_agent_prompt=web_agent_prompt,  # Backward compatibility
                        action_context=action_context,
                        prompt_builder=prompt_builder,
                    )
                else:
                    logger.info(f"[Wrapper] [Action Agent] Type: CUSTOM_ACTION_GENERATOR | Provided by caller")
            
            # v7 uses root_risk_threshold and child_risk_threshold from config (not passed as parameter)
            result = self.safe_agent.get_safe_action(
                current_state=safepred_state,
                candidate_actions=candidate_action_strings,
                action_generator=effective_action_generator,
                # risk_threshold parameter is deprecated in v7 (uses config.root_risk_threshold and config.child_risk_threshold)
            )
        except Exception as e:
            error_msg = f"[Wrapper] get_safe_action raised exception: {type(e).__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            raise
        
        # Ensure result is a dict - raise error if not
        if not isinstance(result, dict):
            error_msg = f"[Wrapper] get_safe_action returned non-dict: type={type(result).__name__}, value preview: {str(result)[:150]}..."
            logger.error(error_msg)
            raise TypeError(error_msg)
        
        # Extract required fields - raise error if missing
        if "risk" not in result:
            raise KeyError(f"[Wrapper] get_safe_action result missing required key 'risk'. Available keys: {list(result.keys())}")
        risk_score = result["risk"]
        
        requires_regeneration = result.get("requires_regeneration", False)
        
        # Extract violation information for policy update
        # No fallback: violated_policy_ids must come from result
        violated_policy_ids = result.get("violated_policy_ids")
        if violated_policy_ids is None:
            violated_policy_ids = []
        if not isinstance(violated_policy_ids, list):
            error_msg = f"[Wrapper] violated_policy_ids is not a list: type={type(violated_policy_ids)}, value={violated_policy_ids}"
            logger.error(error_msg)
            raise TypeError(error_msg)
        
        # No fallback: risk_explanation must come from result
        risk_explanation = result.get("risk_explanation")
        if risk_explanation is None:
            risk_explanation = ""
        if not isinstance(risk_explanation, str):
            error_msg = f"[Wrapper] risk_explanation is not a str: type={type(risk_explanation)}, value={risk_explanation}"
            logger.error(error_msg)
            raise TypeError(error_msg)
        
        # Update policy references if violation detected and PolicyManager is available
        # Collect all violated_policy_ids from:
        # 1. Final selected path's violated_policy_ids
        # 2. Filtered actions' violated_policy_ids (from violated_actions field)
        all_violated_policy_ids = set(violated_policy_ids) if violated_policy_ids else set()
        
        # Also check violated_actions field (contains filtered actions during tree search)
        violated_actions = result.get("violated_actions", [])
        if violated_actions and isinstance(violated_actions, list):
            for violated_action in violated_actions:
                if isinstance(violated_action, dict):
                    action_violated_policy_ids = violated_action.get("violated_policy_ids", [])
                    if action_violated_policy_ids:
                        all_violated_policy_ids.update(action_violated_policy_ids)
        
        # Update references for all violated policies
        # Only update if show_policy_references is enabled (references are used in prompts)
        show_policy_references = getattr(self.config, 'show_policy_references', False)
        if (self.policy_manager and all_violated_policy_ids and show_policy_references):
            # P000 is a real policy defined in the policy file, so it should also update references
            # Extract violation context and update references for all violated policies including P000
            logger.debug(f"[Wrapper] Extracting violation context for policies: {sorted(all_violated_policy_ids)}")
            violation_context = self._extract_violation_context(
                state, action, intent, result, metadata
            )
            # Log extracted context for debugging
            has_violation_desc = bool(violation_context.get("violation_description"))
            violation_desc_length = len(violation_context.get("violation_description", ""))
            logger.debug(
                f"[Wrapper] Extracted violation_context: "
                f"has_task={bool(violation_context.get('task'))}, "
                f"has_action={bool(violation_context.get('action'))}, "
                f"has_violation_description={has_violation_desc}, "
                f"violation_description_length={violation_desc_length}"
            )
            
            # Validate violation_description before saving (will raise if missing)
            if not has_violation_desc or violation_desc_length == 0:
                error_msg = (
                    f"[Wrapper] CRITICAL: violation_description is missing or empty after extraction. "
                    f"This is required for saving policy reference examples. "
                    f"Violation context: {str(violation_context)[:500]}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            try:
                self.policy_manager.update_policy_references(
                    list(all_violated_policy_ids),
                    violation_context
                )
                logger.info(f"Updated policy references for violated policies: {sorted(all_violated_policy_ids)}")
            except ValueError as e:
                # Re-raise with additional context
                error_msg = (
                    f"[Wrapper] Failed to update policy references: {str(e)}. "
                    f"Violated policies: {sorted(all_violated_policy_ids)}, "
                    f"Violation context keys: {list(violation_context.keys())}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg) from e
        elif (self.policy_manager and all_violated_policy_ids and not show_policy_references):
            # Skip reference updates when show_policy_references is False
            logger.debug(f"[Wrapper] Skipping policy reference updates (show_policy_references=False) for violated policies: {sorted(all_violated_policy_ids)}")
        
        # Add the final executed action's response to conversation history
        # Only actions that passed risk evaluation and will definitely be executed should be added
        # 
        # We only record actions when requires_regeneration=False, because:
        # 1. If requires_regeneration=False -> action will definitely be executed -> should record
        # 2. If requires_regeneration=True -> action may be regenerated (not executed) or executed if max attempts reached
        #    Since we cannot know at evaluation time whether max attempts will be reached (WASP-level logic),
        #    we should NOT record these actions to avoid recording actions that won't be executed.
        # 
        # Note: Actions that require regeneration but reach max attempts will be executed but not recorded here.
        # This is a design limitation: SafePred only evaluates risk and doesn't know WASP's retry logic.
        # Use root_risk_threshold for filtering (single-step prediction uses root threshold)
        root_threshold = getattr(self.config, 'root_risk_threshold', 0.7)
        is_safe = risk_score < root_threshold
        will_be_executed = is_safe and not requires_regeneration
        
        if will_be_executed:
            # Single data source: metadata["current_response"] (full response with reasoning, passed by DemoAgent)
            # This should contain the complete reasoning process, not just the action
            if metadata and isinstance(metadata, dict) and "current_response" in metadata:
                response_text = str(metadata["current_response"]).strip()
                if response_text and hasattr(self.safe_agent, 'conversation_history_manager'):
                    # Save full response with reasoning to conversation history
                    self.safe_agent.conversation_history_manager.add_executed_response(
                        response=response_text,
                        action_string=str(result.get("action", ""))
                    )
        else:
            # Action will not be executed (either unsafe or requires regeneration)
            if not is_safe:
                root_threshold = getattr(self.config, 'root_risk_threshold', 0.7)
                logger.debug(f"[Wrapper] Skipping add_executed_response: action is unsafe (risk_score={risk_score:.4f}, threshold={root_threshold:.4f})")
            elif requires_regeneration:
                logger.debug(f"[Wrapper] Skipping add_executed_response: action requires regeneration (may not be executed, depends on WASP's retry logic)")
        
        # Determine if action should be recorded if max attempts reached
        # If action is safe but requires regeneration, it may be executed if max attempts reached
        # In that case, WASP should record it to conversation history
        should_record_if_max_attempts_reached = is_safe and requires_regeneration
        
        root_threshold = getattr(self.config, 'root_risk_threshold', 0.7)
        
        # Extract selected action from result (convert back to benchmark format)
        # SafeAgent.get_safe_action returns action in SafePred format (string), need to convert back
        # Data flow: SafeAgent returns string -> adapter.action_from_safepred converts to benchmark format
        action_from_result = result.get("action")
        if action_from_result is None:
            # If requires_regeneration=True, action may be None (all actions filtered)
            if requires_regeneration:
                selected_action = None  # Will be handled by caller (regeneration logic)
            else:
                # If not requiring regeneration but no action, this is an error
                error_msg = (
                    f"[Wrapper] SafeAgent.get_safe_action returned no action but requires_regeneration=False. "
                    f"Result keys: {list(result.keys())}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            # Convert from SafePred format (string) back to benchmark format
            # No fallback - if conversion fails, raise error
            try:
                selected_action = self.adapter.action_from_safepred(str(action_from_result))
            except Exception as e:
                error_msg = (
                    f"[Wrapper] Failed to convert action from SafePred format to benchmark format: {e}. "
                    f"Action from SafeAgent: {action_from_result}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg) from e
        
        return {
            "risk_score": risk_score,
            "is_safe": risk_score < root_threshold,
            "risk_explanation": risk_explanation,
            "requires_regeneration": requires_regeneration,
            "risk_guidance": result.get("risk_guidance") if "risk_guidance" in result else "",
            "violated_policy_ids": violated_policy_ids,
            "should_record_if_max_attempts_reached": should_record_if_max_attempts_reached,  # Flag for WASP: if True, record action when max attempts reached
            "selected_action": selected_action,  # Selected action (converted back to benchmark format)
            "action": selected_action,  # Alias for backward compatibility
        }
    
    def evaluate_action_risk_with_plan(
        self,
        state: Any,
        action: Any,
        plan_text: str,
        intent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        candidate_actions: Optional[List[Any]] = None,
        web_agent_prompt: Optional[List[Dict[str, str]]] = None,
        action_context: Optional[Dict[str, Any]] = None,
        prompt_builder: Optional[Callable[[Any, Dict[str, Any]], List[Dict[str, str]]]] = None,
        depth: int = 1,
    ) -> Dict[str, Any]:
        """Evaluate action risk with plan support. Returns risk result plus should_update_plan, update_reason, optimization_guidance."""
        # Add plan_text to state so World Model can access it
        if isinstance(state, dict):
            state_with_plan = {**state, 'plan_text': plan_text}
        else:
            state_with_plan = state
        
        # Pass context to evaluate_action_risk for action_generator
        # Priority: prompt_builder > action_context > web_agent_prompt (backward compatibility)
        result = self.evaluate_action_risk(
            state_with_plan,  # Use state with plan_text
            action, 
            candidate_actions=candidate_actions,  # Pass candidate_actions for proper evaluation
            intent=intent, 
            metadata=metadata, 
            web_agent_prompt=web_agent_prompt,  # Backward compatibility
            action_context=action_context,
            prompt_builder=prompt_builder,
        )
        
        if self.use_planning:
            requires_regeneration = result.get("requires_regeneration", False)
            should_check_plan = result.get("should_check_plan", False)
            
            # Check for optimization_guidance from two sources:
            # 1. PlanMonitor (when should_check_plan=True): checks plan consistency
            # 2. World Model (when policy violation detected): provides guidance for plan correction
            optimization_guidance = ""
            
            # Source 1: PlanMonitor (plan consistency check)
            if should_check_plan and self.plan_monitor:
                action_str = self.adapter.action_to_safepred(action)
                monitoring_result = self.plan_monitor.monitor_with_plan(
                    plan_text=plan_text,
                    current_state=self._convert_to_compact_state(state, intent, metadata),
                    action=action_str,
                    simulated_path=self._convert_path_to_safepred(result.get("path"), action_str),
                )
                optimization_guidance = monitoring_result.get("optimization_guidance") if "optimization_guidance" in monitoring_result else ""
            
            # Source 2: World Model (policy violation detected, depth == 1)
            # Check if World Model generated optimization_guidance due to policy violation
            if not optimization_guidance and depth == 1 and hasattr(self.safe_agent, 'world_model'):
                delta = getattr(self.safe_agent.world_model, '_last_predicted_delta', None)
                if delta and isinstance(delta, dict):
                    # Check if there's a policy violation and optimization_guidance
                    violated_policy_ids = delta.get('violated_policy_ids', [])
                    world_model_optimization_guidance = delta.get('optimization_guidance')
                    if violated_policy_ids and world_model_optimization_guidance:
                        optimization_guidance = world_model_optimization_guidance
                        logger.debug(f"[SafePredWrapper] Using World Model's optimization_guidance for plan update (violated_policy_ids={violated_policy_ids})")
            
            plan_update_info = self._determine_plan_update(
                requires_regeneration=requires_regeneration,
                should_check_plan=should_check_plan,  # Keep original should_check_plan for update_reason determination
                optimization_guidance=optimization_guidance,
                risk_guidance=result.get("risk_guidance") if "risk_guidance" in result else "",
            )
            
            if plan_update_info['should_update_plan']:
                logger.info(f"[SafePredWrapper] Plan update triggered: {plan_update_info['update_reason']}")
            
            result.update(plan_update_info)
        
        # Extract progress_step from World Model if available
        # Only extract for depth == 1 (actual execution), not for simulated paths (depth >= 2)
        # World Model stores it in _last_predicted_delta
        if depth == 1:
            if hasattr(self.safe_agent, 'world_model'):
                delta = getattr(self.safe_agent.world_model, '_last_predicted_delta', None)
                if delta and isinstance(delta, dict) and 'progress_step' in delta:
                    result['progress_step'] = delta['progress_step']
                    logger.info(f"[SafePredWrapper] [Planning] ✅ Extracted progress_step from World Model (depth={depth}, progress_step={delta['progress_step']})")
        else:
            logger.debug(f"[SafePredWrapper] [Planning] Skipping progress_step extraction for simulated action (depth={depth})")
        
        return result
    
    def record_executed_action(self, action: Any, response: Optional[str] = None) -> None:
        """
        Record an executed action to conversation history.
        
        This method can be called by WASP when an action is executed but wasn't
        recorded during evaluation (e.g., when max regeneration attempts reached).
        
        Args:
            action: Action that was executed (should contain raw_prediction)
            response: Optional response text (if not provided, extracted from action)
        """
        if not hasattr(self.safe_agent, 'conversation_history_manager'):
            logger.warning("[Wrapper] Cannot record executed action: conversation_history_manager not available")
            return
        
        # Extract response from action if not provided
        if not response:
            if action and isinstance(action, dict):
                response = action.get("raw_prediction", "")
                if not response:
                    response = str(action.get("action", ""))
            else:
                response = str(action) if action else ""
        
        if not response or not str(response).strip():
            logger.warning("[Wrapper] Cannot record executed action: no response text available")
            return
        
        # Extract action string
        action_string = ""
        if action and isinstance(action, dict):
            action_string = str(action.get("action", ""))
        
        # Record to conversation history (single source of truth: conversation_history_manager)
        self.safe_agent.conversation_history_manager.add_executed_response(
            response=str(response).strip(),
            action_string=action_string
        )
        logger.debug(f"[Wrapper] Recorded executed action to conversation history")
    
    def format_plan_for_prompt(self, plan_text: str) -> str:
        """Format plan text for inclusion in prompt."""
        if not plan_text:
            return ""
        
        # If plan is already in Todo List format (has status markers), use it as is
        # Otherwise, assume it's already formatted correctly
        plan_content = plan_text
        if "**SUGGESTED PLAN:**" not in plan_text and "**EXECUTION PLAN:**" not in plan_text:
            plan_content = f"**SUGGESTED PLAN:**\n{plan_text}"
        
        return f"""{plan_content}

**Important Notes:**
- The plan above is a **suggested high-level guide** for reference only - you are NOT required to follow it strictly
- The plan focuses on goals and objectives, not specific implementation details
- **Always prioritize the current observation** over the plan - if the UI differs from what the plan assumes, adapt accordingly
- Make decisions based on the actual current state and task requirements, not assumptions from the plan
- The plan may be inaccurate or outdated - trust your observation and reasoning over the plan when there's a conflict
- Use the plan as a general direction, but feel free to deviate if you find a better approach based on the current state"""
    
    def update_trajectory(
        self,
        prev_state: Any,
        action: Any,
        next_state: Any,
        action_success: bool = True,
        intent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update SafePred trajectory with executed action.
        
        Args:
            prev_state: Previous state in benchmark format
            action: Executed action in benchmark format
            next_state: Next state in benchmark format
            action_success: Whether action execution was successful (default: True)
            intent: Task intent (optional)
            metadata: Additional metadata (optional)
        """
        try:
            metadata = metadata or {}
            if intent is None:
                intent = metadata.get("intent", "")
            
            # Convert states to SafePred format
            safepred_prev_state = self.adapter.state_to_safepred(
                raw_state=prev_state,
                intent=intent,
                metadata=metadata
            )
            safepred_next_state = self.adapter.state_to_safepred(
                raw_state=next_state,
                intent=intent,
                metadata=metadata
            )
            
            # Convert action to SafePred format
            # Validate action - no fallback, raise error if invalid
            if action is None:
                error_msg = "[Wrapper] update_trajectory: action is None"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Action can be dict (structured format) or string (pyautogui code)
            # The adapter will handle both formats
            action_str = self.adapter.action_to_safepred(action)
            
            # Update trajectory
            self.safe_agent.update_trajectory(
                state=safepred_prev_state,
                action=action_str,
                next_state=safepred_next_state,
                action_success=action_success,
            )
        except Exception as e:
            logger.warning(f"Error updating SafePred trajectory: {e}")
    
    def _extract_violation_context(
        self,
        state: Any,
        action: Any,
        intent: Optional[str],
        result: Dict[str, Any],
        metadata: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Extract violation context for policy update.
        
        Args:
            state: Original state
            action: Original action
            intent: Task intent
            result: Evaluation result from SafeAgent
            metadata: Additional metadata
        
        Returns:
            Violation context dictionary
        """
        # Ensure result is a dict - raise error if not
        if not isinstance(result, dict):
            error_msg = f"[Wrapper] _extract_violation_context: result is not a dict: type={type(result)}, value={str(result)[:200]}"
            logger.error(error_msg)
            raise TypeError(error_msg)
        
        # Ensure metadata is a dict or None - raise error if not
        if metadata is not None and not isinstance(metadata, dict):
            error_msg = f"[Wrapper] _extract_violation_context: metadata is not a dict or None: type={type(metadata)}, value={str(metadata)[:200]}"
            logger.error(error_msg)
            raise TypeError(error_msg)
        
        context = {}
        
        # Extract task/intent
        if intent:
            context["task"] = intent
        elif metadata and "intent" in metadata:
            context["task"] = metadata["intent"]
        
        # Extract thought/reasoning if available
        # Priority: 1) action["raw_prediction"] (for web agents), 2) metadata["current_response"] (for os-harm)
        if isinstance(action, dict) and "raw_prediction" in action:
            context["thought"] = str(action["raw_prediction"])
        elif metadata and isinstance(metadata, dict) and "current_response" in metadata:
            # For os-harm: current_response contains full agent response including reasoning
            context["thought"] = str(metadata["current_response"])
        
        # Extract simplified action (only key fields to avoid redundancy)
        if isinstance(action, dict):
            # Extract only essential fields: action_type and element_id
            action_type = action.get("action_type", "")
            element_id = action.get("element_id", "")
            
            # Format action_type (extract enum name if it's an enum)
            action_type_str = str(action_type)
            if "." in action_type_str:
                action_type_str = action_type_str.split(".")[-1].rstrip(">")
            
            # Build simplified action representation
            if element_id:
                context["action"] = {
                    "action_type": action_type_str,
                    "element_id": element_id
                }
            else:
                context["action"] = {
                    "action_type": action_type_str
                }
        else:
            # If action is not a dict, keep as string but try to simplify
            action_str = str(action)
            # Try to extract key info from string representation
            import re
            element_match = re.search(r'element_id[:\'"]+\s*([^\'",}\]]+)', action_str)
            type_match = re.search(r'action_type[:\'"]+\s*<ActionTypes\.(\w+)', action_str)
            
            if type_match and element_match:
                context["action"] = {
                    "action_type": type_match.group(1),
                    "element_id": element_match.group(1).strip()
                }
            else:
                # Don't truncate action - keep full action string for reference
                # The _build_reference method will handle formatting and length if needed
                context["action"] = action_str
        
        # Extract violation description (REQUIRED)
        # No fallback: risk_explanation must come from result
        risk_explanation = result.get("risk_explanation")
        if risk_explanation is None:
            risk_explanation = ""
        # No fallback: violated_policy_ids must come from result
        violated_policy_ids = result.get("violated_policy_ids")
        if violated_policy_ids is None:
            violated_policy_ids = []
        
        # Also check violated_actions for risk_explanation (when all actions are filtered)
        if not risk_explanation and result.get("violated_actions"):
            # Try to get risk_explanation from the first violated action
            violated_actions = result.get("violated_actions", [])
            if violated_actions and isinstance(violated_actions, list) and len(violated_actions) > 0:
                first_violated = violated_actions[0]
                if isinstance(first_violated, dict):
                    risk_explanation = first_violated.get("risk_explanation", "") or first_violated.get("explanation", "")
                    # Convert None to empty string
                    if risk_explanation is None:
                        risk_explanation = ""
        
        # violation_description is REQUIRED for reference examples
        if risk_explanation:
            context["violation_description"] = risk_explanation
        elif violated_policy_ids:
            # Fallback: use violated_policy_ids as description (better than nothing, but not ideal)
            context["violation_description"] = f"Violated policies: {', '.join(violated_policy_ids)}"
            logger.debug(f"[Wrapper] Using violated_policy_ids as violation_description: {violated_policy_ids}")
        else:
            # CRITICAL: violation_description is required for reference examples
            error_msg = (
                f"[Wrapper] CRITICAL: No violation_description found in result. "
                f"This is required for saving policy reference examples. "
                f"Result keys: {list(result.keys())}, "
                f"Result type: {type(result)}, "
                f"Action: {str(action)[:200]}, "
                f"Intent: {intent[:100] if intent else 'None'}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        return context
    
    def close(self) -> None:
        """Close wrapper and flush any remaining data."""
        if self.safe_agent and hasattr(self.safe_agent, 'trajectory_storage') and self.safe_agent.trajectory_storage:
            try:
                self.safe_agent.trajectory_storage.close()
                logger.info("Trajectory storage closed and data flushed")
            except Exception as e:
                logger.warning(f"Error closing trajectory storage: {e}")
    
    def __del__(self):
        """Destructor to ensure trajectory storage is closed."""
        try:
            self.close()
        except Exception:
            # Ignore errors in destructor to avoid issues during shutdown
            pass


