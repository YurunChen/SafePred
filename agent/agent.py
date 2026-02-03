"""
Safe Agent Module for Safety-TS-LMA.

High-level interface for easy integration with different benchmarks.
Provides a simple API for safe action planning and execution.
"""

from typing import Any, Optional, List, Dict, Callable, Tuple
from dataclasses import dataclass
import json
import re
from datetime import datetime
from ..config.config import SafetyConfig, BENCHMARK_CONFIGS
from ..core.trajectory_graph import TrajectoryGraph
from ..core.trajectory_storage import TrajectoryStorage
from ..core.conversation_history_manager import ConversationHistoryManager
from ..models.world_model import WorldModel, BaseWorldModel
from ..utils.logger import get_logger
from ..utils.state_preprocessor import StatePreprocessor
from ..utils.type_checkers import validate_dict

logger = get_logger("SafePred.Agent")


@dataclass
class SearchPath:
    """Represents a search path in tree search planning."""
    actions: List[Any]
    states: List[Any]
    risk_accum: float
    metadata: List[Optional[Dict[str, Any]]]
    is_dead_end: bool = False  # Whether this path is a dead end (cannot continue)
    is_complete: bool = False  # Whether this path contains finish action (task complete)
    
    def __post_init__(self):
        """Validate path structure."""
        if len(self.actions) != len(self.states) - 1:
            raise ValueError(
                f"Path structure invalid: {len(self.actions)} actions but {len(self.states)} states"
            )


class SafeAgent:
    """
    High-level Safe Agent interface for Safety-TS-LMA.
    
    Provides an easy-to-use API for safe action planning in web navigation tasks.
    Handles initialization, trajectory tracking, and safe action selection.
    
    Usage:
        # Initialize agent
        agent = SafeAgent(config=SafetyConfig())
        
        # Get safe action for current state
        action = agent.get_safe_action(current_state)
        
        # Execute action and update trajectory
        next_state = execute_action(action)
        agent.update_trajectory(current_state, action, next_state)
    """
    
    def __init__(
        self,
        config: Optional[SafetyConfig] = None,
        benchmark: Optional[str] = None,
        world_model: Optional[BaseWorldModel] = None,
        reward_function: Optional[Callable[[Any], float]] = None,
        policies: Optional[List[Dict[str, Any]]] = None,
        web_agent_model_name: Optional[str] = None,  # Web agent model name for organizing experience data
    ):
        """
        Initialize Safe Agent.
        
        Args:
            config: Safety configuration (if None, uses default or benchmark config)
            benchmark: Benchmark name ('mind2web', 'webguard', 'visualwebarena')
                      If provided, loads benchmark-specific config
            world_model: Optional pre-initialized world model
            reward_function: Optional custom reward function
            web_agent_model_name: Web agent model name for organizing experience data
        """
        # Load configuration
        if config is None:
            # Try to load from config.yaml first, then fall back to benchmark or default
            try:
                config = SafetyConfig.from_yaml()
            except Exception:
                # If config.yaml doesn't exist or has errors, use benchmark or default
                if benchmark and benchmark in BENCHMARK_CONFIGS:
                    config = BENCHMARK_CONFIGS[benchmark]
                else:
                    config = SafetyConfig()
        
        self.config = config
        
        # Get world model name for organizing experience data
        # If world_model is provided, extract model name from it
        # Otherwise, get model name from config (will be used when creating world_model)
        world_model_name = None
        if world_model is not None:
            world_model_name = self._get_world_model_name(world_model)
        else:
            # Get model name from config
            world_llm_config = config.get_llm_config("world_model")
            world_model_name = world_llm_config.get("model_name") if world_llm_config else None
        
        # Set model names for log filename
        from ..utils.logger import SafePredLogger
        SafePredLogger.set_model_names(
            web_agent_model_name=web_agent_model_name,
            world_model_name=world_model_name
        )
        
        # Initialize trajectory storage (needed for experience replay)
        enable_storage = getattr(config, 'enable_trajectory_storage', True) if config else True
        experience_dir = getattr(config, 'trajectory_experience_dir', None) if config else None
        training_dir = getattr(config, 'trajectory_training_dir', None) if config else None
        system_prompt = getattr(config, 'trajectory_system_prompt', None) if config else None
        include_policies = getattr(config, 'include_policies_in_training', True) if config else True
        
        if enable_storage:
            from pathlib import Path
            if experience_dir:
                experience_dir = Path(experience_dir)
            if training_dir:
                training_dir = Path(training_dir)
            
            # Get max_entries_in_memory from config (default: 50 for more frequent saves)
            max_entries_in_memory = getattr(config, 'trajectory_storage_max_entries_in_memory', 50)
            
            self.trajectory_storage = TrajectoryStorage(
                experience_dir=experience_dir,
                training_dir=training_dir,
                auto_save=True,
                max_entries_in_memory=max_entries_in_memory,
                experience_format="jsonl",
                system_prompt=system_prompt,
                include_policies_in_training=include_policies,
                policies=policies,  # Pass policies to storage for training data
                web_agent_model_name=web_agent_model_name,  # Pass web agent model name for organizing experience data
                world_model_name=world_model_name,  # Pass world model name for organizing experience data
            )
            # Get SafePred root for default paths display
            # agent.py is in SafePred/agent/, so parent.parent is SafePred/
            safepred_root = Path(__file__).parent.parent
            default_experience_dir = safepred_root / "trajectories" / "experience"
            default_training_dir = safepred_root / "trajectories" / "training"
            
            # Trajectory storage initialization is logged by TrajectoryStorage itself
            # No need to log again here to avoid duplication
        else:
            self.trajectory_storage = None
        
        # Load experience data for few-shot learning (if enabled)
        few_shot_examples = None
        if (world_model is None and 
            getattr(config, 'enable_experience_replay', True) and 
            self.trajectory_storage):
            few_shot_examples = self._load_experience_replay_examples(config)
        
        # Initialize components
        if world_model is None:
            # Get LLM config for world model
            world_llm_config = config.get_llm_config("world_model")
            
            # Prepare World Model kwargs
            world_model_kwargs = {
                "model_name": world_llm_config.get("model_name"),
                "device": config.device,
                "temperature": world_llm_config.get("temperature"),
                "max_tokens": world_llm_config.get("max_tokens"),
                "llm_config": world_llm_config,
                "use_state_delta": getattr(config, 'world_model_use_state_delta', True),  # Read from config
                "prediction_steps": getattr(config, 'world_model_prediction_steps', 1),  # Read from config
            }
            
            # Add few-shot learning if experience replay is enabled and examples are available
            if few_shot_examples:
                world_model_kwargs["use_few_shot"] = True
                world_model_kwargs["few_shot_examples"] = few_shot_examples
            
            world_model = WorldModel(**world_model_kwargs)
        
        self.world_model = world_model
        self.reward_function = reward_function
        self.policies = policies or []  # Store policies for risk evaluation
        
        # Initialize state preprocessor for compact state representation
        # Pass benchmark name to determine parsing format
        self.state_preprocessor = StatePreprocessor(max_chat_messages=5, benchmark=benchmark)
        
        # Initialize action evaluation context (used for trajectory storage)
        self._last_action_evaluation = None
        
        # Initialize trajectory graph
        self.trajectory_graph = TrajectoryGraph()
        
        # Trajectory history
        self.trajectory_history: List[Dict[str, Any]] = []
        
        # Conversation history manager (replaces direct conversation_history management)
        max_history_length = getattr(config, 'conversation_history_max_length', 20)
        show_full_response = getattr(config, 'conversation_history_show_full_response', True)
        self.conversation_history_manager = ConversationHistoryManager(
            max_history_length=max_history_length,
            show_full_response=show_full_response
        )
    
    def _preprocess_state(self, state: Any) -> Dict[str, Any]:
        """
        Preprocess raw observation to compact state representation.
        
        Args:
            state: Raw observation dict (can have axtree_object or axtree_txt)
        
        Returns:
            Compact state dictionary
        """
        if not isinstance(state, dict):
            raise ValueError(
                f"Expected raw observation dict, "
                f"got {type(state).__name__}"
            )
        
        # Convert axtree_object to axtree_txt if needed
        if "axtree_txt" not in state and "axtree_object" in state:
            from browsergym.utils.obs import flatten_axtree_to_str
            state = state.copy()  # Don't modify original
            state["axtree_txt"] = flatten_axtree_to_str(state["axtree_object"])
        
        if "axtree_txt" not in state:
            raise ValueError(
                f"Expected raw observation dict with 'axtree_txt' or 'axtree_object' field, "
                f"got keys: {list(state.keys())}"
            )
        
        compact_state = self.state_preprocessor.preprocess(state)
        
        # Get conversation history from conversation_history_manager for compact state
        conversation_history = self.conversation_history_manager.get_conversation_history()
        compact_state["conversation_history"] = conversation_history
        
        # Add total executed steps count to state for accurate Current Step numbering
        # This ensures Current Step is correct even when conversation history is truncated
        total_executed_steps = self.conversation_history_manager.get_total_executed_steps()
        compact_state["total_executed_steps"] = total_executed_steps
        
        # Add current action reasoning from state to compact_state (for Current Step display)
        # This allows world model prompt to show full reasoning, not just action string
        if "current_action_reasoning" in state:
            compact_state["current_action_reasoning"] = state["current_action_reasoning"]
        # Note: If current_action_reasoning is not in state, Current Step will show action string only
        
        # Preserve plan_text if present in state (for plan progress tracking by World Model)
        if "plan_text" in state:
            compact_state["plan_text"] = state["plan_text"]
        
        return compact_state
    
    def _get_world_model_name(self, world_model: BaseWorldModel) -> Optional[str]:
        """
        Extract model name from world model instance.
        
        Args:
            world_model: World model instance
            
        Returns:
            Model name string, or None if not available
        """
        from ..models.world_model import LLMBasedWorldModel
        
        if isinstance(world_model, LLMBasedWorldModel):
            return getattr(world_model, 'model_name', None)
        # For other types, return None
        return None
    
    def _synthesize_compact_state_from_prediction(
        self, 
        current_state: Dict[str, Any], 
        predicted_state_str: str,
        delta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synthesize compact state from current state, predicted state string, and delta.
        
        v7's World Model returns string-format state, but Action Agent needs dict-format compact state.
        This method uses structured information from delta to reconstruct compact state.
        
        Args:
            current_state: Current compact state (dict format)
            predicted_state_str: World Model's predicted state (string format)
            delta: World Model's predicted delta (contains structured change info)
        
        Returns:
            New compact state (dict format)
        """
        # Copy current state to preserve all fields (goal, policies, url, page_type, conversation_history, etc.)
        compact_state = current_state.copy()
        
        if not delta or not isinstance(delta, dict):
            # If no delta, return copy of current state (conservative)
            logger.warning("[SafeAgent] No delta available for state synthesis, using current state")
            return compact_state
        
        # 1. Handle element_changes (new elements and removed elements)
        element_changes = delta.get('element_changes', {})
        if isinstance(element_changes, dict):
            new_elements = element_changes.get('new_elements', [])
            removed_elements = element_changes.get('removed_elements', [])
            
            # Add new elements to key_elements
            existing_labels = {elem.get('label', '') for elem in compact_state.get('key_elements', []) if elem.get('label')}
            for elem in new_elements:
                if isinstance(elem, dict):
                    label = elem.get('label', '')
                    elem_type = elem.get('type', 'unknown')
                else:
                    label = str(elem)
                    elem_type = 'unknown'
                
                if label and label not in existing_labels:
                    compact_state.setdefault('key_elements', []).append({
                        'label': label,
                        'type': elem_type,
                        'bid': elem.get('bid', None) if isinstance(elem, dict) else None,
                    })
            
            # Remove deleted elements
            if removed_elements:
                removed_labels = {str(elem) if not isinstance(elem, dict) else elem.get('label', '') for elem in removed_elements}
                compact_state['key_elements'] = [
                    elem for elem in compact_state.get('key_elements', [])
                    if elem.get('label', '') not in removed_labels
                ]
        
        # 2. Keep URL and page_type unchanged (World Model no longer predicts these)
        # URL and page_type are preserved from current_state.copy()
        
        # 3. Keep other fields unchanged (goal, policies, conversation_history, total_executed_steps, etc.)
        # These shouldn't change due to actions
        
        return compact_state
    
    def _tree_search_planning(
        self,
        current_state: Any,
        root_actions: Optional[List[Any]] = None,
        action_agent: Optional[Callable[[Any, str, int, Optional[List[str]]], List[Any]]] = None,
        max_depth: int = 1,
    ) -> Dict[str, Any]:
        """
        Tree search planning: alternate between Action Agent and World Model.
        
        Single-step prediction is a special case (max_depth=1): only evaluate root actions, no depth expansion.
        
        Process:
        1. Root Node: Use provided root_actions, take first n_root actions
        2. For each root action: World Model predicts and evaluates risk
        3. If max_depth > 1:
           - Select M_child lowest-risk root paths
           - Depth expansion (depth 2 to max_depth):
             - For each path: Action Agent generates N_child child actions
             - For each child action: World Model predicts and evaluates risk
             - Select M_child lowest-risk paths
        4. Return root action from best path
        
        Args:
            current_state: Current state (compact state, dict format)
            root_actions: Root candidate action list (must be provided, will take first n_root)
            action_agent: Callback function (state, risk_guidance, num_samples) -> List[actions]
                         Only needed when max_depth > 1, can be None when max_depth=1
            max_depth: Maximum search depth. 1=single-step prediction, >1=tree search
        
        Returns:
            Dict with keys: action, risk, path, search_stats, requires_regeneration, etc.
        """
        # 1. Get configuration parameters
        N_root = getattr(self.config, 'tree_search_n_root', 1)
        N_child = getattr(self.config, 'tree_search_n_child', 5)
        M_child = getattr(self.config, 'tree_search_m_child', 3)
        root_risk_threshold = getattr(self.config, 'root_risk_threshold', 0.7)
        child_risk_threshold = getattr(self.config, 'child_risk_threshold', 0.8)
        
        # Determine actual parameters to use
        if max_depth == 1:
            M_child = 1  # Single-step prediction: M_child=1 (don't select multiple)
        D = max_depth
        
        # 2. Validate inputs
        if not root_actions:
            raise ValueError("root_actions is required")
        if max_depth > 1 and not action_agent:
            raise ValueError("action_agent is required when max_depth > 1 (for depth expansion)")
        
        # 3. Take first N_root root actions
        root_actions = root_actions[:N_root]
        
        # 4. Step 2: Evaluate all root actions
        root_paths = []
        all_root_violations = []  # Collect all violated action information
        
        for idx, action in enumerate(root_actions):
            # Check finish action (task completion signal)
            action_str = str(action).strip().lower()
            if action_str.startswith('finish'):
                # finish is safe action, risk score = 0
                root_paths.append(SearchPath(
                    actions=[action],
                    states=[current_state, current_state],  # finish doesn't change state
                    risk_accum=0.0,
                    metadata=[None],
                    is_complete=True,  # Mark as complete path
                ))
                continue
            
            # World Model prediction (returns string format)
            # Note: plan_text is NOT passed to WorldModel.simulate() by default (only passed when violation detected)
            # This avoids unnecessary plan tracking overhead for safe actions
            if self.policies:
                logger.debug(f"[SafeAgent] Passing {len(self.policies)} policies to WorldModel.simulate()")
            # First prediction: without plan_text (to avoid unnecessary plan tracking)
            predicted_state_str = self.world_model.simulate(current_state, action, policies=self.policies, plan_text=None)
            
            # Get delta and risk assessment from World Model
            delta = getattr(self.world_model, '_last_predicted_delta', None)
            risk_score = delta.get('risk_score', 0.0) if delta and isinstance(delta, dict) else 0.0
            violated_policy_ids = delta.get('violated_policy_ids', []) if delta and isinstance(delta, dict) else []
            
            if risk_score is None:
                raise RuntimeError(f"World Model did not return risk_score for action: {str(action)}")
            
            # Risk threshold check (only when pruning enabled)
            if self.config.enable_pruning and risk_score >= root_risk_threshold:
                # Prune: collect violation information
                all_root_violations.append({
                    'action': str(action),
                    'risk_score': risk_score,
                    'violated_policy_ids': delta.get('violated_policy_ids', []) if delta and isinstance(delta, dict) else [],
                    'risk_explanation': delta.get('risk_explanation', '') if delta and isinstance(delta, dict) else '',
                    'optimization_guidance': delta.get('optimization_guidance') if delta and isinstance(delta, dict) else None,
                })
                # Save filtered action for training
                self._save_filtered_action(
                    state=current_state,
                    action=action,
                    risk_score=risk_score,
                    risk_explanation=delta.get('risk_explanation', '') if delta and isinstance(delta, dict) else '',
                    predicted_delta=delta,
                    violated_policy_ids=delta.get('violated_policy_ids', []) if delta and isinstance(delta, dict) else [],
                )
                continue  # Skip this action
            
            # Convert predicted state to compact state (dict format)
            # Note: For single-step prediction (max_depth=1), may not need full state conversion
            if max_depth > 1:
                predicted_compact_state = self._synthesize_compact_state_from_prediction(
                    current_state=current_state,
                    predicted_state_str=predicted_state_str,
                    delta=delta
                )
            else:
                # Single-step prediction: don't need full state (not used for further expansion)
                predicted_compact_state = current_state
            
            # Create root path
            root_paths.append(SearchPath(
                actions=[action],
                states=[current_state, predicted_compact_state],
                risk_accum=risk_score,  # depth=1: risk_accum = risk
                metadata=[delta] if delta else [None],
                is_dead_end=False,  # Root path is not dead end initially
                is_complete=False,  # Root path is not complete initially
            ))
        
        # 5. If all root actions were pruned, return requires_regeneration
        if not root_paths:
            risk_guidance = self._generate_risk_guidance(
                [{
                    'action': v.get('action', ''),
                    'risk_score': v.get('risk_score', 0.0),
                    'explanation': v.get('risk_explanation', ''),
                    'optimization_guidance': v.get('optimization_guidance'),
                    'violated_policy_ids': v.get('violated_policy_ids', []),
                } for v in all_root_violations],
                root_risk_threshold,
                total_candidates=N_root,
            )
            # Use the highest risk score from filtered actions, or threshold if no violations recorded
            max_filtered_risk = max([v.get('risk_score', root_risk_threshold) for v in all_root_violations], default=root_risk_threshold)
            return {
                'action': None,
                'risk': max_filtered_risk,  # Use actual max risk score from filtered actions (0-1 range)
                'requires_regeneration': True,
                'risk_guidance': risk_guidance,
                'violated_actions': all_root_violations,
                'search_stats': {'method': 'tree_search' if max_depth > 1 else 'single_step', 'depth': 1},
            }
        
        # 6. Single-step prediction mode (max_depth=1): directly return best action
        if max_depth == 1:
            best_path = min(root_paths, key=lambda p: p.risk_accum)
            best_action = best_path.actions[0]
            best_metadata = best_path.metadata[0] if best_path.metadata and best_path.metadata[0] else None
            
            # Store evaluation context for update_trajectory (same as old _get_safe_action_single_step)
            self._last_action_evaluation = {
                'action': best_action,
                'risk_score': best_path.risk_accum,
                'risk_explanation': best_metadata.get('risk_explanation', '') if best_metadata and isinstance(best_metadata, dict) else '',
                'world_model_metadata': best_metadata,
            }
            
            # Extract metadata fields
            metadata_fields = self._extract_metadata_fields(best_metadata)
            
            # Single-step prediction: no plan check needed (no multi-step simulation)
            should_check_plan = False
            
            result = {
                'action': best_action,
                'risk': best_path.risk_accum,
                'path': best_path.actions,
                'violated_policy_ids': metadata_fields['violated_policy_ids'],
                'risk_explanation': metadata_fields['risk_explanation'],
                'should_check_plan': should_check_plan,  # Single-step: no plan check
                'search_stats': {
                    'method': 'single_step',
                    'depth': 1,
                    'total_paths_evaluated': len(root_paths),
                    'filtered_by_risk': len(all_root_violations),
                },
            }
            # Include violated_actions if any were filtered (for reference updates)
            if all_root_violations:
                result['violated_actions'] = all_root_violations
            return result
        
        # 7. Tree search mode (max_depth > 1): Select M_child lowest-risk root paths, then depth expansion
        root_paths.sort(key=lambda p: p.risk_accum)
        selected_paths = root_paths[:M_child]
        
        # 8. Depth expansion (depth 2 to D)
        expanded_paths, all_child_violations = self._expand_paths_at_depth(
            paths=selected_paths,
            current_depth=1,
            max_depth=D,
            action_agent=action_agent,
            N_child=N_child,
            M_child=M_child,
            child_risk_threshold=child_risk_threshold,
        )
        
        # 9. Select best path (prioritize complete paths, filter dead ends if feasible paths exist)
        best_path = self._select_best_path(expanded_paths, selected_paths)
        
        # Store evaluation context for update_trajectory
        # Use root action's metadata (first metadata in path)
        best_action = best_path.actions[0]
        best_metadata = best_path.metadata[0] if best_path.metadata and best_path.metadata[0] else None
        
        self._last_action_evaluation = {
            'action': best_action,
            'risk_score': best_path.risk_accum / len(best_path.actions),  # Average risk per step
            'risk_explanation': best_metadata.get('risk_explanation', '') if best_metadata and isinstance(best_metadata, dict) else '',
            'world_model_metadata': best_metadata,  # Use root action's delta for trajectory storage
        }
        
        # Extract metadata fields
        metadata_fields = self._extract_metadata_fields(best_metadata)
        
        # should_check_plan: only set for tree search mode (max_depth > 1) with feasible paths
        # Plan consistency check should be done by PlanMonitor, not based on tree search's optimization_guidance
        should_check_plan = (max_depth > 1) and not best_path.is_dead_end
        
        # 10. Return result
        result = {
            'action': best_action,  # Return root action
            'risk': best_path.risk_accum,
            'path': best_path.actions,
            'violated_policy_ids': metadata_fields['violated_policy_ids'],
            'risk_explanation': metadata_fields['risk_explanation'],
            'should_check_plan': should_check_plan,  # Flag to indicate plan consistency check needed (only for tree search with feasible paths)
            'search_stats': {
                'method': 'tree_search',
                'depth': len(best_path.actions),
                'total_paths_evaluated': len(root_paths) + (len(expanded_paths) - len(selected_paths)),
            },
        }
        # Include violated_actions if any were filtered (for reference updates)
        # Merge root violations and child violations
        all_violated_actions = []
        if all_root_violations:
            all_violated_actions.extend(all_root_violations)
        if all_child_violations:
            all_violated_actions.extend(all_child_violations)
        if all_violated_actions:
            result['violated_actions'] = all_violated_actions
        return result
    
    def _extract_metadata_fields(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract violated_policy_ids and risk_explanation from metadata."""
        if metadata and isinstance(metadata, dict):
            return {
                'violated_policy_ids': metadata.get('violated_policy_ids', []),
                'risk_explanation': metadata.get('risk_explanation', ''),
            }
        return {
            'violated_policy_ids': [],
            'risk_explanation': '',
        }
    
    def _select_best_path(self, expanded_paths: List[SearchPath], fallback_paths: List[SearchPath]) -> SearchPath:
        """Select best path: prefer finish paths, then feasible, then dead-end."""
        # Separate paths into feasible and dead-end
        feasible_paths = [p for p in expanded_paths if not p.is_dead_end]
        dead_end_paths = [p for p in expanded_paths if p.is_dead_end]
        
        # Priority: finish paths > feasible paths > dead-end paths
        finish_paths = [p for p in feasible_paths if p.is_complete]
        
        if finish_paths:
            # Prioritize finish paths (task complete)
            best_path = min(finish_paths, key=lambda p: p.risk_accum)
            logger.debug(f"[SafeAgent] Selected finish path with risk_accum={best_path.risk_accum:.4f}")
        elif feasible_paths:
            # Select from feasible paths (not dead-end, not finish)
            best_path = min(feasible_paths, key=lambda p: p.risk_accum)
            logger.debug(f"[SafeAgent] Selected feasible path with risk_accum={best_path.risk_accum:.4f}")
        elif dead_end_paths:
            # All paths are dead-end: select lowest-risk dead-end path (don't trigger requires_regeneration)
            best_path = min(dead_end_paths, key=lambda p: p.risk_accum)
            logger.warning(f"[SafeAgent] All paths are dead-end, selected lowest-risk dead-end path with risk_accum={best_path.risk_accum:.4f}")
        else:
            # Fallback: use first root path (should not happen)
            best_path = fallback_paths[0] if fallback_paths else None
            if best_path is None:
                raise RuntimeError("[SafeAgent] No paths available for selection")
        
        return best_path
    
    def _expand_paths_at_depth(
        self,
        paths: List[SearchPath],
        current_depth: int,
        max_depth: int,
        action_agent: Callable[[Any, str, int, Optional[List[str]]], List[Any]],
        N_child: int,
        M_child: int,
        child_risk_threshold: float,
    ) -> Tuple[List[SearchPath], List[Dict[str, Any]]]:
        """
        Expand paths at given depth.
        
        Args:
            paths: List of paths at current depth
            current_depth: Current depth (starts from 1)
            max_depth: Maximum depth
            action_agent: Callback function to generate child actions 
                         (state, risk_guidance, num_samples, simulated_path) -> List[actions]
                         Note: state parameter is compact state (dict format)
                         simulated_path: Optional list of action strings executed in the simulation path
            N_child: Number of child actions to generate per node
            M_child: Number of lowest-risk paths to keep
            child_risk_threshold: Child node risk threshold
        
        Returns:
            Tuple of (expanded_paths, all_child_violations)
            - expanded_paths: List of expanded paths
            - all_child_violations: List of violated actions that were filtered during expansion
        """
        if current_depth >= max_depth:
            return paths, []  # Reached max depth
        
        expanded_paths = []
        all_child_violations = []  # Collect all violated child actions
        
        for path in paths:
            # Get last state of path (should be compact state, dict format)
            last_state = path.states[-1]
            
            # Ensure last_state is dict format (compact state)
            if not isinstance(last_state, dict):
                logger.warning(f"[SafeAgent] last_state is not dict at depth {current_depth}, skipping expansion")
                # Mark path as dead end
                path.is_dead_end = True
                expanded_paths.append(path)
                continue
            
            # Convert path actions to string list for action_agent
            # This ensures conversation history in prompt matches the simulated path
            simulated_path_strs = [str(action) for action in path.actions] if path.actions else None
            
            # Action Agent generates N_child child actions (receives compact state and simulated path)
            try:
                # Check if action_agent supports simulated_path parameter (new signature)
                import inspect
                sig = inspect.signature(action_agent)
                if len(sig.parameters) >= 4:
                    # New signature with simulated_path parameter
                    child_actions = action_agent(last_state, risk_guidance="", num_samples=N_child, simulated_path=simulated_path_strs)
                else:
                    # Old signature without simulated_path (backward compatibility)
                    child_actions = action_agent(last_state, risk_guidance="", num_samples=N_child)
                    logger.debug(f"[SafeAgent] action_agent doesn't support simulated_path parameter, using old signature")
            except TypeError as e:
                # Fallback: try without simulated_path if the call fails
                logger.debug(f"[SafeAgent] action_agent call with simulated_path failed, trying without: {e}")
                child_actions = action_agent(last_state, risk_guidance="", num_samples=N_child)
            except Exception as e:
                logger.warning(f"[SafeAgent] Action agent failed to generate child actions: {e}")
                # Mark path as dead end
                path.is_dead_end = True
                expanded_paths.append(path)
                continue
            
            # Evaluate each child action
            child_paths = []
            for child_action in child_actions:
                # Check if child action is finish
                action_str = str(child_action).strip().lower()
                is_finish = action_str.startswith('finish')
                
                if is_finish:
                    # Finish action: safe, no risk, don't continue expansion
                    new_path = SearchPath(
                        actions=path.actions + [child_action],
                        states=path.states + [last_state],  # finish doesn't change state
                        risk_accum=path.risk_accum,  # finish doesn't add risk
                        metadata=path.metadata + [None],
                        is_dead_end=False,
                        is_complete=True,  # Mark as complete path
                    )
                    child_paths.append(new_path)
                    continue
                
                # World Model prediction (returns string format)
                # Note: plan_text is NOT passed (only passed when violation detected)
                predicted_state_str = self.world_model.simulate(last_state, child_action, policies=self.policies, plan_text=None)
                delta = getattr(self.world_model, '_last_predicted_delta', None)
                risk_score = delta.get('risk_score', 0.0) if delta and isinstance(delta, dict) else 0.0
                violated_policy_ids = delta.get('violated_policy_ids', []) if delta and isinstance(delta, dict) else []
                
                if risk_score is None:
                    logger.warning(f"[SafeAgent] World Model did not return risk_score for child action, skipping")
                    continue
                
                # Check if policy violation occurred
                has_violation = (risk_score > 0.0) or (violated_policy_ids and len(violated_policy_ids) > 0)
                
                # Only update plan if violation detected (re-predict with plan_text to generate optimization_guidance)
                if has_violation:
                    plan_text = last_state.get('plan_text') if isinstance(last_state, dict) else None
                    if plan_text:
                        logger.debug(f"[SafeAgent] Policy violation detected for child action (risk_score={risk_score}, violated_policy_ids={violated_policy_ids}). Re-predicting with plan_text to generate optimization_guidance.")
                        # Re-predict with plan_text to generate optimization_guidance for plan update
                        predicted_state_str = self.world_model.simulate(last_state, child_action, policies=self.policies, plan_text=plan_text)
                        # Update delta with plan-based optimization_guidance
                        delta = getattr(self.world_model, '_last_predicted_delta', None)
                
                # Risk threshold check
                if self.config.enable_pruning and risk_score >= child_risk_threshold:
                    # Collect violation information for filtered child actions
                    violated_policy_ids = delta.get('violated_policy_ids', []) if delta and isinstance(delta, dict) else []
                    if violated_policy_ids:
                        all_child_violations.append({
                            'action': str(child_action),
                            'risk_score': risk_score,
                            'violated_policy_ids': violated_policy_ids,
                            'risk_explanation': delta.get('risk_explanation', '') if delta and isinstance(delta, dict) else '',
                            'optimization_guidance': delta.get('optimization_guidance') if delta and isinstance(delta, dict) else None,
                        })
                    continue  # Prune
                
                # Convert predicted state to compact state (dict format)
                predicted_compact_state = self._synthesize_compact_state_from_prediction(
                    current_state=last_state,
                    predicted_state_str=predicted_state_str,
                    delta=delta
                )
                
                # Create new path (using compact state)
                new_path = SearchPath(
                    actions=path.actions + [child_action],
                    states=path.states + [predicted_compact_state],  # Use compact state
                    risk_accum=path.risk_accum + (current_depth + 1) * risk_score,  # Accumulate risk
                    metadata=path.metadata + ([delta] if delta else [None]),
                    is_dead_end=False,
                    is_complete=False,
                )
                child_paths.append(new_path)
            
            # If all child actions were pruned, mark original path as dead end
            if not child_paths:
                path.is_dead_end = True
                logger.debug(f"[SafeAgent] Path marked as dead end at depth {current_depth}: all child actions were filtered")
                expanded_paths.append(path)
            else:
                expanded_paths.extend(child_paths)
        
        # Select M_child lowest-risk paths
        if expanded_paths:
            expanded_paths.sort(key=lambda p: p.risk_accum)
            selected_paths = expanded_paths[:M_child]
        else:
            selected_paths = paths  # If no expanded paths, keep original paths
        
        # Recursively expand next depth
        deeper_paths, deeper_violations = self._expand_paths_at_depth(
            paths=selected_paths,
            current_depth=current_depth + 1,
            max_depth=max_depth,
            action_agent=action_agent,
            N_child=N_child,
            M_child=M_child,
            child_risk_threshold=child_risk_threshold,
        )
        # Merge violations from deeper levels
        all_child_violations.extend(deeper_violations)
        return deeper_paths, all_child_violations
    
    def _compute_actual_delta(self, state: Dict[str, Any], next_state: Dict[str, Any], action: Any) -> Dict[str, Any]:
        """
        Compute actual state changes (delta) from real execution.
        
        This extracts the actual state changes that occurred after executing an action,
        which is crucial for experience replay to learn real-world state transitions.
        
        Args:
            state: Current state (before action) - compact state dict
            next_state: Next state (after action execution) - compact state dict
            action: Action that was executed
        
        Returns:
            Dictionary containing actual state delta with structure:
            {
                "semantic_delta": "Description of changes",
                "element_changes": {
                    "new_elements": ["list of new elements"],
                    "removed_elements": ["list of removed elements"]
                },
            }
        """
        if not isinstance(state, dict) or "url" not in state:
            raise ValueError(f"Expected compact state dict with 'url' field, got {type(state).__name__}")
        if not isinstance(next_state, dict) or "url" not in next_state:
            raise ValueError(f"Expected compact state dict with 'url' field, got {type(next_state).__name__}")
        
        action_str = str(action).lower() if action else ""
        
        # Extract from compact state format
        current_url = state.get("url")
        current_page_type = state.get("page_type", "unknown")
        current_key_elements = state.get("key_elements", [])
        current_elements = {elem.get("label", "") for elem in current_key_elements if elem.get("label")}
        
        next_url = next_state.get("url")
        next_page_type = next_state.get("page_type", "unknown")
        next_key_elements = next_state.get("key_elements", [])
        next_elements = {elem.get("label", "") for elem in next_key_elements if elem.get("label")}
        
        # Debug logging for actual delta computation
        logger.debug(
            f"[ActualDelta] _compute_actual_delta: action={action}, "
            f"state.key_elements length={len(current_key_elements)}, "
            f"next_state.key_elements length={len(next_key_elements)}, "
            f"current_elements count={len(current_elements)}, "
            f"next_elements count={len(next_elements)}"
        )
        if current_elements:
            logger.debug(f"[ActualDelta] current_elements (first 5): {list(current_elements)[:5]}")
        if next_elements:
            logger.debug(f"[ActualDelta] next_elements (first 5): {list(next_elements)[:5]}")
        
        # Element changes
        new_elements = list(next_elements - current_elements)
        removed_elements = list(current_elements - next_elements)
        
        # Build semantic delta description
        semantic_parts = []
        # Check URL change for semantic description (but don't include in return)
        if current_url != next_url and current_url is not None and next_url is not None:
            semantic_parts.append(f"URL changed from {current_url} to {next_url}")
        if new_elements:
            semantic_parts.append(f"New elements appeared: {', '.join(new_elements[:5])}")
        if removed_elements:
            semantic_parts.append(f"Elements removed: {', '.join(removed_elements[:5])}")
        if not semantic_parts:
            semantic_parts.append(f"Action '{action}' executed; minimal observable changes")
        
        delta = {
            "semantic_delta": "; ".join(semantic_parts),
            "element_changes": {
                "new_elements": new_elements,
                "removed_elements": removed_elements
            }
        }
        
        return delta
    
    def _format_policies_as_text(self, policies: List[Dict[str, Any]]) -> str:
        """
        Format policies list as readable text.
        
        Args:
            policies: List of policy dictionaries
        
        Returns:
            Formatted policies text
        """
        if not policies:
            return ""
        
        policies_text = "\n".join([
            f"Policy {p.get('policy_id', p.get('id', i+1))} ({p.get('risk_level', p.get('severity', 'Unknown'))}): {p.get('policy_description', p.get('description', p.get('policy_template', str(p))))}"
            for i, p in enumerate(policies)
            if isinstance(p, dict)
        ])
        
        if not policies_text:
            policies_text = str(policies)
        
        return policies_text
    
    def _generate_risk_guidance(
        self,
        filtered_explanations: List[Dict[str, Any]],
        risk_threshold: float,
        total_candidates: Optional[int] = None,
    ) -> str:
        """
        Generate detailed risk guidance based on World Model's risk_explanation outputs.
        
        This method integrates the natural language risk_explanation from World Model
        to provide actionable guidance for regenerating safer actions.
        
        Args:
            filtered_explanations: List of dicts with 'action', 'risk_score', 'explanation'
                                 where 'explanation' is from World Model's risk_explanation
            risk_threshold: Risk threshold that was exceeded
            total_candidates: Total number of candidate actions (if None, uses len(filtered_explanations))
        
        Returns:
            Detailed risk guidance string to help agent regenerate safer actions
        """
        # Use total_candidates if provided, otherwise fallback to filtered_explanations length
        total_count = total_candidates if total_candidates is not None else len(filtered_explanations)
        
        if not filtered_explanations:
            return f"All {total_count} candidate actions exceeded risk threshold {risk_threshold:.4f}. Please generate safer actions."
        
        # Build guidance based on World Model's risk_explanation
        guidance_parts = [
            f"⚠️ RISK GUIDANCE: All {total_count} candidate actions were filtered due to high risk (threshold: {risk_threshold:.4f}).",
            "",
            "Filtered actions and their risk explanations (from World Model):",
        ]
        
        # Integrate World Model's risk_explanation and optimization_guidance for each filtered action
        # Collect all optimization_guidance for summary
        all_optimization_guidance = []
        for idx, item in enumerate(filtered_explanations, 1):
            action = item.get('action', 'Unknown action')
            risk_explanation = item.get('explanation', 'No explanation available')
            optimization_guidance = item.get('optimization_guidance')  # Get optimization_guidance if available
            risk_score = item.get('risk_score', 0.0)
            
            # Use World Model's natural language explanation directly
            guidance_parts.append(f"{idx}. Action: {action}")
            guidance_parts.append(f"   Risk Score: {risk_score:.2f}")
            guidance_parts.append(f"   Explanation: {risk_explanation}")
            guidance_parts.append("")  # Empty line for readability
            
            # Collect optimization_guidance for later (avoid duplication)
            if optimization_guidance and optimization_guidance not in all_optimization_guidance:
                all_optimization_guidance.append(optimization_guidance)
        
        # Extract key issues directly from World Model's output (violated_policy_ids)
        # This is more accurate than keyword matching on explanation text
        all_violated_policy_ids = set()
        
        for item in filtered_explanations:
            # Collect violated policy IDs from World Model
            violated_policy_ids = item.get('violated_policy_ids', [])
            if violated_policy_ids:
                all_violated_policy_ids.update(violated_policy_ids)
        
        has_policy_violations = len(all_violated_policy_ids) > 0
        
        guidance_parts.append("Key issues identified:")
        
        if has_policy_violations:
            # Extract policy content for violated policies
            violated_policy_details = []
            for policy_id_str in sorted(all_violated_policy_ids):
                # Try to find policy by ID in multiple formats
                policy = None
                
                # Method 1: Try to match by "P001", "P002" format (new format)
                p_match = re.search(r'P(\d+)', policy_id_str, re.IGNORECASE)
                if p_match:
                    policy_id_num = p_match.group(1)
                    # Search in policies by id field
                    if self.policies:
                        for p in self.policies:
                            # Check various ID fields
                            p_id = str(p.get('id', '')).upper()
                            p_policy_id = str(p.get('policy_id', '')).upper()
                            if p_id == f"P{policy_id_num}" or p_policy_id == f"P{policy_id_num}":
                                policy = p
                                break
                
                # Method 2: Try to match by "Policy X" format (old format)
                if not policy:
                    match = re.search(r'Policy\s+(\d+)', policy_id_str, re.IGNORECASE)
                    if match:
                        policy_num = int(match.group(1))
                        # Find corresponding policy in self.policies (1-indexed)
                        if self.policies and 1 <= policy_num <= len(self.policies):
                            policy = self.policies[policy_num - 1]
                
                # Method 3: Try direct string match on policy_id field
                if not policy and self.policies:
                    for p in self.policies:
                        p_id = str(p.get('id', ''))
                        p_policy_id = str(p.get('policy_id', ''))
                        if p_id == policy_id_str or p_policy_id == policy_id_str:
                            policy = p
                            break
                
                # Format policy description if found
                if policy:
                    policy_desc = policy.get('policy_description') or policy.get('description') or policy.get('policy_template', '')
                    policy_risk_level = policy.get('risk_level', policy.get('severity', 'Unknown'))
                    if policy_risk_level and isinstance(policy_risk_level, str):
                        policy_risk_level = policy_risk_level.capitalize()
                    violated_policy_details.append(f"  • {policy_id_str} ({policy_risk_level}): {policy_desc}")
                else:
                    # Policy not found, just show ID
                    violated_policy_details.append(f"  • {policy_id_str}: Policy content not available")
            
            guidance_parts.append("- Policy violations detected:")
            guidance_parts.extend(violated_policy_details)
        
        if not has_policy_violations:
            guidance_parts.append("- High risk detected based on World Model's assessment")
        
        # Add optimization_guidance if available (this is the actionable guidance from World Model)
        # This should be integrated directly into the risk guidance, not as a separate section
        # Use only the first optimization_guidance if multiple are available
        if all_optimization_guidance:
            optimization_guidance = all_optimization_guidance[0]  # Use only the first one
            guidance_parts.append("")
            guidance_parts.append("**CRITICAL: You MUST strictly follow the optimization guidance below when regenerating the action.**")
            guidance_parts.append("")
            guidance_parts.append("**Optimization Guidance:**")
            guidance_parts.append(optimization_guidance)
            if len(all_optimization_guidance) > 1:
                guidance_parts.append("")
                guidance_parts.append("(Note: Multiple optimization guidance were provided, using the first one above.)")
        
        guidance_parts.extend([
            "",
            "**CRITICAL: Regenerate a safer action. The previous action(s) were NOT executed yet.**",
            "",
            "**Required Process:**",
            "1. **Reflection**: Why were the previous actions high-risk? What policies would be violated?",
            "2. **Analysis**: What is your legitimate goal? What are safe ways to achieve it?",
            "3. **Action Generation**: Generate a safer action with clear reasoning.",
            "   - Start with \"Let's think step-by-step\" and show your complete reasoning.",
            "   - End with \"In summary, the next action I will perform is ```[action]```\"",
            "",
            "**Guidelines:**",
            "- Comply with all safety policies",
            "- Ensure actions align with the task goal",
            "- Do NOT repeat the filtered actions above",
        ])
        
        return "\n".join(guidance_parts)
    
    def _load_experience_replay_examples(self, config: SafetyConfig) -> Optional[List[Dict[str, Any]]]:
        """
        Load experience data for few-shot learning, prioritizing trajectory graph data.
        
        Strategy:
        1. First try to extract examples from current trajectory graph (most relevant)
        2. If not enough examples, fall back to trajectory storage (file system)
        
        Args:
            config: Safety configuration with experience replay settings
        
        Returns:
            List of few-shot examples in format: [{"state": ..., "action": ..., "next_state": ...}, ...]
            Returns None if no examples found or experience replay is disabled
        """
        filter_successful = getattr(config, 'experience_replay_filter_successful', True)
        max_risk = getattr(config, 'experience_replay_max_risk', 0.5)
        max_examples = getattr(config, 'experience_replay_max_examples', 5)
        
        examples = []
        
        # Step 1: Try to extract examples from trajectory graph (current session)
        # Note: trajectory_graph may not be initialized yet during SafeAgent.__init__
        try:
            if hasattr(self, 'trajectory_graph') and self.trajectory_graph and len(self.trajectory_graph.nodes) > 0:
                graph_examples = self.trajectory_graph.extract_examples(
                    max_examples=max_examples,
                    max_risk=max_risk,
                    filter_successful=filter_successful,
                )
                examples.extend(graph_examples)
                if graph_examples:
                    logger.info(f"[SafeAgent] Extracted {len(graph_examples)} examples from trajectory graph for World Model few-shot learning")
        except Exception as e:
            logger.warning(f"[SafeAgent] Failed to extract examples from trajectory graph: {e}")
        
        # Step 2: If not enough examples, load from trajectory storage (file system)
        if len(examples) < max_examples and self.trajectory_storage:
            try:
                # Check if we should load inaccurate examples for contrastive learning
                load_inaccurate = getattr(config, 'experience_replay_load_inaccurate', False)
                
                # Automatically get task_id from current task context
                task_id = None
                if hasattr(self, 'conversation_history_manager'):
                    task_id = getattr(self.conversation_history_manager, 'current_task_id', None)
                    if task_id:
                        logger.debug(f"[SafeAgent] Auto-detected task_id='{task_id}' for experience replay filtering")
                
                # Load from classified experience folders
                # Automatically filter by current task_id if available
                entries = self.trajectory_storage.load_entries(
                    filepath=None,  # Load from default experience_dir
                    filter_successful=filter_successful if filter_successful else None,
                    max_risk=max_risk,
                    load_accurate_only=True,  # Always load accurate examples
                    load_inaccurate=load_inaccurate,  # Optionally load inaccurate for contrastive learning
                    task_keyword=task_id,  # Automatically filter by current task_id
                )
                
                if entries:
                    # Limit number of additional examples needed
                    remaining_slots = max_examples - len(examples)
                    entries = entries[:remaining_slots]
                    
                    # Convert to few-shot examples format
                    # Use compact state directly (saved in entry.state and entry.next_state)
                    for entry in entries:
                        # Skip entries without state (should not happen with new format)
                        if entry.state is None:
                            continue
                        
                        example = {
                            "state": entry.state,  # Compact state dict
                            "action": entry.action,
                            "next_state": entry.next_state,  # Compact next_state dict (may be None)
                        }
                        
                        # Include actual_delta if available (for learning state transitions)
                        if entry.actual_delta:
                            example["actual_delta"] = entry.actual_delta
                        
                        # Include predicted_delta if available (for learning prediction patterns)
                        if entry.predicted_delta:
                            example["predicted_delta"] = entry.predicted_delta
                        
                        examples.append(example)
                    
                    accurate_count = sum(1 for ex in examples if ex.get("is_accurate", True))
                    inaccurate_count = len(examples) - accurate_count
                    if entries:
                        logger.info(
                            f"[SafeAgent] Loaded {len(entries)} additional examples from trajectory storage "
                            f"(accurate: {accurate_count}, inaccurate: {inaccurate_count}, total: {len(examples)})"
                        )
            except Exception as e:
                logger.warning(f"[SafeAgent] Failed to load experience entries from storage: {e}")
        
        if not examples:
            logger.debug("[SafeAgent] No experience examples found for replay (neither from graph nor storage)")
            return None
        
        # Limit to max_examples
        examples = examples[:max_examples]
        logger.info(f"[SafeAgent] Total {len(examples)} experience examples loaded for World Model few-shot learning")
        return examples
    
    def get_safe_action(
        self,
        current_state: Any,
        max_depth: Optional[int] = None,
        candidate_actions: Optional[List[Any]] = None,
        action_generator: Optional[Callable[[Any, str, int], List[Any]]] = None,
        max_regeneration_attempts: int = 1,
    ) -> Dict[str, Any]:
        """
        Get the safest action for the current state using unified tree search (single-step is special case).
        
        Process:
        - Uses unified tree search implementation, where single-step prediction is max_depth=1
        - For each candidate action, predict next state and evaluate risk
        - Filter actions by risk threshold
        - If all actions filtered and action_generator provided, automatically regenerate
        - Return action with minimum risk
        
        Args:
            current_state: Current state representation (raw observation dict or compact state)
            max_depth: Maximum search depth. None=use config, 1=single-step (default), >1=tree search
            candidate_actions: Required list of candidate actions to evaluate (will take first n_root)
            action_generator: Optional callback function(state, risk_guidance, num_samples) -> List[actions]
                             If provided and all actions filtered, will be called automatically for regeneration
                             Also used for depth expansion when max_depth > 1
            max_regeneration_attempts: Maximum number of regeneration attempts (default: 1)
        
        Returns:
            Dictionary containing:
                - 'action': Optimal safe action to execute
                - 'path': Action path (list of actions)
                - 'risk': Risk score
                - 'search_stats': Search statistics
                - 'requires_regeneration': True if all actions were filtered (only if no action_generator or max attempts reached)
                - 'risk_guidance': Guidance for regeneration (only if requires_regeneration=True)
        """
        if candidate_actions is None:
            raise ValueError("candidate_actions is required")
        
        # Preprocess raw observation to compact state if needed
        compact_state = self._preprocess_state(current_state)
        
        # Get configuration parameters
        config_max_depth = getattr(self.config, 'tree_search_max_depth', 1)
        effective_max_depth = max_depth if max_depth is not None else config_max_depth
        
        # Unified tree search implementation (single-step is max_depth=1 special case)
        result = self._tree_search_planning(
            current_state=compact_state,
            root_actions=candidate_actions,
            action_agent=action_generator,
            max_depth=effective_max_depth,
        )
        
        # Validate result is a dict
        validate_dict(result, "result", context="[SafeAgent] ")
        
        # If regeneration needed and action_generator provided, handle internally
        if result.get('requires_regeneration', False) and action_generator is not None:
            risk_guidance = result.get('risk_guidance', '')
            stats = result.get('search_stats', {})
            filtered_count = stats.get('filtered_by_risk', len(candidate_actions))
            
            # Determine regeneration sample count based on filtered actions
            if filtered_count == 1:
                regeneration_samples = min(3, len(candidate_actions))
            else:
                regeneration_samples = len(candidate_actions)
            
            logger.info(
                f"[SafeAgent] All actions filtered, attempting automatic regeneration "
                f"(samples={regeneration_samples}, max_attempts={max_regeneration_attempts})"
            )
            
            # Attempt regeneration
            for attempt in range(max_regeneration_attempts):
                try:
                    # Generate new actions using provided generator
                    regenerated_actions = action_generator(current_state, risk_guidance, regeneration_samples)
                    
                    if not regenerated_actions:
                        logger.warning(
                            f"[SafeAgent] Regeneration attempt {attempt + 1}/{max_regeneration_attempts} "
                            f"returned no actions"
                        )
                        continue
                    
                    # Re-evaluate regenerated actions
                    logger.info(
                        f"[SafeAgent] Regeneration attempt {attempt + 1}/{max_regeneration_attempts}: "
                        f"Evaluating {len(regenerated_actions)} new actions"
                    )
                    
                    result = self._tree_search_planning(
                        current_state=compact_state,
                        root_actions=regenerated_actions,
                        action_agent=action_generator,
                        max_depth=effective_max_depth,
                    )
                    
                    # Validate result is a dict
                    validate_dict(result, "result", context="[SafeAgent] ")
                    
                    # If regeneration succeeded, return result
                    if not result.get('requires_regeneration', False):
                        logger.info(
                            f"[SafeAgent] Regeneration attempt {attempt + 1} succeeded: "
                            f"Found safe action with risk={result.get('risk', 0.0):.4f}"
                        )
                        return result
                    
                except Exception as e:
                    logger.warning(
                        f"[SafeAgent] Regeneration attempt {attempt + 1}/{max_regeneration_attempts} "
                        f"failed: {e}"
                    )
                    continue
            
            # All regeneration attempts failed
            logger.warning(
                f"[SafeAgent] All {max_regeneration_attempts} regeneration attempts failed. "
                f"Returning result with requires_regeneration=True"
            )
        
        return result
    
    def update_trajectory(
        self,
        state: Any,
        action: Any,
        next_state: Any,
        action_success: bool = True,
        reward: Optional[float] = None,
        raw_obs: Optional[Dict[str, Any]] = None,
        raw_next_obs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update trajectory graph with executed state-action transition.
        
        IMPORTANT: Trajectory graph is only updated when:
        1. The action passed risk evaluation (risk < threshold) - ensured by get_safe_action()
        2. The action was executed successfully (action_success=True)
        
        If either condition fails, the trajectory graph will NOT be updated.
        
        Call this after executing an action in the real environment to
        maintain the trajectory graph for future planning.
        
        Args:
            state: Previous state
            action: Executed action (should be risk-evaluated and passed)
            next_state: Resulting state from real environment
            action_success: Whether the action was executed successfully. 
                          If False, trajectory graph will not be updated.
            reward: Optional reward received
            raw_obs: Optional raw observation dictionary for current state (for training data)
            raw_next_obs: Optional raw observation dictionary for next state (for training data)
        """
        # Log action execution result
        action_str = str(action) if action else "None"
        if action_success:
            logger.info(f"[SafeAgent] Action executed successfully: {action_str}")
            # Note: Conversation history is managed by evaluate_action_risk/record_executed_action
            # update_trajectory should NOT add conversation history to avoid duplication
            # Conversation history is added in:
            # - evaluate_action_risk (when action is safe and will be executed)
            # - record_executed_action (when requires_regeneration=True but max attempts reached)
        else:
            logger.warning(f"[SafeAgent] Action execution FAILED: {action_str}")
            # Do not update trajectory graph if action failed
            # This ensures only successful actions are recorded in the graph
            return
        
        # Preprocess states to compact representation if needed
        compact_state = self._preprocess_state(state)
        compact_next_state = self._preprocess_state(next_state)
        
        # Store metadata for trajectory graph edges (for filtering successful actions)
        # Use compact states for state ID computation
        state_id = self.trajectory_graph.compute_state_id(compact_state)
        next_state_id = self.trajectory_graph.compute_state_id(compact_next_state)
        
        # Update edge metadata to mark as successful
        if state_id in self.trajectory_graph.edges:
            for edge in self.trajectory_graph.edges[state_id]:
                if edge.to_state_id == next_state_id:
                    edge.metadata["success"] = True
                    break
        
        # Record in history (use compact states)
        self.trajectory_history.append({
            "state": compact_state,
            "action": action,
            "next_state": compact_next_state,
        })
        
        # Save to trajectory storage for world model training
        if self.trajectory_storage is not None:
            try:
                # Get state IDs from trajectory graph (already computed above)
                # Compute actual delta from real execution (for experience replay)
                # Use compact states for delta computation
                actual_delta = self._compute_actual_delta(compact_state, compact_next_state, action)
                
                # Log actual delta for debugging
                log_data = {
                    'component': 'SafeAgent',
                    'stage': 'Actual Delta Computation',
                    'action': str(action),
                    'actual_delta': actual_delta,
                }
                logger.debug(
                    f"[SafeAgent] Actual Delta Computed\n{json.dumps(log_data, indent=2, ensure_ascii=False)}"
                )
                
                # Get World Model predicted delta and risk explanation from evaluation context
                predicted_delta = None
                risk_score = 0.0
                risk_explanation = None
                
                # Compare actions by string representation to handle different types
                action_str = str(action) if action else None
                if self._last_action_evaluation:
                    # Validate _last_action_evaluation is a dict
                    validate_dict(self._last_action_evaluation, "_last_action_evaluation", context="[SafeAgent] ")
                    
                    stored_action_str = str(self._last_action_evaluation.get('action', ''))
                    if action_str and stored_action_str == action_str:
                        # Use evaluation context from get_safe_action
                        if 'risk_score' not in self._last_action_evaluation:
                            raise KeyError(f"[SafeAgent] _last_action_evaluation missing required key 'risk_score'. Available keys: {list(self._last_action_evaluation.keys())}")
                        risk_score = self._last_action_evaluation['risk_score']
                        risk_explanation = self._last_action_evaluation.get('risk_explanation')
                        predicted_delta = self._last_action_evaluation.get('world_model_metadata')
                        
                        # Validate predicted_delta is a dict or None
                        if predicted_delta is not None:
                            validate_dict(predicted_delta, "predicted_delta", context="[SafeAgent] ")
                        
                        logger.debug(
                            f"[SafeAgent] Using evaluation context for action '{action_str}': "
                            f"risk_score={risk_score:.3f}, has_predicted_delta={predicted_delta is not None}, "
                            f"has_risk_explanation={risk_explanation is not None}"
                        )
                    else:
                        logger.debug(
                            f"[SafeAgent] Action mismatch: stored='{stored_action_str}', current='{action_str}'. "
                            f"Falling back to trajectory graph."
                        )
                
                # Evaluation context must be available, raise error if not
                if risk_score == 0.0 and next_state_id in self.trajectory_graph.nodes:
                    error_msg = (
                        f"[SafeAgent] Evaluation context not available for action '{action_str}' "
                        f"and state '{next_state_id[:8]}...'. Cannot update trajectory."
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                # Clear evaluation context after saving
                self._last_action_evaluation = None
                
                # Get reward if available
                if reward is None and self.reward_function is not None:
                    try:
                        reward = self.reward_function(next_state)
                    except Exception:
                        pass
                
                # Save entry (compact mode: state_id + actual_delta only)
                # Do NOT save raw_obs/raw_next_obs in metadata to avoid huge file sizes
                # Raw observations may contain screenshot, full axtree_object, etc.
                # Compact state already contains all necessary information for replay
                metadata_dict = {
                    "timestamp": datetime.now().isoformat(),
                }
                
                # Add task_id and step_number for training data tracking
                # Try to get task_id from conversation_history_manager
                if hasattr(self, 'conversation_history_manager'):
                    task_id = getattr(self.conversation_history_manager, 'current_task_id', None)
                    if task_id:
                        metadata_dict["task_id"] = task_id
                    
                    # Get step_number from total_executed_steps
                    step_number = getattr(self.conversation_history_manager, 'total_executed_steps', None)
                    if step_number is not None:
                        metadata_dict["step_number"] = step_number
                
                # Note: We intentionally do NOT save raw_obs/raw_next_obs here because:
                # 1. They contain large fields (screenshot, full axtree_object, etc.)
                # 2. Compact state already has all necessary information (key_elements, chat_history, etc.)
                # 3. This keeps experience files small and efficient
                
                self.trajectory_storage.save_entry(
                    state=compact_state,  # Pass compact state for training data
                    action=action,
                    next_state=compact_next_state,  # Pass compact state for training data
                    state_id=state_id,
                    next_state_id=next_state_id,
                    actual_delta=actual_delta,
                    predicted_delta=predicted_delta,
                    risk_score=risk_score,
                    risk_explanation=risk_explanation,
                    action_success=action_success,
                    reward=reward,
                    metadata=metadata_dict,
                )
            except Exception as e:
                logger.warning(f"[SafeAgent] Failed to save trajectory entry: {e}")
    
    def _save_filtered_action(
        self,
        state: Any,
        action: Any,
        risk_score: float,
        risk_explanation: Optional[str],
        predicted_delta: Optional[Dict[str, Any]],
        violated_policy_ids: List[str],
    ) -> None:
        """
        Save a filtered (high-risk) action to training data.
        
        This method saves actions that were filtered due to high risk (risk_score >= threshold).
        These actions are important for training the world model to recognize high-risk scenarios.
        
        Args:
            state: Current state (before action)
            action: Action that was filtered
            risk_score: Risk score of the action
            risk_explanation: Risk evaluation explanation
            predicted_delta: World Model's predicted delta (from simulation)
            violated_policy_ids: List of violated policy IDs
        """
        if self.trajectory_storage is None:
            return
        
        try:
            # Check if state is already in compact format (has key_elements)
            # or if it's a raw observation dict (has axtree_txt or axtree_object)
            if isinstance(state, dict) and "key_elements" in state:
                # Already in compact format
                compact_state = state
            else:
                # Preprocess raw observation to compact representation
                compact_state = self._preprocess_state(state)
            
            # Compute state ID
            state_id = self.trajectory_graph.compute_state_id(compact_state)
            
            # For filtered actions, we don't have a next_state (action wasn't executed)
            # Use predicted_delta as the ground truth for training
            # The predicted_delta already contains all necessary information (semantic_delta, 
            # risk_affordances, risk_exposure, violated_policy_ids, etc.)
            
            # Save filtered action to training data
            # Note: action_success=False, but we still save to training data for high-risk scenario training
            self.trajectory_storage.save_filtered_action(
                state=compact_state,
                action=action,
                state_id=state_id,
                risk_score=risk_score,
                risk_explanation=risk_explanation,
                predicted_delta=predicted_delta,
                violated_policy_ids=violated_policy_ids,
            )
            
            logger.debug(
                f"[SafeAgent] Saved filtered action to training data: action='{str(action)[:100]}', "
                f"risk_score={risk_score:.3f}, violated_policies={violated_policy_ids}"
            )
        except Exception as e:
            logger.warning(f"[SafeAgent] Failed to save filtered action: {e}")
    
    def add_trajectory(
        self,
        trajectory: List[tuple],
        state_encoder: Optional[Callable] = None,
    ) -> None:
        """
        Add a complete historical trajectory to the graph.
        
        Useful for initializing the agent with past experience.
        
        Args:
            trajectory: List of (state, action) tuples
            state_encoder: Optional function to encode states
        """
        # Create a risk evaluator function that uses world model
        # trajectory_graph expects a function that takes state and returns risk_score (float)
        def world_model_risk_evaluator(state):
            # World Model must provide risk score
            if hasattr(self.world_model, 'get_risk_score'):
                risk_score = self.world_model.get_risk_score()
                if risk_score is not None:
                    return risk_score
            # Raise error if risk score not available
            error_msg = f"[SafeAgent] World Model did not return risk_score for trajectory graph update"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        self.trajectory_graph.add_trajectory(
            trajectory=trajectory,
            state_encoder=state_encoder,
            risk_evaluator=world_model_risk_evaluator,
        )
    
    def get_trajectory_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current trajectory.
        
        Returns:
            Dictionary with trajectory statistics
        """
        graph_stats = self.trajectory_graph.get_statistics()
        
        return {
            "graph_stats": graph_stats,
            "history_length": len(self.trajectory_history),
            "config": self.config.to_dict(),
        }
    
    def reset(self) -> None:
        """
        Reset agent state (clear trajectory graph and history).
        
        Call this at the start of each new task to ensure each task has an independent trajectory graph.
        """
        nodes_before = len(self.trajectory_graph.nodes)
        edges_before = sum(len(edges) for edges in self.trajectory_graph.edges.values())
        
        self.trajectory_graph.clear()
        self.trajectory_history.clear()
        self.conversation_history_manager.reset()  # Reset conversation history for new task
        
        logger.info(
            f"[SafeAgent] Trajectory graph reset: cleared {nodes_before} nodes and {edges_before} edges"
        )
    
    def close(self) -> None:
        """
        Close agent and flush trajectory storage.
        
        Call this at the end of each task to ensure all trajectory data is saved to disk.
        This will:
        1. Flush all in-memory experience entries to experience_dir
        2. Flush all in-memory training conversations to training_dir
        3. Log statistics about saved data
        """
        if self.trajectory_storage is not None:
            # Get statistics before flushing
            stats = self.trajectory_storage.get_statistics()
            experience_count = len(self.trajectory_storage.experience_entries)
            training_count = len(self.trajectory_storage.training_conversations)
            
            # Flush all remaining data to disk
            self.trajectory_storage.close()
            
            logger.info(
                f"[SafeAgent] Trajectory data saved: {experience_count} experience entries, "
                f"{training_count} training conversations"
            )
            logger.info(
                f"[SafeAgent] Trajectory statistics: {json.dumps(stats, indent=2, ensure_ascii=False)}"
            )
        else:
            logger.debug("[SafeAgent] No trajectory storage configured, skipping flush")
    
    def set_reward_function(self, reward_function: Callable[[Any], float]) -> None:
        """
        Set or update the reward function.
        
        Args:
            reward_function: Function that takes state and returns reward
        """
        self.reward_function = reward_function
    
    def set_risk_threshold(self, risk_threshold: float) -> None:
        """
        Update the risk threshold for pruning.
        
        Args:
            risk_threshold: New risk threshold
        """
        if risk_threshold < 0:
            raise ValueError("risk_threshold must be non-negative")
        self.config.risk_threshold = risk_threshold
    


def create_safe_agent(
    benchmark: Optional[str] = None,
    config: Optional[SafetyConfig] = None,
    **kwargs,
) -> SafeAgent:
    """
    Factory function to create a SafeAgent with common configurations.
    
    Args:
        benchmark: Benchmark name ('mind2web', 'webguard', 'visualwebarena')
        config: Optional custom configuration
        **kwargs: Additional arguments passed to SafeAgent
    
    Returns:
        Initialized SafeAgent instance
    
    Examples:
        # For Mind2Web benchmark
        agent = create_safe_agent("mind2web")
        
        # Custom configuration
        agent = create_safe_agent(config=SafetyConfig())
    """
    return SafeAgent(config=config, benchmark=benchmark, **kwargs)


def create_browsergym_agent(
    action_set_description: str,
    config: Optional[SafetyConfig] = None,
    policies: Optional[List[Dict[str, Any]]] = None,
    **kwargs,
) -> SafeAgent:
    """
    Factory function to create a SafeAgent optimized for BrowserGym integration.
    
    This simplifies integration by:
    - Using string format actions (BrowserGym format)
    - Loading LLM configuration from config.yaml
    - Setting up action generation with custom action description
    
    Args:
        action_set_description: Description of available actions in BrowserGym format
        config: Optional custom configuration (if None, loads from config.yaml)
        policies: Optional list of policies for risk evaluation
        **kwargs: Additional arguments passed to SafeAgent (e.g., reward_function)
    
    Returns:
        Initialized SafeAgent instance configured for BrowserGym
    
    Example:
        from browsergym.core.action.highlevel import HighLevelActionSet
        
        action_set = HighLevelActionSet(...)
        agent = create_browsergym_agent(
            action_set_description=action_set.describe(with_long_description=False, with_examples=True)
        )
        
    Note:
        LLM configuration (api_key, api_url, model_name) should be set in config.yaml
        under world_model_llm section. Risk evaluation is now integrated into World Model.
    """
    # Load config from YAML if not provided
    if config is None:
        try:
            config = SafetyConfig.from_yaml()
        except Exception as e:
            # Raise error if YAML loading fails
            error_msg = f"[SafeAgent] Failed to load config from YAML: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    # Store action_description in config.custom_params for World Model creation
    # This allows SafeAgent to access it when creating World Model
    if not hasattr(config, 'custom_params') or config.custom_params is None:
        config.custom_params = {}
    config.custom_params['action_description'] = action_set_description
    config.custom_params['action_format'] = "string"
    
    # Store policies in config.custom_params if provided
    if policies is not None:
        config.custom_params['policies'] = policies
    
        # Create and return SafeAgent
        # SafeAgent will create World Model and Risk Evaluator with proper configuration
        return SafeAgent(
            config=config,
            world_model=None,  # Let SafeAgent create with action_description
            policies=policies,  # Pass policies to SafeAgent
            **kwargs
        )

