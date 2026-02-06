"""
Configuration management for Safety-TS-LMA.

Provides flexible configuration system for different benchmarks and use cases.
Supports loading configuration from YAML files.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import os
import yaml
import logging

logger = logging.getLogger(__name__)

# Load .env from project root if python-dotenv is available
def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        # Project root: parent of config package directory
        package_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(package_dir)
        env_path = os.path.join(project_root, ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path)
    except ImportError:
        pass


@dataclass
class SafetyConfig:
    """
    Configuration for Safety-TS-LMA algorithm.
    
    Attributes:
        risk_threshold: Risk threshold for filtering unsafe actions (default: 0.7)
        enable_pruning: Whether to enable risk-based action filtering (default: True)
        state_encoder_model: Model name/path for state encoding (default: 'qwen2.5-3B')
        device: Device to run models on ('cuda', 'cpu') (default: 'cuda')
        batch_size: Batch size for model inference (default: 8)
        temperature: Temperature for LLM-based world model (default: 0.7)
        max_tokens: Maximum tokens for LLM generation (default: 512)
        custom_params: Custom parameters for specific benchmarks
    """
    
    # Risk filtering parameters
    risk_threshold: float = 0.7  # Risk threshold for filtering actions
    enable_pruning: bool = True  # Enable risk-based action filtering
    
    # Trajectory storage configuration
    enable_trajectory_storage: bool = True  # Enable trajectory data storage for training
    trajectory_experience_dir: Optional[str] = None  # Directory to save experience data (default: ./trajectories/experience/)
    trajectory_training_dir: Optional[str] = None  # Directory to save training data in ShareGPT format (default: ./trajectories/training/)
    trajectory_system_prompt: Optional[str] = None  # System prompt for ShareGPT format training data
    include_policies_in_training: bool = True  # Whether to include policies in training data (recommended for safety-aware models)
    
    # Experience replay configuration
    enable_experience_replay: bool = True  # Enable experience replay for World Model (few-shot learning)
    experience_replay_max_examples: int = 5  # Maximum number of examples to load for few-shot learning
    experience_replay_filter_successful: bool = True  # Only load successful actions for experience replay
    experience_replay_max_risk: Optional[float] = 0.5  # Maximum risk score for experience replay (None = no filter)
    experience_replay_load_inaccurate: bool = False  # Whether to load inaccurate predictions for contrastive learning (default: False, only load accurate)
    # Note: Experience replay automatically filters by current task_id (no configuration needed)
    
    # Trajectory storage configuration
    trajectory_storage_max_entries_in_memory: int = 50  # Maximum entries to keep in memory before flushing (default: 1000, lowered for more frequent saves)
    trajectory_save_inaccurate_for_analysis: bool = True  # Whether to save inaccurate predictions for analysis (Scheme 2)
    trajectory_risk_consistency_threshold: float = 0.7  # Threshold for risk consistency validation (0.0-1.0)
    
    # Model configuration
    state_encoder_model: Optional[str] = "qwen2.5-3B"
    device: str = "cuda"
    batch_size: int = 8
    
    # LLM generation parameters
    temperature: float = 0.7
    max_tokens: int = 512
    
    # LLM API configuration (loaded from config.yaml)
    llm_api_key: Optional[str] = None
    llm_api_url: Optional[str] = None
    llm_provider: str = "openai"  # 'openai' (OpenAI SDK), 'gemini' (Gemini SDK), 'custom' (requests)
    llm_timeout: int = 30
    
    # World model LLM config (overrides global if set)
    world_model_llm_api_key: Optional[str] = None
    world_model_llm_api_url: Optional[str] = None
    world_model_llm_model_name: Optional[str] = None
    world_model_llm_provider: Optional[str] = None
    world_model_llm_temperature: Optional[float] = None
    world_model_llm_max_tokens: Optional[int] = None
    world_model_llm_timeout: Optional[int] = None
    world_model_use_state_delta: bool = True  # Use state delta mode (predict only changes, more efficient)
    world_model_prediction_steps: int = 1  # Number of steps to predict ahead (1 = single-step, >1 = multi-step)
    world_model_log_prompt: bool = False  # Whether to log full prompts (default: False to reduce log size)
    
    # Rule extractor LLM config (overrides global if set)
    rule_extractor_llm_api_key: Optional[str] = None
    rule_extractor_llm_api_url: Optional[str] = None
    rule_extractor_llm_model_name: Optional[str] = None
    rule_extractor_llm_provider: Optional[str] = None
    rule_extractor_llm_temperature: Optional[float] = None
    rule_extractor_llm_max_tokens: Optional[int] = None
    rule_extractor_llm_timeout: int = 60  # Longer timeout for document processing
    
    # Action agent LLM config (for generating child actions in tree search)
    action_agent_llm_api_key: Optional[str] = None
    action_agent_llm_api_url: Optional[str] = None
    action_agent_llm_model_name: Optional[str] = None
    action_agent_llm_provider: Optional[str] = None
    action_agent_llm_temperature: Optional[float] = None
    action_agent_llm_max_tokens: Optional[int] = None
    action_agent_llm_timeout: Optional[int] = None
    action_agent_log_prompt: bool = False  # Whether to log full action-agent prompts (default False to avoid privacy leak in logs)
    
    # Logging configuration
    log_level: str = "INFO"  # Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)
    
    # Conversation history management configuration
    conversation_history_max_length: Optional[int] = 20  # Maximum number of messages to keep in conversation history. When exceeded, older messages are removed (FIFO). Set to None or 0 to disable limiting.
    conversation_history_show_full_response: bool = True  # Whether to show full response or only action in conversation history. True = full response (including reasoning), False = only action string (e.g., "click [42]").
    
    # Planning configuration
    planning_enable: bool = False  # Enable plan monitoring and adaptation
    risk_guidance_enable: bool = True  # Enable risk guidance for action regeneration (default: True)
    enable_short_term_prediction: bool = True  # Enable short-term prediction (semantic_delta) (default: True)
    enable_long_term_prediction: bool = True  # Enable long-term prediction (long_term_impact) (default: True)
    
    # Action sampling configuration
    num_action_samples: int = 1  # Number of candidate actions to generate and evaluate
    use_diverse_sampling: bool = False  # Use different temperatures to increase diversity
    
    # Policy reference storage configuration
    reference_limits: Optional[Dict[str, int]] = None  # Maximum references to store per risk level (e.g., {"high": 10, "medium": 7, "low": 5})
    similarity_threshold: float = 0.85  # Similarity threshold for duplicate detection (0.0-1.0). References with similarity >= this threshold will be considered duplicates and skipped. Default: 0.85
    show_policy_references: bool = False  # Whether to include policy reference examples in prompts (default: False)
    
    # Risk score calculation configuration
    risk_score_violation_penalty: float = 0.5  # Penalty for policy violations (added when violations exist)
    risk_score_irreversible_penalty: float = 0.4  # Penalty for irreversible actions (added when action is irreversible)
    
    # Tree search planning configuration
    tree_search_n_root: int = 1  # Root node candidate action count (N_root)
    tree_search_n_child: int = 5  # Child action count per internal node (N_child)
    tree_search_m_child: int = 3  # Number of lowest-risk paths to keep (M_child)
    tree_search_max_depth: int = 1  # Maximum search depth (D). Default 1 = single-step prediction
    root_risk_threshold: float = 0.7  # Root node risk threshold (T_root, stricter)
    child_risk_threshold: float = 0.8  # Child node risk threshold (T_child, more lenient, should be >= root_risk_threshold)
    
    # Custom parameters for specific benchmarks
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.risk_threshold < 0:
            raise ValueError("risk_threshold must be non-negative")
        if self.child_risk_threshold < self.root_risk_threshold:
            logger.warning(
                f"child_risk_threshold ({self.child_risk_threshold}) < root_risk_threshold ({self.root_risk_threshold}). "
                f"Setting child_risk_threshold = root_risk_threshold."
            )
            self.child_risk_threshold = self.root_risk_threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "risk_threshold": self.risk_threshold,
            "enable_pruning": self.enable_pruning,
            "state_encoder_model": self.state_encoder_model,
            "device": self.device,
            "batch_size": self.batch_size,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "llm_api_key": self.llm_api_key,
            "llm_api_url": self.llm_api_url,
            "llm_provider": self.llm_provider,
            "llm_timeout": self.llm_timeout,
            "custom_params": self.custom_params,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SafetyConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: Optional[str] = None) -> "SafetyConfig":
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML config file. If None, looks for 'config.yaml' 
                      in the package directory.
        
        Returns:
            SafetyConfig instance with values from YAML file
        
        Example:
            config = SafetyConfig.from_yaml("config.yaml")
        """
        if yaml_path is None:
            # Try to find config.yaml in the config package directory
            package_dir = os.path.dirname(os.path.abspath(__file__))
            yaml_path = os.path.join(package_dir, "config.yaml")
            # If not found, try parent directory (for backward compatibility)
            if not os.path.exists(yaml_path):
                parent_dir = os.path.dirname(package_dir)
                yaml_path = os.path.join(parent_dir, "config.yaml")
        
        if not os.path.exists(yaml_path):
            # If file doesn't exist, return default config
            return cls()
        
        _load_dotenv()

        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f) or {}
        
        # Extract LLM configuration
        world_llm_config = yaml_config.get("world_model_llm", {})
        rule_extractor_llm_config = yaml_config.get("rule_extractor_llm", {})
        action_agent_llm_config = yaml_config.get("action_agent_llm", {})
        
        # Helper function to get value or None if empty string
        def get_value_or_none(config_dict, key, default=None):
            value = config_dict.get(key, default)
            # Treat empty string as None (use default)
            if value == "":
                return None
            return value if value is not None else default

        # API key/URL: from .env only (e.g. OPENAI_API_KEY, OPENAI_API_URL for provider "openai"); config.yaml does not set api_key/api_url
        def api_key_for_provider(llm_config: dict) -> Optional[str]:
            """Gets API key from environment variables based on the provider."""
            provider = (llm_config.get("provider") or "openai").strip().lower()
            env_key = f"{provider.upper()}_API_KEY"
            from_env = os.environ.get(env_key)
            if from_env and from_env.strip():
                return from_env.strip()
            return None

        def api_url_for_provider(llm_config: dict) -> Optional[str]:
            provider = (llm_config.get("provider") or "openai").strip().lower()
            env_key = f"{provider.upper()}_API_URL"
            from_env = os.environ.get(env_key)
            if from_env and from_env.strip():
                return from_env.strip()
            return None

        # Build config dict
        config_dict = {
            # Global defaults
            "llm_provider": "openai",
            "llm_timeout": 30,
            "temperature": world_llm_config.get("temperature") or 0.7,
            "max_tokens": world_llm_config.get("max_tokens") or 512,
            "state_encoder_model": get_value_or_none(world_llm_config, "model_name") or "qwen2.5-3B",
            # Risk threshold configuration
            "risk_threshold": yaml_config.get("risk_threshold", 0.7),  # Default: 0.7
            # World model config (api_key/api_url from .env by provider; config.yaml sets provider)
            "world_model_llm_api_key": api_key_for_provider(world_llm_config),
            "world_model_llm_api_url": api_url_for_provider(world_llm_config),
            "world_model_llm_model_name": get_value_or_none(world_llm_config, "model_name"),
            "world_model_llm_provider": get_value_or_none(world_llm_config, "provider", "openai"),
            "world_model_llm_temperature": world_llm_config.get("temperature"),
            "world_model_llm_max_tokens": world_llm_config.get("max_tokens"),
            "world_model_llm_timeout": world_llm_config.get("timeout"),
            "world_model_use_state_delta": world_llm_config.get("use_state_delta", True),  # Default to True
            "world_model_prediction_steps": world_llm_config.get("prediction_steps", 1),  # Default to 1 (single-step)
            "world_model_log_prompt": world_llm_config.get("log_prompt", False),  # Default to False
            # Experience replay config
            "enable_experience_replay": yaml_config.get("experience_replay", {}).get("enable", True),
            "experience_replay_max_examples": yaml_config.get("experience_replay", {}).get("max_examples", 5),
            "experience_replay_filter_successful": yaml_config.get("experience_replay", {}).get("filter_successful", True),
            "experience_replay_max_risk": yaml_config.get("experience_replay", {}).get("max_risk", 0.5),
            "experience_replay_load_inaccurate": yaml_config.get("experience_replay", {}).get("load_inaccurate", False),
            # Trajectory storage config
            "enable_trajectory_storage": yaml_config.get("trajectory_storage", {}).get("enable", True),
            "trajectory_storage_max_entries_in_memory": yaml_config.get("trajectory_storage", {}).get("max_entries_in_memory", 50),
            # Conversation history config
            "conversation_history_max_length": yaml_config.get("conversation_history", {}).get("max_length", 20),
            "conversation_history_show_full_response": yaml_config.get("conversation_history", {}).get("show_full_response", True),
            # Logging config
            "log_level": yaml_config.get("logging", {}).get("level", "INFO"),
            # Rule extractor config (api_key/api_url from .env by provider)
            "rule_extractor_llm_api_key": api_key_for_provider(rule_extractor_llm_config),
            "rule_extractor_llm_api_url": api_url_for_provider(rule_extractor_llm_config),
            "rule_extractor_llm_model_name": get_value_or_none(rule_extractor_llm_config, "model_name"),
            "rule_extractor_llm_provider": get_value_or_none(rule_extractor_llm_config, "provider", "openai"),
            "rule_extractor_llm_temperature": rule_extractor_llm_config.get("temperature"),
            "rule_extractor_llm_max_tokens": rule_extractor_llm_config.get("max_tokens"),
            "rule_extractor_llm_timeout": rule_extractor_llm_config.get("timeout") or 60,
            # Action agent LLM config (api_key/api_url from .env by provider)
            "action_agent_llm_api_key": api_key_for_provider(action_agent_llm_config),
            "action_agent_llm_api_url": api_url_for_provider(action_agent_llm_config),
            "action_agent_llm_model_name": get_value_or_none(action_agent_llm_config, "model_name"),
            "action_agent_llm_provider": get_value_or_none(action_agent_llm_config, "provider", "openai"),
            "action_agent_llm_temperature": action_agent_llm_config.get("temperature"),
            "action_agent_llm_max_tokens": action_agent_llm_config.get("max_tokens"),
            "action_agent_llm_timeout": action_agent_llm_config.get("timeout"),
            "action_agent_log_prompt": action_agent_llm_config.get("log_prompt", False),
            # Planning configuration
            "planning_enable": yaml_config.get("planning", {}).get("enable", False),
            "risk_guidance_enable": yaml_config.get("planning", {}).get("risk_guidance_enable", True),  # Default: True
            "enable_short_term_prediction": yaml_config.get("planning", {}).get("enable_short_term_prediction", True),  # Default: True
            "enable_long_term_prediction": yaml_config.get("planning", {}).get("enable_long_term_prediction", True),  # Default: True
            # Action sampling configuration
            "num_action_samples": yaml_config.get("action_sampling", {}).get("num_samples", 1),
            "use_diverse_sampling": yaml_config.get("action_sampling", {}).get("use_diverse_sampling", False),
            # Policy reference storage configuration (from world_model_llm config block)
            "reference_limits": world_llm_config.get("reference_limits") or {"high": 10, "medium": 7, "low": 5},
            "similarity_threshold": world_llm_config.get("similarity_threshold", 0.85),
            "show_policy_references": world_llm_config.get("show_policy_references", False),
            # Risk score calculation configuration
            "risk_score_violation_penalty": yaml_config.get("risk_score_config", {}).get("violation_penalty", 0.5),
            "risk_score_irreversible_penalty": yaml_config.get("risk_score_config", {}).get("irreversible_penalty", 0.4),
            # Tree search planning configuration
            "tree_search_n_root": yaml_config.get("tree_search", {}).get("n_root", 1),
            "tree_search_n_child": yaml_config.get("tree_search", {}).get("n_child", 5),
            "tree_search_m_child": yaml_config.get("tree_search", {}).get("m_child", 3),
            "tree_search_max_depth": yaml_config.get("tree_search", {}).get("max_depth", 1),
            "root_risk_threshold": yaml_config.get("tree_search", {}).get("root_risk_threshold", 0.7),
            "child_risk_threshold": yaml_config.get("tree_search", {}).get("child_risk_threshold", 0.8),
        }
        
        # Remove None values to use defaults
        config_dict = {k: v for k, v in config_dict.items() if v is not None}
        
        # Create config with YAML values, falling back to defaults
        return cls(**config_dict)
    
    def get_llm_config(self, component: str = "default") -> Optional[Dict[str, Any]]:
        """
        Get LLM configuration for a specific component.
        
        Args:
            component: Component name ('default', 'world_model', 'rule_extractor', 'action_agent')
        
        Returns:
            Dictionary with LLM configuration, or None if component not configured
        
        Raises:
            ValueError: If action_agent component is requested but model_name is not configured
        """
        if component == "action_agent":
            # Require explicit model_name configuration, no fallback
            if not self.action_agent_llm_model_name:
                raise ValueError(
                    "action_agent_llm.model_name is required in config.yaml. "
                    "Please configure action_agent_llm.model_name explicitly. "
                    "No fallback to state_encoder_model is allowed."
                )
            if not self.action_agent_llm_api_key:
                raise ValueError(
                    "action_agent_llm.api_key is required in config.yaml. "
                    "Please configure action_agent_llm.api_key explicitly."
                )
            return {
                "api_key": self.action_agent_llm_api_key,
                "api_url": self.action_agent_llm_api_url,
                "model_name": self.action_agent_llm_model_name,
                "provider": self.action_agent_llm_provider or self.llm_provider,
                "temperature": self.action_agent_llm_temperature if self.action_agent_llm_temperature is not None else self.temperature,
                "max_tokens": self.action_agent_llm_max_tokens if self.action_agent_llm_max_tokens is not None else self.max_tokens,
                "timeout": self.action_agent_llm_timeout if self.action_agent_llm_timeout is not None else self.llm_timeout,
            }
        elif component == "world_model":
            return {
                "api_key": self.world_model_llm_api_key,
                "api_url": self.world_model_llm_api_url,
                "model_name": self.world_model_llm_model_name or self.state_encoder_model,
                "provider": self.world_model_llm_provider or self.llm_provider,
                "temperature": self.world_model_llm_temperature if self.world_model_llm_temperature is not None else self.temperature,
                "max_tokens": self.world_model_llm_max_tokens if self.world_model_llm_max_tokens is not None else self.max_tokens,
                "timeout": self.world_model_llm_timeout if self.world_model_llm_timeout is not None else self.llm_timeout,
                "log_prompt": getattr(self, 'world_model_log_prompt', False),  # Add log_prompt to config
                "prediction_steps": getattr(self, 'world_model_prediction_steps', 1),
                "show_policy_references": getattr(self, 'show_policy_references', False),
                # Risk score calculation configuration
                "risk_score_violation_penalty": getattr(self, 'risk_score_violation_penalty', 0.5),
                "risk_score_irreversible_penalty": getattr(self, 'risk_score_irreversible_penalty', 0.4),
                # Prediction type configuration
                "enable_short_term_prediction": getattr(self, 'enable_short_term_prediction', True),
                "enable_long_term_prediction": getattr(self, 'enable_long_term_prediction', True),
            }
        elif component == "rule_extractor":
            return {
                "api_key": self.rule_extractor_llm_api_key,
                "api_url": self.rule_extractor_llm_api_url,
                "model_name": self.rule_extractor_llm_model_name or self.state_encoder_model,
                "provider": self.rule_extractor_llm_provider or self.llm_provider,
                "temperature": self.rule_extractor_llm_temperature if self.rule_extractor_llm_temperature is not None else self.temperature,
                "max_tokens": self.rule_extractor_llm_max_tokens if self.rule_extractor_llm_max_tokens is not None else self.max_tokens,
                "timeout": self.rule_extractor_llm_timeout,
            }
        else:
            # Default: use world_model config as fallback
            return {
                "api_key": self.world_model_llm_api_key,
                "api_url": self.world_model_llm_api_url,
                "model_name": self.state_encoder_model,
                "provider": self.llm_provider,
                "temperature": self.world_model_llm_temperature if self.world_model_llm_temperature is not None else self.temperature,
                "max_tokens": self.world_model_llm_max_tokens if self.world_model_llm_max_tokens is not None else self.max_tokens,
                "timeout": self.world_model_llm_timeout if self.world_model_llm_timeout is not None else self.llm_timeout,
            }


# Default configuration for quick start
default_config = SafetyConfig()

# Predefined configurations for common benchmarks
BENCHMARK_CONFIGS = {
    "mind2web": SafetyConfig(
        risk_threshold=0.8,
    ),
    "webguard": SafetyConfig(
        risk_threshold=0.6,
    ),
    "visualwebarena": SafetyConfig(
        risk_threshold=1.0,
    ),
}

