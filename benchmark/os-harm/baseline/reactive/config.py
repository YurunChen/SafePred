"""
Configuration management for reactive safety system.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ReactiveConfig:
    """Configuration for reactive safety system."""
    
    # LLM configuration (similar to SafePred)
    api_key: str
    api_url: str
    model_name: str = "gpt-4o"
    provider: str = "openai"  # 'openai', 'qwen', or 'custom'
    max_tokens: int = 2048
    temperature: float = 0.0
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    
    # Policy configuration
    enable_reference_updates: bool = True  # Default to True (automatic updates)
    
    @classmethod
    def from_yaml(cls, yaml_path: Optional[str] = None) -> "ReactiveConfig":
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML config file. If None, looks for config.yaml in current directory.
            
        Returns:
            ReactiveConfig instance
        """
        if yaml_path is None:
            # Look for config.yaml in current directory or parent directories
            current_dir = os.path.dirname(os.path.abspath(__file__))
            yaml_path = os.path.join(current_dir, "config.yaml")
            if not os.path.exists(yaml_path):
                # Try parent directory
                parent_dir = os.path.dirname(current_dir)
                yaml_path = os.path.join(parent_dir, "config.yaml")
        
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_content = f.read()
        
        # Replace environment variables
        yaml_content = cls._replace_env_vars(yaml_content)
        
        config_dict = yaml.safe_load(yaml_content)
        
        # Extract reactive_agent_llm config (similar to SafePred's world_model_llm)
        reactive_llm_config = config_dict.get("reactive_agent_llm", {})
        
        return cls(
            api_key=reactive_llm_config.get("api_key", os.getenv("OPENAI_API_KEY", "")),
            api_url=reactive_llm_config.get("api_url", os.getenv("OPENAI_API_BASE", "")),
            model_name=reactive_llm_config.get("model_name", "gpt-4o"),
            provider=reactive_llm_config.get("provider", "openai"),
            max_tokens=reactive_llm_config.get("max_tokens", 2048),
            temperature=reactive_llm_config.get("temperature", 0.0),
            timeout=reactive_llm_config.get("timeout", 30.0),
            max_retries=reactive_llm_config.get("max_retries", 3),
            retry_delay=reactive_llm_config.get("retry_delay", 1.0),
            retry_backoff=reactive_llm_config.get("retry_backoff", 2.0),
            enable_reference_updates=True,  # Always enabled (automatic updates)
        )
    
    @staticmethod
    def _replace_env_vars(content: str) -> str:
        """Replace environment variable placeholders in YAML content."""
        import re
        def replace_var(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))
        
        return re.sub(r'\$\{([^}]+)\}', replace_var, content)
