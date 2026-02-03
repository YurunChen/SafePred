"""
Configuration management for Safety-TS-LMA.

Provides configuration classes and YAML loading functionality.
"""

from .config import SafetyConfig, default_config, BENCHMARK_CONFIGS

__all__ = [
    "SafetyConfig",
    "default_config",
    "BENCHMARK_CONFIGS",
]


