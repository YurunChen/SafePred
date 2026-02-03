"""
Benchmark Adapters for SafePred Integration.

This module provides adapter interfaces and implementations for integrating
SafePred with different benchmarks. Each benchmark needs to implement a
BenchmarkAdapter to convert its specific state/action formats to SafePred's
standard format.
"""

from .base import BenchmarkAdapter, register_adapter, get_adapter

# Import adapters (they will auto-register when imported)
# This ensures adapters are registered when the module is imported
try:
    from . import visualwebarena  # This will trigger registration
except ImportError:
    # VisualWebArena adapter may have optional dependencies
    pass

try:
    from . import stwebagentbench  # This will trigger registration
except ImportError:
    # STWebAgentBench adapter may have optional dependencies
    pass

try:
    from . import osworld  # This will trigger registration
except ImportError:
    # OSWorld adapter may have optional dependencies
    pass

__all__ = [
    "BenchmarkAdapter",
    "register_adapter",
    "get_adapter",
]

