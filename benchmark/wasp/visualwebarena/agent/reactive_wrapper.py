"""
Reactive Safety Evaluator Integration for WASP - Direct Import.

This module provides a simple import wrapper for Reactive's evaluator.
All format conversion is handled by the wrapper.

Uses Reactive evaluator (gpt-oss-safeguard-20b based safety evaluator).
"""

import logging
from pathlib import Path
import sys

logger = logging.getLogger("ReactiveWrapper")

# Import Reactive only
REACTIVE_AVAILABLE = False
ReactiveWrapper = None
REACTIVE_VERSION = "1.0.0"

try:
    # Add Reactive parent directory to sys.path
    reactive_parent_paths = [
        Path("/data/chenyurun/methods"),
        Path(__file__).parent.parent.parent.parent / "methods",
    ]
    
    reactive_parent_path = None
    reactive_path = None
    
    # Find Reactive
    for path in reactive_parent_paths:
        reactive_dir = path / "reactive"
        if reactive_dir.exists():
            reactive_parent_path = path
            reactive_path = reactive_dir
            break
    
    if not reactive_path:
        raise ImportError("Reactive not found. Please ensure Reactive is installed in the methods directory.")
    
    if reactive_parent_path and str(reactive_parent_path) not in sys.path:
        sys.path.insert(0, str(reactive_parent_path))
    
    # Import Reactive
    from reactive import ReactiveWrapper
    REACTIVE_AVAILABLE = True
    logger.info("Reactive imported successfully")
        
except ImportError as e:
    logger.error(f"Reactive not available: {e}. Install Reactive or disable --use_reactive")
    REACTIVE_AVAILABLE = False
    ReactiveWrapper = None
