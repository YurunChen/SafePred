"""
SafePred Integration for WASP - Outer SafePred (repo root).

This module imports the SafePred wrapper from the outer project root
(SafePred repo), not from a nested SafePred_v9 / SafePred_v10 directory.
All format conversion is handled by adapters in the outer SafePred.
"""

import logging
from pathlib import Path
import sys

logger = logging.getLogger("SafePredWrapper")

SAFEPRED_AVAILABLE = False
SafePredWrapper = None

_e = None
try:
    # Resolve outer SafePred repo root: agent -> visualwebarena -> wasp -> benchmark -> repo root (5 levels up from file)
    this_file = Path(__file__).resolve()
    repo_root = this_file.parent.parent.parent.parent.parent
    wrapper_py = repo_root / "wrapper.py"
    if not wrapper_py.exists():
        raise ImportError(
            "Outer SafePred not found: wrapper.py missing at repo root. "
            "Ensure benchmark/wasp is inside the SafePred repo."
        )
    # Outer SafePred uses relative imports in wrapper.py, so it must be imported as a package.
    # Add repo root's parent to path so repo root dir (e.g. SafePred) is the package name.
    parent_of_repo = str(repo_root.parent)
    if parent_of_repo not in sys.path:
        sys.path.insert(0, parent_of_repo)
    package_name = repo_root.name
    mod = __import__(f"{package_name}.wrapper", fromlist=["SafePredWrapper"])
    SafePredWrapper = getattr(mod, "SafePredWrapper")
    SAFEPRED_AVAILABLE = True
    logger.info("SafePred (outer repo) imported successfully")
except Exception as e:
    _e = e
    SafePredWrapper = None

if not SAFEPRED_AVAILABLE and _e is not None:
    logger.error("SafePred not available: %s. Ensure benchmark/wasp is inside the SafePred repo (wrapper.py at repo root) or disable --use_safepred", _e)
