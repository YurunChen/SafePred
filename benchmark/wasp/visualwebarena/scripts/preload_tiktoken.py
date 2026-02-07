#!/usr/bin/env python3
"""
Preload tiktoken encodings to avoid network issues during runtime.
This script downloads and caches tiktoken BPE files for common OpenAI models.
"""
import sys
import os
import logging

# Try to import tiktoken, with helpful error message if not available
try:
    import tiktoken
except ImportError:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(script_dir, '..', 'venv', 'bin', 'python')
    if os.path.exists(venv_python):
        print(f"Error: tiktoken not found in current environment.")
        print(f"Please run this script using the venv Python:")
        print(f"  {venv_python} {__file__}")
        print(f"\nOr activate the venv first:")
        print(f"  source {os.path.join(script_dir, '..', 'venv', 'bin', 'activate')}")
        print(f"  python {__file__}")
    else:
        print("Error: tiktoken module not found. Please install it:")
        print("  pip install tiktoken")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Common OpenAI models that might be used
MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "o200k_base",  # Common encoding used by newer models
]

def preload_encodings(models=None, max_retries=3, retry_delay=2):
    """Preload tiktoken encodings for specified models with retry mechanism."""
    if models is None:
        models = MODELS
    
    success_count = 0
    fail_count = 0
    
    for model in models:
        # Skip invalid model names
        if model == "o200k_base":
            try:
                logger.info(f"Loading encoding: {model}")
                encoding = tiktoken.get_encoding("o200k_base")
                logger.info(f"✓ Successfully loaded encoding for {model}")
                success_count += 1
                continue
            except Exception as e:
                logger.warning(f"⚠ Could not load {model}: {e}")
                fail_count += 1
                continue
        
        # Retry mechanism for network issues
        last_error = None
        for attempt in range(max_retries):
            try:
                logger.info(f"Loading encoding for model: {model} (attempt {attempt + 1}/{max_retries})")
                encoding = tiktoken.encoding_for_model(model)
                logger.info(f"✓ Successfully loaded encoding for {model}")
                success_count += 1
                break
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"  Failed (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"  Retrying in {wait_time}s...")
                    import time
                    time.sleep(wait_time)
                else:
                    logger.error(f"✗ Failed to load encoding for {model} after {max_retries} attempts: {e}")
                    fail_count += 1
    
    logger.info(f"\nSummary: {success_count} succeeded, {fail_count} failed")
    return fail_count == 0

if __name__ == "__main__":
    # Allow custom models via command line
    custom_models = sys.argv[1:] if len(sys.argv) > 1 else None
    
    success = preload_encodings(custom_models)
    sys.exit(0 if success else 1)

