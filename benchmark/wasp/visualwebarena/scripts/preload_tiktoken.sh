#!/bin/bash
# Wrapper script to preload tiktoken encodings using the correct Python environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/../venv"
PYTHON_SCRIPT="$SCRIPT_DIR/preload_tiktoken.py"

# Check if venv exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Error: Virtual environment not found at $VENV_DIR"
    exit 1
fi

# Use venv Python to run the script
"$VENV_DIR/bin/python" "$PYTHON_SCRIPT" "$@"



