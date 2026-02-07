#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Wrapper script to run evaluate_tasks.py with correct virtual environments

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

# Check if output directory is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <output_dir> [options...]"
    echo ""
    echo "Examples:"
    echo "  $0 logs/gpt-4o-mini_safepred_agentwithpolicy/0"
    echo "  $0 logs/gpt-4o-mini_safepred_agentwithpolicy/0 --step-by-step-only"
    exit 1
fi

OUTPUT_DIR="$1"
shift  # Remove first argument, pass rest to Python script

# Activate webarena_prompt_injections venv for step-by-step evaluator
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Activated webarena_prompt_injections venv"
else
    echo "⚠️  webarena_prompt_injections venv not found, using current Python"
fi

# Note: Environment variables (REDDIT, GITLAB, etc.) should be set before running this script
# Example:
#   export DATASET=webarena_prompt_injections
#   export REDDIT="http://ec2-3-128-35-24.us-east-2.compute.amazonaws.com:9999"
#   export GITLAB="http://ec2-3-128-35-24.us-east-2.compute.amazonaws.com:8023"
#   ./evaluate_tasks.sh logs/your_output_dir/0

# Run the Python script
python evaluate_tasks.py "$OUTPUT_DIR" "$@"

# Deactivate venv
if [ -f "venv/bin/activate" ]; then
    deactivate
fi

