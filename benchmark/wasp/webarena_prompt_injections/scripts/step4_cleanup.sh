#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
set -e

export OUTPUT_DIR=${1:-/tmp/computer-use-agent-logs/}

if [[ "${OUTPUT_DIR}" != */ ]]; then
    OUTPUT_DIR="${OUTPUT_DIR}/"
fi

# Convert to absolute path if relative
if [[ ! "$OUTPUT_DIR" = /* ]]; then
    OUTPUT_DIR="$(pwd)/${OUTPUT_DIR#./}"
fi

PROMPT_INJECTION_CONFIG="${OUTPUT_DIR}instantiated_prompt_injections_config.json"

echo "step 4 | OUTPUT_DIR: $OUTPUT_DIR"
echo "step 4 | PROMPT_INJECTION_CONFIG: $PROMPT_INJECTION_CONFIG"

##### STEP 4: Cleanup environment ######
# Get the script directory and navigate to project root
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR/.."
source venv/bin/activate
python environment_cleanup.py --prompt-injection-config-path "$PROMPT_INJECTION_CONFIG" --gitlab-domain $GITLAB --reddit-domain $REDDIT
deactivate
##### -----------