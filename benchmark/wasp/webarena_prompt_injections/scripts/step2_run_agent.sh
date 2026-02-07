#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Prerequisite: run step1_run_prompt_injector.sh

set -e

export OUTPUT_DIR=${1:-/tmp/computer-use-agent-logs/}

# since the docker process runs as a user not in a known group, we need to open it up
chmod -R 777 $OUTPUT_DIR
if [[ "${OUTPUT_DIR}" != */ ]]; then
    OUTPUT_DIR="${OUTPUT_DIR}/"
fi
AGENT_RUN_SCRIPT="${OUTPUT_DIR}run_agent.sh"

##### STEP 2: Run agents on tasks ######
cd ../../visualwebarena/
source venv/bin/activate
echo "step 2 | Executing agent script at $AGENT_RUN_SCRIPT"
bash "$AGENT_RUN_SCRIPT"
echo "step 2 | Done with agent run script"
deactivate
##### -----------
