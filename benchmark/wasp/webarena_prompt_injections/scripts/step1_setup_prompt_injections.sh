#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Prerequisite 1: python virtual environment and built Docker container (for Claude experiments)
# Prerequisite 2: set the required authentication environment variables (AZURE_API_ENDPOINT and AZURE_API_KEY for GPT/WebArena and AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN for Claude)

set -e

export OUTPUT_DIR=${1:-/tmp/computer-use-agent-logs/}
export MODEL=${2:-gpt-4o}
export SYSTEM_PROMPT=${3:-configs/system_prompts/wa_p_som_cot_id_actree_3s.json}
export CONFIG_PATH=${4:-configs/experiment_config.raw.json}
export USER_GOAL_IDX=${5:-0}
export INJECTION_FORMAT=${6:-goal_hijacking_plain_text}
export OUTPUT_FORMAT=${7:-webarena}

if [[ "${OUTPUT_DIR}" != */ ]]; then
    OUTPUT_DIR="${OUTPUT_DIR}/"
fi

if [ -d "$OUTPUT_DIR" ]; then
  echo "Deleting existing OUTPUT_DIR=${OUTPUT_DIR}"
  rm -rf "$OUTPUT_DIR"
fi

echo "Creating the new OUTPUT_DIR=${OUTPUT_DIR}"
mkdir "$OUTPUT_DIR"

# ----- cleanup after previous run
if [ -f "/tmp/run_step_by_step_asr.json" ]; then
  rm "/tmp/run_step_by_step_asr.json"
fi

if [ -f "/tmp/run_attacker_utility.json" ]; then
  rm "/tmp/run_attacker_utility.json"
fi

if [ -f "/tmp/run_user_utility.json" ]; then
  rm "/tmp/run_user_utility.json"
fi
# -----

echo "step 1 | OUTPUT_DIR: $OUTPUT_DIR"
echo "step 1 | CONFIG_PATH: $CONFIG_PATH"
# use env variables to define gitlab and reddit domains
echo "step 1 | GITLAB_DOMAIN: $GITLAB"
echo "step 1 | REDDIT_DOMAIN: $REDDIT"
echo "step 1 | Model: $MODEL"
echo "step 1 | SYSTEM_PROMPT: $SYSTEM_PROMPT"
echo "step 1 | USER_GOAL_IDX: $USER_GOAL_IDX"
echo "step 1 | INJECTION_FORMAT: $INJECTION_FORMAT"
echo "step 1 | OUTPUT_FORMAT: $OUTPUT_FORMAT"

cd ..
cp $CONFIG_PATH "${OUTPUT_DIR}experiment_config.json"

##### STEP 1: Inject prompts and create tasks in web environment ######
source venv/bin/activate

python prompt_injector.py --config $CONFIG_PATH \
                          --gitlab-domain $GITLAB \
                          --reddit-domain $REDDIT \
                          --model $MODEL \
                          --system_prompt $SYSTEM_PROMPT \
                          --output-dir $OUTPUT_DIR \
                          --user_goal_idx $USER_GOAL_IDX \
                          --injection_format $INJECTION_FORMAT \
                          --output-format $OUTPUT_FORMAT

deactivate
##### -----------
