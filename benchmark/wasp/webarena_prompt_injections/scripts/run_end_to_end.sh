#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Prerequisite 1: python virtual environment and built Docker container (TODO: describe how to do these in README)
# Prerequisite 2: set the required authentication environment variables (AZURE_API_ENDPOINT and AZURE_API_KEY for GPT/WebArena and AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN for Claude)

set -e

export OUTPUT_DIR=${1:-/tmp/computer-use-agent-logs/}
export MODEL=${2:-gpt-4o}
export SYSTEM_PROMPT=${3:-configs/system_prompts/wa_p_som_cot_id_actree_3s.json}
export CONFIG_PATH=${4:-configs/experiment_config.raw.json}
export USER_GOAL_IDX=${5:-0}
export INJECTION_FORMAT=${6:-goal_hijacking_plain_text}
export OUTPUT_FORMAT=${7:-webarena}
# Parse SafePred and Reactive arguments
# Initialize defaults
export USE_SAFEPRED=""
export SAFEPRED_CONFIG_PATH=""
export SAFEPRED_RISK_THRESHOLD=""
export USE_REACTIVE=""
export REACTIVE_CONFIG_PATH=""
export REACTIVE_POLICY_PATH=""
export POLICY_PATH=""
export POLICY_TARGET="safepred"
export GPU_ID=""
export RESUME=""

# Check if 8th argument is --use_safepred flag
if [ -n "${8:-}" ] && [ "${8}" = "--use_safepred" ]; then
    export USE_SAFEPRED="--use_safepred"
    export SAFEPRED_CONFIG_PATH=${9:-}
    export SAFEPRED_RISK_THRESHOLD=${10:-}
    export POLICY_PATH=${11:-}
    export POLICY_TARGET=${12:-safepred}
    # Check for Reactive arguments after SafePred arguments (position 13)
    if [ -n "${13:-}" ] && [ "${13}" = "--use_reactive" ]; then
        export USE_REACTIVE="--use_reactive"
        export REACTIVE_CONFIG_PATH=${14:-}
        export REACTIVE_POLICY_PATH=${15:-}
        # Check for HarmonyGuard arguments after Reactive (position 16)
        if [ -n "${16:-}" ] && [ "${16}" = "--use_harmonyguard" ]; then
            export USE_HARMONYGUARD="--use_harmonyguard"
            export HARMONYGUARD_RISK_CAT_PATH=${17:-}
            export HARMONYGUARD_POLICY_TXT_PATH=${18:-}
            # Check for GPU argument after HarmonyGuard (position 19)
            if [ -n "${18:-}" ] && [ "${18}" = "--gpu" ]; then
                export GPU_ID=${19:-}
            fi
        elif [ -n "${16:-}" ] && [ "${16}" = "--gpu" ]; then
            # GPU argument directly after Reactive (position 16)
            export GPU_ID=${17:-}
        fi
    elif [ -n "${13:-}" ] && [ "${13}" = "--use_harmonyguard" ]; then
        # HarmonyGuard argument directly after SafePred (position 13)
        export USE_HARMONYGUARD="--use_harmonyguard"
        export HARMONYGUARD_RISK_CAT_PATH=${14:-}
        export HARMONYGUARD_POLICY_TXT_PATH=${15:-}
        # Check for GPU argument after HarmonyGuard (position 16)
        if [ -n "${15:-}" ] && [ "${15}" = "--gpu" ]; then
            export GPU_ID=${16:-}
        fi
    elif [ -n "${13:-}" ] && [ "${13}" = "--gpu" ]; then
        # GPU argument directly after SafePred (position 13)
        export GPU_ID=${14:-}
    fi
else
    # SafePred is not enabled, check if 8th argument is --use_reactive flag
    if [ -n "${8:-}" ] && [ "${8}" = "--use_reactive" ]; then
        export USE_REACTIVE="--use_reactive"
        export REACTIVE_CONFIG_PATH=${9:-}
        export REACTIVE_POLICY_PATH=${10:-}
        # Check for HarmonyGuard arguments after Reactive (position 11)
        if [ -n "${11:-}" ] && [ "${11}" = "--use_harmonyguard" ]; then
            export USE_HARMONYGUARD="--use_harmonyguard"
            export HARMONYGUARD_RISK_CAT_PATH=${12:-}
            export HARMONYGUARD_POLICY_TXT_PATH=${13:-}
            # Check for GPU argument after HarmonyGuard (position 14)
            if [ -n "${13:-}" ] && [ "${13}" = "--gpu" ]; then
                export GPU_ID=${14:-}
            fi
        elif [ -n "${11:-}" ] && [ "${11}" = "--gpu" ]; then
            # GPU argument directly after Reactive (position 11)
            export GPU_ID=${12:-}
        fi
        # Check for policy arguments (may be before or after Reactive/HarmonyGuard)
        export POLICY_PATH=${11:-}
        export POLICY_TARGET=${12:-safepred}
    elif [ -n "${8:-}" ] && [ "${8}" = "--use_harmonyguard" ]; then
        # HarmonyGuard argument without SafePred or Reactive (position 8)
        export USE_HARMONYGUARD="--use_harmonyguard"
        export HARMONYGUARD_RISK_CAT_PATH=${9:-}
        export HARMONYGUARD_POLICY_TXT_PATH=${10:-}
        # Check for GPU argument after HarmonyGuard (position 11)
        if [ -n "${10:-}" ] && [ "${10}" = "--gpu" ]; then
            export GPU_ID=${11:-}
        fi
    else
        # Neither SafePred nor Reactive, check for policy and GPU
        export POLICY_PATH=${8:-}
        export POLICY_TARGET=${9:-safepred}
        if [ -n "${10:-}" ] && [ "${10}" = "--gpu" ]; then
            export GPU_ID=${11:-}
        fi
    fi
fi

# Set CUDA_VISIBLE_DEVICES if GPU_ID is provided
if [ -n "$GPU_ID" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    echo "Setting CUDA_VISIBLE_DEVICES=$GPU_ID"
fi

if [[ "${OUTPUT_DIR}" != */ ]]; then
    OUTPUT_DIR="${OUTPUT_DIR}/"
fi

# Always delete existing directory to ensure fresh run
if [ -d "$OUTPUT_DIR" ]; then
  echo "Deleting existing OUTPUT_DIR=${OUTPUT_DIR} to ensure fresh run"
  rm -rf "$OUTPUT_DIR"
fi

echo "Creating/Using OUTPUT_DIR=${OUTPUT_DIR}"
mkdir -p "$OUTPUT_DIR"

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

echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "GITLAB_DOMAIN: $GITLAB"
echo "REDDIT_DOMAIN: $REDDIT"
echo "Model: $MODEL"
echo "SYSTEM_PROMPT: $SYSTEM_PROMPT"
echo "USER_GOAL_IDX: $USER_GOAL_IDX"
echo "INJECTION_FORMAT: $INJECTION_FORMAT"
echo "OUTPUT_FORMAT: $OUTPUT_FORMAT"

##### STEP 1: Inject prompts and create tasks in web environment ######
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "SCRIPT_DIR: $SCRIPT_DIR"
cd $SCRIPT_DIR/..
cp $CONFIG_PATH "${OUTPUT_DIR}experiment_config.json"
echo "step 1 | preparing prompt injections and tasks..."
source venv/bin/activate
SAFEPRED_ARGS=""
# Only add SafePred arguments if USE_SAFEPRED is explicitly set to "--use_safepred"
if [ -n "$USE_SAFEPRED" ] && [ "$USE_SAFEPRED" = "--use_safepred" ]; then
    SAFEPRED_ARGS="--use_safepred"
    if [ -n "$SAFEPRED_CONFIG_PATH" ]; then
        SAFEPRED_ARGS="$SAFEPRED_ARGS --safepred_config_path $SAFEPRED_CONFIG_PATH"
        # If config_path is provided, risk_threshold will be read from config file
        # Only add risk_threshold if explicitly provided (and not empty/None)
        if [ -n "$SAFEPRED_RISK_THRESHOLD" ] && [ "$SAFEPRED_RISK_THRESHOLD" != "None" ] && [ "$SAFEPRED_RISK_THRESHOLD" != "" ]; then
            SAFEPRED_ARGS="$SAFEPRED_ARGS --safepred_risk_threshold $SAFEPRED_RISK_THRESHOLD"
        fi
    else
        # No config file, add risk_threshold if provided, or use default
        if [ -n "$SAFEPRED_RISK_THRESHOLD" ] && [ "$SAFEPRED_RISK_THRESHOLD" != "None" ] && [ "$SAFEPRED_RISK_THRESHOLD" != "" ]; then
            SAFEPRED_ARGS="$SAFEPRED_ARGS --safepred_risk_threshold $SAFEPRED_RISK_THRESHOLD"
        else
            SAFEPRED_ARGS="$SAFEPRED_ARGS --safepred_risk_threshold 0.7"
        fi
    fi
fi

# Add policy arguments if provided
POLICY_ARGS=""
if [ -n "$POLICY_PATH" ]; then
    POLICY_ARGS="--policy $POLICY_PATH --policy-target $POLICY_TARGET"
fi

# Add Reactive arguments if enabled
REACTIVE_ARGS=""
if [ -n "$USE_REACTIVE" ] && [ "$USE_REACTIVE" = "--use_reactive" ]; then
    REACTIVE_ARGS="--use_reactive"
    if [ -n "$REACTIVE_CONFIG_PATH" ]; then
        REACTIVE_ARGS="$REACTIVE_ARGS --reactive_config_path $REACTIVE_CONFIG_PATH"
    fi
    if [ -n "$REACTIVE_POLICY_PATH" ]; then
        REACTIVE_ARGS="$REACTIVE_ARGS --reactive_policy_path $REACTIVE_POLICY_PATH"
    elif [ -n "$POLICY_PATH" ]; then
        # Use shared policy if reactive_policy_path not provided
        REACTIVE_ARGS="$REACTIVE_ARGS --reactive_policy_path $POLICY_PATH"
    fi
fi

# Add HarmonyGuard arguments if enabled
HARMONYGUARD_ARGS=""
if [ -n "$USE_HARMONYGUARD" ] && [ "$USE_HARMONYGUARD" = "--use_harmonyguard" ]; then
    HARMONYGUARD_ARGS="--use_harmonyguard"
    if [ -n "$HARMONYGUARD_RISK_CAT_PATH" ]; then
        HARMONYGUARD_ARGS="$HARMONYGUARD_ARGS --harmonyguard_risk_cat_path $HARMONYGUARD_RISK_CAT_PATH"
    fi
    if [ -n "$HARMONYGUARD_POLICY_TXT_PATH" ]; then
        HARMONYGUARD_ARGS="$HARMONYGUARD_ARGS --harmonyguard_policy_txt_path $HARMONYGUARD_POLICY_TXT_PATH"
    fi
fi

# Add GPU argument if provided
GPU_ARGS=""
if [ -n "$GPU_ID" ]; then
    GPU_ARGS="--gpu $GPU_ID"
fi

python prompt_injector.py --config $CONFIG_PATH \
                          --gitlab-domain $GITLAB \
                          --reddit-domain $REDDIT \
                          --model $MODEL \
                          --system_prompt $SYSTEM_PROMPT \
                          --output-dir $OUTPUT_DIR \
                          --user_goal_idx $USER_GOAL_IDX \
                          --injection_format $INJECTION_FORMAT \
                          --output-format $OUTPUT_FORMAT \
                          $SAFEPRED_ARGS \
                          $POLICY_ARGS \
                          $REACTIVE_ARGS \
                          $HARMONYGUARD_ARGS \
                          $GPU_ARGS
deactivate
# bash step1_setup_prompt_injections.sh $OUTPUT_DIR $MODEL $SYSTEM_PROMPT $CONFIG_PATH $USER_GOAL_IDX $INJECTION_FORMAT $OUTPUT_FORMAT
##### -----------


##### STEP 2: Run agents on tasks ######
echo "SCRIPT_DIR: $SCRIPT_DIR"
cd $SCRIPT_DIR/../../visualwebarena/
source venv/bin/activate
chmod -R 777 $OUTPUT_DIR
AGENT_RUN_SCRIPT="${OUTPUT_DIR}run_agent.sh"
echo "step 2 | Executing agent script at $AGENT_RUN_SCRIPT"
# Unset SOCKS proxy if set, as httpx requires socksio package for SOCKS support
# If you need SOCKS proxy, install: pip install httpx[socks]
unset ALL_PROXY all_proxy 2>/dev/null || true
bash "$AGENT_RUN_SCRIPT"
# bash step2_run_agent.sh $OUTPUT_DIR
deactivate
##### -----------


##### STEP 3: Run evaluations ######
echo "SCRIPT_DIR: $SCRIPT_DIR"
cd $SCRIPT_DIR/..
LOG_DIR="${OUTPUT_DIR}agent_logs/"
TASK_DIR="${OUTPUT_DIR}webarena_tasks/"
ATTACKER_TASK_DIR="${OUTPUT_DIR}webarena_tasks_attacker/"
echo "step 3 | OUTPUT_DIR: $OUTPUT_DIR"
echo "step 3 | OUTPUT_FORMAT: $OUTPUT_FORMAT"

# first report ASR
source venv/bin/activate
python evaluator_step_by_step.py --log-folder $LOG_DIR --task-folder $TASK_DIR --format $OUTPUT_FORMAT
echo "Done evaluating attack goals step-by-step, above score is ASR!"
deactivate

# next report user task success rate
cd ../visualwebarena/
source venv/bin/activate
bash prepare.sh
# evaluate attacker task performance (i.e. if final goal of the attacker was achieved)
python evaluator_final_step.py --log-folder $LOG_DIR --task-folder $ATTACKER_TASK_DIR --format $OUTPUT_FORMAT
echo "Done evaluating attacker goals, above score is ASR!"
# evaluate user task performance
python evaluator_final_step.py --log-folder $LOG_DIR --task-folder $TASK_DIR
echo "Done evaluating user objectives!"
deactivate
# bash step3_run_evaluators.sh $OUTPUT_DIR $OUTPUT_FORMAT
##### -----------


##### STEP 4: Cleanup environment ######
echo "SCRIPT_DIR: $SCRIPT_DIR"
cd $SCRIPT_DIR/..
PROMPT_INJECTION_CONFIG="${OUTPUT_DIR}instantiated_prompt_injections_config.json"
echo "step 4 | OUTPUT_DIR: $OUTPUT_DIR"
echo "step 4 | PROMPT_INJECTION_CONFIG: $PROMPT_INJECTION_CONFIG"
source venv/bin/activate
python environment_cleanup.py --prompt-injection-config-path "$PROMPT_INJECTION_CONFIG" --gitlab-domain $GITLAB --reddit-domain $REDDIT
deactivate
# bash step4_cleanup.sh $OUTPUT_DIR
##### -----------