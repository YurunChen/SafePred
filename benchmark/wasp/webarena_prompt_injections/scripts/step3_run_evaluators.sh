#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Prerequisite: run step1_run_prompt_injector.py and step2_run_agent.sh

set -e

export OUTPUT_DIR=${1:-/tmp/computer-use-agent-logs/}
export OUTPUT_FORMAT=${2:-webarena}

if [[ "${OUTPUT_DIR}" != */ ]]; then
    OUTPUT_DIR="${OUTPUT_DIR}/"
fi
LOG_DIR="${OUTPUT_DIR}agent_logs/"
TASK_DIR="${OUTPUT_DIR}webarena_tasks/"
ATTACKER_TASK_DIR="${OUTPUT_DIR}webarena_tasks_attacker/"

echo "step 3 | OUTPUT_DIR: $OUTPUT_DIR"
echo "step 3 | OUTPUT_FORMAT: $OUTPUT_FORMAT"

##### STEP 3: Run evaluations ######
# first report ASR
cd ..

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
##### -----------