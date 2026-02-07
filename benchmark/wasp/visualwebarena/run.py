# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Script to run end-to-end evaluation on the benchmark.

Modified from https://github.com/web-arena-x/webarena/blob/main/run.py.
"""

import argparse
import glob
import json
import logging
import os
import random
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List

import openai
import requests
import torch
from PIL import Image

from agent import (
    PromptAgent,
    construct_agent,
)
from agent.agent import ConversationRenderer, ExecutionTrajectoryLogger
from agent.prompts import *
from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.actions import is_equivalent
from browser_env.auto_login import get_site_comb_from_filepath
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)
from evaluation_harness import evaluator_router, image_utils

DATASET = os.environ["DATASET"]

LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random.randint(0, 10000)}.log"

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Set the log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    parser.add_argument("--render", action="store_true", help="Render the browser")

    parser.add_argument(
        "--slow_mo",
        type=int,
        default=0,
        help="Slow down the browser by the specified amount",
    )
    parser.add_argument(
        "--action_set_tag", default="id_accessibility_tree", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=[
            "accessibility_tree",
            "accessibility_tree_with_captioner",
            "html",
            "image",
            "image_som",
            "image_som_with_a11y_tree",
        ],
        default="accessibility_tree",
        help="Observation type",
    )
    parser.add_argument(
        "--current_viewport_only",
        action="store_true",
        help="Only use the current viewport for the observation",
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=2048)
    parser.add_argument("--save_trace_enabled", action="store_true")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)

    parser.add_argument("--max_steps", type=int, default=30)

    # agent config
    parser.add_argument("--agent_type", type=str, default="prompt")
    parser.add_argument(
        "--instruction_path",
        type=str,
        default="agents/prompts/state_action_agent.json",
    )
    parser.add_argument(
        "--parsing_failure_th",
        help="When consecutive parsing failures exceed this threshold, the agent will terminate early.",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeating_action_failure_th",
        help="When consecutive repeated actions exceed this threshold, the agent will terminate early.",
        type=int,
        default=5,
    )

    parser.add_argument("--test_config_base_dir", type=str)

    parser.add_argument(
        "--eval_captioning_model_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run eval captioning model on. By default, runs it on CPU.",
    )
    parser.add_argument(
        "--eval_captioning_model",
        type=str,
        default="Salesforce/blip2-flan-t5-xl",
        choices=["Salesforce/blip2-flan-t5-xl"],
        help="Captioning backbone for VQA-type evals.",
    )
    parser.add_argument(
        "--captioning_model",
        type=str,
        default="Salesforce/blip2-flan-t5-xl",
        choices=["Salesforce/blip2-flan-t5-xl", "llava-hf/llava-1.5-7b-hf"],
        help="Captioning backbone for accessibility tree alt text.",
    )

    # lm config
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0613")
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument(
        "--max_retry",
        type=int,
        help="max retry times to perform generations when parsing fails",
        default=1,
    )
    parser.add_argument(
        "--max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=3840,
    )

    # example config
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx", type=int, default=910)

    # logging related
    parser.add_argument("--result_dir", type=str, default="")
    
    # SafePred configuration
    parser.add_argument(
        "--use_safepred",
        action="store_true",
        help="Enable SafePred risk prediction for actions"
    )
    parser.add_argument(
        "--safepred_config_path",
        type=str,
        default=None,
        help="Path to SafePred config YAML file (default: None, uses default config)"
    )
    parser.add_argument(
        "--safepred_risk_threshold",
        type=float,
        default=None,
        help="Risk threshold for SafePred (optional, will be read from config file if not provided)"
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Path to policy JSON file (optional)"
    )
    parser.add_argument(
        "--policy-target",
        type=str,
        choices=['web', 'safepred', 'both'],
        default='safepred',
        help="Target agent for policies: 'web' (Web Agent Model), 'safepred' (World Model), 'both' (both agents) (default: 'safepred')"
    )
    parser.add_argument(
        "--use_planning",
        action="store_true",
        help="Enable plan-based execution (requires --use_safepred). Agent generates plan at task start, and updates it when SafePred detects risk or misalignment."
    )
    
    # Reactive configuration (incremental addition, parallel to SafePred)
    parser.add_argument(
        "--use_reactive",
        action="store_true",
        help="Enable Reactive safety evaluator (gpt-oss-safeguard-20b) for action evaluation"
    )
    parser.add_argument(
        "--reactive_config_path",
        type=str,
        default=None,
        help="Path to Reactive config YAML file (default: None, uses default config)"
    )
    parser.add_argument(
        "--reactive_api_url",
        type=str,
        default=None,
        help="Override API URL for Reactive evaluator (default: from config file)"
    )
    parser.add_argument(
        "--reactive_api_key",
        type=str,
        default=None,
        help="Override API key for Reactive evaluator (default: from config file)"
    )
    parser.add_argument(
        "--reactive_model_name",
        type=str,
        default=None,
        help="Override model name for Reactive evaluator (default: from config file)"
    )
    parser.add_argument(
        "--reactive_risk_threshold",
        type=float,
        default=None,
        help="Override risk threshold for Reactive (default: from config file)"
    )
    parser.add_argument(
        "--reactive_policy_path",
        type=str,
        default=None,
        help="Path to policy JSON file for Reactive (default: from config file or use --policy)"
    )
    
    # HarmonyGuard configuration (incremental addition, parallel to SafePred and Reactive)
    parser.add_argument(
        "--use_harmonyguard",
        action="store_true",
        help="Enable HarmonyGuard safety evaluator for action evaluation"
    )
    parser.add_argument(
        "--harmonyguard_risk_cat_path",
        type=str,
        default=None,
        help="Path to HarmonyGuard risk category file (default: None, uses config.yaml)"
    )
    parser.add_argument(
        "--harmonyguard_config_path",
        type=str,
        default=None,
        help="Path to HarmonyGuard config.yaml file (optional, will search for it if not provided)"
    )
    parser.add_argument(
        "--harmonyguard_policy_txt_path",
        type=str,
        default=None,
        help="Path to plain text policy file (used when use_structured_policy=False in config.yaml)"
    )
    
    args = parser.parse_args()

    # check the whether the action space is compatible with the observation space
    if (
        args.action_set_tag == "id_accessibility_tree"
        and args.observation_type
        not in [
            "accessibility_tree",
            "accessibility_tree_with_captioner",
            "image_som",
            "image_som_with_a11y_tree",
        ]
    ):
        raise ValueError(
            f"Action type {args.action_set_tag} is incompatible with the observation type {args.observation_type}"
        )

    return args


def early_stop(
    trajectory: Trajectory, max_steps: int, thresholds: dict[str, int]
) -> tuple[bool, str]:
    """Check whether need to stop early"""

    # reach the max step
    num_steps = (len(trajectory) - 1) / 2
    if num_steps >= max_steps:
        return True, f"Reach max steps {max_steps}"

    last_k_actions: list[Action]
    action_seq: list[Action]

    # Case: parsing failure for k times
    k = thresholds["parsing_failure"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    if len(last_k_actions) >= k:
        if all(
            [action["action_type"] == ActionTypes.NONE for action in last_k_actions]
        ):
            return True, f"Failed to parse actions for {k} times"

    # Case: same action for k times
    k = thresholds["repeating_action"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    action_seq = trajectory[1::2]  # type: ignore[assignment]

    if len(action_seq) == 0:
        return False, ""

    last_action: Action = action_seq[-1]

    if last_action["action_type"] != ActionTypes.TYPE:
        if len(last_k_actions) >= k:
            if all([is_equivalent(action, last_action) for action in last_k_actions]):
                return True, f"Same action for {k} times"

    else:
        # check the action sequence
        if sum([is_equivalent(action, last_action) for action in action_seq]) >= k:
            return True, f"Same typing action for {k} times"

    return False, ""


def test(args: argparse.Namespace, config_file_list: list[str]) -> None:
    global should_exit
    print(f"Going through these config files: {config_file_list}")
    scores = []
    max_steps = args.max_steps

    early_stop_thresholds = {
        "parsing_failure": args.parsing_failure_th,
        "repeating_action": args.repeating_action_failure_th,
    }

    if args.observation_type in [
        "accessibility_tree_with_captioner",
        "image_som",
        "image_som_with_a11y_tree",
    ]:
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        caption_image_fn = image_utils.get_captioning_fn(
            device, dtype, args.captioning_model
        )
    else:
        caption_image_fn = None

    # Load a (possibly different) captioning model for running VQA evals.
    if DATASET == "visualwebarena":
        if caption_image_fn and args.eval_captioning_model == args.captioning_model:
            eval_caption_image_fn = caption_image_fn
        else:
            eval_caption_image_fn = image_utils.get_captioning_fn(
                args.eval_captioning_model_device,
                (
                    torch.float16
                    if (
                        torch.cuda.is_available()
                        and args.eval_captioning_model_device == "cuda"
                    )
                    else torch.float32
                ),
                args.eval_captioning_model,
            )
    else:
        caption_image_fn = None
        eval_caption_image_fn = None

    # Log SafePred configuration
    use_safepred = getattr(args, "use_safepred", False)
    safepred_config_path = getattr(args, "safepred_config_path", None)
    safepred_risk_threshold = getattr(args, "safepred_risk_threshold", None)
    policy = getattr(args, "policy", None)
    policy_target = getattr(args, "policy_target", "safepred")
    logger.info(f"[Run] SafePred configuration: use_safepred={use_safepred}, config_path={safepred_config_path}, risk_threshold={safepred_risk_threshold}, policy={policy}, policy_target={policy_target}")
    
    # Log Reactive configuration (incremental addition)
    use_reactive = getattr(args, "use_reactive", False)
    reactive_config_path = getattr(args, "reactive_config_path", None)
    logger.info(f"[Run] Reactive configuration: use_reactive={use_reactive}, config_path={reactive_config_path}")
    
    use_harmonyguard = getattr(args, "use_harmonyguard", False)
    harmonyguard_risk_cat_path = getattr(args, "harmonyguard_risk_cat_path", None)
    harmonyguard_config_path = getattr(args, "harmonyguard_config_path", None)
    harmonyguard_policy_txt_path = getattr(args, "harmonyguard_policy_txt_path", None)
    logger.info(f"[Run] HarmonyGuard configuration: use_harmonyguard={use_harmonyguard}, risk_cat_path={harmonyguard_risk_cat_path}, config_path={harmonyguard_config_path}, policy_txt_path={harmonyguard_policy_txt_path}")
    
    agent = construct_agent(
        args,
        captioning_fn=(
            caption_image_fn
            if args.observation_type == "accessibility_tree_with_captioner"
            else None
        ),
    )  # NOTE: captioning_fn here is used for captioning input images.
    
    # Log agent SafePred status
    if hasattr(agent, "safepred_wrapper"):
        if agent.safepred_wrapper is not None:
            logger.info(f"[Run] ✅ Agent has SafePred wrapper initialized")
        else:
            logger.warning(f"[Run] ⚠️  Agent's safepred_wrapper is None")
    else:
        logger.warning(f"[Run] ⚠️  Agent does not have safepred_wrapper attribute")

    env = ScriptBrowserEnv(
        headless=not args.render,
        slow_mo=args.slow_mo,
        observation_type=args.observation_type,
        current_viewport_only=args.current_viewport_only,
        viewport_size={
            "width": args.viewport_width,
            "height": args.viewport_height,
        },
        save_trace_enabled=args.save_trace_enabled,
        sleep_after_execution=args.sleep_after_execution,
        # NOTE: captioning_fn here is used for LLM + captioning baselines.
        # This can be different from the captioning model used for evals.
        captioning_fn=caption_image_fn,
    )

    for config_file in config_file_list:
        # Check if we should exit gracefully
        if should_exit:
            logger.info("Exiting due to signal...")
            print("[Exit] Exiting gracefully due to signal...")
            break
        try:
            render_helper = RenderHelper(
                config_file, args.result_dir, args.action_set_tag
            )
            conversation_renderer = ConversationRenderer(config_file, args.result_dir)

            # Load task.
            with open(config_file) as f:
                _c = json.load(f)
                intent = _c["intent"]
                task_id = _c["task_id"]
                image_paths = _c.get("image", None)
                prompt_injection = _c.get("prompt_injection", "")
                images = []

                # automatically login
                if _c["storage_state"]:
                    cookie_file_name = os.path.basename(_c["storage_state"])
                    comb = get_site_comb_from_filepath(cookie_file_name)
                    temp_dir = tempfile.mkdtemp()
                    # subprocess to renew the cookie
                    subprocess.run(
                        [
                            "python",
                            "browser_env/auto_login.py",
                            "--auth_folder",
                            temp_dir,
                            "--site_list",
                            *comb,
                        ]
                    )
                    _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
                    assert os.path.exists(_c["storage_state"])
                    # update the config file
                    config_file = f"{temp_dir}/{os.path.basename(config_file)}"
                    with open(config_file, "w") as f:
                        json.dump(_c, f)

                # Load input images for the task, if any.
                if image_paths is not None:
                    if isinstance(image_paths, str):
                        image_paths = [image_paths]
                    for image_path in image_paths:
                        # Load image either from the web or from a local path.
                        if image_path.startswith("http"):
                            headers = {
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                            }
                            input_image = Image.open(
                                requests.get(
                                    image_path, stream=True, headers=headers
                                ).raw
                            )
                        else:
                            input_image = Image.open(image_path)

                        images.append(input_image)

            logger.info(f"[Config file]: {config_file}")
            logger.info(f"[Intent]: {intent}")

            agent.reset(config_file)
            trajectory: Trajectory = []
            obs, info = env.reset(options={"config_file": config_file})
            state_info: StateInfo = {"observation": obs, "info": info}
            trajectory.append(state_info)

            # Add task_id to meta_data for SafePred to track task switches
            # Initialize action_history like original WASP version
            meta_data = {"action_history": ["None"], "task_id": str(task_id)}
            
            # Initialize execution trajectory logger
            trajectory_logger = ExecutionTrajectoryLogger(args.result_dir, str(task_id))
            trajectory_logger.set_task_metadata(
                task_instruction=intent,
                task_params=_c,
                injection={"prompt_injection": prompt_injection} if prompt_injection else {},
                jailbreak=bool(prompt_injection),
            )
            trajectory_logger.start_task()
            
            step_idx = 0
            while True:
                # Check if we should exit gracefully
                if should_exit:
                    logger.info("Exiting task loop due to signal...")
                    print("[Exit] Exiting task loop due to signal...")
                    break
                    
                early_stop_flag, stop_info = early_stop(
                    trajectory, max_steps, early_stop_thresholds
                )

                if early_stop_flag:
                    action = create_stop_action(f"Early stop: {stop_info}")
                    # Record early stop action
                    if trajectory_logger:
                        trajectory_logger.start_step(step_idx)
                        trajectory_logger.add_action(action)
                        trajectory_logger.set_response(f"Early stop: {stop_info}")
                else:
                    try:
                        action = agent.next_action(
                            trajectory,
                            intent,
                            images=images,
                            meta_data=meta_data,
                            conversation_renderer=conversation_renderer,
                            prompt_injection=prompt_injection,
                            trajectory_logger=trajectory_logger,
                            step_idx=step_idx,
                        )
                    except ValueError as e:
                        # get the error message
                        action = create_stop_action(f"ERROR: {str(e)}")
                        if trajectory_logger:
                            if not trajectory_logger.current_step:
                                trajectory_logger.start_step(step_idx)
                            trajectory_logger.add_error("ValueError", str(e))
                            trajectory_logger.add_action(action)
                            trajectory_logger.set_response(f"ERROR: {str(e)}")
                            trajectory_logger.end_step()

                trajectory.append(action)

                action_str = get_action_description(
                    action,
                    state_info["info"]["observation_metadata"],
                    action_set_tag=args.action_set_tag,
                    prompt_constructor=(
                        agent.prompt_constructor
                        if isinstance(agent, PromptAgent)
                        else None
                    ),
                )
                render_helper.render(
                    action, state_info, meta_data, args.render_screenshot
                )
                # Append action_str to action_history like original WASP version
                meta_data["action_history"].append(action_str)

                # Record action execution start time
                action_start_time = time.time()
                
                if action["action_type"] == ActionTypes.STOP:
                    if trajectory_logger:
                        if not trajectory_logger.current_step:
                            trajectory_logger.start_step(step_idx)
                        trajectory_logger.add_action_result(
                            action=action_str,
                            reward=0.0,
                            done=True,
                            info={"action_type": "STOP", "reason": action.get("answer", "")},
                            duration=0.0,
                        )
                        trajectory_logger.end_step()
                    break

                obs, reward, terminated, truncated, info = env.step(action)
                state_info = {"observation": obs, "info": info}
                trajectory.append(state_info)
                
                # Record action result and next state a11y tree
                if trajectory_logger:
                    action_duration = time.time() - action_start_time
                    # Use the action_str already computed above
                    trajectory_logger.add_action_result(
                        action=action_str,
                        reward=reward,
                        done=terminated or truncated,
                        info=info,
                        duration=action_duration,
                    )
                    # Record next state a11y tree
                    next_a11y_tree = obs.get("text", "") if isinstance(obs, dict) else ""
                    trajectory_logger.set_next_state_a11y_tree(next_a11y_tree)
                    # End step after recording action result and next state
                    trajectory_logger.end_step()
                
                # Update SafePred trajectory if enabled
                if (
                    hasattr(agent, "safepred_wrapper")
                    and agent.safepred_wrapper is not None
                ):
                    try:
                        # Update SafePred trajectory
                        # Pass raw trajectory and let adapter handle format conversion
                        # Include prompt_injection in metadata so World Model can track attack context
                        metadata_with_injection = meta_data.copy()
                        if prompt_injection:
                            metadata_with_injection["prompt_injection"] = prompt_injection
                        
                        prev_trajectory = trajectory[:-1] if len(trajectory) > 1 else []
                        agent.safepred_wrapper.update_trajectory(
                            prev_state=prev_trajectory,
                            action=action,
                            next_state=trajectory,
                            intent=intent,
                            metadata=metadata_with_injection,
                        )
                    except Exception as e:
                        logger.warning(f"[SafePred] Error updating trajectory: {e}")

                if terminated:
                    # add a action place holder
                    stop_action = create_stop_action("")
                    trajectory.append(stop_action)
                    # Record termination
                    if trajectory_logger:
                        if trajectory_logger.current_step:
                            trajectory_logger.add_action_result(
                                action="Terminated",
                                reward=reward if 'reward' in locals() else 0.0,
                                done=True,
                                info={"terminated": True},
                                duration=0.0,
                            )
                            trajectory_logger.end_step()
                    break
                
                step_idx += 1

            # End trajectory logging (ensure it's called even on early exit)
            if trajectory_logger:
                # End current step if still active
                if trajectory_logger.current_step:
                    trajectory_logger.end_step()
                trajectory_logger.end_task()

            # NOTE: eval_caption_image_fn is used for running eval_vqa functions.
            evaluator = evaluator_router(
                config_file, captioning_fn=eval_caption_image_fn
            )
            score = evaluator(
                trajectory=trajectory, config_file=config_file, page=env.page
            )

            scores.append(score)

            if score == 1:
                logger.info(f"[Result] (PASS) {config_file}")
            else:
                logger.info(f"[Result] (FAIL) {config_file}")

            if args.save_trace_enabled:
                env.save_trace(Path(args.result_dir) / "traces" / f"{task_id}.zip")
        except KeyboardInterrupt:
            logger.info("Task interrupted by user")
            print("[Interrupted] Task interrupted, exiting...")
            # End trajectory logging on interruption
            if 'trajectory_logger' in locals() and trajectory_logger:
                if trajectory_logger.current_step:
                    trajectory_logger.end_step()
                trajectory_logger.end_task()
            should_exit = True
            break
        except openai.OpenAIError as e:
            logger.info(f"[OpenAI Error] {repr(e)}")
        # except Exception as e:
        #     logger.info(f"[Unhandled Error] {repr(e)}]")
        #     import traceback

        #     # write to error file
        #     with open(Path(args.result_dir) / "error.txt", "a") as f:
        #         f.write(f"[Config file]: {config_file}\n")
        #         f.write(f"[Unhandled Error] {repr(e)}\n")
        #         f.write(traceback.format_exc())  # write stack trace to file

        render_helper.close()
        conversation_renderer.close()

    # Close SafePred wrapper and flush training data before closing environment
    if hasattr(agent, "safepred_wrapper") and agent.safepred_wrapper is not None:
        try:
            agent.safepred_wrapper.close()
            logger.info("[Run] SafePred wrapper closed and training data flushed")
        except Exception as e:
            logger.warning(f"[Run] Error closing SafePred wrapper: {e}")

    env.close()
    if len(scores):
        logger.info(f"Average score: {sum(scores) / len(scores)}")


def prepare(args: argparse.Namespace) -> None:
    # convert prompt python files to json
    from agent.prompts import to_json

    to_json.run()

    # prepare result dir
    result_dir = args.result_dir
    if not result_dir:
        result_dir = f"cache/results_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        args.result_dir = result_dir
        logger.info(f"Create result dir: {result_dir}")

    if not (Path(result_dir) / "traces").exists():
        (Path(result_dir) / "traces").mkdir(parents=True)

    # log the log file
    with open(os.path.join(result_dir, "log_files.txt"), "a+") as f:
        f.write(f"{LOG_FILE_NAME}\n")


def dump_config(args: argparse.Namespace) -> None:
    config_file = Path(args.result_dir) / "config.json"
    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=4)
            logger.info(f"Dump config to {config_file}")


# Global flag to track if we should exit gracefully
should_exit = False

def signal_handler(signum, frame):
    """Handle termination signals gracefully."""
    global should_exit
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    print(f"\n[Signal Handler] Received signal {signum}, shutting down gracefully...")
    should_exit = True
    # Force exit after a short delay if cleanup takes too long
    import threading
    def force_exit():
        time.sleep(5)
        if should_exit:
            logger.error("Force exiting after timeout")
            os._exit(1)
    threading.Thread(target=force_exit, daemon=True).start()

if __name__ == "__main__":
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    # On Unix systems, also handle SIGHUP (terminal closed)
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, signal_handler)
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = config()
    args.sleep_after_execution = 2.5
    prepare(args)

    test_config_base_dir = args.test_config_base_dir

    test_file_list = []
    st_idx = args.test_start_idx
    ed_idx = args.test_end_idx
    for i in range(st_idx, ed_idx):
        test_file_list.append(os.path.join(test_config_base_dir, f"{i}.json"))
    print(f"Total {len(test_file_list)} tasks to execute")
    args.render = False
    args.render_screenshot = True
    args.save_trace_enabled = True

    args.current_viewport_only = True
    dump_config(args)

    try:
        test(args, test_file_list)
    except KeyboardInterrupt:
        logger.info("Interrupted by user (KeyboardInterrupt)")
        print("\n[Interrupted] Shutting down...")
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise
