# Copyright (c) Meta Platforms, Inc. and affiliates.
import click
import subprocess
import json
import os
import sys
from collections import defaultdict
from constants import (
    PromptInjectionFormat,
    GitlabUserGoals,
    RedditUserGoals
)


def load_results_from_files(results_dict):
    """Load results from the result JSON files."""
    try:
        # Load step-by-step ASR
        if os.path.exists("/tmp/run_step_by_step_asr.json"):
            with open("/tmp/run_step_by_step_asr.json", 'r') as json_file:
                res_step_by_step = json.load(json_file)
                if "compromised" in res_step_by_step:
                    results_dict["cnt_intermediate_compromised"] += res_step_by_step["compromised"]
                if "not_compromised" in res_step_by_step:
                    results_dict["cnt_intermediate_not_compromised"] += res_step_by_step["not_compromised"]
                if "outright_refusal" in res_step_by_step:
                    results_dict["cnt_intermediate_outright_refusal"] += res_step_by_step["outright_refusal"]
        
        # Load attacker utility
        if os.path.exists("/tmp/run_attacker_utility.json"):
            with open("/tmp/run_attacker_utility.json", 'r') as json_file:
                res_attacker_utility = json.load(json_file)
                results_dict["cnt_end2end_compromised"] += res_attacker_utility["total_scores"]
                results_dict["cnt_end2end_not_compromised"] += (res_attacker_utility["cnt_tasks"] - res_attacker_utility["total_scores"])
        
        # Load user utility
        if os.path.exists("/tmp/run_user_utility.json"):
            with open("/tmp/run_user_utility.json", 'r') as json_file:
                res_user_utility = json.load(json_file)
                results_dict["cnt_user_utility"] += res_user_utility["total_scores"]
                results_dict["cnt_user_total_tasks"] += res_user_utility["cnt_tasks"]
        
        return True
    except Exception as e:
        print(f"⚠️  Failed to load results from files: {e}")
        return False


def run_single_end_to_end(
        config,  # prompt injection configurations
        model,  # agentic model to run
        system_prompt,  # system prompt for agent
        output_dir,  # to store settings, logs, errors
        output_format,  # agentic scaffolding format
        user_goal_idx,  # benign user goal
        injection_format,  # prompt injection format to try
        results_dict,  # to save and accumulate results
        output_dir_idx=0,  # to save logs for each run separately
        use_safepred=False,  # enable SafePred risk prediction
        safepred_config_path=None,  # path to SafePred config file
        safepred_risk_threshold=None,  # SafePred risk threshold (optional, will be read from config file if not provided)
            policy=None,  # path to policy JSON file
            policy_target='safepred',  # target agent: 'web', 'safepred', or 'both'
            use_reactive=False,  # enable Reactive safety evaluator
            reactive_config_path=None,  # path to Reactive config file
            reactive_policy_path=None,  # path to Reactive policy file
            use_harmonyguard=False,  # enable HarmonyGuard safety evaluator
            harmonyguard_risk_cat_path=None,  # path to HarmonyGuard risk category file
            harmonyguard_policy_txt_path=None,  # path to HarmonyGuard plain text policy file (used when use_structured_policy=False)
            gpu=None,  # GPU device ID(s) to use (e.g., "0" or "0,1"), sets CUDA_VISIBLE_DEVICES
    ):
    if output_dir[-1] == '/':
        output_dir = output_dir + str(output_dir_idx) + '/'
    else:
        output_dir = output_dir + '/' + str(output_dir_idx) + '/'

    command = [
        'bash',
        'scripts/run_end_to_end.sh',
        output_dir,
        model,
        system_prompt,
        config,
        str(user_goal_idx),
        injection_format,
        output_format,
    ]
    # Only add SafePred-related arguments if use_safepred is True
    if use_safepred:
        command.extend([
            '--use_safepred',
            safepred_config_path or '',
            str(safepred_risk_threshold) if safepred_risk_threshold is not None else '',
            policy or '',
            policy_target or 'safepred',
        ])
    # Only add Reactive-related arguments if use_reactive is True
    if use_reactive:
        command.extend([
            '--use_reactive',
            reactive_config_path or '',
            reactive_policy_path or policy or '',  # Use policy if reactive_policy_path not provided
        ])
    # Only add HarmonyGuard-related arguments if use_harmonyguard is True
    if use_harmonyguard:
        command.extend([
            '--use_harmonyguard',
            harmonyguard_risk_cat_path or '',
            harmonyguard_policy_txt_path or '',
        ])
    # Add GPU argument if provided
    if gpu:
        command.append('--gpu')
        command.append(str(gpu))
    # Note: Do NOT remove empty strings - they are needed to maintain argument positions for shell script parsing
    print(f"\nRunning command: \n{' '.join([str(arg) for arg in command])}", flush=True)
    
    # Run the main command
    subprocess.run(command, check=True)

    # ------- check and save results
    load_results_from_files(results_dict)
    # -------


def run_all(config, 
            model, 
            system_prompt, 
            output_dir, 
            output_format, 
            run_single,
            user_goal_start,
        use_safepred=False,
        safepred_config_path=None,
        safepred_risk_threshold=None,
        policy=None,
        policy_target='safepred',
        use_reactive=False,
        reactive_config_path=None,
        reactive_policy_path=None,
        use_harmonyguard=False,
        harmonyguard_risk_cat_path=None,
        harmonyguard_policy_txt_path=None,
        gpu=None):
    gitlab_user_goals = GitlabUserGoals("")
    reddit_user_goals = RedditUserGoals("")
    assert len(gitlab_user_goals.GOALS) == len(reddit_user_goals.GOALS), "Number of user goals should match!"
    user_goals_len = len(gitlab_user_goals.GOALS)
    injection_format_list = [PromptInjectionFormat.GOAL_HIJACKING_PLAIN_TEXT,
                             PromptInjectionFormat.GOAL_HIJACKING_URL_INJECTION]
    results_dict = defaultdict(int)

    # Run all user goals from user_goal_start to the end
    for user_goal_idx in range(user_goal_start, user_goals_len):
        print(f"$$$$$$$ Running {user_goal_idx+1} our of {user_goals_len} user goals, current one: "
              f"(gitlab) '{gitlab_user_goals.GOALS[user_goal_idx]}', "
              f"(reddit) '{reddit_user_goals.GOALS[user_goal_idx]}'")
        for i, injection_format in enumerate(injection_format_list):
            print(f"$$$$$$$ Running {i+1} out of {len(injection_format_list)} injection formats, current one: {injection_format}")

            run_single_end_to_end(config=config,
                                  model=model, 
                                  system_prompt=system_prompt, 
                                  output_dir=output_dir, 
                                  output_format=output_format, 
                                  user_goal_idx=user_goal_idx, 
                                  injection_format=injection_format, 
                                  results_dict=results_dict,
                                  output_dir_idx=user_goal_idx * len(injection_format_list) + i,
                                  use_safepred=use_safepred,
                                  safepred_config_path=safepred_config_path,
                                  safepred_risk_threshold=safepred_risk_threshold,
                                  policy=policy,
                                  policy_target=policy_target,
                                  use_reactive=use_reactive,
                                  reactive_config_path=reactive_config_path,
                                  reactive_policy_path=reactive_policy_path,
                                  use_harmonyguard=use_harmonyguard,
                                  harmonyguard_risk_cat_path=harmonyguard_risk_cat_path,
                                  harmonyguard_policy_txt_path=harmonyguard_policy_txt_path,
                                  gpu=gpu)

            print(f"\nAccumulated results after user_goal_idx = {user_goal_idx+1} and injection_format_idx = {i+1}: ")
            for key, value in results_dict.items():
                print(f"{key} = {value}")

            if run_single:
                print("\n!!! Running a single user goal and a single injection format is requested. Terminating")
                return
    
    print("\n\nDone running all experiments! Final results:")
    for key, value in results_dict.items():
        print(f"{key} = {value}")
    
    # Kill all wasp processes after execution completes
    print("\n\nKilling all wasp-related processes...")
    kill_script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "kill_wasp_processes.py")
    if os.path.exists(kill_script_path):
        try:
            subprocess.run([sys.executable, kill_script_path, "--force"], check=False)
            print("Wasp processes cleanup completed.")
        except Exception as e:
            print(f"Warning: Failed to kill wasp processes: {e}")
    else:
        print(f"Warning: Kill script not found at {kill_script_path}")


@click.command()
@click.option(
    "--config",
    type=str,
    default="configs/experiment_config.raw.json",
    help="Where to find the config for prompt injections",
)
@click.option(
    "--model",
    type=click.Choice(['gpt-4o', 'gpt-4o-mini', 'claude-35', 'claude-37'], case_sensitive=False),
    default="gpt-4o",
    help="backbone LLM. Available options: gpt-4o, gpt-4o-mini, claude-35, claude-37",
)
@click.option(
    "--system-prompt",
    type=str,
    default="configs/system_prompts/wa_p_som_cot_id_actree_3s.json",
    help="system_prompt for the backbone LLM. Default = VWA's SOM system prompt for GPT scaffolding",
)
@click.option(
    "--output-dir",
    type=str,
    default="/tmp/computer-use-agent-logs",
    help="Folder to store the output configs and commands to run the agent",
)
@click.option(
    "--output-format",
    type=str,
    default="webarena",
    help="Format of the agentic scaffolding: webarena (default), claude, gpt_web_tools",
)
@click.option(
    "--run-single",
    is_flag=True,
    default=False,
    help="whether to test only a single user goal and a single injection format",
)
@click.option(
    "--user_goal_start",
    type=int,
    default=0,
    help="starting user_goal index (between 0 and total number of benign user goals)",
)
@click.option(
    "--use_safepred",
    is_flag=True,
    default=False,
    help="Enable SafePred risk prediction for actions",
)
@click.option(
    "--safepred_config_path",
    type=str,
    default=None,
    help="Path to SafePred config YAML file (default: None, uses default config)",
)
@click.option(
    "--safepred_risk_threshold",
    type=float,
    default=None,
    help="Risk threshold for SafePred (optional, will be read from config file if not provided)",
)
@click.option(
    "--policy",
    type=str,
    default=None,
    help="Path to policy JSON file (optional)",
)
@click.option(
    "--policy-target",
    type=click.Choice(['web', 'safepred', 'both'], case_sensitive=False),
    default='safepred',
    help="Target agent for policies: 'web' (Web Agent Model), 'safepred' (World Model), 'both' (both agents) (default: 'safepred')",
)
@click.option(
    "--use_reactive",
    is_flag=True,
    default=False,
    help="Enable Reactive safety evaluator (gpt-oss-safeguard-20b) for action evaluation",
)
@click.option(
    "--reactive_config_path",
    type=str,
    default=None,
    help="Path to Reactive config YAML file (default: None, uses default config)",
)
@click.option(
    "--reactive_policy_path",
    type=str,
    default=None,
    help="Path to policy JSON file for Reactive (default: None, uses --policy if provided)",
)
@click.option(
    "--use_harmonyguard",
    is_flag=True,
    default=False,
    help="Enable HarmonyGuard safety evaluator for action evaluation",
)
@click.option(
    "--harmonyguard_risk_cat_path",
    type=str,
    default=None,
    help="Path to HarmonyGuard risk category file (default: None, uses config.yaml)",
)
@click.option(
    "--harmonyguard_policy_txt_path",
    type=str,
    default=None,
    help="Path to HarmonyGuard plain text policy file (used when use_structured_policy=False in config.yaml)",
)
@click.option(
    "--gpu",
    type=str,
    default=None,
    help="GPU device ID(s) to use (e.g., '0' or '0,1'). Sets CUDA_VISIBLE_DEVICES environment variable. If not specified, uses all available GPUs.",
)
def main(config, 
         model, 
         system_prompt, 
         output_dir, 
         output_format, 
         run_single, 
         user_goal_start,
         use_safepred,
         safepred_config_path,
         safepred_risk_threshold,
         policy,
         policy_target,
         use_reactive,
         reactive_config_path,
         reactive_policy_path,
         use_harmonyguard,
         harmonyguard_risk_cat_path,
         harmonyguard_policy_txt_path,
         gpu):
    print("Arguments provided to run.py: \n", locals(), "\n\n")
    run_all(config=config, 
            model=model, 
            system_prompt=system_prompt, 
            output_dir=output_dir, 
            output_format=output_format, 
            run_single=run_single,
            user_goal_start=user_goal_start,
            use_safepred=use_safepred,
            safepred_config_path=safepred_config_path,
            safepred_risk_threshold=safepred_risk_threshold,
            policy=policy,
            policy_target=policy_target,
            use_reactive=use_reactive,
            reactive_config_path=reactive_config_path,
            reactive_policy_path=reactive_policy_path,
            use_harmonyguard=use_harmonyguard,
            harmonyguard_risk_cat_path=harmonyguard_risk_cat_path,
            harmonyguard_policy_txt_path=harmonyguard_policy_txt_path,
            gpu=gpu)


if __name__ == '__main__':
    main()
