#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Standalone script to evaluate completed tasks in a given output directory.
This allows evaluating tasks while they are still running, without waiting for all tasks to complete.
"""
import os
import sys
import subprocess
import argparse
import json
from pathlib import Path


def find_completed_tasks(output_dir):
    """Find all completed tasks in the output directory.
    
    Returns:
        tuple: (completed_task_ids, all_task_ids) - sets of task IDs
    """
    agent_logs_dir = os.path.join(output_dir, "agent_logs")
    task_dir = os.path.join(output_dir, "webarena_tasks")
    
    # Initialize return values
    all_task_ids = set()
    completed_task_ids = set()
    
    if not os.path.exists(agent_logs_dir):
        print(f"⚠️  Agent logs directory not found: {agent_logs_dir}")
        return completed_task_ids, all_task_ids
    
    if not os.path.exists(task_dir):
        print(f"⚠️  Task directory not found: {task_dir}")
        return completed_task_ids, all_task_ids
    
    # Get all task files
    task_files = [f for f in os.listdir(task_dir) if f.endswith(".json")]
    all_task_ids = set([f.replace(".json", "") for f in task_files])
    
    # Get completed tasks (check for conversation files)
    conversation_files = [f for f in os.listdir(agent_logs_dir) if f.startswith("conversation_raw_") and f.endswith(".jsonl")]
    for f in conversation_files:
        # conversation_raw_1000.jsonl -> 1000
        task_id = f.replace("conversation_raw_", "").replace(".jsonl", "")
        completed_task_ids.add(task_id)
    
    return completed_task_ids, all_task_ids


def run_evaluator_step_by_step(log_dir, task_dir, output_format):
    """Run step-by-step evaluator for attack goals (ASR)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    evaluator_script = os.path.join(script_dir, "evaluator_step_by_step.py")
    venv_python = os.path.join(script_dir, "venv", "bin", "python")
    
    if not os.path.exists(evaluator_script):
        print(f"⚠️  Evaluator script not found: {evaluator_script}")
        return False
    
    # Use venv Python if available, otherwise use sys.executable
    python_cmd = venv_python if os.path.exists(venv_python) else sys.executable
    
    print(f"\n{'='*80}")
    print("Running step-by-step evaluator (ASR)...")
    print(f"{'='*80}")
    
    try:
        # Prepare environment: unset SOCKS proxy to avoid httpx[socks] requirement
        env = os.environ.copy()
        env.pop("ALL_PROXY", None)
        env.pop("all_proxy", None)
        
        # Change to script directory to match original behavior
        result = subprocess.run([
            python_cmd, evaluator_script,
            "--log-folder", log_dir,
            "--task-folder", task_dir,
            "--format", output_format
        ], check=False, capture_output=False, cwd=script_dir, env=env)
        
        if result.returncode == 0:
            print("✅ Step-by-step evaluator completed successfully")
            return True
        else:
            print(f"⚠️  Step-by-step evaluator exited with code {result.returncode}")
            return False
    except Exception as e:
        print(f"❌ Error running step-by-step evaluator: {e}")
        return False


def run_evaluator_final_step(log_dir, task_dir, output_format=None):
    """Run final step evaluator for task performance."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    visualwebarena_dir = os.path.join(os.path.dirname(script_dir), "visualwebarena")
    evaluator_script = os.path.join(visualwebarena_dir, "evaluator_final_step.py")
    venv_python = os.path.join(visualwebarena_dir, "venv", "bin", "python")
    prepare_script = os.path.join(visualwebarena_dir, "prepare.sh")
    
    if not os.path.exists(evaluator_script):
        print(f"⚠️  Final step evaluator not found: {evaluator_script}")
        return False
    
    # Use venv Python if available, otherwise use sys.executable
    python_cmd = venv_python if os.path.exists(venv_python) else sys.executable
    
    print(f"\n{'='*80}")
    print("Running final step evaluator...")
    print(f"{'='*80}")
    
    try:
        # Prepare environment: set DATASET and required URLs for visualwebarena
        env = os.environ.copy()
        # Use DATASET from environment if set, otherwise default to "webarena"
        env["DATASET"] = os.environ.get("DATASET", "webarena")
        
        # Get domain URLs from environment
        # Priority: GITLAB_DOMAIN/REDDIT_DOMAIN (webarena_prompt_injections) > GITLAB/REDDIT > defaults
        gitlab_domain = os.environ.get("GITLAB_DOMAIN") or os.environ.get("GITLAB", "")
        reddit_domain = os.environ.get("REDDIT_DOMAIN") or os.environ.get("REDDIT", "")
        
        # Set required environment variables for webarena evaluator
        # Use default localhost values if not set (these are typical development values)
        # The evaluator will check these, so we must provide non-empty values
        required_vars = {
            "REDDIT": reddit_domain or os.environ.get("REDDIT", "http://localhost:8000"),
            "GITLAB": gitlab_domain or os.environ.get("GITLAB", "http://localhost:8023"),
            "SHOPPING": os.environ.get("SHOPPING", "http://localhost:7770"),
            "SHOPPING_ADMIN": os.environ.get("SHOPPING_ADMIN", "http://localhost:7780"),
            "WIKIPEDIA": os.environ.get("WIKIPEDIA", "http://localhost:8888"),
            "MAP": os.environ.get("MAP", "http://localhost:3000"),
            "HOMEPAGE": os.environ.get("HOMEPAGE", "http://localhost:4399"),
        }
        
        # Check if using default values (warn user)
        using_defaults = []
        if not (reddit_domain or os.environ.get("REDDIT")):
            using_defaults.append("REDDIT")
        if not (gitlab_domain or os.environ.get("GITLAB")):
            using_defaults.append("GITLAB")
        for var in ["SHOPPING", "SHOPPING_ADMIN", "WIKIPEDIA", "MAP", "HOMEPAGE"]:
            if not os.environ.get(var):
                using_defaults.append(var)
        
        if using_defaults:
            print(f"⚠️  Warning: Using default localhost values for: {', '.join(using_defaults)}")
            print("   If these are incorrect, please set environment variables:")
            print("   export REDDIT=<reddit_domain>")
            print("   export GITLAB=<gitlab_domain>")
            print("   export SHOPPING=<shopping_domain>")
            print("   export SHOPPING_ADMIN=<shopping_admin_domain>")
            print("   export WIKIPEDIA=<wikipedia_domain>")
            print("   export MAP=<map_domain>")
            print("   export HOMEPAGE=<homepage_domain>")
            print("   Or set GITLAB_DOMAIN and REDDIT_DOMAIN if using webarena_prompt_injections")
        else:
            print(f"✅ Using environment variables for domains:")
            if required_vars["REDDIT"]:
                print(f"   REDDIT: {required_vars['REDDIT']}")
            if required_vars["GITLAB"]:
                print(f"   GITLAB: {required_vars['GITLAB']}")
        
        # Set environment variables (always set, even if using defaults)
        for var, value in required_vars.items():
            env[var] = value
        
        # Run prepare.sh first (as in original script)
        # prepare.sh needs to run in visualwebarena venv, but 'source' may not work in subprocess
        # Since we've already set all environment variables, we can skip prepare.sh or run it differently
        if os.path.exists(prepare_script):
            print("Running prepare.sh...")
            # Try to run prepare.sh directly (it should work if environment variables are set)
            # Use bash -c to handle the script properly
            try:
                prepare_result = subprocess.run(
                    ["bash", prepare_script],
                    check=False,
                    cwd=visualwebarena_dir,
                    env=env,
                    capture_output=True,
                    text=True
                )
                if prepare_result.returncode != 0:
                    print(f"⚠️  Warning: prepare.sh exited with code {prepare_result.returncode}")
                    if prepare_result.stderr:
                        print(f"   Error: {prepare_result.stderr[:200]}")
                    print("   Continuing anyway - environment variables are already set")
            except Exception as e:
                print(f"⚠️  Warning: Failed to run prepare.sh: {e}")
                print("   Continuing anyway - environment variables are already set")
        
        cmd = [
            python_cmd, evaluator_script,
            "--log-folder", log_dir,
            "--task-folder", task_dir
        ]
        if output_format:
            cmd.extend(["--format", output_format])
        
        result = subprocess.run(cmd, check=False, capture_output=False, cwd=visualwebarena_dir, env=env)
        
        if result.returncode == 0:
            print("✅ Final step evaluator completed successfully")
            return True
        else:
            print(f"⚠️  Final step evaluator exited with code {result.returncode}")
            return False
    except Exception as e:
        print(f"❌ Error running final step evaluator: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate completed tasks in a given output directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate tasks in a specific output directory
  python evaluate_tasks.py logs/gpt-4o-mini_safepred_agentwithpolicy/0
  
  # Evaluate with specific output format
  python evaluate_tasks.py logs/gpt-4o-mini_safepred_agentwithpolicy/0 --format webarena
  
  # Only run step-by-step evaluator (ASR)
  python evaluate_tasks.py logs/gpt-4o-mini_safepred_agentwithpolicy/0 --step-by-step-only
  
  # Only run final step evaluator
  python evaluate_tasks.py logs/gpt-4o-mini_safepred_agentwithpolicy/0 --final-step-only
        """
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the output directory containing completed tasks (e.g., logs/gpt-4o-mini_safepred_agentwithpolicy/0)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="webarena",
        help="Output format (default: webarena)"
    )
    parser.add_argument(
        "--step-by-step-only",
        action="store_true",
        help="Only run step-by-step evaluator (ASR), skip final step evaluator"
    )
    parser.add_argument(
        "--final-step-only",
        action="store_true",
        help="Only run final step evaluator, skip step-by-step evaluator"
    )
    parser.add_argument(
        "--attacker-tasks",
        action="store_true",
        help="Also evaluate attacker tasks (webarena_tasks_attacker)"
    )
    
    args = parser.parse_args()
    
    # Normalize output directory path
    output_dir = os.path.abspath(args.output_dir)
    if not os.path.exists(output_dir):
        print(f"❌ Output directory does not exist: {output_dir}")
        sys.exit(1)
    
    # Find completed tasks
    print(f"\n{'='*80}")
    print(f"Scanning output directory: {output_dir}")
    print(f"{'='*80}")
    
    completed_task_ids, all_task_ids = find_completed_tasks(output_dir)
    
    print(f"\nTask Status:")
    print(f"  Total tasks: {len(all_task_ids)}")
    print(f"  Completed tasks: {len(completed_task_ids)}")
    print(f"  Pending tasks: {len(all_task_ids) - len(completed_task_ids)}")
    
    if len(completed_task_ids) == 0:
        print("\n⚠️  No completed tasks found. Nothing to evaluate.")
        sys.exit(0)
    
    print(f"\nCompleted task IDs: {sorted([int(tid) for tid in completed_task_ids])[:10]}{'...' if len(completed_task_ids) > 10 else ''}")
    
    # Set up directories
    log_dir = os.path.join(output_dir, "agent_logs")
    task_dir = os.path.join(output_dir, "webarena_tasks")
    attacker_task_dir = os.path.join(output_dir, "webarena_tasks_attacker")
    
    if not os.path.exists(log_dir):
        print(f"❌ Agent logs directory not found: {log_dir}")
        sys.exit(1)
    
    if not os.path.exists(task_dir):
        print(f"❌ Task directory not found: {task_dir}")
        sys.exit(1)
    
    # Run evaluators
    success = True
    
    # Step 1: Run step-by-step evaluator (ASR)
    if not args.final_step_only:
        if not run_evaluator_step_by_step(log_dir, task_dir, args.format):
            success = False
    
    # Step 2: Run final step evaluator for attacker tasks (if requested)
    if args.attacker_tasks and os.path.exists(attacker_task_dir) and not args.step_by_step_only:
        if not run_evaluator_final_step(log_dir, attacker_task_dir, args.format):
            success = False
    
    # Step 3: Run final step evaluator for user tasks
    if not args.step_by_step_only:
        if not run_evaluator_final_step(log_dir, task_dir, args.format):
            success = False
    
    # Summary
    print(f"\n{'='*80}")
    print("Evaluation Summary")
    print(f"{'='*80}")
    print(f"  Completed tasks evaluated: {len(completed_task_ids)}")
    print(f"  Pending tasks: {len(all_task_ids) - len(completed_task_ids)}")
    
    if success:
        print("\n✅ Evaluation completed successfully!")
        print(f"\nResults are saved in:")
        print(f"  - {log_dir}/action_classifications.csv (step-by-step ASR)")
        print(f"  - /tmp/run_step_by_step_asr.json (step-by-step ASR summary)")
        print(f"  - /tmp/run_attacker_utility.json (attacker utility, if evaluated)")
        print(f"  - /tmp/run_user_utility.json (user utility)")
    else:
        print("\n⚠️  Evaluation completed with some errors. Check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

