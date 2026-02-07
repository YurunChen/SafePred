# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import time
from environment_setup import AGENT_ACCOUNTS
from environment_editors.gitlab_editor import GitlabEditor
from environment_editors.reddit_editor import RedditEditor
from environment_editors.base_environment_editor import WebArenaEditorException

from utils import load_prompt_injection_config
from tqdm import tqdm
import click


def delete_all_reddit_posts_by_user(
    reddit_editor: RedditEditor,
    username: str,
    password: str,
    max_attempts: int,
    **kwargs,
):
    try:
        with reddit_editor:
            reddit_editor.login(username, password)
            reddit_editor.delete_all_posts_by_username(username, max_attempts)
    except WebArenaEditorException as e:
        if "Failed to login" in str(e):
            print(
                f"Did not delete {username} with password {password}. Likely does not exist or password is wrong."
            )
        else:
            raise


def delete_reddit_post_by_url(
    reddit_editor: RedditEditor,
    username: str,
    password: str,
    user_post_url: str,
    **kwargs,
):
    try:
        with reddit_editor:
            reddit_editor.login(username, password)
            reddit_editor.delete_post_by_url(url=user_post_url)
    except WebArenaEditorException as e:
        if "Failed to login" in str(e):
            print(
                f"Did not delete {username} with password {password}. Likely does not exist or password is wrong."
            )
        else:
            raise


def delete_all_comments_on_post_by_user(
    reddit_editor: RedditEditor,
    username: str,
    password: str,
    instantiated_action_url: str,
    max_attempts: int,
    **kwargs,
):
    try:
        with reddit_editor:
            reddit_editor.login(username, password)
            reddit_editor.delete_all_comments_on_post_by_user(
                instantiated_action_url, max_attempts
            )
    except WebArenaEditorException as e:
        if "Failed to login" in str(e):
            print(
                f"Did not delete {username} with password {password}. Likely does not exist or password is wrong."
            )
        else:
            raise


def reset_reddit_user_email(
    reddit_editor: RedditEditor, username: str, password: str, **kwargs
):
    try:
        with reddit_editor:
            reddit_editor.login(username, password)
            reddit_editor.reset_email(username)
    except WebArenaEditorException as e:
        print(f"Error in resetting reddit account email: {e}")


def delete_reddit_user_messages(
    reddit_editor: RedditEditor, username: str, password: str, **kwargs
):
    try:
        with reddit_editor:
            reddit_editor.login(username, password)
            reddit_editor.delete_all_messages_by_user(username)
    except WebArenaEditorException as e:
        print(f"Error in resetting reddit account email: {e}")


def delete_reddit_instantiated_posts_and_comments(
    reddit_editor: RedditEditor,
    attacker_username: str,
    attacker_password: str,
    user_username: str,
    user_password: str,
    **kwargs,
):
    delete_all_reddit_posts_by_user(
        reddit_editor, attacker_username, attacker_password, max_attempts=3
    )
    delete_all_reddit_posts_by_user(
        reddit_editor, user_username, user_password, max_attempts=1
    )
    reset_reddit_user_email(reddit_editor, user_username, user_password)
    delete_reddit_user_messages(reddit_editor, user_username, user_password)
    if "user_post_url" in kwargs:
        user_post_url = kwargs["user_post_url"]
        delete_reddit_post_by_url(
            reddit_editor, user_username, user_password, user_post_url
        )
    if "instantiated_action_url":
        instantiated_action_url = kwargs["instantiated_action_url"]
        delete_all_comments_on_post_by_user(
            reddit_editor,
            user_username,
            user_password,
            instantiated_action_url,
            max_attempts=3,
        )


def delete_attacker_account(
    gitlab_editor: GitlabEditor, attacker_username: str, password: str, **kwargs
):
    try:
        with gitlab_editor:
            gitlab_editor.login(attacker_username, password)
            gitlab_editor.delete_account(password)
    except WebArenaEditorException as e:
        if "Invalid login or password" in str(e):
            print(
                f"Did not delete {attacker_username} with password {password}. Likely does not exist or password is wrong."
            )
        else:
            print(
                f"Unknown exception raised when trying to delete the account {attacker_username}. Skipping..."
            )


def delete_project_owned_by_agent_user(
    gitlab_editor: GitlabEditor, project_name: str, **kwargs
):
    with gitlab_editor:
        gitlab_editor.login(
            AGENT_ACCOUNTS["gitlab"]["username"], AGENT_ACCOUNTS["gitlab"]["password"]
        )
        gitlab_editor.delete_project(AGENT_ACCOUNTS["gitlab"]["username"], project_name)


def delete_project_owned_by_agent_user_and_attacker_account(
    gitlab_editor: GitlabEditor,
    project_name: str,
    attacker_username: str,
    password: str,
    **kwargs,
):
    delete_project_owned_by_agent_user(gitlab_editor, project_name)
    delete_attacker_account(
        gitlab_editor,
        attacker_username,
        password,
    )


def attempt_to_delete_project_with_different_namespaces(
    gitlab_editor: GitlabEditor,
    project_name: str,
    new_namespace: str,
    attacker_username: str,
    password: str,
    **kwargs,
):
    failed_deletion = False
    try:
        delete_project_owned_by_agent_user(gitlab_editor, project_name)
    except WebArenaEditorException as e:
        if str(e).startswith("Currently logged in user cannot"):
            failed_deletion = True
        else:
            raise
    if failed_deletion:
        with gitlab_editor:
            gitlab_editor.login(
                attacker_username,
                password,
            )
            gitlab_editor.delete_project(new_namespace, project_name)

    with gitlab_editor:
        gitlab_editor.login(
            AGENT_ACCOUNTS["gitlab"]["username"],
            AGENT_ACCOUNTS["gitlab"]["password"],
        )
        gitlab_editor.delete_group(new_namespace)


def attempt_to_delete_project_with_different_namespaces_and_attacker_account(
    gitlab_editor: GitlabEditor,
    project_name: str,
    new_namespace: str,
    attacker_username: str,
    password: str,
    **kwargs,
):
    attempt_to_delete_project_with_different_namespaces(
        gitlab_editor, project_name, new_namespace, attacker_username, password
    )
    delete_attacker_account(
        gitlab_editor,
        attacker_username,
        password,
    )


def make_agent_user_profile_public(gitlab_editor: GitlabEditor, **kwargs):
    """
    Makes the agent user's profile public.
    """
    with gitlab_editor:
        gitlab_editor.login(
            AGENT_ACCOUNTS["gitlab"]["username"], AGENT_ACCOUNTS["gitlab"]["password"]
        )
        gitlab_editor.toggle_private_profile(False)
    return {}


def reset_zshrc_file_in_repo(
    gitlab_editor: GitlabEditor, project_owner: str, project_name: str, **kwargs
):
    """
    Takes the file at `environment_editors/prisitine_dotfiles_zshrc.txt`
    and replaces the .zshrc on the main branch in the specified project
    with the contents of this file

    Args:
        gitlab_editor (GitLabEditor): an object providing functionalities for basic scripted actions in the GitLab environment via Playwright

    """
    with gitlab_editor:
        gitlab_editor.login(
            AGENT_ACCOUNTS["gitlab"]["username"], AGENT_ACCOUNTS["gitlab"]["password"]
        )

        gitlab_editor.replace_file_with_local_file(
            project_owner,
            project_name,
            "main",
            ".zshrc",
            "environment_editors/prisitine_dotfiles_zshrc.txt",
        )

    return {}


def delete_merge_request_and_branch(
    gitlab_editor: GitlabEditor,
    project_owner: str,
    project_name: str,
    branch_name: str,
    merge_request_number: int,
):
    with gitlab_editor:
        gitlab_editor.login(
            AGENT_ACCOUNTS["gitlab"]["username"], AGENT_ACCOUNTS["gitlab"]["password"]
        )
        gitlab_editor.close_merge_request(
            project_owner, project_name, merge_request_number
        )
        gitlab_editor.delete_branch(project_owner, project_name, branch_name)


def delete_deploy_key(
    gitlab_editor: GitlabEditor, project_owner: str, project_name: str, **kwargs
):
    with gitlab_editor:
        gitlab_editor.login(
            AGENT_ACCOUNTS["gitlab"]["username"], AGENT_ACCOUNTS["gitlab"]["password"]
        )
        gitlab_editor.delete_deploy_key(project_owner, project_name)


def delete_deploy_token(
    gitlab_editor: GitlabEditor, project_owner: str, project_name: str, **kwargs
):
    with gitlab_editor:
        gitlab_editor.login(
            AGENT_ACCOUNTS["gitlab"]["username"], AGENT_ACCOUNTS["gitlab"]["password"]
        )
        gitlab_editor.delete_deploy_token(project_owner, project_name)


def delete_webhook(
    gitlab_editor: GitlabEditor, project_owner: str, project_name: str, **kwargs
):
    with gitlab_editor:
        gitlab_editor.login(
            AGENT_ACCOUNTS["gitlab"]["username"], AGENT_ACCOUNTS["gitlab"]["password"]
        )
        gitlab_editor.delete_webhook(project_owner, project_name)


def delete_ssh_key(gitlab_editor: GitlabEditor, **kwargs):
    with gitlab_editor:
        gitlab_editor.login(
            AGENT_ACCOUNTS["gitlab"]["username"], AGENT_ACCOUNTS["gitlab"]["password"]
        )
        gitlab_editor.delete_ssh_key()


def delete_access_token(gitlab_editor: GitlabEditor, **kwargs):
    with gitlab_editor:
        gitlab_editor.login(
            AGENT_ACCOUNTS["gitlab"]["username"], AGENT_ACCOUNTS["gitlab"]["password"]
        )
        gitlab_editor.delete_access_token()


CLEANUP_ACTION_NAME_TO_FUNCTION = {
    "attempt_to_delete_project_with_different_namespaces_and_attacker_account": attempt_to_delete_project_with_different_namespaces_and_attacker_account,
    "make_agent_user_profile_public": make_agent_user_profile_public,
    "reset_zshrc_file_in_repo": reset_zshrc_file_in_repo,
    "delete_project_owned_by_agent_user": delete_project_owned_by_agent_user,
    "delete_project_owned_by_agent_user_and_attacker_account": delete_project_owned_by_agent_user_and_attacker_account,
    "delete_reddit_instantiated_posts_and_comments": delete_reddit_instantiated_posts_and_comments,
    "delete_merge_request_and_branch": delete_merge_request_and_branch,
    "delete_deploy_key": delete_deploy_key,
    "delete_deploy_token": delete_deploy_token,
    "delete_webhook": delete_webhook,
    "delete_ssh_key": delete_ssh_key,
    "delete_access_token": delete_access_token,
}


@click.command()
@click.option(
    "--gitlab-domain",
    default="none",
    show_default=True,
    help="GitLab domain URL (e.g., http://example.com:port)",
)
@click.option(
    "--reddit-domain",
    default="none",
    show_default=True,
    help="Reddit domain URL (e.g., http://example.com:port)",
)
@click.option(
    "--prompt-injection-config-path",
    required=True,
    default="/tmp/instantiated_prompt_injections_config.json",
    help="Path to the prompt injection config JSON file.",
)
@click.option(
    "--start-index",
    type=int,
    default=0,
    help="Perform environment cleanup starting from the given index in the config list. If negative or greater than the length of the list, skips.",
)
@click.option(
    "--skip-delete-gitlab-issues",
    is_flag=True,
    default=False,
    help="Skip deleting all issues in the project.",
)
@click.option(
    "--delete-attacker-account",
    is_flag=True,
    default=False,
    help="Skip deleting the attacker's account.",
)
@click.option(
    "--max-num-retries",
    type=int,
    default=3,
    show_default=True,
    help="Maximum number of retries for cleanup actions.",
)
def cleanup(
    gitlab_domain,
    reddit_domain,
    prompt_injection_config_path,
    start_index,
    skip_delete_gitlab_issues,
    delete_attacker_account,
    max_num_retries,
):
    if gitlab_domain == "none":  # try to get it from env var
        gitlab_domain = os.environ.get("GITLAB")
    if reddit_domain == "none":
        reddit_domain = os.environ.get("REDDIT")

    # Initialize Editors
    gitlab_editor = GitlabEditor(gitlab_domain)
    reddit_editor = RedditEditor(reddit_domain)

    prompt_injection_configs = load_prompt_injection_config(
        prompt_injection_config_path
    )
    if start_index >= 0 and start_index < len(prompt_injection_configs):
        print(
            f"Performing environment cleanup starting from index {start_index} "
            f"in the list at {prompt_injection_config_path}"
        )
        for idx, prompt_injection_config in enumerate(
            tqdm(prompt_injection_configs[start_index:])
        ):
            cleanup_fn_name = prompt_injection_config["cleanup_fn"]
            if cleanup_fn_name:
                for cleanup_attempt_i in range(max_num_retries):
                    try:
                        print(
                            f"Processing cleanup config at index {idx}. Environment: {prompt_injection_config['environment']}"
                        )
                        cleanup_fn = CLEANUP_ACTION_NAME_TO_FUNCTION[cleanup_fn_name]
                        if prompt_injection_config["environment"] == "gitlab":
                            cleanup_fn(
                                gitlab_editor, **prompt_injection_config["parameters"]
                            )

                        elif prompt_injection_config["environment"] == "reddit":
                            cleanup_fn(
                                reddit_editor, **prompt_injection_config["parameters"]
                            )
                        print(f"Cleanup was successfull, moving to the next one...")
                        break
                    except Exception as e:
                        if cleanup_attempt_i + 1 == max_num_retries:
                            print(f"Cleanup idx={idx} failed. Skipping...")
                            continue
                        print(
                            f"Error during cleanup: {e}. Attempting to cleanup idx={idx} again: {cleanup_attempt_i+1}"
                        )
                        time.sleep(
                            5
                        )  # fail typically occurs due to timeout, put some sleep before retrying

        print("Done with cleanup actions for prompt injection objectives.")

    if not skip_delete_gitlab_issues:
        print("Cleaning up issues.")
        with gitlab_editor:
            try:
                gitlab_editor.login(
                    AGENT_ACCOUNTS["gitlab"]["username"],
                    AGENT_ACCOUNTS["gitlab"]["password"],
                )
                gitlab_editor.delete_all_issues_in_project(
                    AGENT_ACCOUNTS["gitlab"]["username"], "dotfiles"
                )
            except Exception as e:
                print(f"[Warning!] Deleting malicious issues failed: {e}.")
                print(
                    "That is fine, next run will create new gitlab issues. Skippping..."
                )
        print("Done deleting all issues in the project.")

    if delete_attacker_account:
        print("Deleting attacker account.")
        delete_attacker_account(
            gitlab_editor, "arsenelupin", "notaverycommonlyusedpassword"
        )
        print("Done deleting the attacker's account.")


if __name__ == "__main__":
    cleanup()
