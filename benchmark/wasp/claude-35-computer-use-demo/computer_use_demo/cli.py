# Copyright (c) Meta Platforms, Inc. and affiliates.
import asyncio
from enum import StrEnum
import json
import os

import click
from computer_use_demo.loop import sampling_loop
from anthropic.types.beta import BetaContentBlockParam, BetaTextBlockParam
from computer_use_demo.tools import ToolResult
import httpx


class Sender(StrEnum):
    USER = "user"
    BOT = "assistant"
    TOOL = "tool"


def _output_callback(message: BetaContentBlockParam):
    pass


def _tool_callback(message: ToolResult, some_other_str: str):
    pass


class APIRequestsLogger:
    def __init__(self, file_path_to_write_to: str):
        self.file_path_to_write_to = file_path_to_write_to
        self.turn = 0

    def log_claude(
        self,
        request: httpx.Request,
        response: httpx.Response | object | None,
        error: Exception | None,
    ):
        with open(self.file_path_to_write_to, "a") as f:
            latest_request = json.loads(request.content.decode("utf-8"))

            latest_messages = latest_request["messages"]
            system_message_content = latest_request["system"]
            system_message = {"role": "system", "content": system_message_content}
            latest_messages.insert(0, system_message)
            f.write(json.dumps(latest_messages) + "\n")

            if not error:
                latest_response = json.loads(response.content)["content"]

                latest_response_first_text = ""
                for member in latest_response:
                    if member["type"] == "text":
                        latest_response_first_text = member["text"]
                print(
                    f"[Model response number {self.turn}] {latest_response_first_text}"
                )
                self.turn += 1

            else:
                raise error


def load_json_file(config_file_path: str):
    with open(config_file_path, "r") as config_file:
        config_data = json.load(config_file)

    return config_data


@click.command()
@click.option(
    "--user-prompt",
    required=True,
    help="The user prompt to give to the agent",
)
@click.option(
    "--conversation-log-file-path",
    default="computer_use_demo/conversation_logs.json",
    help="Where to write the conversation log, in the container's filesystem",
)
@click.option(
    "--max-actions",
    type=int,
    default=100,
    help="The maximum number of loop iterations (each of which is roughly an action by the agent)",
)
@click.option(
    "--model",
    default="arn:aws:bedrock:us-west-2:302263051492:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    help="The model to run that will back the agent",
)
@click.option(
    "--system-prompt-suffix",
    default="",
    help="A suffix to add to the defaul system prompt",
)
@click.option(
    "--only-n-most-recent-images",
    default=10,
    help="how many images to include at most in a request to the agent",
)
def main(
    user_prompt: str,
    conversation_log_file_path: str,
    max_actions: int,
    system_prompt_suffix: str,
    model: str,
    only_n_most_recent_images: int,
):

    initial_messages = [
        {
            "role": Sender.USER,
            "content": [BetaTextBlockParam(type="text", text=user_prompt)],
        }
    ]

    api_requests_accumulator = APIRequestsLogger(conversation_log_file_path)

    asyncio.run(
        sampling_loop(
            system_prompt_suffix=system_prompt_suffix,
            model=model,
            provider=os.getenv("API_PROVIDER", "bedrock"),
            messages=initial_messages,
            output_callback=_output_callback,
            tool_output_callback=_tool_callback,
            api_response_callback=api_requests_accumulator.log_claude,
            api_key="",
            only_n_most_recent_images=only_n_most_recent_images,
            max_actions=max_actions,
        )
    )

try:
    main()
except Exception as e:
    print(f"[!!! Error] An error occurred: {e}. Proceeding to the next task.")
