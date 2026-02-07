# WASP: Benchmarking Web Agent Security Against Prompt Injection Attacks
[Ivan Evtimov](https://ivanevtimov.eu/),
[Arman Zharmagambetov](https://arman-z.github.io),
[Aaron Grattafiori](https://scholar.google.com/citations?user=j8PDwLwAAAAJ),
[Chuan Guo](https://sites.google.com/view/chuanguo),
[Kamalika Chaudhuri](https://cseweb.ucsd.edu/~kamalika)


This repo is an official implementation of the **WASP** ([arxiv:2504.18575](https://arxiv.org/abs/2504.18575)). 

Autonomous web navigation agents hold great potential for unlocking human productivity by automating mundane tasks on the internet. However, these agents can also be taken advantage of as new attack vectors for adversaries when deployed without proper mitigation. One emerging threat &mdash; prompt injection attacks, where the adversary embeds malicious instructions in the web environment that divert the agent to instead perform tasks for the adversary. To measure the extent of this risk, we construct this new benchmark that simulates real world attacks in a highly realistic, end-to-end executable web environment, without exposing the agent to real harm. 

Who did we develop this benchmark for?
- Developers of autonomous web navigation agents, broadly defined: engineers, researchers, and startups who want to evaluate their agentic systems against prompt injection attacks.
- Red-teamers and security researchers who want to test novel prompt injection attack mechanisms designed for such agents.


Please 🌟star🌟 this repo and cite our paper 📜 if you like (and/or use) our work, thank you!


## Installation

What our setup does not handle (you need to do this yourself before beginning setup):
* Install docker (required for the Claude Computer Use only)
* Install python 3.10 (required by the `visualwebarena` codebase's dependencies). It needs to be available as a `python3.10` command
* `playwright` installations often require system dependencies to be installed after running `playwright install`. We run this command in our script but check the stdout in case it is asking you to manually finish the dependency setup

We provide a setup script to install all required packages:   
```bash
cd webarena_prompt_injections/
bash setup.sh
```

You can also run the unit tests to ensure that VisualWebArena is installed correctly:
```bash
cd visualwebarena/
pytest -x
```


## Running an end-to-end prompt injection tests.

1. Setup the standalone environments.
Please check out [this page](visualwebarena/environment_docker/README.md) for details. IMPORTANT: Please follow instructions on that page as we modify several settings from the original (visual)webarena! We use `gitlab` and `reddit` environments, so you can discard setup instructions for all other websites.

2. Configurate the urls for each website and export the `DATASET` to be `webarena_prompt_injections` (to skip this step next time, add this to your `.bashrc`):
```bash
export DATASET=webarena_prompt_injections
export REDDIT="<your_reddit_domain>:9999"
export GITLAB="<your_gitlab_domain>:8023"
```

3. Set up API keys.

* Set a valid OpenAI API key (starting with `sk-`) as the environment variable: 
    ```bash
    export OPENAI_API_KEY='your_key'
    ```
    Alternatively, set Azure API key and Azure Endpoint if you are accessing OpenAI models via Azure:
    ```bash
    export AZURE_API_KEY='your_azure_api_key'
    export AZURE_API_ENDPOINT='your_azure_endpoint'
    export AZURE_API_VERSION='2024-10-21'
    ``` 
* For Claude, the code expects `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN`. Additionally, provide the full AWS Bedrock name for the Claude model to the following config file: `webarena_prompt_injections/system_prompts/claude_*_agent.json`
    
  Note that we still need OpenAI/Azure API keys to run evaluators.

4. Run end-to-end evaluation! The main entrypoint is `webarena_prompt_injections/run.py`. Below is the example command to run GPT-4o with VisualWebArena scaffolding using accessibility tree + SOM (set-of-marks) representation discussed in the paper:
    ```bash
    cd webarena_prompt_injections
    python run.py --config configs/experiment_config.raw.json \
                  --model gpt-4o \
                  --system-prompt configs/system_prompts/wa_p_som_cot_id_actree_3s.json \
                  --output-dir DIR_TO_STORE_RESULTS_AND_LOGS \
                  --output-format webarena
    ```
#### Explanation for the hyperparameters:

`--config` configurations for setting up attacker's goals and how to evaluate them.

`--model` backbone LLM model, currently available: `gpt-4o`, `gpt-4o-mini`, `claude-35`, `claude-37`.

`--system-prompt` system prompt used by specific scaffolding. System prompts used for (Visual)WebArena are located in `configs/system_prompts/wa_*`, whereas for Claude it is `configs/system_prompts/claude_*`. System prompt with defensive instructions ends with *defense* (e.g. `claude_35_agent_generic_defense.json`).

`--output-dir` this will store logs and agentic traces, which can be important for debugging.

`--output-format` spits out the appropriate commands to run an agent. Available options: `webarena`, `claude`, `gpt_web_tools`. The latter (gpt_web_tools) is similar to webarena but puts observation (accessibility tree) under separate tool_output role defined in OpenAI API. This can be useful to test Instruction Hierarchy defense mechanisms implemented in some OpenAI models. 

**Important note**: currently, single run takes approximately 4-6 hours to run. Sometimes it is useful to test only subset of user goals (`--user_goal_start`) or run on a single user_goal / prompt_injection_format (`--run-single`). 

### Running Claude agent

Example script to run Claude-3.5-Sonnet computer use agent
```bash
python run.py --model claude-35 \
              --system-prompt configs/system_prompts/claude_35_agent.json \
              --output-dir DIR_TO_STORE_RESULTS_AND_LOGS \
              --output-format claude
```

### Running GPT with tool use
Example script to run GPT-4o-mini agent with tool/function use capability: 
```bash
python run.py --model gpt-4o-mini \
              --output-dir DIR_TO_STORE_RESULTS_AND_LOGS \
              --output-format gpt_web_tools
```


### Resetting environments

Our scripts automatically cleans up environment after each run. However, sometimes manual reset is required. To do so, please follow steps on [this page](visualwebarena/environment_docker/README.md#environment-reset).

### How to test a new prompt injection attack?

The main logic for injecting malicious prompts into the web environment is implemented in [prompt_injector.py](./webarena_prompt_injections/prompt_injector.py). It relies on defining attacker's goals which can be found in [this config](./webarena_prompt_injections/configs/experiment_config.raw.json). The malicious prompts themselves are defined [here](./webarena_prompt_injections/constants.py#L10-L14). Inspect the above scripts/configs to try your novel attack method(s) and test them on this benchmark on a variety of web agents that we support!

## Acknowledgements

Our code is heavily based off the <a href="https://github.com/web-arena-x/webarena">WebArena</a>, <a href="https://github.com/web-arena-x/visualwebarena">VisualWebArena</a> and <a href="https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo">Anthropic Computer Use Demo</a> codebases.

## License
The majority of WASP is licensed under [CC-BY-NC 4.0 license](./LICENSE), however portions of the project are available under separate license terms: visualwebarena is licensed under the MIT license. More information available [via this link](./visualwebarena/LICENSE). The [data](./webarena_prompt_injections/configs/experiment_config.raw.json) is intended for benchmarking purposes and is licensed CC-BY-NC. License for using Claude computer use demo code is available [via this link](./claude-35-computer-use-demo/LICENSE).
