# SafePred

SafePred (Safety-TS-LMA) evaluates action risk for web agents using a world model and policy-based risk scoring. Use it to filter unsafe actions before execution or to get guidance for safer alternatives.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Configuration

**1. API keys (per provider)**

Copy `.env.example` to `.env` and set keys for the providers you use:

```bash
cp .env.example .env
# Edit .env: OPENAI_API_KEY, OPENAI_API_URL, QWEN_API_KEY, QWEN_API_URL, etc.
```

**2. Which provider per component**

In `config/config.yaml`, each LLM block has a `provider` field. SafePred reads the matching env vars (e.g. `provider: "qwen"` → `QWEN_API_KEY`, `QWEN_API_URL`).

- `world_model_llm`: state prediction and risk evaluation  
- `rule_extractor_llm`: extracting policies from documents (optional)  
- `action_agent_llm`: generating candidate actions in tree search (optional)

Adjust `model_name`, `temperature`, `max_tokens`, and risk thresholds in the same file as needed.

---

## Usage

### 1. Wrapper (recommended for benchmarks)

Use `SafePredWrapper` when integrating with a benchmark. It handles state/action format conversion via adapters.

**Initialize**

```python
from SafePred import SafePredWrapper

wrapper = SafePredWrapper(
    benchmark="visualwebarena",   # or "stwebagentbench", "osworld"
    config_path="config/config.yaml",
    policy_path="policies/my_policies.json",
    # optional:
    # use_planning=True,
    # web_agent_model_name="gpt-4",
)
```

**Evaluate action risk**

Call this **before** executing an action. Pass the current state, the action to evaluate, and candidate actions (e.g. from your web agent).

```python
result = wrapper.evaluate_action_risk(
    state=benchmark_state,           # benchmark-specific state
    action=action_to_evaluate,
    candidate_actions=[action1, action2, ...],
    intent="User task description",
    metadata={
        "task_id": "task_001",
        "action_history": [...],
        "current_response": "Agent's reasoning for this step",
    },
)
```

**Use the result**

- `result["risk_score"]` — 0.0–1.0  
- `result["is_safe"]` — True if below threshold  
- `result["risk_explanation"]` — why it was (un)safe  
- `result["requires_regeneration"]` — True if a safer action should be generated  
- `result["risk_guidance"]` — text to feed back into the agent for regeneration  
- `result["selected_action"]` — lowest-risk action among candidates (SafePred format; convert back with your adapter if needed)  
- `result["violated_policy_ids"]` — list of policy IDs violated, if any  

If `requires_regeneration` is True, prompt your agent again with `risk_guidance` and get new candidates; then call `evaluate_action_risk` again until you get a safe action or hit your retry limit.

**With plan (e.g. WASP)**

If your agent uses a plan and you want plan consistency checks and plan-update hints:

```python
result = wrapper.evaluate_action_risk_with_plan(
    state=state,
    action=action,
    plan_text=plan_text,
    intent=intent,
    metadata=metadata,
    candidate_actions=candidate_actions,
)
# Same risk fields as above, plus:
# result["should_update_plan"], result["update_reason"], result["optimization_guidance"]
```

**Policies and plan in the agent prompt**

- Policies string for the agent prompt: `wrapper.format_policies_for_prompt()`  
- Plan string for the agent prompt: `wrapper.format_plan_for_prompt(plan_text)`

**After execution**

- Record an executed action (e.g. when it wasn’t recorded during evaluation):  
  `wrapper.record_executed_action(action, response=...)`  
- Append to trajectory for training/replay:  
  `wrapper.update_trajectory(prev_state, action, next_state, action_success=..., intent=..., metadata=...)`

---

### 2. SafeAgent (direct use)

Use `SafeAgent` when you already have SafePred-format state and action strings (e.g. from your own adapter).

```python
from SafePred import SafeAgent, SafetyConfig

config = SafetyConfig.from_yaml("config/config.yaml")
agent = SafeAgent(config=config, policies=policies_list)

# state: SafePred format (e.g. dict with goal, url, axtree_txt / key_elements, etc.)
# candidate_actions: list of SafePred action strings, e.g. ["click [42]", "type [7] hello"]
result = agent.get_safe_action(
    current_state=state,
    candidate_actions=candidate_actions,
)

# result["action"], result["risk"], result["risk_explanation"], result["requires_regeneration"], result["risk_guidance"], ...
```

---

### 3. Supported benchmarks (adapters)

- `visualwebarena`  
- `stwebagentbench`  
- `osworld`  

State and action types are benchmark-specific; the wrapper converts them via the adapter. To add a benchmark, implement `BenchmarkAdapter` (see `adapters/base.py`) and register it.

---

### 4. Using SafePred on OS-Harm

OS-Harm is a benchmark under `benchmark/os-harm` that evaluates safety of computer-use agents (OSWorld-based). It uses the **outer SafePred package** (this repo’s `config/`, `models/`, etc.) automatically: from `benchmark/os-harm`, the path is set so that SafePred is imported from the repo root.

**Prerequisites**

- OSWorld/OS-Harm environment (VM, dependencies). See [benchmark/os-harm/README.md](benchmark/os-harm/README.md) and [OSWorld installation](https://github.com/xlang-ai/OSWorld#-installation).
- SafePred config and API keys: at the **repo root** (the method directory), ensure `.env` is set and `config/config.yaml` exists (or pass `--safepred_config_path` to point to a custom config).

**Run with SafePred enabled**

From the **SafePred repo root**:

```bash
cd benchmark/os-harm
python run.py \
  --path_to_vm /path/to/Ubuntu/Ubuntu.vmx \
  --observation_type screenshot_a11y_tree \
  --model o4-mini \
  --result_dir ./results \
  --test_all_meta_path evaluation_examples/test_misuse.json \
  --enable_safety_check
```

Optional SafePred-related arguments:

- `--safepred_policy_path`: path to policy rules JSON (e.g. repo root `policies/...` or a file under `benchmark/os-harm`). If omitted, no policies are loaded.
- `--safepred_config_path`: path to SafePred `config.yaml`. If omitted, the default is the outer method’s `config/config.yaml` (repo root).

**Task categories**

- Deliberate user misuse: `--test_all_meta_path evaluation_examples/test_misuse.json`; add `--jailbreak` to wrap prompts with a jailbreak template.
- Prompt injection: `--test_all_meta_path evaluation_examples/test_injection.json --inject`
- Model misbehavior: `--test_all_meta_path evaluation_examples/test_misbehavior.json`

**Example with policy file**

If you have policies at the repo root:

```bash
cd benchmark/os-harm
python run.py \
  --path_to_vm /path/to/Ubuntu/Ubuntu.vmx \
  --observation_type screenshot_a11y_tree \
  --model o4-mini \
  --result_dir ./results \
  --test_all_meta_path evaluation_examples/test_misuse.json \
  --enable_safety_check \
  --safepred_policy_path ../../policies/my_policies.json
```

Results (screenshots, logs, judge output) are written to `--result_dir`. More options and manual judge usage are in [benchmark/os-harm/README.md](benchmark/os-harm/README.md) and [benchmark/os-harm/QUICK_START_COMMANDS.md](benchmark/os-harm/QUICK_START_COMMANDS.md).

---

### 5. Using SafePred on WASP

WASP (benchmark under `benchmark/wasp`) evaluates web agents against prompt injection attacks. It uses the **outer SafePred** (this repo root): `benchmark/wasp` lives inside the SafePred repo, and the agent imports `SafePredWrapper` from the repo root via `visualwebarena/agent/safepred_wrapper.py`. No nested SafePred_v9 / SafePred_v10 directory is required.

**Prerequisites**

- WASP environment: Python 3.10, Playwright, Docker (for Claude computer use). See [benchmark/wasp/README.md](benchmark/wasp/README.md).
- Set `DATASET=webarena_prompt_injections` and website URLs (e.g. `REDDIT`, `GITLAB`) as in the WASP README.
- API keys: `OPENAI_API_KEY` (or Azure) for the agent and evaluators.
- SafePred: at the **SafePred repo root**, ensure `.env` and `config/config.yaml` exist (or pass `--safepred_config_path` to a custom config).

**Run prompt-injection evaluation with SafePred**

From the **SafePred repo root**:

```bash
cd benchmark/wasp/webarena_prompt_injections
python run.py --config configs/experiment_config.raw.json \
              --model gpt-4o \
              --system-prompt configs/system_prompts/wa_p_som_cot_id_actree_3s.json \
              --output-dir DIR_TO_STORE_RESULTS \
              --output-format webarena \
              --use_safepred \
              --safepred_config_path ../../../config/config.yaml
```

- `--use_safepred`: enable SafePred action risk evaluation.
- `--safepred_config_path`: path to SafePred `config.yaml` (e.g. repo root `config/config.yaml`; use a path relative to your cwd).

**Run VisualWebArena with SafePred (single run)**

From the **SafePred repo root**:

```bash
cd benchmark/wasp/visualwebarena
export DATASET=webarena_prompt_injections
export REDDIT="<your_reddit_domain>:9999"
export GITLAB="<your_gitlab_domain>:8023"

python run.py --test_config_base_dir ../webarena_prompt_injections/configs \
              --model gpt-4o \
              --result_dir ./results \
              --use_safepred \
              --safepred_config_path ../../config/config.yaml
```

Optional SafePred-related arguments (in `visualwebarena/run.py`):

- `--safepred_risk_threshold`: risk threshold (optional; can be read from config).
- `--policy`, `--policy_target`: policy file path and target (optional).
- `--use_planning`: plan-based execution (optional; planning can also be enabled via `config.yaml` only — see below).

**Planning**

Planning is controlled by SafePred’s config only: set `planning.enable: true` in the YAML passed via `--safepred_config_path`. There is no separate WASP CLI flag for planning.

**More**

- WASP setup, environments, and attack configs: [benchmark/wasp/README.md](benchmark/wasp/README.md).
- SafePred integration details (path resolution, package name): [benchmark/wasp/SAFEPRED_INTEGRATION.md](benchmark/wasp/SAFEPRED_INTEGRATION.md).

---

## Dependencies

Core: `numpy`, `pyyaml`, `requests`, `python-dotenv`.  
Optional: `python-docx`, `beautifulsoup4`, PDF libraries, and LLM SDKs depending on provider.

---

## Other docs

- **LLM Rule Extractor** (extract policies from PDF/DOC/TXT): see [USAGE_GUIDE.md](USAGE_GUIDE.md).  
- **Project layout**: `core/` (trajectory graph, plan monitor, trajectory storage), `models/` (world model, LLM client, prompts), `agent/` (SafeAgent), `config/`, `adapters/`.

---

## License

MIT.
