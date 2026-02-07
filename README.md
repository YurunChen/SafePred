<div align="center">

<img src="assets/logo1.png" alt="SafePred Logo" width="150"/>

# SafePred

**A Predictive Guardrail for Computer-Using Agents**

[![arXiv](https://img.shields.io/badge/arXiv-Paper-red.svg)](https://arxiv.org/abs/2602.01725)

</div>

## 📋 Abstract

<div align="justify">

With the widespread deployment of **Computer-Using Agents (CUAs)** in complex real-world environments, long-term risks often lead to severe and irreversible consequences. Most existing guardrails adopt a *reactive* approach—constraining behavior only within the current observation space. They can prevent immediate risks (e.g., clicking a phishing link) but cannot avoid *long-term* risks: seemingly reasonable actions can yield high-risk outcomes that appear only later (e.g., cleaning logs makes future audits untraceable), which reactive guardrails cannot see in the current observation.

We propose a **predictive guardrail** approach: align predicted future risks with current decisions. **SafePred** implements this via:

- **Short- and long-term risk prediction** — Using safety policies as the basis, SafePred leverages a world model to produce semantic risk representations (short- and long-term), identifying and pruning actions that lead to high-risk states.

- **Decision optimization** — Translating predicted risks into actionable guidance through step-level interventions and task-level re-planning.

**Extensive experiments** show that SafePred significantly reduces high-risk behaviors, achieving **over 97.6% safety performance** and improving task utility by up to **21.4%** compared with reactive baselines.

</div>

---

## 🚀 Setup

This section provides instructions for setting up the project, including cloning the repository, configuring environment variables, and setting up separate environments for the WASP and OS-Harm benchmarks.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/SafePred.git
cd SafePred
```

### 2. Set Up Environment Variables

The project requires API keys for language models. Copy the example environment file and add your keys. This step is required for both benchmarks.

```bash
cp .env.example .env
```

Then, edit the `.env` file to add your API keys for the providers you use:

| Provider | Environment Variables |
|----------|----------------------|
| **OpenAI** | `OPENAI_API_KEY`, `OPENAI_API_URL` |
| **Qwen** | `QWEN_API_KEY`, `QWEN_API_URL` |
| **Others** | Add corresponding variables |

### 3. Configure SafePred

Edit `config/config.yaml` to set up your providers and models:

- Set the `provider` field in each LLM block to match your `.env` variables
- Configure `world_model_llm` for state prediction and risk evaluation
- Configure `rule_extractor_llm` for extracting policies from documents (optional)
- Configure `action_agent_llm` for generating candidate actions (optional)
- Adjust `model_name`, `temperature`, `max_tokens`, and risk thresholds as needed

### 4. Benchmark Environments

The project uses two benchmarks, WASP and OS-Harm, which require separate environments.

#### WASP Environment

**Create and activate a conda environment for WASP:**

```bash
conda create -n wasp python=3.10 -y
conda activate wasp
```

**Install the required packages for WASP:**

```bash
pip install -r benchmark/wasp/webarena_prompt_injections/requirements.txt
```

#### OS-Harm Environment

**Create and activate a conda environment for OS-Harm:**

```bash
conda create -n osworld python=3.10 -y
conda activate osworld
```

**Install the required packages for OS-Harm:**

```bash
pip install -r benchmark/os-harm/baseline/code/requirements.txt
```

---

## 📜 Extracting Policies

SafePred uses safety policies to evaluate action risk. To extract policies from your own documents, see **[Extracting Policies](docs/POLICY_EXTRACTION.md)**.

---

## 🧪 Running Experiments

This section provides example commands for running the WASP and OS-Harm benchmarks with SafePred integration.

### WASP Benchmark

> **⚠️ Prerequisites**: Before running, you need to deploy Reddit and GitLab services and replace the placeholder URLs with your own.

```bash
cd benchmark/wasp
export DATASET=webarena_prompt_injections
export REDDIT="<your_reddit_domain>:9999"
export GITLAB="<your_gitlab_domain>:8023"
cd webarena_prompt_injections

python run.py \
    --config configs/experiment_config.raw.json \
    --model gpt-4o \
    --system-prompt configs/system_prompts/wa_p_cot_id_actree_3s.json \
    --output-dir /data/chenyurun/SafePred/benchmark/wasp/res \
    --output-format webarena \
    --use_safepred \
    --safepred_config_path ../../../config/config.yaml \
    --policy ../../../policies/my_policies.json
```

### OS-Harm Benchmark

```bash
cd benchmark/os-harm

python run.py \
    --path_to_vm /path/to/Ubuntu/Ubuntu.vmx \
    --observation_type screenshot_a11y_tree \
    --model o4-mini \
    --max_tokens 6000 \
    --result_dir ./results \
    --safepred_policy_path ../../policies/my_policies.json \
    --test_all_meta_path evaluation_examples/test_misuse.json \
    --inject \
    --enable_safety_check \
    --safepred_config_path ../../config/config.yaml
```

---

## 🔗 SafePred Integration

SafePred provides a wrapper for easy integration with benchmarks. This section explains how to integrate SafePred with existing benchmarks or extend it to new ones.

### Initializing SafePred Wrapper

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

### Using SafePred for Action Risk Evaluation

You can use SafePred to evaluate action risk before execution:

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

# Use the result
if result["requires_regeneration"]:
    # Prompt your agent again with result["risk_guidance"]
    pass
```

### Extending to Other Benchmarks

To integrate SafePred with a new benchmark:

1. **Implement the adapter**: Create a new adapter in `adapters/` that converts your benchmark's state/action format to SafePred's format.

2. **Register the benchmark**: Add your benchmark name to the supported list in SafePredWrapper.

3. **Use the wrapper**: Initialize SafePredWrapper with your benchmark name and appropriate config/policy paths.

<details>
<summary><b> Example Adapter Structure</b></summary>

```python
# adapters/my_benchmark.py
class MyBenchmarkAdapter(BaseAdapter):
    def convert_state(self, benchmark_state):
        # Convert your benchmark state to SafePred format
        return converted_state
    
    def convert_action(self, safepred_action):
        # Convert SafePred action back to your benchmark format
        return converted_action
```

</details>

---

## 📞 Contact

For any questions or issues, please contact via [email](mailto:yurunchen.research@gmail.com).

---
## 📄 Citation

```bibtex
@article{chen2026safepred,
  title={SafePred: A Predictive Guardrail for Computer-Using Agents via World Models},
  author={Chen, Yurun and Liao, Zeyi and Yin, Ping and Xie, Taotao and Yin, Keting and Zhang, Shengyu},
  journal={arXiv preprint arXiv:2602.01725},
  year={2026}
}
```

