# Reactive Safety System

A lightweight, prompt-based safety evaluation system extracted from HarmonyGuard's UtilityAgent, designed for integration with OS-Harm's mmagent.

## Overview

Reactive is a simplified safety system that focuses on:
1. **Prompt-based evaluation**: Uses LLM to evaluate actions based on security policies
2. **Dual-aspect validation**: Policy compliance check + Task alignment check
3. **Optional policy reference updates**: Can update policy references when violations are detected

## Key Features

- ✅ **Lightweight**: No complex World Model or trajectory tracking
- ✅ **Prompt-based**: Pure LLM-based evaluation using structured prompts
- ✅ **Policy-driven**: Evaluates actions against security policies
- ✅ **Task alignment**: Checks if actions align with task objectives
- ✅ **Optimization guidance**: Provides actionable feedback for regeneration
- ✅ **Optional reference updates**: Can update policy references when violations detected

## Architecture

```
baseline/reactive/
├── __init__.py
├── reactive_agent.py      # Core evaluation agent (extracted from UtilityAgent)
├── prompt_builder.py      # Prompt construction logic
├── policy_loader.py       # Policy loading and formatting
├── config.py              # Configuration management
├── wrapper.py             # mmagent integration wrapper
├── config.yaml            # Configuration file
└── README.md
```

## Components

### 1. ReactiveAgent
Core evaluation agent that:
- Loads security policies
- Builds evaluation prompts
- Calls LLM for evaluation
- Returns structured evaluation results

### 2. PromptBuilder
Handles prompt construction:
- Policy formatting
- Evaluation protocol (3 phases)
- JSON output format specification

### 3. PolicyLoader
Manages policy loading:
- Reads policy JSON files
- Formats policies for prompts
- Handles policy updates (optional)

### 4. Wrapper
Integration with mmagent:
- Compatible interface with SafePred's SafetyWrapper
- Action filtering
- Regeneration support

## Usage

### Basic Usage

```python
from baseline.reactive.wrapper import ReactiveWrapper

wrapper = ReactiveWrapper(
    enabled=True,
    policy_path="path/to/policies.json",
    config_path="baseline/reactive/config.yaml"
)

# Filter actions
filtered_actions, safety_info, risk_guidance = wrapper.filter_actions(
    obs=obs,
    actions=actions,
    instruction=instruction,
    current_response=response,
    action_generator=action_generator
)
```

### Configuration

Edit `config.yaml`:

```yaml
openai:
  reactive_agent:
    api_key: "${OPENAI_API_KEY}"
    base_url: "${OPENAI_API_BASE}"
    model: "gpt-4o"
    max_tokens: 2048
    temperature: 0

policy:
  enable_reference_updates: false  # Optional: enable policy reference updates
```

## Integration with OS-Harm

Use with `run.py`:

```bash
python run.py \
  --enable_safety_check \
  --safety_system reactive \
  --reactive_policy_path path/to/policies.json \
  --reactive_config_path baseline/reactive/config.yaml \
  --model gpt-4o
```

## Differences from HarmonyGuard

| Feature | HarmonyGuard | Reactive |
|---------|--------------|----------|
| Policy updates | ✅ Automatic | ⚠️ Optional |
| MCP Server | ✅ Required | ❌ Not needed |
| Policy Agent | ✅ Required | ❌ Not needed |
| Complexity | High | Low |
| Dependencies | Many | Minimal |

## Advantages

1. **Simpler**: No MCP server or Policy Agent dependencies
2. **Faster**: Direct LLM calls, no complex orchestration
3. **Focused**: Only prompt-based evaluation
4. **Flexible**: Can enable/disable policy updates as needed
