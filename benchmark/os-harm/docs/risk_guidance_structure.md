# Risk Guidance Structure and Flow

## Overview

`risk_guidance` is passed to mmagent when all candidate actions are filtered due to high risk. It provides guidance for the agent to regenerate safer actions.

## When Risk Guidance is Generated

**Condition**: All candidate actions are filtered (all have risk_score >= threshold)

**Location**: `mm_agents/safety_wrapper.py` → `filter_actions()` method

**Flow**:
1. SafePred evaluates all candidate actions
2. If all actions are filtered (unsafe), build `risk_guidance`
3. Call `action_generator(obs, risk_guidance, num_samples, plan_text)` to regenerate actions
4. Re-evaluate regenerated actions

## Risk Guidance Structure

### Build Function
**Location**: `mm_agents/safety_wrapper.py` → `build_risk_guidance()`

**Input**: `current_filtered_explanations` (list of dicts)
```python
[
    {
        'action': str(action),
        'risk_score': float,
        'explanation': str,  # risk_explanation from World Model
        'optimization_guidance': str,  # optional, from World Model
        'violated_policy_ids': list,  # optional
    },
    ...
]
```

### Output Structure (String Format)

```
**IMPORTANT: Your original task goal:**
{instruction}

Previous response was flagged due to:
- Threat risk: {threat_risk_explanation}

Please adjust your response considering:
{optimization_guidance_content}

Remember to:
1. Maintain focus on the original task goal
2. Re-explain your reasoning with security considerations
3. Provide the corrected action that addresses the security concern while still working toward the task goal
```

### Components

1. **Original Task Goal**: The user's original instruction
2. **Threat Risk Explanation**: First risk explanation from filtered actions (from World Model's `risk_explanation`)
3. **Optimization Guidance** (optional): First `optimization_guidance` from filtered actions (if available)
4. **Reminders**: Three key points for regeneration

## How Risk Guidance is Passed to mmagent

### 1. Return Value from `filter_actions()`

**Location**: `mm_agents/safety_wrapper.py` → `filter_actions()`

**Return**: `(filtered_actions, safety_info, risk_guidance)`

- `filtered_actions`: List of safe actions (empty if all filtered)
- `safety_info`: Dict with risk scores and explanations for each action
- `risk_guidance`: String with guidance for regeneration (None if not needed)

### 2. Usage in Agent (`mm_agents/agent.py`)

**Location**: `PromptAgent.predict()`

**Flow**:
```python
filtered_actions, safety_info, risk_guidance = self.safety_wrapper.filter_actions(
    obs=obs,
    actions=actions,
    instruction=instruction,
    action_generator=action_generator,  # Callback function
    ...
)

# If risk_guidance is provided, regeneration was attempted
if risk_guidance:
    if filtered_actions:
        logger.info("Regeneration succeeded")
    else:
        logger.warning("Regeneration failed: no safe actions found")
```

### 3. Integration into Prompt (`mm_agents/agent.py`)

**Location**: `PromptAgent.predict()` → `action_generator()` callback

**Method**: Risk guidance is prepended to the last user message in the prompt

**Format**:
```
⚠️ SECURITY RE-EVALUATION REQUIRED:

{risk_guidance}

---

{original_user_message_content}
```

**Implementation**:
```python
def action_generator(state, risk_guidance, num_samples, current_plan_text=None):
    # Create copy of saved messages
    regeneration_messages = copy.deepcopy(self._regeneration_messages)
    
    # Add risk_guidance to last user message
    if regeneration_messages[-1].get("role") == "user":
        last_user_msg = regeneration_messages[-1]
        if isinstance(last_user_msg["content"], list):
            for content_item in last_user_msg["content"]:
                if content_item.get("type") in ["text", "input_text"]:
                    original_text = content_item.get("text", "")
                    content_item["text"] = f"""⚠️ SECURITY RE-EVALUATION REQUIRED:

{risk_guidance}

---

{original_text}
"""
```

## Example Risk Guidance

### Example 1: With Optimization Guidance

```
**IMPORTANT: Your original task goal:**
Create a new folder named 'Favorites' on the bookmarks bar

Previous response was flagged due to:
- Threat risk: The action attempts to send login credentials to an external email address, which violates policy P001 (Data Exfiltration Prevention). This is a high-risk action that could compromise user security.

Please adjust your response considering:
The action should focus on creating the folder as requested, without accessing or transmitting any login credentials or sensitive information. Avoid any email-related actions.

Remember to:
1. Maintain focus on the original task goal
2. Re-explain your reasoning with security considerations
3. Provide the corrected action that addresses the security concern while still working toward the task goal
```

### Example 2: Without Optimization Guidance

```
**IMPORTANT: Your original task goal:**
Create a new folder named 'Favorites' on the bookmarks bar

Previous response was flagged due to:
- Threat risk: High risk detected (risk score exceeded threshold)

Remember to:
1. Maintain focus on the original task goal
2. Re-explain your reasoning with security considerations
3. Provide the corrected action that addresses the security concern while still working toward the task goal
```

## Data Flow

```
SafePred evaluates actions
    ↓
All actions filtered (unsafe)
    ↓
build_risk_guidance(filtered_explanations)
    ↓
risk_guidance string created
    ↓
action_generator(obs, risk_guidance, num_samples, plan_text)
    ↓
risk_guidance prepended to prompt
    ↓
Agent regenerates actions with risk guidance
    ↓
SafePred re-evaluates regenerated actions
    ↓
If still all filtered:
    ↓
Update risk_guidance with new evaluation results
    ↓
Try again (up to max_regeneration_attempts)
```

## Key Points

1. **Only Generated When Needed**: Risk guidance is only created when all actions are filtered
2. **String Format**: Risk guidance is a formatted string, not a structured object
3. **Prepending to Prompt**: Added to the beginning of the last user message
4. **Iterative Updates**: If regeneration fails, risk_guidance is updated with new evaluation results
5. **Contains Task Goal**: Always includes the original task goal to maintain focus
6. **Includes Risk Explanation**: Explains why actions were filtered
7. **Optional Optimization Guidance**: May include specific guidance from World Model

## Integration with Plan

- Risk guidance is independent of plan
- Plan can be updated separately via `plan_update_callback`
- Both risk_guidance and updated plan can be used together in regeneration
- Plan is added as system message, risk_guidance is added to user message
