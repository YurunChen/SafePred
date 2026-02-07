# Reactive Data Flow Analysis

## Complete Data Flow from run.py to reactive_agent

### 1. Initialization Flow

```
run.py
  └─> PromptAgent.__init__(
        enable_safety_check=True,
        safety_system="reactive",
        reactive_policy_path="...",
        reactive_config_path="..."
      )
      └─> mm_agents/agent.py:481-491
          └─> ReactiveWrapper.__init__(
                enabled=True,
                policy_path=reactive_policy_path,
                config_path=reactive_config_path,
                max_regeneration_attempts=2
              )
              └─> baseline/reactive/wrapper.py:38-90
                  └─> ReactiveAgent.__init__(
                        policy_path=policy_path,
                        config=config
                      )
                      └─> baseline/reactive/reactive_agent.py:18-67
                          ├─> PolicyLoader(policy_path)
                          └─> LLMClient(...)
```

### 2. Action Evaluation Flow (Each Step)

```
lib_run_single.py:254
  └─> agent.predict(instruction, obs)
      └─> mm_agents/agent.py:536-1217
          ├─> Generate actions using LLM
          ├─> response = all_responses[0]  # Line 867: Full response with reasoning
          └─> if self.safety_wrapper and actions:  # Line 900
              └─> self.safety_wrapper.filter_actions(
                    obs=obs,
                    actions=actions,
                    instruction=instruction,
                    action_generator=action_generator,
                    current_response=response,  # Line 1103: Pass full response
                    ...
                  )
                  └─> baseline/reactive/wrapper.py:251-506
                      └─> for i, action in enumerate(actions):  # Line 303
                          └─> self.evaluate_action_risk(
                                obs=obs,
                                action=action,
                                instruction=instruction,
                                current_response=current_response,  # Line 313
                                ...
                              )
                              └─> baseline/reactive/wrapper.py:98-249
                                  ├─> Extract thought from current_response  # Line 155-180
                                  └─> self.reactive_agent.evaluate_action(
                                        instruction=instruction,
                                        thought=thought,  # Extracted reasoning
                                        last_step_message="..."
                                      )
                                      └─> baseline/reactive/reactive_agent.py:69-183
                                          ├─> Build prompt with policy + instruction + thought
                                          └─> self.llm_client.generate(prompt)
                                              └─> baseline/reactive/llm_client.py
```

### 3. Key Data Points

#### 3.1 `current_response` Format

From `mm_agents/agent.py:867`:
- `response = all_responses[0]` - This is the raw LLM response
- For most models: `response` is a **string** containing the full response
- For `computer-use-preview` model: `response` is a **list** of dicts with `type` and `text` fields

#### 3.2 Thought Extraction in Reactive

From `baseline/reactive/wrapper.py:155-180`:
- Handles string, list, and dict formats
- For list format: Extracts `reasoning` and `text` type items
- Falls back to `str(action)` if no response provided

#### 3.3 Comparison with SafePred

SafePred's `evaluate_action_risk` (mm_agents/safety_wrapper.py:200-324):
- Also receives `current_response` parameter
- Passes it directly to SafePredWrapper's `evaluate_action_risk`
- SafePredWrapper extracts reasoning internally

Reactive's approach:
- Extracts thought in `wrapper.py` before calling `reactive_agent.evaluate_action`
- More explicit extraction logic for different response formats

### 4. Potential Issues

#### 4.1 Response Format Handling

**Issue**: `current_response` might be in different formats:
- String (most models)
- List of dicts (computer-use-preview)
- Empty/None

**Current Fix**: Added comprehensive extraction logic in `wrapper.py:155-180`

#### 4.2 Logging Visibility

**Issue**: Logs might not be visible if logger level is too high

**Current Fix**: Added `print()` statements alongside `logger.info()` for critical paths

#### 4.3 Missing Evaluation Logs

**Issue**: Evaluation results not printed for each step

**Current Fix**: Added detailed logging at:
- `filter_actions` entry
- `evaluate_action_risk` entry
- Evaluation result (JSON format)
- Action filtering decision

### 5. Verification Checklist

- [x] `run.py` passes `reactive_policy_path` and `reactive_config_path`
- [x] `PromptAgent.__init__` creates `ReactiveWrapper`
- [x] `ReactiveWrapper.__init__` creates `ReactiveAgent`
- [x] `agent.predict()` calls `safety_wrapper.filter_actions()`
- [x] `filter_actions()` calls `evaluate_action_risk()` for each action
- [x] `evaluate_action_risk()` extracts thought from `current_response`
- [x] `reactive_agent.evaluate_action()` receives instruction + thought
- [x] Evaluation results are logged in JSON format
- [x] Action filtering decisions are logged

### 6. Next Steps

1. Verify `current_response` format matches expected format
2. Ensure all logs are visible (check logger configuration)
3. Test with different model types (string vs list response)
4. Verify evaluation results are printed for each step
