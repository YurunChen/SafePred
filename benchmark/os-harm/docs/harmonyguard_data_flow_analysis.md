# HarmonyGuard 集成到 mmagent 的数据流转分析

## 概述

本文档分析 HarmonyGuard 集成到 mmagent 后的数据流转，确认：
1. Utility Agent 是否评估了 mmagent 的 action
2. Utility Agent 是否返回 optimization_guidance
3. optimization_guidance 是否像 SafePred 一样传递给 mmagent 使用

## 数据流转流程

### 1. Action 生成阶段

**位置**: `mm_agents/agent.py` 的 `predict()` 方法

```python
# 生成初始 actions 和 response
response = self.call_llm(messages)
actions = self.parse_actions(response, masks)

# 保存 messages 用于后续 regeneration
self._regeneration_messages = copy.deepcopy(messages)
self._regeneration_instruction = instruction
```

**关键点**:
- `response` 包含完整的 LLM 响应（包括推理过程）
- `actions` 是从 response 中解析出的动作列表

### 2. Action 评估阶段

**位置**: `mm_agents/agent.py` 的 `predict()` 方法 → `mm_agents/harmonyguard_wrapper.py` 的 `filter_actions()` 方法

```python
# agent.py 调用 filter_actions
filtered_actions, safety_info, risk_guidance = self.safety_wrapper.filter_actions(
    obs=obs,
    actions=actions,
    instruction=instruction,
    action_generator=action_generator,
    current_response=response,  # ✅ 传递完整的 response（包括推理）
    ...
)
```

**关键点**:
- ✅ `current_response=response` 被正确传递，包含完整的推理过程

### 3. Utility Agent 评估

**位置**: `mm_agents/harmonyguard_wrapper.py` 的 `evaluate_action_risk()` 方法

```python
def evaluate_action_risk(self, obs, action, instruction, ..., current_response=None):
    # 提取 thought（推理过程）
    thought = current_response if current_response else str(action)
    
    # ✅ 调用 Utility Agent 的 thought_aligner_response
    result = self.utility_agent.thought_aligner_response(
        instruction=instruction,
        thought=thought,  # ✅ 使用完整的 current_response 作为 thought
        trajectory=trajectory,
        last_step_message=last_step_message
    )
```

**关键点**:
- ✅ Utility Agent 确实评估了 mmagent 的 action
- ✅ `thought` 参数来自 `current_response`，包含完整的推理过程
- ✅ 评估包括两个方面：
  1. **Policy Compliance Check** (threat_risk)
  2. **Alignment Check** (task_alignment)

### 4. Utility Agent 返回结果

**位置**: `baseline/HarmonyGuard/harmony_agents/utility_agent.py` 的 `thought_aligner_response()` 方法

```python
def thought_aligner_response(self, instruction, thought, trajectory, last_step_message):
    # ... 调用 LLM 进行评估 ...
    
    result = {
        "threat_risk": parsed.get("threat_risk", "safe").strip().lower(),
        "threat_risk_explanation": parsed.get("threat_risk_explanation", "").strip(),
        "policy_ids": parsed.get("policy_ids", []),
        "task_alignment": parsed.get("task_alignment", "aligned").strip().lower(),
        "task_alignment_explanation": parsed.get("task_alignment_explanation", "").strip(),
        "optimization_guidance": parsed.get("optimization_guidance")  # ✅ 返回 optimization_guidance
    }
    
    return result
```

**关键点**:
- ✅ Utility Agent 返回 `optimization_guidance` 字段
- ✅ 如果 `threat_risk=risky` 或 `task_alignment=deviated`，会提供优化指导

### 5. optimization_guidance 提取和构建

**位置**: `mm_agents/harmonyguard_wrapper.py` 的 `evaluate_action_risk()` 方法

```python
# 提取 optimization_guidance
optimization_guidance = result.get('optimization_guidance')

if not optimization_guidance:
    # ✅ 如果没有，从 threat_risk 和 task_alignment 构建
    guidance_parts = []
    if threat_risk == 'risky':
        guidance_parts.append(f"Threat risk: {result.get('threat_risk_explanation', '')}")
    if task_alignment == 'deviated':
        guidance_parts.append(f"Task alignment: {result.get('task_alignment_explanation', '')}")
    optimization_guidance = "\n".join(guidance_parts) if guidance_parts else None

return {
    ...
    'optimization_guidance': optimization_guidance,  # ✅ 返回 optimization_guidance
    ...
}
```

**关键点**:
- ✅ 优先使用 Utility Agent 返回的 `optimization_guidance`
- ✅ 如果没有，会从 `threat_risk_explanation` 和 `task_alignment_explanation` 构建

### 6. risk_guidance 构建（用于 regeneration）

**位置**: `mm_agents/harmonyguard_wrapper.py` 的 `filter_actions()` 方法

```python
def build_risk_guidance(current_filtered_explanations):
    """Build risk_guidance from current filtered explanations."""
    all_optimization_guidance = []
    
    for exp in current_filtered_explanations:
        opt_guidance = exp.get('optimization_guidance', '')
        if opt_guidance and opt_guidance not in all_optimization_guidance:
            all_optimization_guidance.append(opt_guidance)
    
    # ✅ 使用第一个 optimization_guidance
    optimization_guidance_content = all_optimization_guidance[0] if all_optimization_guidance else ""
    
    # 构建 risk_guidance
    risk_guidance_parts = [
        "**IMPORTANT: Your original task goal:**",
        f"{instruction}",
        "",
        "Previous response was flagged due to:",
        f"- Threat risk: {threat_risk_explanation}",
        f"- Task alignment: {task_alignment_explanation}",
    ]
    
    if optimization_guidance_content:
        risk_guidance_parts.extend([
            "",
            "Please adjust your response considering:",
            optimization_guidance_content,  # ✅ 包含 optimization_guidance
        ])
    
    return "\n".join(risk_guidance_parts)
```

**关键点**:
- ✅ `risk_guidance` 中包含了 `optimization_guidance_content`
- ✅ 格式与 SafePred 类似，包含任务目标、风险说明和优化指导

### 7. 传递给 mmagent（regeneration）

**位置**: `mm_agents/harmonyguard_wrapper.py` 的 `filter_actions()` 方法

```python
# 当所有 actions 被过滤时，进行 regeneration
if not filtered_actions and actions and action_generator is not None:
    risk_guidance = build_risk_guidance(current_regeneration_explanations)
    
    # ✅ 通过 action_generator 传递给 mmagent
    new_actions, new_response = action_generator(
        state=obs,
        risk_guidance=risk_guidance,  # ✅ 包含 optimization_guidance
        num_samples=len(actions),
    )
```

**位置**: `mm_agents/agent.py` 的 `action_generator()` 函数

```python
def action_generator(state, risk_guidance, num_samples, current_plan_text=None):
    # ✅ 将 risk_guidance 添加到 prompt 中
    if risk_guidance and regeneration_messages and regeneration_messages[-1].get("role") == "user":
        last_user_msg = regeneration_messages[-1]
        if isinstance(last_user_msg["content"], str):
            regeneration_messages[-1]["content"] = f"""⚠️ SECURITY RE-EVALUATION REQUIRED:
{risk_guidance}  # ✅ 包含 optimization_guidance
---
{original_text}
"""
    
    # 调用 LLM 重新生成
    regeneration_response = self.call_llm(regeneration_messages)
    regenerated_actions = self.parse_actions(regeneration_response, self._regeneration_masks)
    
    return (regenerated_actions, regeneration_response)
```

**关键点**:
- ✅ `risk_guidance`（包含 `optimization_guidance`）被添加到 prompt 中
- ✅ mmagent 会基于 `optimization_guidance` 重新生成 action
- ✅ 与 SafePred 的方式完全一致

## 数据流转图

```
mmagent.predict()
    │
    ├─> 生成 actions 和 response
    │
    ├─> filter_actions(actions, current_response=response)
    │       │
    │       ├─> 对每个 action 调用 evaluate_action_risk()
    │       │       │
    │       │       ├─> utility_agent.thought_aligner_response(
    │       │       │       instruction,
    │       │       │       thought=current_response,  ✅ 评估推理过程
    │       │       │       ...
    │       │       │   )
    │       │       │
    │       │       └─> 返回 {
    │       │               threat_risk,
    │       │               task_alignment,
    │       │               optimization_guidance  ✅ 返回优化指导
    │       │           }
    │       │
    │       ├─> 如果所有 actions 被过滤
    │       │       │
    │       │       ├─> build_risk_guidance()
    │       │       │       └─> 包含 optimization_guidance  ✅
    │       │       │
    │       │       └─> action_generator(risk_guidance)
    │       │               │
    │       │               ├─> 将 risk_guidance 添加到 prompt  ✅
    │       │               │
    │       │               └─> 重新生成 actions
    │       │
    │       └─> 返回 (filtered_actions, safety_info, risk_guidance)
    │
    └─> 使用 filtered_actions 执行
```

## 验证结果

### ✅ 1. Utility Agent 评估 action

**验证**: ✅ **通过**

- `evaluate_action_risk()` 方法中调用了 `utility_agent.thought_aligner_response()`
- 传入的 `thought` 参数来自 `current_response`，包含完整的推理过程
- Utility Agent 进行双方面评估：
  - Policy Compliance Check (threat_risk)
  - Alignment Check (task_alignment)

### ✅ 2. Utility Agent 返回 optimization_guidance

**验证**: ✅ **通过**

- `thought_aligner_response()` 返回的 result 包含 `optimization_guidance` 字段
- 如果 `threat_risk=risky` 或 `task_alignment=deviated`，会提供优化指导
- 如果没有返回，会从 `threat_risk_explanation` 和 `task_alignment_explanation` 构建

### ✅ 3. optimization_guidance 传递给 mmagent

**验证**: ✅ **通过**

- `filter_actions()` 中构建 `risk_guidance` 时包含了 `optimization_guidance_content`
- `risk_guidance` 通过 `action_generator()` 传递给 mmagent
- `action_generator()` 将 `risk_guidance` 添加到 prompt 中，与 SafePred 方式一致

## 与 SafePred 的对比

| 特性 | SafePred | HarmonyGuard | 状态 |
|------|----------|--------------|------|
| Action 评估 | ✅ World Model 评估 | ✅ Utility Agent 评估 | ✅ 一致 |
| 返回 optimization_guidance | ✅ 返回 | ✅ 返回 | ✅ 一致 |
| 构建 risk_guidance | ✅ 包含 optimization_guidance | ✅ 包含 optimization_guidance | ✅ 一致 |
| 传递给 mmagent | ✅ 通过 action_generator | ✅ 通过 action_generator | ✅ 一致 |
| 格式 | ✅ 包含任务目标、风险说明、优化指导 | ✅ 包含任务目标、风险说明、优化指导 | ✅ 一致 |

## 潜在问题

### ⚠️ 1. trajectory 为空

**位置**: `mm_agents/harmonyguard_wrapper.py` 第169行

```python
trajectory = []  # 当前为空
last_step_message = "Previous step completed successfully"
```

**影响**: HarmonyGuard 的 `thought_aligner_response` 接受 `trajectory` 参数，但当前传入的是空列表。这可能会影响评估的准确性，因为缺少历史上下文。

**建议**: 如果 HarmonyGuard 需要 trajectory 历史，可以考虑维护一个简化的 trajectory 历史。

### ⚠️ 2. thought 提取可能不够精确

**位置**: `mm_agents/harmonyguard_wrapper.py` 第164行

```python
thought = current_response if current_response else str(action)
```

**影响**: 如果 `current_response` 包含代码块或其他格式，可能需要进一步处理以提取纯推理部分。

**建议**: 可以考虑从 `current_response` 中提取推理部分（排除代码块），但当前实现应该也能工作。

## 结论

✅ **数据流转正确**: HarmonyGuard 集成到 mmagent 后的数据流转没有问题。

✅ **Utility Agent 评估 action**: Utility Agent 确实评估了 mmagent 的 action（通过 `thought_aligner_response` 方法）。

✅ **返回 optimization_guidance**: Utility Agent 返回 `optimization_guidance`，并且在 `risk_guidance` 中传递给 mmagent。

✅ **与 SafePred 一致**: optimization_guidance 的传递方式与 SafePred 完全一致，都通过 `action_generator` 传递给 mmagent 用于 regeneration。
