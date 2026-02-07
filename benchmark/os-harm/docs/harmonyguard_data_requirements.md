# HarmonyGuard Utility Agent 数据需求分析

## Utility Agent 需要的参数

`thought_aligner_response` 方法需要以下参数：

```python
def thought_aligner_response(self, instruction, thought, trajectory, last_step_message):
```

### 1. `instruction` (任务指令)
- **用途**: 在 prompt 中作为 "TASK OBJECTIVE"
- **当前状态**: ✅ **已正确传递**
- **来源**: `evaluate_action_risk()` 方法的 `instruction` 参数

### 2. `thought` (Agent 推理过程)
- **用途**: 在 prompt 中作为 "AGENT REASONING"，用于评估推理和动作
- **当前状态**: ✅ **已传递，但可能需要优化**
- **来源**: `current_response`（完整的 LLM 响应，包括推理过程）
- **当前实现**:
  ```python
  thought = current_response if current_response else str(action)
  ```
- **潜在问题**: 
  - `current_response` 可能包含代码块，需要确认是否应该提取纯推理部分
  - 如果 `current_response` 为空，回退到 `str(action)`，这可能不够完整

### 3. `trajectory` (轨迹历史)
- **用途**: 在方法签名中，但在当前 prompt 构建中**没有直接使用**
- **当前状态**: ❌ **传递的是空列表**
- **当前实现**:
  ```python
  trajectory = []  # 空列表
  ```
- **分析**:
  - 从 `_get_prompt_with_evaluation` 方法来看，`trajectory` 参数没有被使用
  - 但在某些场景下，历史上下文可能有助于更准确的评估
  - 如果需要，可以从 `agent.actions`、`agent.thoughts`、`agent.observations` 构建

### 4. `last_step_message` (上一步消息)
- **用途**: 在 prompt 中用于 "taking into account the previous step"
- **当前状态**: ⚠️ **传递的是固定字符串，不够准确**
- **当前实现**:
  ```python
  last_step_message = "Previous step completed successfully"
  ```
- **问题**:
  - 固定字符串无法反映实际的上一步执行结果
  - 应该传递实际的上一步动作和执行结果
- **建议**:
  - 如果有上一步的动作，应该传递类似 "Previous action: {action}, Result: {result}" 的信息
  - 如果是第一步，可以传递 "This is the first step" 或类似信息

## Prompt 构建分析

从 `_get_prompt_with_evaluation` 方法来看，实际使用的数据：

1. **`policy_content`**: 从 `self._formatted_policy_content` 获取（已初始化）
2. **`instruction`**: ✅ 已传递
3. **`thought`**: ✅ 已传递
4. **`last_step_message`**: ⚠️ 已传递但不够准确
5. **`additional_guideline`**: 固定字符串（在方法内部设置）

**注意**: `trajectory` 参数在 prompt 构建中**没有被使用**，但保留在方法签名中可能是为了未来扩展。

## 当前数据传递状态

| 参数 | 需要 | 已传递 | 准确性 | 备注 |
|------|------|--------|--------|------|
| `instruction` | ✅ | ✅ | ✅ 准确 | 任务指令正确传递 |
| `thought` | ✅ | ✅ | ⚠️ 可能需优化 | 从 `current_response` 提取，可能包含代码块 |
| `trajectory` | ❓ | ❌ | ❌ 空列表 | Prompt 中未使用，但可能对某些场景有用 |
| `last_step_message` | ✅ | ⚠️ | ❌ 固定字符串 | 应该传递实际的上一步信息 |

## 改进建议

### 1. 优化 `thought` 提取
如果 `current_response` 包含代码块，可以考虑提取纯推理部分：

```python
# 提取推理部分（排除代码块）
def extract_thought(response: str) -> str:
    # 移除代码块，保留推理文本
    import re
    # 移除 ```python ... ``` 等代码块
    thought = re.sub(r'```[\w]*\n.*?```', '', response, flags=re.DOTALL)
    return thought.strip() or str(action)
```

### 2. 改进 `last_step_message`
如果有历史信息，应该传递实际的上一步结果：

```python
# 如果有上一步动作
if hasattr(self, 'last_action') and self.last_action:
    last_step_message = f"Previous action: {self.last_action}, Result: {self.last_result or 'completed'}"
elif hasattr(self, 'step_count') and self.step_count == 0:
    last_step_message = "This is the first step"
else:
    last_step_message = "Previous step completed successfully"
```

### 3. 构建 `trajectory`（可选）
如果需要历史上下文，可以从 agent 的历史构建：

```python
# 从 agent 历史构建 trajectory（如果可用）
trajectory = []
if hasattr(self, 'agent') and hasattr(self.agent, 'actions'):
    # 构建简化的轨迹历史
    for i, (action, thought, obs) in enumerate(zip(
        self.agent.actions[-3:],  # 最近3步
        self.agent.thoughts[-3:],
        self.agent.observations[-3:]
    )):
        trajectory.append({
            'step': i + 1,
            'action': str(action),
            'thought': str(thought)[:200],  # 截断
            'observation': '...'  # 简化
        })
```

## 结论

✅ **核心数据已传递**: `instruction` 和 `thought` 已正确传递，这是评估所需的核心数据。

⚠️ **可优化项**:
1. `last_step_message` 应该传递实际的上一步信息，而不是固定字符串
2. `thought` 提取可能需要优化，排除代码块
3. `trajectory` 虽然当前未使用，但可以考虑构建以提供历史上下文

🔍 **需要确认**:
- `current_response` 的格式是否包含代码块，是否需要提取纯推理部分
- 是否有可用的上一步动作和执行结果信息
- 是否需要构建 `trajectory` 历史（虽然 prompt 中未使用）
