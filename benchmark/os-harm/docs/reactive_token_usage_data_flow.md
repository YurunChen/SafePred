# Reactive Token Usage 数据流转验证

本文档验证 Reactive 系统的 token usage 数据流转是否正确。

## 数据流转路径

### 1. LLM API 调用 → LLMClient
**位置**: `baseline/reactive/llm_client.py`

- `_generate_openai()`: 从 OpenAI SDK 响应中提取 `response.usage`，存储到 `self._last_token_usage`
- `_generate_qwen()`: 从 Qwen SDK 响应中提取 usage，存储到 `self._last_token_usage`
- `_generate_custom()`: 从 requests 响应 JSON 中提取 `result["usage"]`，存储到 `self._last_token_usage`

**存储格式**:
```python
{
    "model": self.model_name,
    "prompt_tokens": ...,
    "completion_tokens": ...,
    "total_tokens": ...
}
```

### 2. LLMClient → ReactiveAgent
**位置**: `baseline/reactive/reactive_agent.py`

- `evaluate_action()`: 调用 `self.llm_client.generate()` 后，从 `llm_client._last_token_usage` 获取并存储到 `self._last_token_usage`

### 3. ReactiveAgent → ReactiveWrapper
**位置**: `baseline/reactive/wrapper.py`

- `evaluate_action_risk()`: 调用 `reactive_agent.evaluate_action()`
- `filter_actions()`: 
  - 每次 `evaluate_action_risk()` 后，从 `reactive_agent._last_token_usage` 获取并追加到 `_step_reactive_token_usages` 列表
  - 在主循环结束后（第401行），调用 `_aggregate_step_reactive_token_usage()` 聚合所有 token usage
  - 在 regeneration 循环中，每次重评估后也追加 token usage（第517行）
  - 在所有返回点前重新聚合（第542行、第561行），确保包含所有评估的 token usage

**聚合逻辑**:
```python
def _aggregate_step_reactive_token_usage(self) -> Optional[Dict[str, Any]]:
    usages = self._step_reactive_token_usages
    if not usages:
        return None
    return {
        "model": usages[0].get("model", "reactive_agent"),
        "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usages),
        "completion_tokens": sum(u.get("completion_tokens", 0) for u in usages),
        "total_tokens": sum(u.get("total_tokens", 0) for u in usages),
    }
```

### 4. ReactiveWrapper → lib_run_single
**位置**: `lib_run_single.py`

- `get_reactive_token_usage(agent)`: 从 `agent.safety_wrapper._last_step_reactive_token_usage` 获取聚合后的 token usage
- 在每步执行后（第707行）获取 `reactive_token_usage`
- 累加到 `cumulative_stats['total_reactive_tokens_used']`（第920-921行）
- 传递给 `save_execution_trajectory_log()`（第933行）

### 5. lib_run_single → execution_trajectory.json
**位置**: `lib_run_single.py` → `save_execution_trajectory_log()`

- 将 `reactive_token_usage` 添加到 `step_entry['reactive_token_usage']`（第374-375行）
- 保存到 `execution_trajectory.json` 的 `steps` 数组中
- 在任务结束时，`cumulative_stats` 包含 `total_reactive_tokens_used`，保存到顶层

## 返回路径验证

### ReactiveWrapper.filter_actions() 的所有返回路径：

1. **早期返回（disabled/no agent）**（第313行、第317行）:
   - ✅ 设置 `_last_step_reactive_token_usage = None`

2. **主循环结束后（无 regeneration）**（第401行 → 第564行）:
   - ✅ 第401行：聚合初始评估的 token usage
   - ✅ 第564行：返回（此时已有值）

3. **Regeneration 成功**（第542行 → 第543行）:
   - ✅ 第517行：收集 regeneration 评估的 token usage
   - ✅ 第542行：重新聚合（包含初始 + regeneration）
   - ✅ 第543行：返回

4. **Regeneration 失败**（第561行 → 第562行）:
   - ✅ 第517行：收集 regeneration 评估的 token usage
   - ✅ 第561行：重新聚合（包含初始 + regeneration）
   - ✅ 第562行：返回

## 数据完整性检查

### ✅ 初始化
- `LLMClient.__init__`: `self._last_token_usage = None`
- `ReactiveAgent.__init__`: `self._last_token_usage = None`
- `ReactiveWrapper.__init__`: `self._step_reactive_token_usages = []`, `self._last_step_reactive_token_usage = None`
- `lib_run_single`: `cumulative_stats['total_reactive_tokens_used'] = 0`

### ✅ 异常处理
- LLM API 调用失败时，`_last_token_usage` 保持为 `None` 或上次的值
- `get_reactive_token_usage()` 有异常处理，返回 `None` 而不是崩溃

### ✅ 数据保存
- 每步的 `reactive_token_usage` 保存到 `steps[i]['reactive_token_usage']`
- 累积统计保存到 `cumulative_stats['total_reactive_tokens_used']`

## 验证结果

✅ **数据流转完整**: 从 LLM API 响应 → LLMClient → ReactiveAgent → ReactiveWrapper → lib_run_single → execution_trajectory.json

✅ **所有返回路径覆盖**: 所有 `filter_actions()` 的返回路径都正确设置了 `_last_step_reactive_token_usage`

✅ **聚合逻辑正确**: 正确聚合每步内所有 LLM 调用的 token usage（包括初始评估和 regeneration 重评估）

✅ **数据保存完整**: 每步数据和累积统计都正确保存到 trajectory JSON

✅ **异常处理完善**: 所有关键点都有异常处理，不会因 token usage 获取失败而崩溃

## 使用场景

- **正常流程**: 每步评估 action → 收集 token usage → 聚合 → 保存到 trajectory
- **Regeneration 流程**: 初始评估 + regeneration 重评估 → 收集所有 token usage → 聚合 → 保存到 trajectory
- **无评估场景**: 如果某步没有调用 LLM（disabled 或异常），`reactive_token_usage` 为 `None`，不会保存到 trajectory
