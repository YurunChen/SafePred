# Execution Trajectory 字段说明

## 当前已包含的字段

### 顶层字段
- `total_duration_seconds`: 整个任务的总执行时间（秒）
- `total_duration_minutes`: 整个任务的总执行时间（分钟）
- `steps`: 步骤列表

### 每个步骤 (step) 的字段
- `step_idx`: 步骤索引（从 0 开始）
- `timestamp`: 步骤时间戳（格式：YYYYMMDD@HHMMSS）
- `response`: Agent 的完整响应（包含推理过程）
- `actions`: 该步骤生成的所有 actions 列表
- `screenshot_file`: 该步骤的截图文件路径（执行 action 前的状态）
- `a11y_tree`: 该步骤的可访问性树（执行 action 前的状态）
- `world_model_output`: World Model 的预测输出（如果启用 SafePred）
  - `semantic_delta`: 短期预测（UI 状态变化）
  - `element_changes`: 元素变化详情
  - `long_term_impact`: 长期影响评估
  - `risk_explanation`: 风险解释
  - `violated_policy_ids`: 违反的政策 ID 列表
  - `optimization_guidance`: 优化建议
  - `risk_score`: 风险分数
- `safety_info`: 安全评估信息（从 SafetyWrapper）
  - `action_0`, `action_1`, ...: 每个 action 的安全评估
    - `risk_score`: 风险分数
    - `risk_explanation`: 风险解释
    - `is_safe`: 是否安全
    - `world_model_output`: World Model 输出（如果可用）
- `token_usage`: Agent 的 LLM API 调用的 token 使用情况
  - `model`: 使用的模型名称
  - `prompt_tokens`: Prompt tokens 数量
  - `completion_tokens`: Completion tokens 数量
  - `total_tokens`: 总 tokens 数量
- `world_model_token_usage`: World Model 的 LLM API 调用的 token 使用情况（SafePred）
  - `model`: 使用的模型名称
  - `prompt_tokens`: Prompt tokens 数量
  - `completion_tokens`: Completion tokens 数量
  - `total_tokens`: 总 tokens 数量
- `harmonyguard_token_usage`: HarmonyGuard UtilityAgent 的 LLM 调用的 token 使用情况（当 safety_system 为 harmonyguard 时）
  - `model`: 使用的模型名称
  - `prompt_tokens`: Prompt tokens 数量（本步内多次评估的合计）
  - `completion_tokens`: Completion tokens 数量
  - `total_tokens`: 总 tokens 数量
- `reactive_token_usage`: Reactive Agent 的 LLM 调用的 token 使用情况（当 safety_system 为 reactive 时）
  - `model`: 使用的模型名称
  - `prompt_tokens`: Prompt tokens 数量（本步内多次评估的合计）
  - `completion_tokens`: Completion tokens 数量
  - `total_tokens`: 总 tokens 数量
- `step_duration_seconds`: 该步骤的执行时间（秒）

## 可以添加的其他字段

### 1. 任务元数据（顶层）
```json
{
  "task_id": "任务 ID",
  "task_instruction": "任务指令",
  "task_params": {
    "platform": "ubuntu",
    "model": "gpt-4o",
    "action_space": "pyautogui",
    "observation_type": "a11y_tree",
    ...
  },
  "injection": {...},
  "jailbreak": false
}
```

### 2. 每个步骤的额外信息
```json
{
  "step_idx": 0,
  ...
  // 执行结果信息
  "action_results": [
    {
      "action": "具体 action 内容",
      "action_timestamp": "20260123@120000",
      "reward": 0.0,
      "done": false,
      "info": {},
      "action_duration_seconds": 1.23,
      "next_state_a11y_tree": "执行后的 a11y_tree",
      "next_state_screenshot_file": "step_1_xxx.png"
    }
  ],
  // 或者简化为单个 action 的结果（如果每步只有一个 action）
  "reward": 0.0,
  "done": false,
  "info": {},
  "action_timestamp": "20260123@120000",
  "next_state_a11y_tree": "...",
  "next_state_screenshot_file": "..."
}
```

### 3. 计划信息（如果启用 planning）
```json
{
  "step_idx": 0,
  ...
  "plan": {
    "current_plan": "当前计划文本",
    "plan_steps": ["步骤1", "步骤2", ...]
  }
}
```

### 4. 错误信息
```json
{
  "step_idx": 0,
  ...
  "errors": [
    {
      "type": "error_type",
      "message": "error message",
      "timestamp": "20260123@120000"
    }
  ]
}
```

### 5. 累积统计信息
```json
{
  "step_idx": 0,
  ...
  "cumulative_stats": {
    "total_tokens_used": 12345,
    "total_world_model_tokens_used": 6789,
    "total_harmonyguard_tokens_used": 0,
    "total_reactive_tokens_used": 0,
    "total_steps": 5,
    "total_actions_executed": 8
  }
}
```

### 6. 环境状态信息
```json
{
  "step_idx": 0,
  ...
  "env_state": {
    "step_number": 1,
    "action_history_length": 5
  }
}
```

## 建议添加的字段（按优先级）

### 高优先级
1. **任务元数据**（顶层）- 便于识别和分析任务
2. **action_results** - 记录每个 action 的执行结果（reward, done, info）
3. **next_state_a11y_tree** - 执行 action 后的状态（当前只有执行前的）
4. **action_timestamp** - 每个 action 的时间戳（已在 traj.jsonl 中，但未在 execution_trajectory.json 中）

### 中优先级
5. **plan** - 如果启用了 planning 功能
6. **errors** - 错误信息（如果有）
7. **cumulative_stats** - 累积统计信息

### 低优先级
8. **env_state** - 环境状态信息
9. **action_duration_seconds** - 每个 action 的执行时间（如果每步有多个 actions）

## 注意事项

1. **数据大小**: 添加太多信息可能会使 JSON 文件变得很大，特别是 `a11y_tree` 和截图路径
2. **向后兼容**: 添加新字段时应该保持向后兼容，使用可选字段
3. **性能**: 频繁写入 JSON 文件可能影响性能，考虑批量写入或异步写入
4. **隐私**: 确保不记录敏感信息（如密码、API keys 等）
