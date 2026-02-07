# Execution Trajectory 数据流转说明

## 文件位置

所有数据都保存在任务文件夹下的 `execution_trajectory.json` 文件中：
```
{example_result_dir}/execution_trajectory.json
```

其中 `example_result_dir` 的路径结构通常是：
```
{result_dir}/{action_space}/{observation_type}/{model_name}/{domain}/{example_id}/
```

## 数据流转时序

### 1. 任务开始阶段
**调用时机**：任务开始，环境初始化后
**函数**：`save_task_metadata(example_result_dir, task_metadata)`
**保存内容**：
```json
{
  "task_metadata": {
    "task_id": "...",
    "task_instruction": "...",
    "task_params": {...},
    "injection": {...},
    "jailbreak": false
  },
  "steps": []
}
```

### 2. 每个步骤执行阶段
**调用时机**：每个步骤的所有 actions 执行完成后
**函数**：`save_execution_trajectory_log(...)`
**保存内容**：追加步骤数据到 `steps` 数组
```json
{
  "task_metadata": {...},  // 保留已有数据
  "steps": [
    {
      "step_idx": 0,
      "timestamp": "...",
      "response": "...",
      "actions": [...],
      "screenshot_file": "...",
      "a11y_tree": "...",
      "world_model_output": {...},
      "safety_info": {...},
      "token_usage": {...},
      "world_model_token_usage": {...},
      "step_duration_seconds": 2.34,
      "action_results": [
        {
          "action": "...",
          "action_timestamp": "...",
          "reward": 0.0,
          "done": false,
          "info": {},
          "action_duration_seconds": 1.23
        }
      ],
      "next_state_a11y_tree": "...",
      "plan": {
        "current_plan": "..."
      },
      "errors": [...]
    }
  ]
}
```

### 3. 任务结束阶段
**调用时机**：所有步骤执行完成，任务结束
**函数**：`save_total_duration(example_result_dir, total_duration, cumulative_stats)`
**保存内容**：添加总持续时间和累积统计
```json
{
  "task_metadata": {...},  // 保留已有数据
  "steps": [...],  // 保留已有数据
  "total_duration_seconds": 45.67,
  "total_duration_minutes": 0.76,
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

## 数据完整性保证

### 1. 文件加载机制
所有三个保存函数都使用相同的加载逻辑：
- 如果文件存在，尝试加载现有数据
- 如果加载失败，创建新的空结构（但会丢失已有数据，这是异常情况）
- 确保 `steps` 键存在

### 2. 数据保留机制
- `save_task_metadata`: 保留已有的 `steps` 数据
- `save_execution_trajectory_log`: 保留已有的 `task_metadata` 和其他顶层字段
- `save_total_duration`: 保留已有的 `task_metadata` 和 `steps` 数据

### 3. 数据更新机制
- `task_metadata`: 在任务开始时设置一次，后续不会更新
- `steps`: 每个步骤追加一个新条目
- `total_duration_*`: 在任务结束时设置
- `cumulative_stats`: 在任务结束时设置

## 数据收集流程

### 步骤数据收集
1. **步骤开始**：记录 `step_start_time`
2. **Agent 预测**：调用 `agent.predict()` 获取 `response` 和 `actions`
3. **获取信息**：
   - `world_model_output`: 从 World Model 获取
   - `safety_info`: 从 Agent 获取
   - `token_usage`: 从 Agent 获取
   - `world_model_token_usage`: 从 World Model 获取（SafePred）
   - `harmonyguard_token_usage`: 从 HarmonyGuard Wrapper 获取（当 safety_system 为 harmonyguard）
   - `reactive_token_usage`: 从 Reactive Wrapper 获取（当 safety_system 为 reactive）
   - `plan`: 从 `agent.current_plan` 获取
4. **执行 Actions**：
   - 对每个 action 执行 `env.step()`
   - 收集 `action_results`（包含 reward, done, info, timestamp, duration）
   - 收集 `errors`（如果有）
   - 获取 `next_state_a11y_tree`（最后一个 action 执行后）
5. **步骤结束**：计算 `step_duration`，更新 `cumulative_stats`
6. **保存步骤数据**：调用 `save_execution_trajectory_log()`

### 累积统计更新
- `total_tokens_used`: 每个步骤累加 Agent 的 token 使用
- `total_world_model_tokens_used`: 每个步骤累加 World Model 的 token 使用（SafePred）
- `total_harmonyguard_tokens_used`: 每个步骤累加 HarmonyGuard UtilityAgent 的 token 使用（harmonyguard）
- `total_reactive_tokens_used`: 每个步骤累加 Reactive Agent 的 token 使用（reactive）
- `total_steps`: 每个步骤递增，任务结束时设置为最终值
- `total_actions_executed`: 每个 action 执行时递增

## 注意事项

1. **文件路径**：所有函数都使用 `os.path.join(example_result_dir, "execution_trajectory.json")`，确保文件保存在正确的任务文件夹下

2. **数据覆盖**：由于每个函数都会加载现有文件并保留已有数据，不会出现数据覆盖问题

3. **异常处理**：如果文件加载失败，会创建新的空结构，但会记录警告日志

4. **数据完整性**：最终 JSON 文件包含：
   - `task_metadata`: 任务元数据
   - `steps`: 所有步骤的详细数据
   - `total_duration_seconds`: 总执行时间（秒）
   - `total_duration_minutes`: 总执行时间（分钟）
   - `cumulative_stats`: 累积统计信息

5. **文件编码**：使用 UTF-8 编码，支持中文等非 ASCII 字符

## 已知问题与修复

### screenshot_file 与实际保存文件名不一致（已修复）

**问题**：`step_data.screenshot_file` 使用当前步骤的 `action_timestamp` 构造（如 `step_1_{当前步action_ts}.png`），但 step N 的「执行前」截图是在执行 step N-1 的 action 时保存的，实际文件名为 `step_N_{上一步action_ts}.png`。两者时间戳不同，导致记录的 `screenshot_file` 与 `traj.jsonl` 及磁盘上的实际文件不一致。

**修复**：引入 `last_screenshot_file`，在每次保存截图后更新。step 0 固定为 `"step_0.png"`；step N（N>0）的 `screenshot_file` 使用上一迭代保存时的 `last_screenshot_file`，与真正写入的文件名一致。
