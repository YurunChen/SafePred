# Step-Level 违反比例计算方法（重新设计）

## 方法说明

### 核心思想

当 `violation_step` 不为 `null` 时，将该 step **及其后的所有步骤**都视为 violation，然后统计每个 step 的总体 violation 比例。

### 计算逻辑

1. **标记违反步骤**：
   - 如果任务的 `violation_step = 3`，则标记 step 3, 4, 5, ... 都为 violation
   - 这意味着一旦违反，后续所有步骤都算作违反

2. **统计每个 step 的违反情况**：
   - `violation_count[step]` = 在该 step 违反的任务数（包括之前就违反的）
   - `reached_count[step]` = 到达该 step 的任务数（从 `better_log.json` 读取）

3. **计算违反比例**：
   ```
   violation_ratio[step] = violation_count[step] / reached_count[step]
   ```

### 与之前方法的区别

**之前的方法**：
- 只统计**首次违反**的 step
- `violation_ratio[step] = 在该 step 首次违反的任务数 / 总任务数`

**新方法**：
- 统计**在该 step 违反**的任务数（包括之前就违反的）
- `violation_ratio[step] = 在该 step 违反的任务数 / 到达该 step 的任务数`

## 使用方法

### 基本用法

```bash
python scripts/calculate_step_level_violation_ratio.py \
    results_v8/pyautogui/a11y_tree/safepred@custom:o4-mini/batch_judgment_results_gpt-4.1_aer_v3.json
```

### 指定根目录

脚本会自动从 `batch_results` 的 `config.root_dir` 读取根目录，也可以手动指定：

```bash
python scripts/calculate_step_level_violation_ratio.py \
    batch_results.json \
    --root-dir /path/to/results/root
```

### 保存报告

```bash
python scripts/calculate_step_level_violation_ratio.py \
    batch_results.json \
    --output violation_ratio_report.txt
```

### 导出 JSON

```bash
python scripts/calculate_step_level_violation_ratio.py \
    batch_results.json \
    --json violation_ratio_stats.json
```

## 输出说明

### Step-Level Violation Ratios

```
Step       Violations      Reached         Ratio           Percentage     
----------------------------------------------------------------------
0          5               100             0.0500          5.00%
1          8               100             0.0800          8.00%
2          10              95              0.1053          10.53%
3          12              90              0.1333          13.33%
```

**说明**：
- **Step**: 步骤索引（从 0 开始）
- **Violations**: 在该 step 违反的任务数（包括之前就违反的）
- **Reached**: 到达该 step 的任务数（从 `better_log.json` 读取）
- **Ratio**: 违反比例 = Violations / Reached
- **Percentage**: 违反百分比

### 关键特点

1. **累积性**：violation_count 是累积的，一旦违反，后续步骤都算违反
2. **基于实际到达数**：分母是实际到达该 step 的任务数，不是总任务数
3. **更准确的比例**：反映了在每个 step 时，有多少任务处于违反状态

## 示例

### 示例 1: 单个任务

假设有一个任务：
- 总步骤数：10
- violation_step：3

则：
- Step 0-2: 不违反
- Step 3-9: 违反（violation_step 及之后）

### 示例 2: 多个任务

假设有 100 个任务：
- 任务 A: violation_step = 2, 总步骤 = 5
- 任务 B: violation_step = 4, 总步骤 = 8
- 任务 C: violation_step = null（安全）

统计结果：
- Step 0: violations = 0, reached = 100, ratio = 0%
- Step 1: violations = 0, reached = 100, ratio = 0%
- Step 2: violations = 1 (任务A), reached = 100, ratio = 1%
- Step 3: violations = 1 (任务A), reached = 100, ratio = 1%
- Step 4: violations = 2 (任务A+B), reached = 100, ratio = 2%
- Step 5: violations = 2 (任务A+B), reached = 99, ratio = 2.02% (任务A在step4结束，但仍在step5统计)
- Step 6: violations = 2 (任务A+B), reached = 99, ratio = 2.02% (任务A在step4结束，但仍在step6统计)
- Step 7: violations = 2 (任务A+B), reached = 99, ratio = 2.02% (任务A在step4结束，但仍在step7统计)

**关键点**：
- 任务 A 在 step 2 违反，即使它在 step 4 就结束了，在 step 5, 6, 7... 中仍然会被统计为 violation
- 比例会随着 reached_count 的减少而逐渐上升（因为到达后续 step 的任务数减少）

## 实际应用

### 1. 识别高风险步骤

找出 violation 比例最高的步骤：

```bash
python scripts/calculate_step_level_violation_ratio.py \
    batch_results.json \
    --output report.txt

# 查看报告
grep -A 20 "Step-Level Violation Ratios" report.txt
```

### 2. 对比不同版本

对比 v8 和 v9 的违反比例趋势：

```bash
# v8
python scripts/calculate_step_level_violation_ratio.py \
    results_v8/.../batch_results.json \
    --json v8_stats.json

# v9
python scripts/calculate_step_level_violation_ratio.py \
    results_v9/.../batch_results.json \
    --json v9_stats.json

# 对比分析
python -c "
import json
with open('v8_stats.json') as f:
    v8 = json.load(f)
with open('v9_stats.json') as f:
    v9 = json.load(f)
# 对比代码...
"
```

### 3. 按类别分析

查看特定类别的违反趋势：

```bash
python scripts/calculate_step_level_violation_ratio.py \
    batch_results.json \
    --output report.txt

# 查看 chrome 类别
grep -A 20 "CHROME" report.txt
```

## 注意事项

1. **需要 better_log.json**：脚本需要读取每个任务的 `better_log.json` 来获取实际步骤数
2. **路径匹配**：确保 `root_dir` 正确，以便找到 `better_log.json` 文件
3. **累积性**：violation_count 是累积的，一旦违反，后续步骤都算违反
4. **分母是到达数**：比例的分母是实际到达该 step 的任务数，不是总任务数

## 与之前方法的对比

| 特性 | 之前的方法 | 新方法 |
|------|-----------|--------|
| 统计对象 | 首次违反的 step | 在该 step 违反的任务（累积） |
| 分母 | 总任务数 | 到达该 step 的任务数 |
| 含义 | 每个 step 首次违反的比例 | 每个 step 时处于违反状态的比例 |
| 适用场景 | 分析首次违反模式 | 分析违反状态随时间的变化 |

## 相关文件

- `scripts/calculate_step_level_violation_ratio.py`: 新方法实现
- `scripts/calculate_violation_step_ratio.py`: 之前的方法（首次违反统计）
- `judge/run_judge.py`: 评估脚本
- `docs/step_level_violation_evaluation.md`: Step-level 评估指南
