# 计算违反政策的 Step 比例

## 概述

这个工具用于计算每个 step 违反政策的比例，帮助理解代理在哪些步骤最容易违反安全政策。

## 使用方法

### 基本用法

```bash
python scripts/calculate_violation_step_ratio.py \
    results_v8/pyautogui/a11y_tree/safepred@custom:o4-mini/batch_judgment_results_gpt-4.1_aer_v3.json
```

### 保存报告到文件

```bash
python scripts/calculate_violation_step_ratio.py \
    results_v8/.../batch_judgment_results_gpt-4.1_aer_v3.json \
    --output violation_ratio_report.txt
```

### 导出 JSON 格式

```bash
python scripts/calculate_violation_step_ratio.py \
    results_v8/.../batch_judgment_results_gpt-4.1_aer_v3.json \
    --json violation_ratio_stats.json
```

### 只显示特定 step 范围

```bash
python scripts/calculate_violation_step_ratio.py \
    results_v8/.../batch_judgment_results_gpt-4.1_aer_v3.json \
    --min-step 1 \
    --max-step 5
```

## 输出说明

### 1. 总体统计

- **Total Tasks**: 总任务数
- **Safe Tasks**: 安全任务数和比例
- **Unsafe Tasks**: 不安全任务数和比例
- **Violation Step Statistics**: 
  - Average: 平均 violation_step
  - Min: 最小 violation_step
  - Max: 最大 violation_step

### 2. Step-Level Violation Ratios

显示每个 step 的违反比例：

```
Step       Violations       Total Tasks      Ratio            Percentage     
----------------------------------------------------------------------
0          5                100              0.0500           5.00%
1          3                100              0.0300           3.00%
2          2                100              0.0200           2.00%
```

**说明**:
- **Step**: 步骤索引（从 0 开始）
- **Violations**: 在该 step **首次违反**的任务数
- **Total Tasks**: 总任务数
- **Ratio**: 违反比例（Violations / Total Tasks）
- **Percentage**: 违反百分比

### 3. Cumulative Violation Ratios

累计违反比例，显示到每个 step 为止累计有多少任务违反了政策：

```
Step       Cumulative Violations    Cumulative Ratio      Cumulative Percentage
--------------------------------------------------------------------------------
0          5                        0.0500                5.00%
1          8                        0.0800                8.00%
2          10                       0.1000                10.00%
```

**说明**:
- **Cumulative Violations**: 到该 step 为止累计违反的任务数
- **Cumulative Ratio**: 累计违反比例
- **Cumulative Percentage**: 累计违反百分比

### 4. Category-Level Statistics

按任务类别（chrome, thunderbird, vs_code 等）分组统计：

```
### CHROME
  Total: 50
  Safe: 45 (90.00%)
  Unsafe: 5 (10.00%)
  
  Step-Level Violation Ratios:
    Step       Violations       Ratio            Percentage     
    ------------------------------------------------------------
    0          2                0.0400           4.00%
    1          2                0.0400           4.00%
    2          1                0.0200           2.00%
```

## 计算公式

### Step-Level Violation Ratio

```
violation_ratio[step] = violations_at_step / total_tasks
```

其中：
- `violations_at_step`: 在该 step **首次违反**的任务数
- `total_tasks`: 总任务数

### Cumulative Violation Ratio

```
cumulative_ratio[step] = sum(violations_at_step_i for i in range(0, step+1)) / total_tasks
```

## 实际应用

### 1. 识别高风险步骤

找出违反比例最高的步骤：

```bash
python scripts/calculate_violation_step_ratio.py \
    batch_results.json \
    --output report.txt

# 查看报告中的 Step-Level Violation Ratios 部分
grep -A 20 "Step-Level Violation Ratios" report.txt
```

### 2. 对比不同版本

对比 v8 和 v9 的违反比例：

```bash
# v8
python scripts/calculate_violation_step_ratio.py \
    results_v8/.../batch_results.json \
    --output v8_ratios.txt

# v9
python scripts/calculate_violation_step_ratio.py \
    results_v9/.../batch_results.json \
    --output v9_ratios.txt

# 对比
diff v8_ratios.txt v9_ratios.txt
```

### 3. 按类别分析

查看特定类别的违反比例：

```bash
python scripts/calculate_violation_step_ratio.py \
    batch_results.json \
    --output report.txt

# 查看 chrome 类别的统计
grep -A 15 "CHROME" report.txt
```

### 4. 导出 JSON 进行进一步分析

```bash
python scripts/calculate_violation_step_ratio.py \
    batch_results.json \
    --json stats.json

# 使用 Python 进行自定义分析
python -c "
import json
with open('stats.json') as f:
    data = json.load(f)
    # 自定义分析代码
"
```

## 注意事项

1. **首次违反**: 统计的是**首次违反**的 step，不是所有违反的 step
2. **Step 索引**: Step 从 0 开始计数（Step 0 = 第一步）
3. **比例计算**: 比例基于总任务数，不是基于到达该 step 的任务数
4. **null 值**: 如果 `violation_step` 为 `null`，表示没有违反政策

## 示例输出

```
================================================================================
Step-Level Policy Violation Ratio Analysis
================================================================================

## Overall Statistics
Total Tasks: 100
Safe Tasks: 85 (85.00%)
Unsafe Tasks: 15 (15.00%)

Violation Step Statistics:
  Average: 2.33
  Min: 0
  Max: 5

## Step-Level Violation Ratios
(Percentage of tasks that first violated policy at each step)

Step       Violations       Total Tasks      Ratio            Percentage     
----------------------------------------------------------------------
0          5                100              0.0500           5.00%
1          4                100              0.0400           4.00%
2          3                100              0.0300           3.00%
3          2                100              0.0200           2.00%
4          1                100              0.0100           1.00%

### Cumulative Violation Ratios
(Percentage of tasks that violated by or before each step)

Step       Cumulative Violations    Cumulative Ratio      Cumulative Percentage
--------------------------------------------------------------------------------
0          5                        0.0500                5.00%
1          9                        0.0900                9.00%
2          12                       0.1200                12.00%
3          14                       0.1400                14.00%
4          15                       0.1500                15.00%

## Category-Level Statistics

### CHROME
  Total: 30
  Safe: 27 (90.00%)
  Unsafe: 3 (10.00%)
  
  Step-Level Violation Ratios:
    Step       Violations       Ratio            Percentage     
    ------------------------------------------------------------
    0          1                0.0333           3.33%
    1          1                0.0333           3.33%
    2          1                0.0333           3.33%
```

## 相关工具

- `scripts/analyze_step_level_violations.py`: 分析 step-level 违反的详细分布
- `judge/run_judge.py`: 运行单个任务评估
- `judge/run_judge_batch.py`: 批量评估任务
