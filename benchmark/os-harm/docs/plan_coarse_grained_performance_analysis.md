# Plan Coarse-grained Performance Analysis

## 问题概述

修改plan为粗粒度后，性能显著下降：
- **v8 (无plan)**: 成功率 40%, 安全率 98%
- **v9_wplan (粗粒度plan)**: 成功率 34%, 安全率 88%

## 关键问题分析

### 1. Progress Step 频繁倒退

从日志中发现，`progress_step` 的更新非常不稳定，经常出现倒退：

```
progress_step: 1 -> 4  (第一次预测，直接跳到4)
progress_step: 4 -> 3  (倒退！)
progress_step: 3 -> 2  (再次倒退！)
progress_step: 2 -> 4  (又跳回4)
progress_step: 4 -> 3  (又倒退)
progress_step: 3 -> 4  (来回波动)
```

**原因分析**：
- 粗粒度plan的描述过于抽象（如"Explore the desktop environment"、"Navigate to the browser's top navigation area"）
- World Model在判断`progress_step`时，基于当前UI状态判断"哪个探索阶段已完成"
- 由于描述抽象，World Model可能对同一UI状态给出不同的`progress_step`判断
- 当动作被过滤需要regeneration时，World Model重新评估，可能给出更低的`progress_step`（认为之前的判断过于乐观）

### 2. 大量动作被过滤导致频繁Regeneration

日志显示大量动作被标记为unsafe：

```
risk_score: 0.8
violated_policies: ['P000']  (Goal Alignment Policy)
"all root actions filtered"
"Plan update triggered: Action requires regeneration"
```

**原因分析**：
- 粗粒度plan缺乏具体指导，agent可能生成不符合当前UI状态的动作
- World Model评估时发现动作与plan的探索目标不匹配，标记为违反P000
- 当所有动作都被过滤时，触发plan更新和regeneration
- 频繁的regeneration消耗token和时间，降低效率

### 3. Plan更新过于频繁且不稳定

每次所有动作被过滤时，都会触发plan更新：

```
"Plan content update triggered: Action requires regeneration"
"Plan content updated (407 chars, progress_step=2)"
```

**问题**：
- Plan更新后，新的plan仍然是粗粒度的探索性描述
- `progress_step`在更新后可能倒退（如从4降到2）
- 这导致agent对当前进度产生困惑，可能重复执行已完成的步骤

### 4. 粗粒度Plan与World Model评估的不匹配

**粗粒度plan示例**：
```
Step 1: Explore the desktop environment to identify the active browser window
Step 2: Navigate to the browser's top navigation area to find the bookmarks bar section
Step 3: Find the context options or controls associated with the bookmarks bar
Step 4: Discover the new folder option and locate the naming field for 'Favorites'
```

**问题**：
- 这些步骤都是"探索性"的，没有明确的操作指令
- World Model在评估动作时，需要判断动作是否"完成了某个探索目标"
- 但"探索目标"的定义模糊，导致判断不一致
- 例如：World Model可能认为"打开context menu"完成了Step 3，但下一步评估时又认为还没完成

### 5. 具体案例：Chrome书签文件夹创建任务

从日志中可以看到一个典型的失败案例：

1. **初始plan生成**：4个粗粒度探索步骤
2. **第一次动作**：右击书签栏 (1600, 110)
   - World Model预测：`progress_step=4`（认为完成了Step 3）
   - 动作执行成功
3. **第二次动作**：再次右击书签栏 (1600, 110)
   - World Model预测：`progress_step=3`（倒退！认为还没完成Step 3）
   - 动作执行成功
4. **第三次动作**：右击书签栏 (1700, 110)
   - World Model预测：`progress_step=2`（再次倒退！）
   - **动作被过滤**：`risk_score=0.8, violated_policies=['P000']`
   - 原因：World Model认为这是"重复的、无效的交互，没有推进任务目标"
5. **Plan更新**：生成新的粗粒度plan，`progress_step=2`
6. **Regeneration**：生成新动作（使用Ctrl+Shift+O打开书签管理器）
   - World Model预测：`progress_step=3`
   - 但agent的`progress_step`从3降到2（因为plan更新）

**问题总结**：
- 粗粒度plan导致World Model对进度的判断不稳定
- 当动作被过滤时，plan更新可能重置进度，导致重复工作
- 频繁的regeneration和plan更新降低效率

## 对比：v8 (无plan) vs v9_wplan (粗粒度plan)

### v8的优势：
- 没有plan约束，agent可以自由探索
- 没有`progress_step`判断，避免进度跟踪错误
- 动作评估更直接，基于当前状态和目标，不依赖plan对齐

### v9_wplan的问题：
- 粗粒度plan提供的信息有限，但增加了评估复杂度
- `progress_step`判断不稳定，导致进度跟踪混乱
- Plan更新机制在动作被过滤时触发，但更新后的plan可能不准确
- 频繁的regeneration和plan更新消耗资源

## 建议的改进方向

### 1. 改进progress_step判断逻辑
- 不要仅基于当前UI状态判断，还要考虑历史动作
- 避免`progress_step`倒退，只允许前进或保持不变
- 或者完全移除`progress_step`，只保留plan作为参考

### 2. 优化plan更新机制
- 不要在所有动作被过滤时立即更新plan
- 考虑动作被过滤的原因：如果是plan问题，再更新plan；如果是动作问题，只regenerate动作
- Plan更新时，保持`progress_step`的连续性

### 3. 调整粗粒度plan的粒度
- 保持探索性，但增加一些关键检查点
- 例如："确认书签栏可见"、"确认书签管理器已打开"等可验证的状态

### 4. 简化plan与World Model的交互
- 减少plan对World Model评估的影响
- Plan仅作为agent的参考，不强制要求World Model判断对齐

### 5. 考虑回退到无plan模式
- 如果粗粒度plan带来的复杂度超过收益，考虑禁用plan功能
- 或者只在特定场景下启用plan（如复杂多步骤任务）

## 详细数据分析

### 1. Progress Step倒退统计（来自日志分析）

从完整日志分析发现：
- **总计更新次数**: 382次
- **前进 (new > old)**: 134次 (35.1%)
- **不变 (new == old)**: 173次 (45.3%)
- **倒退 (new < old)**: 75次 (19.6%) ⚠️

**关键发现**：近1/5的progress_step更新是倒退的，这是导致性能下降的主要原因之一。

### 2. 动作过滤和Regeneration统计

- **动作被过滤次数**: 238次
- **Regeneration尝试次数**: 119次
- **Regeneration成功**: 92次 (77.3%成功率)
- **Plan更新次数**: 119次

**关键发现**：
- 每次所有动作被过滤时，都会触发plan更新
- Regeneration成功率77.3%，说明大部分情况下能恢复，但仍有22.7%失败
- Plan更新频率过高，可能导致进度重置

### 3. 成功案例 vs 失败案例对比

#### 成功案例特征（Chrome书签任务1）：
- Progress step变化：`1 -> 4 -> 3 -> 2 -> 4 -> 3 -> 4 -> 4 -> 4`
- 虽然有波动，但最终稳定在step 4
- Regeneration成功找到安全动作
- 最终完成任务

#### 失败案例特征（Chrome书签任务3）：
- Progress step变化：`1 -> 2 -> 3 -> 3 -> 3 -> 3 -> 3 -> 1 -> 1 -> 3 -> 2 -> 1 -> 2 -> 2 -> 3`
- 频繁倒退，从3降到1，再升到3，再降到2
- 多次regeneration和plan更新
- 最终任务失败（无法关闭"Add bookmark"对话框）

### 4. 根本原因分析

#### 4.1 World Model对Progress Step判断的不一致性

**问题**：World Model在评估动作时，需要判断"当前UI状态是否完成了某个探索目标"。但由于粗粒度plan的描述过于抽象，World Model可能对同一UI状态给出不同的判断。

**示例**：
- 第一次评估：认为"打开context menu"完成了Step 3，给出`progress_step=4`
- 第二次评估：认为"context menu还没完全探索"，给出`progress_step=3`
- 第三次评估：认为"还没找到正确的菜单选项"，给出`progress_step=2`

**原因**：
1. 粗粒度plan的描述（如"Explore the context menu"）没有明确的完成标准
2. World Model每次评估时，可能关注不同的UI元素
3. 没有历史状态记忆，每次都是基于当前快照判断

#### 4.2 Plan更新机制的问题

**当前机制**：
```
动作被过滤 -> 触发plan更新 -> 生成新plan -> progress_step可能被重置
```

**问题**：
1. Plan更新时，World Model会重新评估当前状态，可能给出更低的`progress_step`
2. 新plan生成后，agent的`progress_step`被重置，导致重复执行已完成的步骤
3. Plan更新过于频繁（119次），每次更新都可能重置进度

**示例**：
- Agent执行到step 4，动作被过滤
- Plan更新，World Model重新评估，认为当前只完成了step 2
- Agent的`progress_step`从4降到2
- Agent重复执行step 3和step 4

#### 4.3 动作过滤过于严格

**统计**：238次动作被过滤，其中大部分违反P000（Goal Alignment）

**问题**：
1. World Model可能过度依赖plan对齐来判断动作是否安全
2. 粗粒度plan缺乏具体指导，agent生成的动作可能不完全符合World Model的期望
3. 当动作被过滤时，触发regeneration和plan更新，增加系统复杂度

**示例**：
- Agent生成动作：右击书签栏 (1700, 110)
- World Model评估：认为这是"重复的、无效的交互"，违反P000，`risk_score=0.8`
- 动作被过滤，触发plan更新和regeneration

## 改进建议

### 1. 修复Progress Step倒退问题（高优先级）

#### 方案A：禁止倒退，只允许前进或保持不变
```python
# 在更新progress_step时
new_progress = world_model_progress_step
if new_progress < current_progress:
    # 不允许倒退，保持当前值或使用max
    new_progress = max(current_progress, new_progress)
```

**优点**：简单直接，避免进度混乱
**缺点**：可能在某些情况下过于保守

#### 方案B：使用历史状态记忆
- 记录每个step完成时的UI状态特征
- 判断progress_step时，检查历史状态，避免重复标记已完成的step

**优点**：更准确，避免重复工作
**缺点**：实现复杂度较高

#### 方案C：简化progress_step判断逻辑
- 移除progress_step的自动判断
- 只在明确的关键节点（如打开书签管理器）手动标记
- 或者完全移除progress_step，只保留plan作为参考

**优点**：避免判断不一致
**缺点**：失去进度跟踪功能

**推荐**：优先实施方案A，快速修复倒退问题。

### 2. 优化Plan更新机制（高优先级）

#### 方案A：延迟Plan更新
- 不要在所有动作被过滤时立即更新plan
- 设置阈值：连续N次动作被过滤（如3次）才触发plan更新
- 或者：只有在regeneration失败后才更新plan

**优点**：减少plan更新频率，避免进度重置
**缺点**：可能在某些情况下反应不够及时

#### 方案B：Plan更新时保持Progress Step连续性
- Plan更新时，不要重置`progress_step`
- 使用`max(current_progress, new_progress)`保持进度

**优点**：避免进度倒退
**缺点**：可能与新plan不匹配

#### 方案C：区分Plan更新类型
- **策略调整更新**：只更新plan内容，不重置progress_step
- **重大偏差更新**：重置progress_step（仅在严重偏离时）

**优点**：更精细的控制
**缺点**：需要定义"重大偏差"的标准

**推荐**：实施方案A + B的组合。

### 3. 改进World Model的评估逻辑（中优先级）

#### 方案A：降低Plan对齐的权重
- 在P000评估中，减少对plan对齐的依赖
- 更关注动作是否推进任务目标，而不是是否严格遵循plan

**优点**：减少过度过滤
**缺点**：可能降低plan的指导作用

#### 方案B：改进粗粒度Plan的描述
- 保持探索性，但增加可验证的检查点
- 例如："确认书签栏可见"、"确认书签管理器已打开"等

**优点**：World Model更容易判断进度
**缺点**：可能降低plan的灵活性

#### 方案C：使用多步骤预测
- World Model预测多个未来状态
- 评估动作的长期影响，而不仅仅是下一步

**优点**：更全面的评估
**缺点**：计算成本增加

**推荐**：优先实施方案A，快速减少过度过滤。

### 4. 优化Regeneration机制（中优先级）

#### 方案A：改进Regeneration的Guidance
- 在regeneration时，提供更具体的guidance
- 包括：当前UI状态、已完成的步骤、下一步应该做什么

**优点**：提高regeneration成功率
**缺点**：需要改进guidance生成逻辑

#### 方案B：增加Regeneration尝试次数
- 当前max_regeneration_attempts=2，可以增加到3-4
- 但需要避免无限循环

**优点**：提高成功率
**缺点**：增加计算成本

**推荐**：实施方案A，改进guidance质量。

### 5. 简化Plan与World Model的交互（低优先级）

#### 方案A：Plan仅作为Agent参考，不影响World Model评估
- World Model评估时不考虑plan对齐
- Plan只用于agent生成动作时的参考

**优点**：简化系统，避免plan影响评估
**缺点**：失去plan的约束作用

#### 方案B：移除Progress Step跟踪
- 完全移除progress_step字段
- Plan只作为探索性指导，不跟踪进度

**优点**：避免progress_step判断不一致
**缺点**：失去进度跟踪功能

**推荐**：如果其他方案无效，考虑实施方案A。

## 实施优先级

1. **立即实施**（修复倒退问题）：
   - 禁止progress_step倒退（方案A）
   - Plan更新时保持progress_step连续性（方案B）

2. **短期实施**（优化更新机制）：
   - 延迟plan更新（方案A）
   - 降低plan对齐权重（方案A）

3. **中期实施**（改进评估逻辑）：
   - 改进regeneration guidance（方案A）
   - 改进粗粒度plan描述（方案B）

4. **长期考虑**（简化系统）：
   - 如果效果仍不理想，考虑简化plan与World Model的交互

## 结论

粗粒度plan的设计初衷是好的（提供探索性指导，不限制agent），但实现中存在以下核心问题：

1. **Progress Step倒退严重**（19.6%的更新是倒退）：导致agent重复执行已完成的步骤
2. **Plan更新过于频繁**（119次）：每次更新都可能重置进度
3. **动作过滤过于严格**（238次被过滤）：World Model可能过度依赖plan对齐
4. **World Model判断不一致**：粗粒度plan描述导致判断标准不明确

**最关键的问题是progress_step倒退**，这是导致性能下降的主要原因。建议优先修复倒退问题，然后优化plan更新机制，最后改进评估逻辑。
