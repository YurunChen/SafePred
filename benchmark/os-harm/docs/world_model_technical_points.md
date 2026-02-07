# SafePred World Model 技术点分析

## 概述

SafePred 的 World Model 是一个基于 LLM 的状态预测和风险评估系统，用于在动作执行前预测状态变化并评估安全风险。

---

## 核心技术点

### 1. **State Delta 模式（状态增量预测）**

**技术原理**：
- 不预测完整的状态，而是预测状态变化（delta）
- 更高效：只需要预测变化的部分，而不是整个状态
- 更准确：专注于变化，减少冗余信息

**实现**：
```python
# SafePred_v10/models/world_model.py
self.use_state_delta = use_state_delta if use_state_delta is not None else True  # Default to state delta mode
```

**预测输出格式**：
```json
{
  "semantic_delta": "预测的状态变化描述",
  "element_changes": {
    "new_elements": ["新增的UI元素"],
    "removed_elements": ["移除的UI元素"]
  },
  "long_term_impact": "长期影响评估",
  "risk_explanation": "风险解释",
  "violated_policy_ids": ["P001", "P002"],
  "optimization_guidance": "优化建议"
}
```

**状态合成**：
- 将 delta 应用到当前状态以合成下一状态
- 保持 Goal 和 Policies 不变
- 更新 Accessibility Tree 和 Chat History

---

### 2. **双时间尺度预测（Short-term + Long-term）**

**短期预测（Semantic Delta）**：
- **目标**：预测动作执行后的立即 UI 状态变化
- **内容**：描述可见的 UI 变化（如对话框打开、按钮状态改变等）
- **用途**：识别立即的安全风险（如打开恶意网站、执行危险命令）

**长期预测（Long-term Impact）**：
- **目标**：评估动作对任务目标的长期影响
- **考虑因素**：
  1. 动作是否推进、阻碍或对任务目标无影响
  2. 潜在的未来影响
  3. 是否会导致不可逆变化或阻碍后续步骤
- **用途**：识别需要多步才能显现的风险（如数据泄露、系统配置修改）

**配置控制**：
```python
# SafePred_v10/config/config.yaml
planning:
  enable_short_term_prediction: True  # 启用短期预测
  enable_long_term_prediction: True   # 启用长期预测
```

**动态 Prompt 生成**：
- 根据配置动态生成 JSON 输出格式
- 根据配置动态生成字段要求
- 根据配置动态生成推理过程说明

---

### 3. **多步预测（Multi-step Prediction）**

**技术原理**：
- 支持预测多个步骤的状态变化
- 使用加权风险评分（exponential decay weights）
- 步骤权重：step 0 = 1.0, step 1 = 0.5, step 2 = 0.25, ...

**加权风险计算**：
```python
# SafePred_v10/models/world_model.py
def _calculate_weighted_risk(self, step_risks: List[float]) -> float:
    weights = []
    for i in range(len(step_risks)):
        weight = 1.0 / (2 ** i)  # 1.0, 0.5, 0.25, 0.125, ...
        weights.append(weight)
    weighted_sum = sum(step_risks[i] * weights[i] for i in range(len(step_risks)))
    weighted_risk = weighted_sum / total_weight
    return weighted_risk
```

**用途**：
- 识别需要多步才能显现的风险
- 评估动作序列的整体风险

---

### 4. **基于规则的风险评分（Rule-based Risk Scoring）**

**技术原理**：
- 不依赖 LLM 生成风险评分，而是基于策略违反情况计算
- 规则：有策略违反 → risk_score = violation_penalty（默认 0.5）
- 规则：无策略违反 → risk_score = 0.0

**实现**：
```python
# SafePred_v10/models/world_model.py
def _calculate_risk_score_by_rules(self, parsed_delta: Dict[str, Any]) -> Dict[str, Any]:
    violated_policy_ids = parsed_delta.get('violated_policy_ids', [])
    has_violations = violated_policy_ids and len(violated_policy_ids) > 0
    
    risk_score = 0.0
    if has_violations:
        risk_score = self.risk_score_violation_penalty  # Default: 0.5
    
    risk_score = max(0.0, min(1.0, risk_score))  # Clamp to [0.0, 1.0]
    parsed_delta['risk_score'] = risk_score
    return parsed_delta
```

**优势**：
- 可解释性强：风险评分直接对应策略违反
- 一致性高：不依赖 LLM 的主观判断
- 可配置：可以通过 `risk_score_violation_penalty` 调整风险阈值

---

### 5. **多格式解析（Multi-format Parsing）**

**支持的格式**：
1. **JSON**（主要格式）：带错误修复的健壮解析
2. **XML**（格式不敏感的回退）
3. **结构化文本**（格式不敏感的回退）

**解析优先级**：
```python
# SafePred_v10/models/world_model.py
def _parse_json_output(self, text: str) -> Optional[Dict[str, Any]]:
    # 1. Try JSON format first (primary format, with robust error fixing)
    parsed = parse_json_from_text(text)
    if isinstance(parsed, dict):
        return parsed
    
    # 2. Try XML format (format-insensitive fallback)
    xml_result = parse_xml_output(text)
    if xml_result:
        return convert_xml_to_world_model_format(xml_result)
    
    # 3. Try Structured Text format (format-insensitive fallback)
    structured_result = parse_structured_text(text)
    if structured_result:
        return convert_structured_text_to_world_model_format(structured_result)
    
    return None
```

**优势**：
- 提高鲁棒性：即使 LLM 输出格式不标准也能解析
- 减少重试：格式错误时自动尝试其他格式

---

### 6. **自动重试机制（Automatic Retry）**

**技术原理**：
- 检测 JSON 解析错误（格式问题）
- 自动重试 API 调用（默认最多 2 次重试，共 3 次尝试）
- 非格式错误立即抛出，不重试

**实现**：
```python
# SafePred_v10/models/world_model.py
max_format_retries = 2  # Default: 2 retries (3 total attempts)
format_retry_delay = 0.5  # Default: 0.5 seconds

while retries <= max_format_retries:
    try:
        generated_text = self._llm_client.generate(prompt)
        next_state = self._parse_llm_output(generated_text, state, action)
        return next_state
    except ValueError as e:
        is_format_error = "Failed to parse JSON" in str(e)
        if is_format_error and retries < max_format_retries:
            retries += 1
            time.sleep(format_retry_delay)
            continue  # Retry
        else:
            raise  # Not format error or max retries exceeded
```

**优势**：
- 提高成功率：格式错误时自动重试
- 减少人工干预：不需要手动修复格式错误

---

### 7. **策略参考机制（Policy Reference）**

**技术原理**：
- 策略中包含历史违反案例（reference examples）
- World Model 可以使用这些案例来识别类似的攻击模式
- 可配置是否显示参考案例（`show_policy_references`）

**Prompt 集成**：
```python
# SafePred_v10/models/prompts.py
if show_policy_references:
    policy_header = (
        "Each policy entry contains the following fields:\n"
        "  - `Reference`: Historical violation examples showing similar attack patterns. "
        "Use these to identify similar attack patterns.\n"
    )
    reference_important_note = (
        "- **IMPORTANT**: Review the **reference examples** to identify similar attack patterns.\n"
        "- Compare the current action with historical violations in reference examples."
    )
```

**优势**：
- 提高识别准确率：基于历史案例识别类似攻击
- 可解释性强：可以指出违反的具体策略和参考案例

---

### 8. **状态预处理（State Preprocessing）**

**技术原理**：
- 状态在传递给 World Model 前进行预处理
- 提取关键信息（key_elements）或完整 accessibility tree（axtree_txt）
- 保持 conversation history 完整

**状态格式**：
```python
state = {
    "goal": "任务目标",
    "key_elements": ["关键UI元素列表"],  # 或
    "axtree_txt": "完整的accessibility tree文本",
    "conversation_history": [
        {"role": "user", "message": "..."},
        {"role": "assistant", "message": "..."}
    ],
    "total_executed_steps": 5,
    "current_action_reasoning": "当前动作的完整推理"
}
```

**优势**：
- 减少 token 消耗：使用 key_elements 时只包含关键信息
- 提高准确性：使用 axtree_txt 时包含完整环境状态

---

### 9. **优化指导生成（Optimization Guidance）**

**技术原理**：
- 当检测到策略违反时，生成具体的优化建议
- 建议基于策略违反类型和当前状态
- 用于指导 agent 重新生成安全的动作

**生成时机**：
- 仅在检测到策略违反时生成（`violated_policy_ids` 不为空）
- 可以基于 plan_text 生成更具体的指导

**示例**：
```
"optimization_guidance": "Do not execute arbitrary terminal commands unrelated to the stated task. To achieve left/right alignment with tab stops in LibreOffice Writer, the agent should: (1) Open LibreOffice Writer, (2) Access paragraph or ruler settings, (3) Set a left-aligned tab at position 0 and a right-aligned tab near the right margin, (4) Input the sentence with a tab between the third word and the rest."
```

---

### 10. **统一 LLM 客户端（Unified LLM Client）**

**技术原理**：
- 使用统一的 `LLMClient` 支持多种 LLM 提供商
- 支持 OpenAI、Qwen、DeepSeek、Custom 等
- 自动处理 API URL 拼接（如 `/v1/chat/completions`）

**实现**：
```python
# SafePred_v10/models/llm_client.py
class LLMClient:
    def generate(self, prompt: str, ...):
        if self.provider == "openai":
            return self._generate_openai_requests(...)
        elif self.provider == "qwen":
            return self._generate_qwen(...)
        elif self.provider == "deepseek":
            return self._generate_deepseek(...)
        elif self.provider == "custom":
            return self._generate_custom(...)
```

**优势**：
- 统一接口：所有 LLM 提供商使用相同的接口
- 易于扩展：添加新的 LLM 提供商只需实现一个方法
- 自动处理：自动处理不同提供商的特殊要求（如 qwen3 的 `enable_thinking` 参数）

---

## 技术优势总结

1. **高效性**：State Delta 模式只预测变化，减少 token 消耗
2. **准确性**：双时间尺度预测（短期+长期）提供全面的风险评估
3. **鲁棒性**：多格式解析和自动重试机制提高成功率
4. **可解释性**：基于规则的风险评分和策略参考机制提供清晰的解释
5. **灵活性**：可配置的预测类型（短期/长期）和策略参考显示
6. **可扩展性**：统一的 LLM 客户端支持多种提供商

---

## 与其他系统的对比

| 特性 | Reactive（反应式） | SafePred World Model（预测式） |
|------|------------------|------------------------------|
| **预测能力** | ❌ 无 | ✅ 短期+长期预测 |
| **状态感知** | ❌ 仅文本 | ✅ 完整环境状态（accessibility tree） |
| **风险评分** | 基于文本分析 | 基于策略违反规则 |
| **优化指导** | ✅ 有 | ✅ 有（更详细） |
| **Plan 支持** | ❌ 无 | ✅ 支持 plan 更新 |
| **多步预测** | ❌ 无 | ✅ 支持（加权风险） |
| **策略参考** | ❌ 无 | ✅ 支持历史案例参考 |

---

## 参考文献

- `SafePred_v10/models/world_model.py`: World Model 核心实现
- `SafePred_v10/models/prompts.py`: Prompt 模板定义
- `SafePred_v10/models/llm_client.py`: 统一 LLM 客户端
- `SafePred_v10/config/config.yaml`: 配置文件
