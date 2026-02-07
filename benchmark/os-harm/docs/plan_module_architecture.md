# Plan Module Architecture and Design

## Overview

The plan module provides a **coarse-grained, suggestive planning system** that guides agent exploration without constraining specific actions. Plans use high-level exploration goals (e.g., "Explore the menu options") rather than detailed step-by-step actions.

## Core Design Principles

1. **Coarse-grained**: Plans describe **what to explore**, not **how to do it**
2. **Suggestive**: Plans are for reference only, not mandatory constraints
3. **Flexible**: Agents can try different approaches to achieve exploration goals
4. **Progress Tracking**: Tracks which exploration stages are completed based on UI state

## Architecture Components

### 1. Plan Generation (`mm_agents/agent.py`)

**Location**: `PromptAgent.generate_plan()`

**When Generated**:
- **Initial Plan**: Generated at task start (first `predict()` call, when `len(observations) == 0`)
- **Plan Updates**: Generated when `optimization_guidance` is provided (via `generate_plan_with_guidance()`)

**Format**:
```
Step 1: [Exploration goal, e.g., 'Explore the menu options']
Step 2: [Exploration goal, e.g., 'Navigate to the settings area']
Step 3: [Exploration goal, e.g., 'Find the configuration dialog']
Step 4: [Exploration goal, e.g., 'Locate the text input field']
```

**Requirements**:
- Use exploration language: "Explore...", "Navigate to...", "Find...", "Discover...", "Locate..."
- DO NOT include detailed actions (click, type, press, etc.)
- DO NOT include specific coordinates
- Keep it coarse-grained and high-level

**Storage**:
- Stored in `agent.current_plan` (without [COMPLETED] markers)
- Markers are added only for display/formatting

### 2. Plan Formatting (`mm_agents/safety_wrapper.py`)

**Location**: `SafetyWrapper.format_plan_for_prompt()`

**Purpose**: Format plan for inclusion in agent's prompt

**Format**:
```
**SUGGESTED PLAN:**

Step 1: Explore the menu options
Step 2: Navigate to the settings area
Step 3: Find the configuration dialog
Step 4: Locate the text input field

**Progress Step:** 2
(Steps 1 to 1 are completed. Current step to execute: 2)

**Note:**
- The plan above contains exploration goals for reference
- You can try different approaches to achieve these exploration goals based on the current observation
- Make decisions based on the current observation and task requirements
- If you use any actions from the plan, find actual coordinates or element identifiers from the current observation
- You MUST output actions in the exact format specified in the system prompt (Python code in code blocks), NOT plan descriptions or natural language
```

**Key Features**:
- Adds "SUGGESTED PLAN:" header
- Includes progress_step information
- Adds instructions emphasizing flexibility and exploration
- Removes [COMPLETED] markers before formatting (markers added only for display)

### 3. Plan Integration into Agent Prompt (`mm_agents/agent.py`)

**Location**: `PromptAgent.predict()`

**When Added**:
- Added as system message before the last user message
- Added during initial action generation
- Added during action regeneration (if plan is updated)

**Flow**:
1. Check if planning is enabled (`planning_enabled`)
2. If enabled and `current_plan` exists:
   - Call `safety_wrapper.format_plan_for_prompt(current_plan, progress_step)`
   - Insert formatted plan as system message
   - Remove any existing plan messages first

### 4. Plan Monitoring (`SafePred_v9/core/plan_monitor.py`)

**Location**: `PlanMonitor.monitor_with_plan()`

**Purpose**: Evaluate if action path has fundamental problems (not enforce plan adherence)

**Input**:
- `plan_text`: Suggested plan
- `current_state`: Current UI state
- `action` or `simulated_path`: Action(s) to evaluate

**Evaluation Criteria**:
- Is the path making progress toward completing the task?
- Are the actions reasonable and appropriate for the current state?
- Is the path unnecessarily repeating steps?
- Are there clear errors or problems?

**Output**:
- `optimization_guidance`: Guidance text if fundamental problem detected, null otherwise
- Only flags **fundamental problems**, not plan misalignment

**Key Feature**: 
- Does NOT enforce plan adherence
- Agent can try different approaches to achieve exploration goals

### 5. Plan Progress Tracking (`SafePred_v9/models/prompts.py`)

**Location**: World Model prompt template

**Purpose**: Track which exploration stage is completed based on UI state

**Field**: `progress_step` (REQUIRED)

**Logic**:
- World Model examines current UI state
- Determines if exploration goal at current `progress_step` is achieved
- Example: If goal is "Explore the menu options" and menu is visible, goal is achieved
- If achieved: increment `progress_step` to next stage
- If not achieved: keep `progress_step` unchanged

**Key Points**:
- Based on **CURRENT UI STATE**, not action trajectory
- Tracks completion of **exploration stages**, not detailed steps
- Used to mark completed steps with [COMPLETED] for display

### 6. Plan Update Mechanism (`mm_agents/agent.py`)

**Location**: `PromptAgent.generate_plan_with_guidance()`

**When Updated**:
- When `optimization_guidance` is provided by PlanMonitor or World Model
- During action regeneration if plan needs correction

**Update Process**:
1. Receive `optimization_guidance` (from PlanMonitor or World Model)
2. Call `generate_plan_with_guidance(instruction, current_state, optimization_guidance, base_plan)`
3. Generate revised plan using same coarse-grained format
4. Update `agent.current_plan`
5. Plan is re-formatted and added to prompt for next action

**Update Rules**:
- Keep same exploration goal format
- DO NOT update steps marked with [COMPLETED]
- Only update incomplete steps based on guidance
- Can add new steps if needed

## Data Flow

```
Task Start
    ↓
Generate Initial Plan (coarse-grained exploration goals)
    ↓
Store in agent.current_plan
    ↓
Format Plan for Prompt (add progress_step, instructions)
    ↓
Add to Agent Prompt (as system message)
    ↓
Agent Generates Actions (can try different approaches)
    ↓
World Model Evaluates Actions
    ├─→ Tracks progress_step (based on UI state)
    └─→ May provide optimization_guidance
    ↓
PlanMonitor Evaluates Actions (optional)
    └─→ May provide optimization_guidance (if fundamental problem)
    ↓
If optimization_guidance provided:
    ↓
Update Plan (generate_plan_with_guidance)
    ↓
Update agent.current_plan
    ↓
Re-format and add to next prompt
    ↓
Continue...
```

## Key Design Decisions

### 1. Coarse-grained Exploration Goals
**Rationale**: 
- Allows flexibility in how goals are achieved
- Reduces plan brittleness (plan doesn't break if UI changes slightly)
- Focuses on "what to explore" rather than "how to do it"

**Implementation**:
- Use exploration language: "Explore...", "Navigate to...", "Find..."
- Prohibit detailed actions: no "click", "type", "press"
- Prohibit specific coordinates

### 2. Suggestive, Not Mandatory
**Rationale**:
- Allows agent to adapt to actual UI state
- Prevents plan from becoming a constraint
- Encourages exploration and experimentation

**Implementation**:
- Title: "SUGGESTED PLAN" (not "EXECUTION PLAN")
- Instructions: "for reference", "can try different approaches"
- PlanMonitor: Only flags fundamental problems, not plan misalignment

### 3. Progress Tracking Based on UI State
**Rationale**:
- Exploration goals are about discovering UI elements/areas
- Completion should be determined by UI state, not actions taken
- More robust than tracking specific actions

**Implementation**:
- World Model examines current UI state
- Determines if exploration goal is achieved (e.g., menu visible = "Explore menu" achieved)
- Updates progress_step accordingly

### 4. Plan Updates Use Same Format
**Rationale**:
- Consistency between initial and updated plans
- Maintains coarse-grained nature
- Easier for agent to understand

**Implementation**:
- Updated plans use same exploration goal format
- Same language requirements
- Same prohibitions (no detailed actions)

## Integration Points

### With SafePred
- **PlanMonitor**: Part of SafePred_v9, evaluates action paths
- **World Model**: Tracks progress_step as part of state delta prediction
- **SafePredWrapper**: Formats plan for prompt, handles plan updates

### With Agent
- **PromptAgent**: Generates plan, stores current_plan, integrates plan into prompt
- **Action Generation**: Plan is included in prompt as system message
- **Action Regeneration**: Plan can be updated and re-included

## Configuration

**Enable/Disable**:
- Controlled by `safety_wrapper.planning_enabled` property
- Set via SafePred config: `planning_enable: true/false`

**Plan Generation Model**:
- Uses same LLM as agent (configurable)
- System prompt: "You are a planning assistant for desktop automation tasks..."

## Example Flow

1. **Task Start**: Agent receives instruction "Create a folder named 'Favorites' on bookmarks bar"
2. **Plan Generation**: 
   ```
   Step 1: Explore the browser interface
   Step 2: Navigate to the bookmarks area
   Step 3: Find the folder creation option
   Step 4: Locate the folder naming input
   ```
3. **Plan Display**: Formatted with progress_step=1, added to prompt
4. **Agent Action**: Agent tries clicking bookmarks bar (one way to "explore browser interface")
5. **World Model**: Evaluates action, sees bookmarks bar is now visible, updates progress_step=2
6. **Next Action**: Agent sees plan shows "Navigate to bookmarks area" (step 2), tries different approach
7. **Plan Update** (if needed): If agent gets stuck, PlanMonitor may provide guidance to update plan
8. **Continue**: Process repeats until task complete or plan exhausted

## Benefits

1. **Flexibility**: Agent can try different actions to achieve same exploration goal
2. **Robustness**: Plan doesn't break if UI changes slightly
3. **Guidance**: Provides direction without constraining specific actions
4. **Progress Tracking**: Helps agent understand which exploration stages are complete
5. **Adaptability**: Plan can be updated based on actual execution

## Summary

The plan module is a **coarse-grained, suggestive planning system** that:
- Generates high-level exploration goals (not detailed actions)
- Provides plans as reference (not mandatory constraints)
- Tracks progress based on UI state (not actions taken)
- Allows flexible exploration approaches
- Can be updated during execution if needed

This design balances guidance with flexibility, allowing agents to explore different paths while still providing useful direction.
