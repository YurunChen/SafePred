# Plan Module Summary - Coarse-grained & Suggestive

## Overview

The entire plan module has been redesigned to be:
1. **Coarse-grained**: Uses high-level exploration goals instead of detailed actions
2. **Suggestive**: Plan is provided as reference only, not mandatory constraints
3. **Flexible**: Allows agent to explore different actions and approaches

## Components Status

### 1. Plan Generation (`mm_agents/agent.py`)

**Initial Plan Generation:**
- ✅ Uses high-level exploration goals
- ✅ Format: "Step 1: [Exploration goal, e.g., 'Explore the menu options']"
- ✅ Language: "Explore...", "Navigate to...", "Find...", "Discover...", "Locate..."
- ✅ Prohibits: Detailed actions (click, type, press, etc.) or specific coordinates

**Plan Updates:**
- ✅ Same format as initial plan (coarse-grained exploration goals)
- ✅ Uses exploration language, not detailed actions
- ✅ Prohibits detailed actions

### 2. Plan Display (`mm_agents/safety_wrapper.py`)

**Title:**
- ✅ "**SUGGESTED PLAN:**" (not "EXECUTION PLAN")

**Instructions:**
- ✅ "The plan above is **suggested** for reference only - you are not required to follow it strictly"
- ✅ "Make decisions based on the current observation and task requirements, not based on what the plan says"
- ✅ Emphasizes flexibility and observation-based decisions

### 3. PlanMonitor (`SafePred_v9/core/plan_monitor.py`)

**Title:**
- ✅ "**SUGGESTED PLAN:**"

**Evaluation Criteria:**
- ✅ Only flags **fundamental problems** (not plan misalignment)
- ✅ Focuses on: incorrect actions, unnecessary repetition, actions that don't make sense
- ✅ Does NOT enforce plan adherence

### 4. World Model (`SafePred_v9/models/prompts.py`)

**Title:**
- ✅ "### Suggested Plan"

**Progress Tracking:**
- ✅ Tracks completion of **exploration stages** (not detailed steps)
- ✅ Example: If goal is "Explore the menu options" and menu is visible, mark as completed
- ✅ Based on UI state, not action trajectory

**Note:**
- ✅ "provided for reference only"
- ✅ "does not affect your evaluation - focus on safety and task requirements"

### 5. SafePred Wrapper (`SafePred_v9/wrapper.py`)

**Title:**
- ✅ "**SUGGESTED PLAN:**"

**Instructions:**
- ✅ "suggested for reference only - you are not required to follow it strictly"
- ✅ "Make decisions based on the current observation and task requirements"

## Key Features

### Coarse-grained Exploration Goals
- Plans describe **what to explore**, not **how to do it**
- Examples: "Explore the menu options", "Navigate to settings", "Find the configuration dialog"
- No specific actions like "click", "type", "press"

### Suggestive, Not Mandatory
- All prompts emphasize plan is "for reference only"
- Agent is "not required to follow it strictly"
- Decisions should be based on "current observation and task requirements"

### Flexible Exploration
- Agent can try different actions to achieve exploration goals
- Plan provides direction, not constraints
- Agent can deviate if better approaches are found

## Benefits

1. **More Flexible**: Agent can explore different paths to achieve goals
2. **Less Constraining**: Plan doesn't force specific actions
3. **Better Adaptation**: Agent can adapt to actual UI state
4. **Progress Tracking**: Still tracks which exploration stages are completed
5. **Safety First**: Safety evaluation is independent of plan adherence

## Example Plan Format

```
**SUGGESTED PLAN:**

Step 1: Explore the menu options
Step 2: Navigate to the settings area
Step 3: Find the configuration dialog
Step 4: Locate the text input field

**Progress Step:** 2
(Steps 1 to 1 are completed. Current step to execute: 2)

**Note:**
- The plan above is **suggested** for reference only - you are not required to follow it strictly
- Make decisions based on the current observation and task requirements, not based on what the plan says
- If you use any actions from the plan, find actual coordinates or element identifiers from the current observation
```

## Conclusion

✅ **All components are now:**
- Coarse-grained (exploration goals, not detailed actions)
- Suggestive (reference only, not mandatory)
- Flexible (allows different exploration approaches)

This design allows the agent to:
- Use the plan as a guide for exploration direction
- Try different actions to achieve exploration goals
- Adapt to actual UI state
- Make decisions based on observations, not plan constraints
