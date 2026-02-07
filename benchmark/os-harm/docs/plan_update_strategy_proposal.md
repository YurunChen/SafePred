# Plan Update Strategy Proposal

## Current Plan Update Logic

### Current Triggers

1. **Risk-based Update** (`requires_regeneration = True`):
   - All actions filtered due to safety risks
   - Update reason: "Action requires regeneration (all root actions filtered)"
   - Uses `risk_guidance` (contains safety-related optimization_guidance)

2. **Plan Misalignment Update** (`should_check_plan and optimization_guidance`):
   - Actions are safe but don't align with plan
   - Update reason: "Path feasible but plan misaligned"
   - Uses `optimization_guidance` from PlanMonitor

### Current Flow

```
World Model Evaluation
  ↓
PlanMonitor Check (if enabled)
  ↓
  ├─→ Actions safe + plan consistent → No update
  ├─→ Actions safe + plan inconsistent → Update plan (misalignment)
  └─→ Actions unsafe → Update plan (risk guidance)
```

## Problem with Current Strategy

Since we've changed the plan to be **suggestive only**:

1. **"Plan misalignment" should NOT trigger updates**: If plan is just a suggestion, agent can deviate without needing plan update
2. **PlanMonitor should only flag fundamental problems**: Not "inconsistency", but actual errors
3. **Updates should be rare**: Only when plan has actual errors or needs improvement

## Proposed Strategy

### Option 1: Remove Plan Misalignment Updates (Recommended)

**Approach**: Only update plan when there are **safety risks** or **plan errors**, not when agent deviates.

**Triggers**:
1. ✅ **Safety Risk**: All actions filtered → Update plan with risk guidance
2. ❌ **Plan Misalignment**: Remove this trigger (agent can deviate freely)
3. ✅ **Plan Errors**: If PlanMonitor detects actual errors in plan → Update plan

**Implementation**:
- Remove "Path feasible but plan misaligned" update reason
- PlanMonitor should only return `optimization_guidance` for **fundamental problems**, not deviations
- Only update when `requires_regeneration = True` (safety risk)

**Code Changes**:
```python
# In _determine_plan_update()
should_update_plan = requires_regeneration  # Remove plan misalignment check

# PlanMonitor should only flag errors, not deviations
# Current: "inconsistent with plan" → Update
# New: "has fundamental problems" → Update (only if safety-related)
```

### Option 2: Keep Updates but Make Them Optional

**Approach**: Keep plan updates, but make them **optional** and **less frequent**.

**Triggers**:
1. ✅ **Safety Risk**: Always update
2. ⚠️ **Plan Misalignment**: Update only if deviation is **significant** and **problematic**
3. ✅ **Plan Errors**: Always update

**Implementation**:
- Add threshold for "significant misalignment"
- Only update if misalignment causes problems (not just deviation)
- Make updates less aggressive

### Option 3: No Automatic Updates (Manual Only)

**Approach**: Remove automatic plan updates, only update manually or on explicit errors.

**Triggers**:
1. ✅ **Safety Risk**: Update plan (critical)
2. ❌ **Plan Misalignment**: No update
3. ❌ **Plan Errors**: Log but don't auto-update

**Implementation**:
- Only update on `requires_regeneration`
- Remove all plan misalignment logic
- Keep plan as static reference

## Recommendation: Option 1 (Remove Plan Misalignment Updates)

I recommend **Option 1** because:

1. **Aligns with "suggestive plan" philosophy**: Agent can deviate without triggering updates
2. **Reduces unnecessary updates**: Only update when there are real problems (safety risks)
3. **Simpler logic**: Less complexity, easier to maintain
4. **Better performance**: Fewer LLM calls for plan generation

## Implementation Details for Option 1

### 1. Update `_determine_plan_update()` in `SafePred_v9/wrapper.py`

**Current**:
```python
should_update_plan = requires_regeneration or (should_check_plan and bool(optimization_guidance))
```

**New**:
```python
# Only update plan for safety risks, not plan misalignment
should_update_plan = requires_regeneration
```

**Remove**:
```python
else:  # should_check_plan and optimization_guidance
    return {
        'should_update_plan': True,
        'update_reason': "Path feasible but plan misaligned",
        'optimization_guidance': optimization_guidance,
    }
```

### 2. Update PlanMonitor Logic

**Current**: Returns `optimization_guidance` when plan is "inconsistent"

**New**: Only return `optimization_guidance` for **fundamental problems**:
- Clear errors in action sequence
- Unnecessary repetition
- Actions that don't make sense for the task
- **NOT** for simple deviations from plan

**Prompt Update** (already done):
- Changed from "inconsistent with plan" to "has fundamental problems"
- Only flag actual problems, not deviations

### 3. Update SafetyWrapper Logic

**Current**: Updates plan on "plan misalignment"

**New**: Only update on safety risks:
```python
# Remove this logic:
plan_updated_for_misalignment = (
    planning_enabled and 
    plan_update_info.get('update_reason') == "Path feasible but plan misaligned"
)

# Keep only safety-based updates
if requires_regeneration:
    # Update plan with risk guidance
```

### 4. Keep PlanMonitor for Information Only (Optional)

**Option A**: Keep PlanMonitor but don't use it for updates
- PlanMonitor still checks for problems
- But `optimization_guidance` is only used for logging/debugging
- Not used to trigger plan updates

**Option B**: Remove PlanMonitor entirely
- Since plan is suggestive, no need to check consistency
- Only use World Model's risk assessment

**Recommendation**: **Option A** (keep for information)
- Useful for debugging
- Can detect actual problems (not just deviations)
- But don't use it to trigger updates

## Summary of Changes

### What to Remove:
1. ❌ "Path feasible but plan misaligned" update trigger
2. ❌ Plan misalignment check in `_determine_plan_update()`
3. ❌ Logic that updates plan based on PlanMonitor's `optimization_guidance` (unless it's a safety issue)

### What to Keep:
1. ✅ Safety risk-based plan updates (`requires_regeneration`)
2. ✅ PlanMonitor for detecting fundamental problems (information only)
3. ✅ Progress tracking (optional, for reference)

### What to Add:
1. ✅ Clear documentation that plan updates only happen for safety risks
2. ✅ Logging to distinguish between "plan deviation" (ok) and "plan error" (update needed)

## Decision Needed

Please choose:
- **Option 1**: Remove plan misalignment updates (recommended)
- **Option 2**: Keep updates but make them optional/less frequent
- **Option 3**: No automatic updates except safety risks
