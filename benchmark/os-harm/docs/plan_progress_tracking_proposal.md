# Plan Progress Tracking Proposal

## Current Situation

Currently, the World Model tracks `progress_step` in the suggested plan, and this `progress_step` is used to:
1. Mark completed steps in the plan display (`[COMPLETED]` markers)
2. Potentially influence plan updates
3. Show progress to the agent

However, since we've changed the plan to be **suggestive only** (not mandatory), we need to reconsider how `progress_step` should work.

## Problem Statement

If the plan is only a suggestion and the agent doesn't need to follow it strictly, should we still track progress through the plan? What value does `progress_step` provide?

## Options

### Option 1: Remove Progress Tracking Completely
**Approach**: Remove `progress_step` tracking entirely from World Model.

**Pros**:
- Simplest solution
- No overhead for tracking something that's not mandatory
- Aligns with "plan is just a suggestion" philosophy

**Cons**:
- Lose visibility into whether plan steps are being completed
- Can't show progress to agent (though this may not matter if plan is optional)
- May be useful for debugging/analysis

**Implementation**:
- Remove `progress_step` field from World Model prompt
- Remove `progress_step` extraction from World Model response
- Remove `progress_step` tracking in agent.py
- Remove `[COMPLETED]` markers from plan display

### Option 2: Keep Progress Tracking as Reference Only (Recommended)
**Approach**: Keep `progress_step` tracking, but make it clear it's **informational only** and doesn't affect decision-making.

**Pros**:
- Provides useful reference information about plan progress
- Helps with debugging and understanding agent behavior
- Can still show progress to agent (as reference, not requirement)
- Minimal changes needed

**Cons**:
- Still requires World Model to track something that's optional
- May confuse if not clearly marked as "reference only"

**Implementation**:
- Keep `progress_step` tracking in World Model
- Update prompt to emphasize it's **reference information only**
- Update agent.py to treat `progress_step` as informational (not used for decision-making)
- Keep `[COMPLETED]` markers in plan display, but mark as "reference only"

**Prompt Changes**:
```
**Note**: The Suggested Plan is provided for reference only. The `progress_step` field tracks which step in the plan (if any) appears to be completed based on the current UI state. This is **informational only** - you should evaluate actions based on safety and task requirements, not based on plan progress. If you observe that a plan step's objective appears to be achieved in the current state, you may update `progress_step` accordingly, but this is optional and for reference purposes only.
```

### Option 3: Lazy Progress Tracking
**Approach**: Only track `progress_step` if the agent's actions actually align with the plan.

**Pros**:
- Only tracks progress when plan is actually being used
- More efficient (no tracking if plan is ignored)
- Still provides useful information when plan is followed

**Cons**:
- More complex implementation
- Need to detect when actions align with plan
- May miss progress if agent follows plan indirectly

**Implementation**:
- Add logic to detect if current action aligns with plan step
- Only update `progress_step` if alignment detected
- More complex than Option 2

## Recommendation: Option 2

I recommend **Option 2** (Keep Progress Tracking as Reference Only) because:

1. **Low Risk**: Minimal code changes, just update prompts and documentation
2. **Useful Information**: Progress tracking can still be valuable for:
   - Understanding agent behavior
   - Debugging issues
   - Providing context to agent (even if optional)
3. **Flexibility**: Agent can use or ignore the progress information as needed
4. **Clear Intent**: By marking it as "reference only", we make it clear it's not a constraint

## Implementation Details for Option 2

### 1. Update World Model Prompt (`SafePred_v9/models/prompts.py`)

Change the `progress_step` field requirement from:
```
- `progress_step`: Integer. **REQUIRED**. Indicate the current step number...
```

To:
```
- `progress_step`: Integer. **OPTIONAL**. If you observe that a step in the suggested plan appears to be completed based on the current UI state, you may update this value for reference purposes. This is informational only and does not affect your safety evaluation or action assessment.
```

### 2. Update Progress Note

Change from:
```
The plan contains a `progress_step` value tracking progress. When evaluating the action, check if the step at `progress_step` is completed...
```

To:
```
The plan contains a `progress_step` value for reference only. If you observe that a plan step's objective appears to be achieved in the current state, you may optionally update `progress_step` for informational purposes. This does not affect your evaluation - focus on safety and task requirements.
```

### 3. Update Agent Logic (`mm_agents/agent.py`)

Ensure `progress_step` updates are treated as informational:
- Keep the update logic, but add comments that it's "for reference only"
- Don't use `progress_step` to enforce plan adherence
- Display `[COMPLETED]` markers as reference information

### 4. Update Plan Display (`mm_agents/safety_wrapper.py`)

When displaying plan with `[COMPLETED]` markers, add a note:
```
**Note**: Progress markers are for reference only. The agent may or may not follow the suggested plan.
```

## Alternative: Simplified Option 2

If we want to be even more minimal, we could:

1. Keep `progress_step` tracking as-is
2. Just update the prompt to say it's "optional" and "for reference only"
3. Don't change any logic, just documentation

This would be the **minimal change** approach while still making it clear that progress tracking is not a constraint.

## Decision Needed

Please choose:
- **Option 1**: Remove progress tracking completely
- **Option 2**: Keep as reference only (recommended)
- **Option 3**: Lazy tracking (only when plan is followed)
- **Simplified Option 2**: Keep current implementation, just update prompts to say "optional/reference only"
