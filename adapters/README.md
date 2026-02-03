# Benchmark Adapters Guide

This guide explains how to create a new benchmark adapter for SafePred integration.

## Quick Start

To integrate SafePred with a new benchmark, you only need to implement a simple adapter class.

### Step 1: Create Your Adapter

Create a new file `adapters/your_benchmark.py`:

```python
from .base import BenchmarkAdapter, register_adapter

class YourBenchmarkAdapter(BenchmarkAdapter):
    """Adapter for YourBenchmark."""
    
    def state_to_safepred(self, raw_state, intent=None, metadata=None):
        """Convert your benchmark's state to SafePred format."""
        # SafePred expects:
        # {
        #     "axtree_txt": str,  # Page content/accessibility tree
        #     "url": str,         # Current URL (optional)
        #     "goal": str,        # Task goal/intent
        #     "action_history": List,  # Previous actions (optional)
        #     "chat_messages": List[Dict],  # Conversation history (optional)
        # }
        
        return {
            "axtree_txt": extract_page_content(raw_state),
            "url": extract_url(raw_state),
            "goal": intent or "",
            "action_history": metadata.get("action_history", []) if metadata else [],
        }
    
    def action_to_safepred(self, action):
        """Convert your benchmark's action to SafePred string format."""
        # Examples:
        # - "click [element_id]"
        # - "type [element_id] [text]"
        # - "navigate [url]"
        
        if action.type == "click":
            return f"click [{action.element_id}]"
        elif action.type == "type":
            return f"type [{action.element_id}] {action.text}"
        # ... handle other action types
        
        return str(action)  # Fallback
    
    def action_from_safepred(self, action_str):
        """Convert SafePred action string back to your benchmark's format."""
        # Parse action_str and reconstruct your action object
        # This is optional if you don't need to convert back
        pass

# Register the adapter
register_adapter("your_benchmark", YourBenchmarkAdapter)
```

### Step 2: Use the Wrapper

```python
from SafePred_v3 import SafePredWrapper

# Initialize wrapper with your benchmark
wrapper = SafePredWrapper(
    benchmark="your_benchmark",
    config_path="config/config.yaml",
    risk_threshold=0.7,
    policy_path="policies/my_policies.json"
)

# Evaluate action risk
result = wrapper.evaluate_action_risk(
    state=your_benchmark_state,
    action=your_benchmark_action,
    candidate_actions=[action1, action2, ...],
    intent="Task description",
    metadata={"action_history": [...]}
)

# Check result
if result["is_safe"]:
    execute_action(action)
else:
    print(f"Action rejected: {result['risk_explanation']}")
```

## Adapter Interface

All adapters must implement three methods:

### 1. `state_to_safepred(raw_state, intent=None, metadata=None)`

Converts benchmark-specific state to SafePred format.

**Input:**
- `raw_state`: Your benchmark's state representation (any format)
- `intent`: Task intent/instruction (optional)
- `metadata`: Additional metadata dict (optional)

**Output:**
- Dict with keys: `axtree_txt`, `url`, `goal`, `action_history`, `chat_messages`, etc.

### 2. `action_to_safepred(action)`

Converts benchmark-specific action to SafePred string format.

**Input:**
- `action`: Your benchmark's action representation (any format)

**Output:**
- String like `"click [id]"`, `"type [id] [text]"`, etc.

### 3. `action_from_safepred(action_str)`

Converts SafePred action string back to benchmark format (optional).

**Input:**
- `action_str`: Action string in SafePred format

**Output:**
- Your benchmark's action representation

## SafePred State Format

SafePred expects states in this standard format:

```python
{
    "axtree_txt": str,           # Required: Page content/accessibility tree as text
    "url": str,                  # Optional: Current page URL
    "goal": str,                 # Required: Task goal/intent
    "intent": str,               # Optional: Same as goal (for compatibility)
    "action_history": List,      # Optional: Previous actions
    "chat_messages": List[Dict], # Optional: Conversation history
    # ... other fields as needed
}
```

## SafePred Action Format

Actions should be strings like:
- `"click [element_id]"`
- `"type [element_id] [text]"`
- `"navigate [url]"`
- `"scroll [direction]"`
- `"go_back"`
- `"stop"`

## Examples

See `adapters/visualwebarena.py` for a complete example.

## Benefits

- **Low Coupling**: SafePred doesn't need to know about your benchmark's internals
- **Easy Integration**: Just implement 2-3 simple methods
- **Reusable**: Your adapter can be used by anyone integrating SafePred with your benchmark
- **Maintainable**: Format conversion logic is isolated in one place





