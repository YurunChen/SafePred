"""
PlanMonitor: checks action/path consistency with plan via LLM; returns optimization_guidance for plan correction.
"""

from typing import Dict, Any, Optional, List
import json
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from ..utils.formatters import format_object_to_string

logger = get_logger("SafePred.PlanMonitor")


class PlanMonitor:
    """Checks action/path consistency with plan via LLM; returns optimization_guidance for plan correction."""
    
    def __init__(
        self,
        llm_config: Dict[str, Any],
    ):
        """
        Initialize PlanMonitor.
        
        Args:
            llm_config: LLM configuration dict (required)
        """
        if llm_config is None:
            raise ValueError("llm_config is required for PlanMonitor initialization")
        
        # Validate required fields
        required_fields = ['api_key', 'api_url']
        missing_fields = [field for field in required_fields if not llm_config.get(field)]
        if missing_fields:
            raise ValueError(f"llm_config missing required fields: {missing_fields}")
        
        # Initialize LLM client for plan consistency checking
        self.llm_client = LLMClient(
            provider=llm_config.get('provider', 'openai'),
            api_key=llm_config.get('api_key'),
            api_url=llm_config.get('api_url'),
            model_name=llm_config.get('model_name', 'gpt-4'),
            temperature=llm_config.get('temperature', 0.7),
            max_tokens=llm_config.get('max_tokens', 512),
            timeout=llm_config.get('timeout', 30),
        )
    
    def _check_plan_consistency(
        self,
        plan_text: str,
        current_state: Dict,
        simulated_path: List[str],
    ) -> str:
        """Check if simulated path is consistent with plan via LLM; return optimization_guidance if not."""
        # Format state for prompt
        state_str = format_object_to_string(current_state)
        
        # Format simulated path
        path_str = "\n".join([f"  Step {i+1}: {action}" for i, action in enumerate(simulated_path)])
        
        # Build prompt for plan consistency check
        # Note: simulated_path may contain multiple future steps (depth >= 2 in tree search)
        # These are predicted future actions that haven't been executed yet
        num_steps = len(simulated_path)
        is_multi_step = num_steps > 1
        
        prompt = f"""You are a plan advisor. Your task is to evaluate if the simulated action path has any fundamental problems.

**SUGGESTED PLAN:**
{plan_text}

**CURRENT STATE:**
{state_str}

**SIMULATED ACTION PATH:**
{path_str}
{"**NOTE:** This is a multi-step simulated path (predicted future actions that haven't been executed yet). Evaluate if the sequence makes sense for the task." if is_multi_step else "**NOTE:** This is a single-step action. Evaluate if it makes sense for the task."}

**TASK:**
Evaluate if the simulated action path has any fundamental problems. The suggested plan is a high-level guide for reference - the agent can and should try different approaches to achieve the goals. The plan may be inaccurate or incomplete. **Only flag as inconsistent if there is a fundamental problem** with the action path itself (e.g., clearly incorrect actions, repeating completed steps unnecessarily, or actions that don't make sense for the task). Do NOT flag inconsistency just because the path deviates from the plan if the path is still reasonable. Consider:
1. Is the path making progress toward completing the task goal?
2. Are the actions reasonable and appropriate for the current state?
3. Is the path unnecessarily repeating steps that have already been completed?
4. Are there clear errors or problems with the action sequence?
5. Would the path work even if it doesn't match the plan exactly?

**OUTPUT:**
- If the path is reasonable and making progress: return consistent (no guidance needed)
- Only if there is a **fundamental problem** with the path (e.g., clear errors, unnecessary repetition, or actions that don't make sense): provide guidance explaining:
  * What the problem is
  * Why the path is problematic
  * Suggest improvements if needed

Return ONLY a JSON object with this structure:
{{
  "is_consistent": true/false,
  "optimization_guidance": "Guidance text if inconsistent, null if consistent"
}}

Return ONLY the JSON object, no additional text or markdown formatting."""

        try:
            # Call LLM for plan consistency check
            response = self.llm_client.generate(prompt)
            
            # Parse JSON response
            response_text = response.strip()
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parse JSON
            try:
                result = json.loads(response_text)
                optimization_guidance = result.get("optimization_guidance") or ""
                is_consistent = result.get("is_consistent", True)
                
                if not is_consistent and optimization_guidance:
                    logger.info(f"[PlanMonitor] Action inconsistent with plan: {optimization_guidance}")
                    return optimization_guidance
                else:
                    logger.debug("[PlanMonitor] Action is consistent with plan")
                    return ""
            except json.JSONDecodeError as e:
                logger.warning(f"[PlanMonitor] Failed to parse LLM response as JSON: {e}")
                # Fallback: check if response contains guidance-like text
                if response_text and len(response_text) > 20:
                    logger.info(f"[PlanMonitor] Using raw response as guidance: {response_text}")
                    return response_text
                return ""
                
        except Exception as e:
            logger.warning(f"[PlanMonitor] Error checking plan consistency: {e}")
            return ""
    
    def monitor_with_plan(
        self,
        plan_text: str,
        current_state: Dict,
        action: str,
        simulated_path: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Monitor action/path against plan; return optimization_guidance if inconsistent."""
        # Use simulated_path if available (tree search mode), otherwise use single action
        # Convert to path format: simulated_path takes priority, fallback to single action
        path_to_check = simulated_path if simulated_path and len(simulated_path) > 1 else ([action] if action else [])
        optimization_guidance = self._check_plan_consistency(
            plan_text=plan_text,
            current_state=current_state,
            simulated_path=path_to_check,
        )
        
        return {"optimization_guidance": optimization_guidance}

