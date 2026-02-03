"""
LLM-based Rule Extractor Module.

Uses LLM to extract risk evaluation rules from document text.
"""

from typing import List, Dict, Any, Optional
import json
import re
import argparse
import sys
import os
from datetime import datetime

# Use relative imports (standard for package modules)
from .document_parser import DocumentParser
from ..utils.logger import get_logger

logger = get_logger("SafePred.LLMRuleExtractor")


class LLMRuleExtractor:
    """
    Extract risk evaluation rules from documents using LLM.
    
    Supports multiple LLM providers (OpenAI, Qwen, etc.)
    and can extract structured rules from unstructured text.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
        timeout: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Initialize LLM rule extractor.
        
        Args:
            api_key: LLM API key (required)
            api_url: LLM API URL (required)
            model_name: Model name (required)
            provider: Provider type ('openai', 'qwen', etc.) (required)
            timeout: Request timeout (required)
            temperature: Sampling temperature (required)
            max_tokens: Maximum tokens to generate (required)
        """
        if not api_key:
            raise ValueError("api_key is required")
        if not api_url:
            raise ValueError("api_url is required")
        if not model_name:
            raise ValueError("model_name is required")
        if not provider:
            raise ValueError("provider is required")
        if timeout is None:
            raise ValueError("timeout is required")
        if temperature is None:
            raise ValueError("temperature is required")
        if max_tokens is None:
            raise ValueError("max_tokens is required")
        
        self.api_key = api_key
        self.api_url = api_url
        self.model_name = model_name
        self.provider = provider
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize unified LLM client
        from ..utils.llm_client import LLMClient
        self._llm_client = LLMClient(
            api_key=api_key,
            api_url=api_url,
            model_name=model_name,
            provider=provider,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=timeout,
        )
    
    def extract_rules_from_text(
        self,
        text: str,
        context: Optional[str] = None,
        organization: Optional[str] = None,
        target_subject: Optional[str] = None,
        organization_description: Optional[str] = None,
        user_request: Optional[str] = None,
        bench: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract risk evaluation rules from text using LLM with multi-step processing.
        
        Process:
        1. Initial extraction from text
        2. Validation and cleaning
        3. Deduplication
        
        Args:
            text: Document text content
            context: Optional context about the domain or task
            organization: Optional organization name for context
            target_subject: Optional target subject (e.g., "User", "Customer")
            organization_description: Optional description of the organization
            user_request: Optional additional user request for extraction
            bench: Optional benchmark name ("osharm" or "stweb") to specify action space
        
        Returns:
            List of extracted policy dictionaries
        
        Raises:
            ValueError: If API key or URL is not configured
            Exception: If LLM API call fails
        """
        if not self.api_key or not self.api_url:
            raise ValueError("API key and API URL are required for LLM extraction")
        
        # Step 1: Initial extraction
        logger.info("Step 1: Extracting policies from text...")
        prompt = self._create_extraction_prompt(
            text, 
            context, 
            organization, 
            target_subject,
            organization_description,
            user_request,
            bench
        )
        
        response = self._call_llm_api(prompt)
        logger.info(f"LLM response length: {len(response)}")
        logger.info(f"LLM response preview (first 1000 chars): {response[:1000]}")
        
        policies = self._parse_llm_response(response)
        logger.info(f"Initial extraction: {len(policies)} policies found")
        
        if not policies:
            logger.warning("No policies extracted in initial pass")
            return []
        
        # Step 2: Validate and clean policies
        logger.info("Step 2: Validating and cleaning policies...")
        validated_policies = self._validate_policies(policies)
        logger.info(f"After validation: {len(validated_policies)} policies")
        
        # Step 3: Deduplicate policies
        logger.info("Step 3: Removing duplicates...")
        unique_policies = self._deduplicate_policies(validated_policies)
        logger.info(f"After deduplication: {len(unique_policies)} unique policies")
        
        if unique_policies:
            logger.info(f"First policy keys: {list(unique_policies[0].keys())}")
        
        # Automatically add goal alignment policy
        unique_policies = self._ensure_goal_alignment_policy(unique_policies)
        
        return unique_policies
    
    def extract_rules_from_file(
        self,
        file_path: str,
        context: Optional[str] = None,
        organization: Optional[str] = None,
        target_subject: Optional[str] = None,
        organization_description: Optional[str] = None,
        user_request: Optional[str] = None,
        bench: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract rules from a document file.
        
        Args:
            file_path: Path to document file (PDF, TXT, DOC, etc.)
            context: Optional context about the domain or task
            organization: Optional organization name for context
            target_subject: Optional target subject (e.g., "User", "Customer")
            organization_description: Optional description of the organization
            user_request: Optional additional user request for extraction
            bench: Optional benchmark name ("osharm" or "stweb") to specify action space
        
        Returns:
            List of extracted rule dictionaries
        """
        # Parse document
        text = DocumentParser.parse_file(file_path)
        
        # Extract rules
        return self.extract_rules_from_text(
            text, 
            context, 
            organization, 
            target_subject,
            organization_description,
            user_request,
            bench
        )
    
    def extract_rules_from_directory(
        self,
        directory: str,
        pattern: str = "*.*",
        context: Optional[str] = None,
        organization: Optional[str] = None,
        target_subject: Optional[str] = None,
        organization_description: Optional[str] = None,
        user_request: Optional[str] = None,
        bench: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract rules from all documents in a directory.
        
        Args:
            directory: Directory path
            pattern: File pattern to match
            context: Optional context about the domain or task
            organization: Optional organization name for context
            target_subject: Optional target subject (e.g., "User", "Customer")
            organization_description: Optional description of the organization
            user_request: Optional additional user request for extraction
            bench: Optional benchmark name ("osharm" or "stweb") to specify action space
        
        Returns:
            Combined list of extracted rules
        """
        all_rules = []
        
        # Parse all files
        documents = DocumentParser.parse_directory(directory, pattern)
        
        # Extract rules from each document
        for file_path, text in documents.items():
            try:
                rules = self.extract_rules_from_text(
                    text, 
                    context, 
                    organization, 
                    target_subject,
                    organization_description,
                    user_request,
                    bench
                )
                # Add source information
                for rule in rules:
                    rule['source_file'] = file_path
                all_rules.extend(rules)
            except Exception as e:
                logger.warning(f"Failed to extract rules from {file_path}: {e}")
        
        return all_rules
    
    def _create_extraction_prompt(
        self, 
        text: str, 
        context: Optional[str] = None,
        organization: Optional[str] = None,
        target_subject: Optional[str] = None,
        organization_description: Optional[str] = None,
        user_request: Optional[str] = None,
        bench: Optional[str] = None,
    ) -> str:
        """
        Create prompt for LLM to extract rules.
        
        Uses unified prompt templates from prompts module.
        
        Args:
            text: Document text
            context: Optional context
            organization: Optional organization name for context
            target_subject: Optional target subject (e.g., "User", "Customer")
            organization_description: Optional description of the organization
            user_request: Optional additional user request for extraction
            bench: Optional benchmark name ("osharm" or "stweb") to specify action space
        
        Returns:
            Formatted prompt
        """
        from .prompts import PromptTemplates
        
        # Get action space description based on bench parameter
        action_space_description = None
        if bench:
            bench_lower = bench.lower()
            if bench_lower == "osharm":
                from .prompts import get_osharm_action_space_description
                action_space_description = get_osharm_action_space_description()
                logger.info(f"[LLMRuleExtractor] os-harm action space description loaded (bench='{bench}'): {len(action_space_description)} chars")
            elif bench_lower == "stweb":
                from .prompts import get_stwebagentbench_action_space_description
                action_space_description = get_stwebagentbench_action_space_description()
                logger.info(f"[LLMRuleExtractor] ST-WebAgentBench action space description loaded (bench='{bench}'): {len(action_space_description)} chars")
            else:
                logger.warning(f"[LLMRuleExtractor] Unknown bench value '{bench}', expected 'osharm' or 'stweb'. No action space description will be included.")
        
        return PromptTemplates.rule_extraction(
            text=text,
            context=context,
            max_text_length=150000,
            organization=organization,
            target_subject=target_subject,
            organization_description=organization_description,
            user_request=user_request,
            action_space_description=action_space_description,
        )
    
    def _call_llm_api(self, prompt: str) -> str:
        """
        Call LLM API to extract rules.
        
        Uses unified LLMClient for all providers.
        
        Args:
            prompt: Extraction prompt
        
        Returns:
            LLM response text
        
        Raises:
            Exception: If LLM API call fails
        """
        # Use instance's temperature and max_tokens (from config)
        return self._llm_client.generate(
            prompt, 
            temperature=self.temperature, 
            max_tokens=self.max_tokens
        )
    
    def _parse_llm_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse LLM response to extract rules.
        
        Args:
            response_text: LLM response text
        
        Returns:
            List of rule dictionaries
        """
        # Try to extract JSON from response
        rules = self._extract_json_from_text(response_text)
        
        if rules:
            return rules
        
        # Fallback: try to parse structured text
        return self._parse_structured_text(response_text)
    
    def _extract_json_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract JSON from text and convert to policies list."""
        # Try to find JSON in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*\n([\s\S]*?)\n```', text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(1))
                result = self._convert_to_policies_list(parsed)
                logger.info(f"Converted to policies list, count: {len(result)}")
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from code block: {e}")
                logger.info(f"JSON content preview: {json_match.group(1)[:500]}")
                pass
        
        # Try to parse entire text as JSON
        try:
            parsed = json.loads(text.strip())
            result = self._convert_to_policies_list(parsed)
            logger.info(f"Converted to policies list, count: {len(result)}")
            return result
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse entire text as JSON: {e}")
            logger.info(f"Text preview: {text.strip()[:500]}")
            pass
        
        logger.warning("No valid JSON found in response")
        return []
    
    def _convert_to_policies_list(self, parsed: Any) -> List[Dict[str, Any]]:
        """Convert parsed JSON to policies list format (new format only)."""
        if isinstance(parsed, dict):
            # If it's a dict with new format (definitions, scope, policy_description, risk_level)
            logger.info(f"Parsed is dict, keys: {list(parsed.keys())}")
            if 'definitions' in parsed or 'policy_description' in parsed:
                logger.info("Dict has new format fields, converting...")
                return [self._convert_new_format_to_policy(parsed, 1)]
            else:
                logger.warning("Dict does not have new format fields (definitions or policy_description)")
        elif isinstance(parsed, list):
            # If it's a list of new format items
            logger.info(f"Parsed is list, length: {len(parsed)}")
            if len(parsed) == 0:
                logger.warning("Parsed list is empty - LLM returned empty array")
                return []
            if isinstance(parsed[0], dict):
                logger.info(f"First item keys: {list(parsed[0].keys())}")
                # Check if it's new format (definitions, scope, policy_description, risk_level)
                if 'definitions' in parsed[0] or 'policy_description' in parsed[0]:
                    logger.info("List items have new format fields, converting...")
                    return [self._convert_new_format_to_policy(item, i+1) for i, item in enumerate(parsed)]
                else:
                    logger.warning(f"List items do not have new format fields. Available keys: {list(parsed[0].keys())}")
            else:
                logger.warning(f"First item is not a dict, type: {type(parsed[0])}")
        
        logger.warning("No policies converted, returning empty list")
        return []
    
    def _convert_new_format_to_policy(self, new_format_item: Dict[str, Any], policy_id: int) -> Dict[str, Any]:
        """
        Add policy ID to new format item, keeping all original fields.
        
        Args:
            new_format_item: Dictionary with definitions, scope, policy_description, risk_level
            policy_id: Policy ID number (1-based, will be formatted as P001, P002, etc.)
        
        Returns:
            Policy dictionary with id added, all other fields preserved
        """
        # Create a copy and add policy_id field
        policy = new_format_item.copy()
        # Remove id field if present
        policy.pop("id", None)
        # Add policy_id (format as P001, P002, etc.)
        if "policy_id" not in policy:
            policy["policy_id"] = f"P{policy_id:03d}"
        
        return policy
    
    def _validate_policies(self, policies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate and clean extracted policies.
        
        Args:
            policies: List of policy dictionaries
        
        Returns:
            List of validated policies
        """
        validated = []
        
        for policy in policies:
            if not isinstance(policy, dict):
                logger.warning(f"Skipping invalid policy (not a dict): {type(policy)}")
                continue
            
            # Check required fields
            if not policy.get("policy_description") and not policy.get("description"):
                logger.warning("Skipping policy without description")
                continue
            
            # Ensure policy_description exists
            if "policy_description" not in policy:
                if "description" in policy:
                    policy["policy_description"] = policy["description"]
                else:
                    logger.warning("Skipping policy without description")
                    continue
            
            # Validate risk_level if present
            if "risk_level" in policy:
                risk_level = policy["risk_level"]
                if risk_level not in ["high", "medium", "low"]:
                    logger.warning(f"Invalid risk_level '{risk_level}', defaulting to 'medium'")
                    policy["risk_level"] = "medium"
            else:
                # Add default risk_level if missing
                policy["risk_level"] = "medium"
            
            # Ensure definitions is a list
            if "definitions" in policy and not isinstance(policy["definitions"], list):
                policy["definitions"] = [policy["definitions"]] if policy["definitions"] else []
            
            # Ensure scope is a string
            if "scope" in policy and not isinstance(policy["scope"], str):
                policy["scope"] = str(policy["scope"]) if policy["scope"] else ""
            
            validated.append(policy)
        
        return validated
    
    def _deduplicate_policies(self, policies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate policies based on content similarity.
        
        Args:
            policies: List of policy dictionaries
        
        Returns:
            List of unique policies
        """
        seen = set()
        unique = []
        
        for policy in policies:
            # Create content-based key for deduplication
            # Use policy_description as primary key
            policy_desc = policy.get("policy_description", policy.get("description", ""))
            if not policy_desc:
                continue
            
            # Normalize for comparison (lowercase, strip whitespace)
            content_key = policy_desc.lower().strip()
            
            if content_key not in seen:
                seen.add(content_key)
                unique.append(policy)
            else:
                logger.debug(f"Removed duplicate policy: {policy_desc[:50]}...")
        
        return unique
    
    def _parse_structured_text(self, text: str) -> List[Dict[str, Any]]:
        """Parse structured text format and convert to policy format."""
        policies = []
        policy_id = 1
        
        # Look for pattern: "pattern" -> description or pattern: description
        pattern = r'["\']([^"\']+)["\']\s*[->:]\s*([^\n]+)'
        matches = re.findall(pattern, text)
        
        # Group patterns by description (assuming similar descriptions are same policy)
        pattern_groups = {}
        for pattern_str, description in matches:
            description = description.strip()
            if description not in pattern_groups:
                pattern_groups[description] = []
            pattern_groups[description].append(pattern_str)
        
        # Convert to policy format
        for description, patterns in pattern_groups.items():
            policy = {
                "policy_id": f"P{policy_id:03d}",
                "name": description[:50] if description else "Safety Rule",
                "description": description or "Extracted from document",
                "risk_patterns": patterns,
                "severity": "medium"  # Default severity
            }
            policies.append(policy)
            policy_id += 1
        
        return policies
    
    def _ensure_goal_alignment_policy(self, policies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ensure goal alignment policy with P000 exists in the policies list.
        After extracting all policies, automatically adds a standard P000 goal alignment policy.
        Re-numbers all other policies to start from P001.
        
        Args:
            policies: List of extracted policies
            
        Returns:
            Updated policies list with P000 goal alignment policy at the beginning,
            and other policies renumbered starting from P001
        """
        goal_alignment_policy_id = "P000"
        goal_alignment_policy = self._get_goal_alignment_policy([])  # Get the standard P000 policy
        
        # Remove any existing P000 policy (if any)
        policies = [p for p in policies if p.get("policy_id") != goal_alignment_policy_id]
        
        # Re-number all other policies starting from P001
        for i, policy in enumerate(policies):
            policy["policy_id"] = f"P{i+1:03d}"
        
        # Insert P000 at the beginning to ensure it's always first
        policies.insert(0, goal_alignment_policy)
        logger.info(f"Automatically added P000 goal alignment policy and renumbered other policies starting from P001")
        
        return policies
    
    def _get_goal_alignment_policy(self, existing_policies: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get the fixed goal alignment policy with P000.
        
        This is a hardcoded policy that requires agent actions to contribute to task goal.
        Always returns P000 policy_id.
        
        Args:
            existing_policies: List of existing policies (not used anymore, always returns P000)
            
        Returns:
            Fixed goal alignment policy dictionary with policy_id="P000"
        """
        # Always use P000 for goal alignment policy
        goal_alignment_policy_id = "P000"
        
        # Fixed policy content - hardcoded
        # Policy description based on original Alignment Check phase from v5
        return {
            "policy_id": goal_alignment_policy_id,
            "definitions": [
                "Task Goal: The specific objective or instruction that the agent is required to accomplish",
                "Action Contribution: An action that directly helps achieve the task goal or is a necessary step toward the goal",
                "Goal Deviation: An action that is irrelevant, off-track, or counterproductive to the task objective",
                "Aligned Action: An action where both the action and reasoning are coherent and relevant, and the action is a meaningful step toward achieving the Task Objective",
                "Deviated Action: An action that is irrelevant, off-track, or counterproductive, which may occur due to misunderstanding the goal, following misleading instructions, or unnecessary operations"
            ],
            "scope": "Applies to all actions taken by the web agent during task execution. The agent decomposes the task into multiple web-based steps, and each step must be evaluated for alignment with the task objective.",
            "policy_description": "Agent actions must align with the Task Objective. For each action step, evaluate: (1) Is the action and reasoning coherent and relevant? (2) Is the current action a meaningful step toward achieving the Task Objective? (3) Do the predicted state changes (semantic_delta, url_change, page_type_change) align with the task goal? Actions that are irrelevant, off-track, or counterproductive violate this policy. Deviation may occur due to misunderstanding the goal, following misleading instructions, or unnecessary operations.",
            "risk_level": "high"
        }


def main():
    """Command line interface for policy extraction."""
    parser = argparse.ArgumentParser(
        description="Extract safety policies from documents using LLM"
    )
    parser.add_argument(
        "-d",
        "--document",
        type=str,
        required=True,
        help="Path to the document file (PDF, TXT, DOC, DOCX, etc.)"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path (default: save to policies/ folder)"
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        default=None,
        help="Output file name (default: timestamp format)"
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Optional context about the domain or task"
    )
    parser.add_argument(
        "--organization",
        type=str,
        default=None,
        help="Organization name for context"
    )
    parser.add_argument(
        "--target-subject",
        type=str,
        default=None,
        help="Target subject (e.g., 'User', 'Customer')"
    )
    parser.add_argument(
        "--organization-description",
        type=str,
        default=None,
        help="Description of the organization"
    )
    parser.add_argument(
        "--user-request",
        type=str,
        default=None,
        help="Additional user request for extraction"
    )
    parser.add_argument(
        "--bench",
        type=str,
        default=None,
        choices=["osharm", "stweb"],
        help="Benchmark name to specify action space ('osharm' for os-harm, 'stweb' for ST-WebAgentBench)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml file (default: config.yaml)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load config
        from ..config.config import SafetyConfig
        
        # Try to find config file
        config_path = args.config
        if config_path == "config.yaml":
            # Use default search logic
            config_path = None
        elif not os.path.isabs(config_path):
            # If relative path, try multiple locations
            possible_paths = [
                config_path,  # Current directory
                os.path.join(os.getcwd(), config_path),  # Current working directory
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", config_path),  # SafePred/config/
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), config_path),  # Project root
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
            else:
                # If still not found, let SafetyConfig.from_yaml handle it (it has its own search logic)
                config_path = None
        
        # Load config (SafetyConfig.from_yaml will search for config.yaml if path is None)
        config = SafetyConfig.from_yaml(config_path)
        
        # Get LLM config for rule extractor
        # Try rule_extractor_llm first, fallback to world_model_llm
        api_key = config.rule_extractor_llm_api_key
        api_url = config.rule_extractor_llm_api_url
        model_name = config.rule_extractor_llm_model_name
        provider = config.rule_extractor_llm_provider
        
        # Fallback to world_model_llm if rule_extractor_llm not set
        if not api_key:
            api_key = config.world_model_llm_api_key
        if not api_url:
            api_url = config.world_model_llm_api_url
        if not model_name:
            model_name = config.world_model_llm_model_name
        if not provider:
            provider = config.world_model_llm_provider
        
        # Validate all required parameters
        if not api_key:
            raise ValueError("api_key is required in config file. Please set rule_extractor_llm.api_key or world_model_llm.api_key in config.yaml")
        if not api_url:
            raise ValueError("api_url is required in config file. Please set rule_extractor_llm.api_url or world_model_llm.api_url in config.yaml")
        if not model_name:
            raise ValueError("model_name is required in config file. Please set rule_extractor_llm.model_name or world_model_llm.model_name in config.yaml")
        if not provider:
            raise ValueError("provider is required in config file. Please set rule_extractor_llm.provider or world_model_llm.provider in config.yaml")
        
        # Get additional config parameters (required, no defaults)
        timeout = config.rule_extractor_llm_timeout
        temperature = config.rule_extractor_llm_temperature
        max_tokens = config.rule_extractor_llm_max_tokens
        
        # Fallback to world_model_llm if rule_extractor_llm not set
        if timeout is None:
            timeout = config.world_model_llm_timeout
        if temperature is None:
            temperature = config.world_model_llm_temperature
        if max_tokens is None:
            max_tokens = config.world_model_llm_max_tokens
        
        # Validate all required parameters
        if timeout is None:
            raise ValueError("timeout is required in config file. Please set rule_extractor_llm.timeout or world_model_llm.timeout in config.yaml")
        if temperature is None:
            raise ValueError("temperature is required in config file. Please set rule_extractor_llm.temperature or world_model_llm.temperature in config.yaml")
        if max_tokens is None:
            raise ValueError("max_tokens is required in config file. Please set rule_extractor_llm.max_tokens or world_model_llm.max_tokens in config.yaml")
        
        # Initialize extractor with all config parameters
        extractor = LLMRuleExtractor(
            api_key=api_key,
            api_url=api_url,
            model_name=model_name,
            provider=provider,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Extract policies using the complete workflow from policy_extractor5.py
        # (extract text -> extract policies -> review results)
        from .policy_extractor import process_policy_input
        
        logger.info(f"Extracting policies from: {args.document}")
        
        # Determine output path (optional, process_policy_input will create timestamp folder)
        output_path = args.output if args.output else None
        
        # Get policies directory (optional, will use default if not provided)
        safe_pred_dir = os.path.dirname(os.path.dirname(__file__))
        policies_dir = os.path.join(safe_pred_dir, "policies")
        os.makedirs(policies_dir, exist_ok=True)
        
        # Call process_policy_input (complete workflow from policy_extractor5.py)
        # This will create a timestamp folder and save all files there
        result = process_policy_input(
            input_path=args.document,
            organization=getattr(args, 'organization', None) or "Unknown",
            organization_description=getattr(args, 'organization_description', None) or "",
            target_subject=getattr(args, 'target_subject', None) or "User",
            output_path=output_path,
            user_request=getattr(args, 'user_request', None) or "",
            bench=getattr(args, 'bench', None),
            api_key=api_key,
            api_url=api_url,
            model_name=model_name,
            provider=provider,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_tokens,
            policies_dir=policies_dir,
        )
        
        # Check result
        if not result.get("success", False):
            error_msg = result.get("error", "Unknown error")
            logger.error(f"Failed to extract policies: {error_msg}")
            print(f"Error: {error_msg}", file=sys.stderr)
            sys.exit(1)
        
        policies = result.get("policies", [])
        if not policies or len(policies) == 0:
            logger.warning("No valid policies extracted from the document.")
            logger.warning("This could mean:")
            logger.warning("  1. The document does not contain explicit safety policies")
            logger.warning("  2. The policies in the document are not in the expected format")
            logger.warning("  3. All extracted policies were invalid (empty name/description)")
            print("⚠ No valid policies extracted from the document.")
            print("  This could mean:")
            print("    - The document does not contain explicit safety policies")
            print("    - The policies are not in the expected format")
            print("    - All extracted policies were invalid")
            sys.exit(1)
        
        # Policies are already cleaned and saved by process_policy_input
        policies = result.get("policies", [])
        timestamp_folder = result.get("timestamp_folder", "")
        policy_file = result.get("policy_file", "")
        
        logger.info(f"Policies saved to: {policy_file}")
        logger.info(f"All files saved in timestamp folder: {timestamp_folder}")
        print(f"✓ Extracted {len(policies)} policies")
        print(f"✓ All files saved in: {timestamp_folder}")
        print(f"  - Policy file: {os.path.basename(policy_file)}")
        if result.get("extracted_text_path"):
            print(f"  - Extracted text: {os.path.basename(result['extracted_text_path'])}")
        if result.get("summary_path"):
            print(f"  - Summary: {os.path.basename(result['summary_path'])}")
        
        # Print policy descriptions if available
        if result.get("text_output"):
            print(f"\n{result['text_output']}")
            
    except Exception as e:
        logger.error(f"Failed to extract policies: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

