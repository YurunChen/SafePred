"""
Policy Extractor Module - Adapted from rule/policy_extractor5.py and rule/mcp_server.py.

Provides complete policy extraction workflow:
1. Extract text from PDF/Webpage/TXT
2. Extract policies from text
3. Review and clean results (remove duplicates, add policy_id)

Direct API calls without MCP dependency.
"""

import os
import json
import re
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from .document_parser import DocumentParser
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger

logger = get_logger("SafePred.PolicyExtractor")


def extract_text_from_pdf(pdf_path: str, output_txt_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract text from a PDF file.
    
    Adapted from policy_extractor5.py:extract_text_from_pdf.
    
    Args:
        pdf_path: Path to PDF file
        output_txt_path: Optional output path for text
    
    Returns:
        Extraction result with txt_path
    """
    try:
        import pdfplumber
        
        # Extract text from PDF
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n\n"
        
        # Save extracted text
        if not output_txt_path:
            output_txt_path = os.path.join(
                os.path.dirname(pdf_path),
                f"{os.path.splitext(os.path.basename(pdf_path))[0]}_extracted_text.txt"
            )
        
        os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        return {
            "status": "success",
            "message": f"Extracted text from PDF and saved to {output_txt_path}",
            "txt_path": output_txt_path
        }
    except Exception as e:
        error_msg = f"Error extracting text from PDF: {str(e)}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}


def extract_text_from_webpage(url: str, output_txt_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract text content from a webpage.
    
    Adapted from policy_extractor5.py:extract_text_from_webpage.
    
    Args:
        url: Webpage URL
        output_txt_path: Optional output path for text
    
    Returns:
        Extraction result with txt_path
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        
        # Fetch webpage content
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML and extract text
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            element.decompose()
        
        # Get text and clean up
        text = soup.get_text()
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Remove excessive newlines
        text = text.strip()
        
        # Save extracted text
        if not output_txt_path:
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.replace(".", "_")
            output_txt_path = os.path.join(
                os.getcwd(),
                f"{domain}_extracted_text.txt"
            )
        
        os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        return {
            "status": "success",
            "message": f"Extracted text from webpage and saved to {output_txt_path}",
            "txt_path": output_txt_path
        }
    except Exception as e:
        error_msg = f"Error extracting text from webpage: {str(e)}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}


def extract_policies_from_file(
    file_path: str,
    organization: str,
    organization_description: str,
    target_subject: str,
    policy_db_path: Optional[str] = None,
    user_request: str = "",
    bench: Optional[str] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
    timeout: Optional[int] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Extract structured policies from a text file and save them to a central JSON file.
    
    Completely based on mcp_server.py:extract_policies_from_file.
    Direct API calls without MCP dependency.
    
    Args:
        file_path: Path to the file containing text to extract policies from
        organization: Name of the organization
        organization_description: Description of the organization
        target_subject: Target subject (e.g., "Web Agent", "User")
        policy_db_path: Optional path to central policy database file
        user_request: Optional additional user request for extraction
        bench: Optional benchmark name ("osharm" or "stweb") to specify action space
        api_key: LLM API key
        api_url: LLM API URL
        model_name: Model name
        provider: Provider type ('openai', 'qwen', etc.)
        timeout: Request timeout
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
    
    Returns:
        Dict: A dictionary containing extracted policies and metadata
    """
    # Check if file exists
    if not os.path.exists(file_path):
        return {
            "success": False,
            "error": f"File does not exist: {file_path}",
            "policies": [],
            "count": 0
        }
    
    # Read text from file (support PDF, TXT, etc. via DocumentParser)
    try:
        text = DocumentParser.parse_file(file_path)
        logger.info(f"Read {len(text)} characters from {file_path}")
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to parse file: {str(e)}",
            "policies": [],
            "count": 0
        }
    
    # Get file info for reference
    file_info = os.path.basename(file_path)
    
    # Use unified prompt template from prompts module
    from .prompts import PromptTemplates
    
    # Get action space description based on bench parameter
    action_space_description = None
    if bench:
        bench_lower = bench.lower()
        if bench_lower == "osharm":
            from .prompts import get_osharm_action_space_description
            action_space_description = get_osharm_action_space_description()
            logger.info(f"[PolicyExtractor] os-harm action space description loaded (bench='{bench}'): {len(action_space_description)} chars")
        elif bench_lower == "stweb":
            from .prompts import get_stwebagentbench_action_space_description
            action_space_description = get_stwebagentbench_action_space_description()
            logger.info(f"[PolicyExtractor] ST-WebAgentBench action space description loaded (bench='{bench}'): {len(action_space_description)} chars")
        else:
            logger.warning(f"[PolicyExtractor] Unknown bench value '{bench}', expected 'osharm' or 'stweb'. No action space description will be included.")
    else:
        logger.info(f"[PolicyExtractor] No bench parameter provided, no action space description will be included")
    
    # Generate prompt using the updated rule_extraction template
    full_prompt = PromptTemplates.rule_extraction(
        text=text,
        context=None,
        max_text_length=150000,
        organization=organization,
        target_subject=target_subject,
        organization_description=organization_description,
        user_request=user_request,
        action_space_description=action_space_description,
    )
    
    # Debug: Log if action space is in prompt
    if action_space_description:
        if "ACTION SPACE" in full_prompt or "action_space" in full_prompt.lower():
            logger.info(f"[PolicyExtractor] ✓ Action space found in prompt (length: {len(full_prompt)} chars)")
            # Log a snippet to verify
            action_space_start = full_prompt.find("ACTION SPACE")
            if action_space_start != -1:
                snippet = full_prompt[max(0, action_space_start-50):action_space_start+200]
                logger.debug(f"[PolicyExtractor] Action space section snippet: ...{snippet}...")
        else:
            logger.warning(f"[PolicyExtractor] ⚠ Action space description provided but NOT found in prompt!")
    else:
        logger.info(f"[PolicyExtractor] No action space description to include in prompt")
    
    # Initialize policies
    policies = None
    
    # Initialize LLM client
    if not all([api_key, api_url, model_name, provider, timeout is not None, temperature is not None, max_tokens is not None]):
        return {
            "success": False,
            "error": "Missing required LLM configuration parameters",
            "policies": [],
            "count": 0
        }
    
    llm_client = LLMClient(
        api_key=api_key,
        api_url=api_url,
        model_name=model_name,
        provider=provider,
        temperature=temperature or 0.2,
        max_tokens=max_tokens or 20000,
        timeout=timeout,
    )
    
    # Try up to 5 times to get a valid JSON response (exactly as in mcp_server.py)
    for attempt in range(5):
        try:
            # Call LLM API (full_prompt already contains system + user prompts from PromptTemplates.rule_extraction)
            response = llm_client.generate(full_prompt, temperature=temperature or 0.2, max_tokens=max_tokens or 20000)
            
            # Extract the JSON part from the response (exactly as in mcp_server.py:507-513)
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
        
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                # Validate JSON by parsing it
                policies = json.loads(json_str)
                logger.info(f"Extracted {len(policies)} policies from {file_path}")
                break
            
        except Exception as e:
            logger.error(f"Error extracting policies from file (attempt {attempt + 1}): {str(e)}")
            continue
    
    if policies is None:
        return {
            "success": False,
            "error": f"Failed to extract policies from {file_path} after 5 attempts",
            "policies": [],
            "count": 0
        }
    
    # Add source file information to each policy (exactly as in mcp_server.py:532-535)
    # Remove id field if present, only use policy_id
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for idx, policy in enumerate(policies):
        # Remove id field if present
        policy.pop("id", None)
        policy["source_file"] = file_path
        policy["extraction_time"] = timestamp
        # Ensure reference field exists (initialize as empty list if not present)
        if "reference" not in policy:
            policy["reference"] = []
    
    # For very large files, handle them in chunks if needed (exactly as in mcp_server.py:537-540)
    if len(text) > 100000:
        logger.info(f"Large file detected ({len(text)} chars). Processing first 100k characters.")
        logger.info(f"Extracted {len(policies)} policies from the first chunk.")
    
    # Create a local file with just the extracted policies from this file
    # If policy_db_path is provided, use the same directory; otherwise use file's directory
    if policy_db_path:
        output_dir = os.path.dirname(policy_db_path)
    else:
        output_dir = os.path.dirname(file_path)
    local_policies_file = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_policies.json")
    with open(local_policies_file, 'w', encoding='utf-8') as f:
        json.dump(policies, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved extracted policies to {local_policies_file}")
    
    # Save to central JSON file if path is provided (exactly as in mcp_server.py:550-576)
    if policy_db_path:
        all_policies = []
        num_existing_policies = 0
        # Load existing policies if the file exists
        if os.path.exists(policy_db_path):
            try:
                with open(policy_db_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                    # Handle both list and dict formats
                    if isinstance(existing_data, list):
                        all_policies = existing_data
                    elif isinstance(existing_data, dict) and "policies" in existing_data:
                        all_policies = existing_data["policies"]
                    num_existing_policies = len(all_policies)
                    # Ensure all existing policies have reference field
                    for existing_policy in all_policies:
                        if "reference" not in existing_policy:
                            existing_policy["reference"] = []
            except json.JSONDecodeError:
                logger.warning(f"Could not load existing policies from {policy_db_path}. Creating new file.")
        
        # Check if P000 exists in existing policies (reserved for goal alignment policy)
        has_p000 = any(p.get("policy_id") == "P000" for p in all_policies)
        
        # Add new policies
        all_policies.extend(policies)
        
        # Add policy IDs (format as P001, P002, etc., P000 reserved for goal alignment policy)
        # Remove id field if present, only use policy_id
        # Start from 1 if P000 exists, otherwise start from 0 (but P000 will be added later)
        policy_counter = 1 if has_p000 else 1  # Always start from 1, P000 will be added by _ensure_goal_alignment_policy
        for local_policy, global_policy in zip(policies, all_policies[-len(policies):]):
            # Remove id field if present
            local_policy.pop("id", None)
            global_policy.pop("id", None)
            # Skip if already has P000 (shouldn't happen, but just in case)
            existing_id = local_policy.get("policy_id", "")
            if existing_id == "P000":
                logger.debug("Skipping P000 in extract_policies_from_file, will be added later")
                continue
            # Add policy_id in P001 format (P000 reserved)
            policy_id_str = f"P{policy_counter:03d}"
            global_policy["policy_id"] = policy_id_str
            local_policy["policy_id"] = policy_id_str
            # Ensure reference field exists (initialize as empty list if not present)
            if "reference" not in global_policy:
                global_policy["reference"] = []
            if "reference" not in local_policy:
                local_policy["reference"] = []
            policy_counter += 1
        
        # Save all policies
        with open(policy_db_path, 'w', encoding='utf-8') as f:
            json.dump(all_policies, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Added {len(policies)} policies to policy database file at {policy_db_path}")
    
    # Extract policy descriptions for the text portion of the return (exactly as in mcp_server.py:579-581)
    policy_descriptions = [policy.get("policy_description", "No description available") for policy in policies]
    descriptions_text = "\n\n".join([f"{i+1}. {desc}" for i, desc in enumerate(policy_descriptions)])
    
    # Return structured data with all important information (exactly as in mcp_server.py:584-595)
    return {
        "success": True,
        "count": len(policies),
        "policies": policies,
        "policy_descriptions": policy_descriptions,
        "local_file": local_policies_file,
        "policy_db_path": policy_db_path,
        "text_output": f"""EXTRACTED {len(policies)} POLICIES FROM {file_path}:

{descriptions_text}
"""
    }


def review_results(
    policy_file: str,
    organization: str,
    process_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process and clean policy data with the following improvements:
    1. Adds sequential policy_id (P001, P002, ...)
    2. Removes metadata fields (source_file, extraction_time)
    3. Removes id field, only keeps policy_id
    4. Maintains clean output structure
    5. Enhanced error handling
    
    Adapted from policy_extractor5.py:review_results.
    
    Args:
        policy_file: Path to policy JSON file
        organization: Organization name
        process_dir: Optional processing directory for summary
    
    Returns:
        Review result with cleaned policies
    """
    try:
        if not os.path.exists(policy_file):
            raise FileNotFoundError(f"Policy file not found: {policy_file}")

        # Load and validate data
        with open(policy_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # Normalize data structure
        policies = raw_data["policies"] if isinstance(raw_data, dict) and "policies" in raw_data else raw_data
        
        if not isinstance(policies, list):
            raise ValueError("Invalid policy data format: expected list or dict with 'policies' key")

        # Data cleaning and transformation
        cleaned_policies = []
        seen_content = set()
        policy_counter = 1  # Start from 1, P000 will be added later if needed
        
        for policy in policies:
            # Convert to dict if necessary
            if not isinstance(policy, dict):
                policy = {"content": str(policy)}
            
            # Remove unwanted metadata fields
            policy.pop("source_file", None)
            policy.pop("extraction_time", None)
            # Remove id field if present, only use policy_id
            policy.pop("id", None)
            
            # Skip P000 if it exists (will be added later by _ensure_goal_alignment_policy)
            existing_policy_id = policy.get("policy_id", "")
            if existing_policy_id == "P000":
                logger.debug("Skipping P000 in review_results, will be added later by _ensure_goal_alignment_policy")
                continue
            
            # Create content-based deduplication key
            # Exclude policy_id and reference fields for deduplication
            # Only compare core policy content: definitions, scope, policy_description, risk_level
            core_fields = {k: v for k, v in policy.items() 
                          if k not in ["policy_id", "reference", "id", "source_file", "extraction_time"]}
            content_key = json.dumps(core_fields, sort_keys=True)
            
            if content_key not in seen_content:
                seen_content.add(content_key)
                # Add sequential ID in P001 format (P000 reserved for goal alignment policy)
                policy["policy_id"] = f"P{policy_counter:03d}"
                # Ensure reference field exists (initialize as empty list if not present)
                if "reference" not in policy:
                    policy["reference"] = []
                cleaned_policies.append(policy)
                policy_counter += 1

        # Save cleaned data
        output_data = {"policies": cleaned_policies} if isinstance(raw_data, dict) else cleaned_policies
        with open(policy_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        # Generate statistics
        stats = {
            "total_policies": len(cleaned_policies),
            "duplicates_removed": len(policies) - len(cleaned_policies),
            "sample_policy_id": cleaned_policies[0]["policy_id"] if cleaned_policies else None
        }

        # Create summary
        summary_content = (
            f"# Policy Processing Report\n\n"
            f"## Organization: {organization}\n"
            f"## Processing Summary\n"
            f"- Final Policies: {stats['total_policies']}\n"
            f"- Duplicates Removed: {stats['duplicates_removed']}\n"
            f"- First Policy ID: {stats['sample_policy_id']}\n\n"
            f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        summary_path = None
        if process_dir:
            os.makedirs(process_dir, exist_ok=True)
            summary_path = os.path.join(process_dir, "processing_summary.md")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary_content)

        result = {
            "status": "success",
            "policy_file": policy_file,
            "summary_path": summary_path,
            "summary": summary_content,
            "stats": stats
        }
        
        logger.info(f"Review completed. Summary: {summary_path}")
        return result

    except Exception as e:
        error_msg = f"Policy review failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "status": "error",
            "message": error_msg,
            "error_type": type(e).__name__
        }


def _ensure_goal_alignment_policy(policies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensure goal alignment policy exists in the policies list.
    If it doesn't exist, add the fixed goal alignment policy.
    
    Args:
        policies: List of existing policies
        
    Returns:
        Updated policies list with goal alignment policy
    """
    # Check if goal alignment policy already exists by policy_id
    # Use a fixed policy_id for goal alignment policy (e.g., P000 or check if P000 exists)
    goal_alignment_policy_id = "P000"
    has_goal_alignment = any(
        p.get("policy_id") == goal_alignment_policy_id for p in policies
    )
    
    # Also check by description keywords as fallback (more strict matching)
    # Only consider it a goal alignment policy if it explicitly mentions goal alignment/deviation
    if not has_goal_alignment:
        goal_alignment_phrases = [
            "goal alignment",
            "goal deviation", 
            "align with the task",
            "contribute to accomplishing the task goal",
            "actions must contribute to",
            "task objective",
            "deviated action"
        ]
        for policy in policies:
            policy_desc = policy.get("policy_description", "").lower()
            # Check if description contains explicit goal alignment phrases
            if any(phrase in policy_desc for phrase in goal_alignment_phrases):
                has_goal_alignment = True
                logger.debug(f"Goal alignment policy already exists: {policy.get('policy_id')}")
                break
    
    if not has_goal_alignment:
        # Add the fixed goal alignment policy
        goal_alignment_policy = _get_goal_alignment_policy(policies)
        # Insert P000 at the beginning of the list, not at the end
        policies.insert(0, goal_alignment_policy)
        logger.info(f"Automatically added goal alignment policy: {goal_alignment_policy.get('policy_id')} at the beginning of the policy list")
    
    return policies


def _get_goal_alignment_policy(existing_policies: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get the fixed goal alignment policy.
    
    This is a hardcoded policy that requires agent actions to contribute to task goal.
    
    Args:
        existing_policies: List of existing policies (used to determine policy_id if P000 is taken)
        
    Returns:
        Fixed goal alignment policy dictionary
    """
    # Try to use P000, but if it exists, find the next available ID
    goal_alignment_policy_id = "P000"
    
    if existing_policies:
        existing_ids = {p.get("policy_id") for p in existing_policies if p.get("policy_id")}
        if goal_alignment_policy_id in existing_ids:
            # Find the highest policy_id and add 1
            max_policy_id = 0
            for policy in existing_policies:
                policy_id = policy.get("policy_id", "")
                if policy_id and policy_id.startswith("P") and len(policy_id) > 1:
                    try:
                        num = int(policy_id[1:])
                        max_policy_id = max(max_policy_id, num)
                    except ValueError:
                        pass
            goal_alignment_policy_id = f"P{max_policy_id + 1:03d}"
    
    # Fixed policy content - hardcoded (detailed version matching WASP policies)
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
        "risk_level": "high",
        "reference": []  # Initialize empty reference list for policy reference tracking
    }


def determine_input_type(input_path: str) -> str:
    """
    Determine the type of input (PDF, URL, or TXT).
    
    Adapted from policy_extractor5.py:determine_input_type.
    
    Args:
        input_path: Input path or URL
    
    Returns:
        Input type: 'pdf', 'url', or 'txt'
    """
    # Check if it's a URL
    if input_path.startswith("http://") or input_path.startswith("https://"):
        return "url"
    
    # Check file extension
    input_lower = input_path.lower()
    if input_lower.endswith(".pdf"):
        return "pdf"
    elif input_lower.endswith(".txt"):
        return "txt"
    
    # Default to txt if unknown
    return "txt"


def process_policy_input(
    input_path: str,
    organization: str,
    organization_description: str,
    target_subject: str,
    output_path: Optional[str] = None,
    user_request: str = "",
    bench: Optional[str] = None,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    model_name: Optional[str] = None,
    provider: Optional[str] = None,
    timeout: Optional[int] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    policies_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Complete policy processing workflow:
    1. Extract text from PDF/Webpage/TXT
    2. Extract policies from text
    3. Review and clean results
    
    All files are saved in a timestamp-based folder under policies directory.
    
    Adapted from policy_extractor5.py:PolicyProcessor.process_policy_input.
    
    Args:
        input_path: Path to input file (PDF/TXT) or URL
        organization: Organization name
        organization_description: Organization description
        target_subject: Target subject
        output_path: Optional output path for policies (if not provided, will use timestamp folder)
        user_request: Optional additional user request
        bench: Optional benchmark name ("osharm" or "stweb") to specify action space
        api_key: LLM API key
        api_url: LLM API URL
        model_name: Model name
        provider: Provider type
        timeout: Request timeout
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        policies_dir: Base directory for policies (default: SafePred_v3/policies/)
    
    Returns:
        Processing result with policies
    """
    # Create timestamp-based folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine policies directory
    if not policies_dir:
        # Default: SafePred_v3/policies/
        # Get SafePred package directory (policy_extractor.py is in SafePred/models/)
        # __file__ = SafePred/models/policy_extractor.py
        # dirname(__file__) = SafePred/models/
        # dirname(dirname(__file__)) = SafePred/
        safe_pred_dir = os.path.dirname(os.path.dirname(__file__))
        policies_dir = os.path.join(safe_pred_dir, "policies")
    
    # Create timestamp folder
    timestamp_folder = os.path.join(policies_dir, timestamp)
    os.makedirs(timestamp_folder, exist_ok=True)
    logger.info(f"Created timestamp folder: {timestamp_folder}")
    
    # Determine input type
    input_type = determine_input_type(input_path)
    logger.info(f"Processing input: {input_path} (type: {input_type})")
    
    # Step 1: Extract text if needed
    txt_path = None
    if input_type == "pdf":
        # Save extracted text to timestamp folder
        output_txt_path = os.path.join(timestamp_folder, f"{os.path.splitext(os.path.basename(input_path))[0]}_extracted_text.txt")
        result = extract_text_from_pdf(input_path, output_txt_path=output_txt_path)
        if result["status"] != "success":
            return {
                "success": False,
                "error": result.get("message", "Failed to extract text from PDF"),
                "policies": [],
                "count": 0,
                "timestamp_folder": timestamp_folder
            }
        txt_path = result["txt_path"]
    elif input_type == "url":
        # Save extracted text to timestamp folder
        from urllib.parse import urlparse
        parsed_url = urlparse(input_path)
        domain = parsed_url.netloc.replace(".", "_")
        output_txt_path = os.path.join(timestamp_folder, f"{domain}_extracted_text.txt")
        result = extract_text_from_webpage(input_path, output_txt_path=output_txt_path)
        if result["status"] != "success":
            return {
                "success": False,
                "error": result.get("message", "Failed to extract text from webpage"),
                "policies": [],
                "count": 0,
                "timestamp_folder": timestamp_folder
            }
        txt_path = result["txt_path"]
    else:
        # TXT file: copy to timestamp folder if not already there
        if not os.path.dirname(os.path.abspath(input_path)) == os.path.abspath(timestamp_folder):
            import shutil
            txt_filename = os.path.basename(input_path)
            txt_path = os.path.join(timestamp_folder, txt_filename)
            shutil.copy2(input_path, txt_path)
            logger.info(f"Copied TXT file to timestamp folder: {txt_path}")
        else:
            txt_path = input_path
    
    # Step 2: Extract policies from text
    if not output_path:
        output_path = os.path.join(timestamp_folder, f"{organization}_policies.json")
    
    result = extract_policies_from_file(
        file_path=txt_path,
        organization=organization,
        organization_description=organization_description,
        target_subject=target_subject,
        policy_db_path=output_path,
        user_request=user_request,
        bench=bench,
        api_key=api_key,
        api_url=api_url,
        model_name=model_name,
        provider=provider,
        timeout=timeout,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    if not result.get("success", False):
        result["timestamp_folder"] = timestamp_folder
        return result
    
    # Step 3: Review and clean results
    review_result = review_results(
        policy_file=output_path,
        organization=organization,
        process_dir=timestamp_folder
    )
    
    if review_result["status"] != "success":
        logger.warning(f"Review step had issues: {review_result.get('message')}")
        # Still return the extraction result even if review fails
    
    # Reload cleaned policies
    with open(output_path, 'r', encoding='utf-8') as f:
        cleaned_data = json.load(f)
        cleaned_policies = cleaned_data["policies"] if isinstance(cleaned_data, dict) and "policies" in cleaned_data else cleaned_data
    
    # Automatically add goal alignment policy
    cleaned_policies = _ensure_goal_alignment_policy(cleaned_policies)
    
    # Save updated policies with goal alignment policy
    output_data = {"policies": cleaned_policies} if isinstance(cleaned_data, dict) else cleaned_policies
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return {
        "success": True,
        "count": len(cleaned_policies),
        "policies": cleaned_policies,
        "policy_file": output_path,
        "summary_path": review_result.get("summary_path"),
        "stats": review_result.get("stats", {}),
        "text_output": result.get("text_output", ""),
        "timestamp_folder": timestamp_folder,
        "extracted_text_path": txt_path if input_type in ["pdf", "url"] else None
    }
