"""
Text cleaning utilities for SafePred_v3.

Provides unified functions for cleaning template fields from LLM responses.
"""
import re
import logging

logger = logging.getLogger(__name__)


def clean_template_fields(text: str) -> str:
    """
    Remove template fields (OBSERVATION:, OBJECTIVE:, URL:, PREVIOUS ACTION:)
    if LLM echoes the input prompt format at the beginning of the response.
    
    This filters out cases where LLM echoes the prompt template structure,
    keeping only the actual reasoning and action.
    
    Args:
        text: Raw text that may contain template fields
    
    Returns:
        Cleaned text without template fields
    """
    if not text or not isinstance(text, str) or not text.strip():
        return text
    
    text = text.strip()
    
    # Check if text starts with OBSERVATION: (likely LLM echoing input prompt)
    if not text.startswith("OBSERVATION:"):
        return text
    
    lines = text.split('\n')
    cleaned_lines = []
    skip_template = True
    found_reasoning = False
    
    for line in lines:
        line_stripped = line.strip()
        
        if skip_template:
            # Check if this line is a template field
            is_template_field = (
                line_stripped.startswith("OBSERVATION:") or
                line_stripped.startswith("URL:") or
                line_stripped.startswith("OBJECTIVE:") or
                line_stripped.startswith("PREVIOUS ACTION:") or
                # Also skip lines that look like accessibility tree elements ([id] ...)
                (line_stripped.startswith("[") and "]" in line_stripped and 
                 (line_stripped.count("[") == 1 or len(line_stripped) > 100))
            )
            
            if is_template_field:
                continue  # Skip this template line
            elif not line_stripped:
                continue  # Skip empty lines in template section
            else:
                # Check if this is actual reasoning (not a template field)
                # Look for reasoning keywords that indicate actual content
                reasoning_keywords = [
                    "let's", "think", "step", "need to", "will", "according",
                    "objective is", "i will", "i need", "i should", "i can",
                    "in summary", "the next action", "reflect", "analyze",
                    "consider", "examine", "plan", "decide"
                ]
                
                if any(keyword in line.lower() for keyword in reasoning_keywords):
                    skip_template = False
                    found_reasoning = True
                    cleaned_lines.append(line)
                elif found_reasoning:
                    # Already found reasoning, this might be continuation or content
                    skip_template = False
                    cleaned_lines.append(line)
                # If we haven't found reasoning yet and this doesn't look like template,
                # it might be content (allow through after a few template lines)
                else:
                    # Allow through non-template lines that don't start with [
                    if not line_stripped.startswith("["):
                        skip_template = False
                        found_reasoning = True
                        cleaned_lines.append(line)
        else:
            # We're past the template section, keep all remaining lines
            cleaned_lines.append(line)
    
    # If we removed everything, return original text
    if not cleaned_lines:
        logger.warning("[TextCleaner] Template cleaning removed all content, keeping original text")
        return text
    
    cleaned_text = '\n'.join(cleaned_lines).strip()
    if cleaned_text:
        logger.debug("[TextCleaner] Cleaned template fields from text (removed OBSERVATION/OBJECTIVE/URL/PREVIOUS ACTION)")
        return cleaned_text
    else:
        # If cleaning resulted in empty string, return original
        return text

