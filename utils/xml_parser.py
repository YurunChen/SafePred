"""
XML-based output parser for World Model.

Provides a format-insensitive output format using XML tags, which is then
converted to JSON. This avoids JSON format errors (missing commas, truncation, etc.).
"""

import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from ..utils.logger import get_logger

logger = get_logger("SafePred.XMLParser")


def parse_xml_output(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse XML output from LLM and convert to JSON format.
    
    Expected XML format:
    <world_model_prediction>
      <semantic_delta>...</semantic_delta>
      <risk_affordances>
        <new_elements>...</new_elements>
        <removed_elements>...</removed_elements>
        <risk_relevant>...</risk_relevant>
      </risk_affordances>
      <risk_exposure>
        <exposed_risks>...</exposed_risks>
        <risk_level>...</risk_level>
      </risk_exposure>
      <reversibility>
        <is_reversible>...</is_reversible>
        <reversibility_description>...</reversibility_description>
        <return_difficulty>...</return_difficulty>
      </reversibility>
      <violated_policy_ids>...</violated_policy_ids>
      <risk_explanation>...</risk_explanation>
      <url_change>...</url_change>
      <page_type_change>...</page_type_change>
    </world_model_prediction>
    
    Args:
        text: Raw text from LLM containing XML
    
    Returns:
        Parsed JSON dict or None if parsing fails
    """
    # Try to extract XML from markdown code blocks
    xml_str = _extract_xml_from_text(text)
    if not xml_str:
        return None
    
    try:
        # Parse XML
        root = ET.fromstring(xml_str)
        
        # Convert to JSON format
        result = _xml_to_dict(root)
        
        # Validate required fields
        if _validate_world_model_output(result):
            return result
        else:
            logger.warning("XML parsed but missing required fields")
            return None
            
    except ET.ParseError as e:
        logger.error(f"XML parsing failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error parsing XML: {e}")
        return None


def _extract_xml_from_text(text: str) -> Optional[str]:
    """
    Extract XML from text, handling markdown code blocks.
    
    Args:
        text: Text containing XML
    
    Returns:
        Extracted XML string or None
    """
    # Try to find XML in markdown code blocks
    code_block_patterns = [
        r'```xml\s*\n([\s\S]*?)\n```',
        r'```xml\s*([\s\S]*?)```',
        r'```\s*\n([\s\S]*?)\n```',
        r'```\s*([\s\S]*?)```',
    ]
    
    for pattern in code_block_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            xml_str = match.group(1).strip()
            if xml_str.startswith('<'):
                return xml_str
    
    # Try to find XML directly (look for <world_model_prediction>)
    xml_match = re.search(r'<world_model_prediction>[\s\S]*?</world_model_prediction>', text, re.DOTALL)
    if xml_match:
        return xml_match.group(0)
    
    # Try to find any XML structure starting with <
    xml_match = re.search(r'<[\s\S]*?>', text, re.DOTALL)
    if xml_match:
        # Try to find the closing tag
        tag_name_match = re.search(r'<(\w+)', xml_match.group(0))
        if tag_name_match:
            tag_name = tag_name_match.group(1)
            full_match = re.search(rf'<{tag_name}[\s\S]*?</{tag_name}>', text, re.DOTALL)
            if full_match:
                return full_match.group(0)
    
    return None


def _xml_to_dict(element: ET.Element) -> Dict[str, Any]:
    """
    Convert XML element to dictionary.
    
    Args:
        element: XML element
    
    Returns:
        Dictionary representation
    """
    # Check if this element only has <item> children (array format)
    children = list(element)
    if children and all(child.tag == 'item' for child in children):
        # This is an array, return list of item values
        items = []
        for child in children:
            item_value = _xml_to_dict(child)
            items.append(item_value)
        return items
    
    result = {}
    
    # Handle child elements
    for child in children:
        tag = child.tag
        value = _xml_to_dict(child)
        
        # Handle arrays (multiple elements with same tag, or <item> tags)
        if tag == 'item':
            # Should not happen if we handled array case above, but just in case
            if 'items' not in result:
                result['items'] = []
            result['items'].append(value)
        elif tag in result:
            if not isinstance(result[tag], list):
                result[tag] = [result[tag]]
            result[tag].append(value)
        else:
            result[tag] = value
    
    # Handle text content (after processing children)
    text_content = element.text.strip() if element.text else None
    tail_content = element.tail.strip() if element.tail else None
    
    # If we have items collected, return as list
    if 'items' in result:
        return result['items']
    
    # If no children, return text value (parsed)
    if not result and text_content:
        return _parse_text_value(text_content)
    
    # If we have both children and text, text is usually whitespace between tags
    # For leaf nodes with text, the text is the value
    if result:
        # Has children, text is usually whitespace - ignore
        return result
    else:
        # No children, return parsed text or empty dict
        if text_content:
            return _parse_text_value(text_content)
        return {}
    
    return result


def _parse_text_value(text: str) -> Any:
    """
    Parse text value to appropriate Python type.
    
    Args:
        text: Text string
    
    Returns:
        Parsed value (bool, int, float, None, or str)
    """
    text = text.strip()
    if text.lower() == 'true':
        return True
    elif text.lower() == 'false':
        return False
    elif text.lower() == 'null' or text.lower() == 'none' or text == '':
        return None
    elif text.isdigit():
        return int(text)
    elif re.match(r'^\d+\.\d+$', text):
        return float(text)
    else:
        return text


def _validate_world_model_output(result: Dict[str, Any]) -> bool:
    """
    Validate that the parsed output has required fields.
    
    Args:
        result: Parsed dictionary
    
    Returns:
        True if valid, False otherwise
    """
    required_fields = [
        'semantic_delta',
        'risk_affordances',
        'risk_exposure',
        'reversibility',
        'violated_policy_ids',
        'risk_explanation',
    ]
    
    for field in required_fields:
        if field not in result:
            logger.warning(f"Missing required field: {field}")
            return False
    
    # Validate nested structures
    if not isinstance(result.get('risk_affordances'), dict):
        return False
    if not isinstance(result.get('risk_exposure'), dict):
        return False
    if not isinstance(result.get('reversibility'), dict):
        return False
    if not isinstance(result.get('violated_policy_ids'), list):
        return False
    
    return True


def convert_xml_to_world_model_format(xml_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert XML dictionary to World Model's expected JSON format.
    
    Args:
        xml_dict: Dictionary parsed from XML
    
    Returns:
        Dictionary in World Model format
    """
    # Handle nested structures
    risk_affordances = xml_dict.get('risk_affordances', {})
    if isinstance(risk_affordances, dict):
        # Convert list fields
        for key in ['new_elements', 'removed_elements', 'risk_relevant']:
            if key in risk_affordances:
                value = risk_affordances[key]
                if isinstance(value, str):
                    # Try to parse as JSON array if it's a string
                    import json
                    try:
                        risk_affordances[key] = json.loads(value)
                    except:
                        # If not JSON, split by comma or newline
                        risk_affordances[key] = [v.strip() for v in re.split(r'[,\n]', value) if v.strip()]
                elif not isinstance(value, list):
                    risk_affordances[key] = [value] if value else []
    
    risk_exposure = xml_dict.get('risk_exposure', {})
    if isinstance(risk_exposure, dict):
        exposed_risks = risk_exposure.get('exposed_risks', [])
        if isinstance(exposed_risks, str):
            import json
            try:
                risk_exposure['exposed_risks'] = json.loads(exposed_risks)
            except:
                risk_exposure['exposed_risks'] = [v.strip() for v in re.split(r'[,\n]', exposed_risks) if v.strip()]
        elif not isinstance(exposed_risks, list):
            risk_exposure['exposed_risks'] = [exposed_risks] if exposed_risks else []
    
    reversibility = xml_dict.get('reversibility', {})
    if isinstance(reversibility, dict):
        is_reversible = reversibility.get('is_reversible')
        if isinstance(is_reversible, str):
            reversibility['is_reversible'] = is_reversible.lower() in ('true', '1', 'yes')
    
    violated_policy_ids = xml_dict.get('violated_policy_ids', [])
    if isinstance(violated_policy_ids, str):
        import json
        try:
            violated_policy_ids = json.loads(violated_policy_ids)
        except:
            violated_policy_ids = [v.strip() for v in re.split(r'[,\n]', violated_policy_ids) if v.strip()]
    elif not isinstance(violated_policy_ids, list):
        violated_policy_ids = [violated_policy_ids] if violated_policy_ids else []
    
    # Convert risk_affordances to element_changes (new format)
    element_changes = {
        'new_elements': risk_affordances.get('new_elements', []),
        'removed_elements': risk_affordances.get('removed_elements', []),
    }
    
    return {
        'semantic_delta': xml_dict.get('semantic_delta', ''),
        'element_changes': element_changes,  # Convert from risk_affordances
        'long_term_impact': xml_dict.get('long_term_impact', 'Long-term impact assessment not available in XML format'),
        'violated_policy_ids': violated_policy_ids,
        'risk_explanation': xml_dict.get('risk_explanation', ''),
        'optimization_guidance': xml_dict.get('optimization_guidance'),  # May not be in XML format
    }

