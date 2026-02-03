"""
State Preprocessor Module for SafePred.

Preprocesses raw observations into compact, structured state representations
by extracting core features and removing redundant Accessibility Tree details.
"""

import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any
from ..utils.logger import get_logger

logger = get_logger("SafePred.StatePreprocessor")


class StatePreprocessor:
    """
    Preprocesses raw observations into compact, structured state representations.
    
    Extracts core features (URL, key elements, page type) and removes
    redundant Accessibility Tree details to reduce data size and improve efficiency.
    """
    
    def __init__(self, max_chat_messages: int = 5, benchmark: Optional[str] = None):
        """
        Initialize State Preprocessor.
        
        Args:
            max_chat_messages: Maximum number of chat messages to keep in simplified state
            benchmark: Benchmark name (e.g., "osworld", "visualwebarena") to determine parsing format
        """
        self.max_chat_messages = max_chat_messages
        self.benchmark = benchmark.lower() if benchmark else None
        
        # Keywords indicating critical operations
        self.critical_keywords = [
            'save', 'submit', 'create', 'delete', 'confirm', 'purchase',
            'pay', 'transfer', 'remove', 'cancel', 'terminate', 'activate',
            'deactivate', 'approve', 'reject', 'publish', 'commit'
        ]
        
        # System-level containers to filter out (reduce redundancy for desktop environments)
        self.system_containers = {
            'gnome-shell', 'overview', 'system', 'notification', 'calendar',
            'settings', 'power', 'network', 'bluetooth', 'privacy',
            'display', 'accessibility', 'language', 'region', 'date',
            'time', 'sound', 'keyboard', 'mouse', 'touchpad'
        }
        
        # Maximum number of key elements to extract (to avoid token overflow)
        self.max_key_elements = 200
    
    def preprocess(self, raw_obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess raw observation into compact state representation.
        
        Args:
            raw_obs: Raw observation dictionary containing:
                - goal: Task goal string
                - policies: List of policy dictionaries
                - axtree_txt: Full Accessibility Tree text
                - chat_messages: List of chat message dictionaries
        
        Returns:
            Compact state dictionary with structure:
            {
                "goal": str,
                "policies": List[Dict],
                "url": Optional[str],
                "page_type": str,  # "form", "list", "detail", "navigation"
                "key_elements": List[Dict],  # Key interactive elements
                "axtree_txt": str,  # Full accessibility tree (preserved for world model)
                # Note: conversation_history will be added by SafeAgent._preprocess_state()
            }
        """
        if not raw_obs:
            return {}
        
        # Extract core information
        goal = raw_obs.get("goal", "")
        policies = raw_obs.get("policies", [])
        
        # Extract key elements from Accessibility Tree
        axtree_txt = raw_obs.get("axtree_txt", "")
        
        key_elements = self._extract_key_elements(axtree_txt)
        url = self._extract_url(axtree_txt)
        page_type = self._infer_page_type(axtree_txt, key_elements)
        
        # Use accessibility tree directly (already processed by mm_agents or adapter)
        # mm_agents processes the accessibility tree and passes it via linearized_accessibility_tree
        # The adapter handles the conversion, so we use axtree_txt directly
        
        # Note: chat_history is not generated here anymore
        # conversation_history will be added by SafeAgent._preprocess_state() from chat_messages
        # This avoids duplication and ensures we use the full conversation history maintained by SafeAgent
        
        compact_state = {
            "goal": goal,
            "policies": policies,
            "url": url,
            "page_type": page_type,
            "key_elements": key_elements,
            # Use axtree_txt directly (already processed by mm_agents or adapter)
            "axtree_txt": axtree_txt,
            # chat_history removed - use conversation_history from SafeAgent instead
        }
        
        return compact_state
    
    def _extract_key_elements(self, axtree_txt: str) -> List[Dict[str, Any]]:
        """
        Extract key interactive elements from Accessibility Tree.
        
        Only extracts:
        - Buttons (button)
        - Links (link) - especially navigation and action links
        - Form fields (textbox, combobox, checkbox)
        - Tree items (tree-item) - interactive items in tree views (e.g., bookmarks, file browser)
        - Critical operation elements (Save, Submit, Delete, etc.)
        
        Args:
            axtree_txt: Full Accessibility Tree text
        
        Returns:
            List of element dictionaries with structure:
            [
                {
                    "type": "button",
                    "label": "Save",
                    "bid": "2081",  # For web environments (browser element ID)
                    "screencoord": "(100, 200)",  # For desktop environments (position)
                    "critical": True,  # Whether this is a critical operation
                    "disabled": False,
                    "url": Optional[str]  # For links
                },
                ...
            ]
            Note: For desktop environments (OSWorld), use screencoord as identifier instead of bid.
        """
        elements = []
        
        if not axtree_txt:
            return elements
        
        # Determine parsing format based on benchmark type
        # osworld/os-harm: use linearized table format (tag\tname\ttext\tclass) or XML format
        # visualwebarena/wasp: use [bid] [TYPE] [label] format
        if self.benchmark == "osworld" or self.benchmark == "os-harm":
            # OSWorld may use either linearized table format or XML format
            # Try linearized table format first (if starts with "tag\t")
            if axtree_txt and axtree_txt.strip().startswith("tag\t"):
                elements = self._extract_from_linearized_table(axtree_txt)
                # Return elements even if empty (format was correct, just no interactive elements found)
                # This prevents false warnings when format is correct but no extractable elements exist
                return elements
            
            # Fallback to XML format if linearized table format fails or not detected
            if axtree_txt and (axtree_txt.strip().startswith("<") or axtree_txt.strip().startswith("<?xml")):
                elements = self._extract_from_xml(axtree_txt)
                # Return elements even if empty (format was correct, just no interactive elements found)
                return elements
            
            # If format doesn't match either expected format, log warning and return empty
            if axtree_txt and not axtree_txt.strip().startswith("tag\t") and not (axtree_txt.strip().startswith("<") or axtree_txt.strip().startswith("<?xml")):
                logger.warning(
                    f"[StatePreprocessor] Unable to parse axtree_txt for osworld/os-harm benchmark. "
                    f"Expected linearized table format (starts with 'tag\\t') or XML format (starts with '<'). "
                    f"Preview: {repr(axtree_txt[:100])}"
                )
            return elements
        elif self.benchmark == "visualwebarena" or self.benchmark == "wasp":
            # VisualWebArena/WASP uses [bid] [TYPE] [label] format
            visualwebarena_pattern = r'\[(\d*)\]\s+\[(\w+)\]\s+\[([^\]]*)\]'
            for match in re.finditer(visualwebarena_pattern, axtree_txt):
                bid = match.group(1) if match.group(1) else ""
                element_type = match.group(2).lower()
                label = match.group(3).strip()
                
                # Map VisualWebArena types to standard types
                type_mapping = {
                    'a': 'link',
                    'button': 'button',
                    'input': 'textbox',
                    'textarea': 'textarea',
                    'select': 'combobox',
                    'checkbox': 'checkbox',
                    'radio': 'radio',
                }
                element_type = type_mapping.get(element_type, element_type)
                
                # Skip non-interactive elements
                # Include tree-item as it's an interactive element in tree views
                if element_type not in ['button', 'link', 'textbox', 'combobox', 'checkbox', 'radio', 'textarea', 'tree-item']:
                    continue
                
                # Create element dict
                elem_dict = {
                    "type": element_type,
                    "label": label,
                    "bid": bid if bid else "unknown",
                    "critical": any(keyword in label.lower() for keyword in self.critical_keywords),
                    "disabled": False,  # VisualWebArena format doesn't show disabled state
                }
                
                if element_type == 'link':
                    elem_dict["url"] = None  # URL not available in this format
                
                elements.append(elem_dict)
            return elements
        else:
            # Unknown benchmark type
            logger.warning(f"[StatePreprocessor] Unknown benchmark type: {self.benchmark}, returning empty elements")
            return elements
    
    def _extract_from_linearized_table(self, axtree_txt: str) -> List[Dict[str, Any]]:
        """
        Extract key interactive elements from linearized table format accessibility tree.
        
        Format: tab-separated values with header "tag\tname\ttext\tclass"
        Each row represents an element in the accessibility tree.
        
        Args:
            axtree_txt: Linearized table format string
            
        Returns:
            List of element dictionaries
        """
        elements = []
        
        # Debug logging for format check
        axtree_txt_length = len(axtree_txt) if axtree_txt else 0
        axtree_txt_preview = (axtree_txt[:200] if axtree_txt else "None")
        starts_with_tag = axtree_txt.strip().startswith("tag\t") if axtree_txt else False
        logger.info(
            f"[StatePreprocessor] _extract_from_linearized_table: "
            f"axtree_txt length={axtree_txt_length}, "
            f"starts_with_tag_tab={starts_with_tag}"
        )
        
        # Print full a11y tree for debugging
        if axtree_txt:
            logger.info(f"[StatePreprocessor] Full a11y tree (first 1000 chars):\n{axtree_txt[:1000]}")
            if len(axtree_txt) > 1000:
                logger.info(f"[StatePreprocessor] ... (truncated, total length: {len(axtree_txt)})")
        
        if not axtree_txt or not axtree_txt.strip().startswith("tag\t"):
            logger.debug("[StatePreprocessor] Not a linearized table format, returning empty elements")
            return elements
        
        lines = axtree_txt.strip().split('\n')
        if len(lines) < 2:
            logger.warning(f"[StatePreprocessor] Not enough lines in axtree_txt: {len(lines)} lines")
            return elements
        
        logger.info(f"[StatePreprocessor] Processing {len(lines)} lines from axtree_txt")
        
        # Print header line for debugging
        if len(lines) > 0:
            logger.info(f"[StatePreprocessor] Header line: {repr(lines[0])}")
        
        # Track statistics
        total_lines_processed = 0
        skipped_empty = 0
        skipped_insufficient_parts = 0
        skipped_no_element_type = 0
        skipped_no_label = 0
        
        # Skip header line (tag\tname\ttext\tclass)
        for line_idx, line in enumerate(lines[1:], start=1):
            if not line.strip():
                skipped_empty += 1
                continue
            
            total_lines_processed += 1
            parts = line.split('\t')
            if len(parts) < 4:
                skipped_insufficient_parts += 1
                if line_idx <= 5:  # Log first few problematic lines
                    logger.debug(f"[StatePreprocessor] Line {line_idx} has insufficient parts ({len(parts)} < 4): {repr(line[:100])}")
                continue
            
            tag = parts[0].strip().lower()
            name = parts[1].strip()
            text = parts[2].strip()
            # class_name = parts[3].strip() if len(parts) > 3 else ""  # Not used currently
            
            # Determine element type from tag
            # Note: OSWorld uses various tag formats, including hyphenated versions (e.g., "push-button", "toggle-button")
            element_type = None
            tag_lower = tag.lower()
            
            # Handle button types (including hyphenated variants)
            if tag_lower in ['button', 'pushbutton', 'push-button', 'push_button']:
                element_type = 'button'
            elif tag_lower in ['toggle-button', 'togglebutton', 'toggle_button']:
                element_type = 'button'  # Toggle buttons are also interactive buttons
            elif tag_lower in ['link', 'hyperlink']:
                element_type = 'link'
            elif tag_lower in ['textbox', 'text', 'entry', 'text-field', 'textfield']:
                element_type = 'textbox'
            elif tag_lower in ['combobox', 'combo', 'combo-box', 'combo_box']:
                element_type = 'combobox'
            elif tag_lower in ['checkbox', 'check-box', 'check_box']:
                element_type = 'checkbox'
            elif tag_lower in ['radio', 'radiobutton', 'radio-button', 'radio_button']:
                element_type = 'radio'
            elif tag_lower in ['tree-item', 'treeitem', 'tree_item']:
                element_type = 'tree-item'
            
            # Skip non-interactive elements
            if not element_type:
                skipped_no_element_type += 1
                if line_idx <= 10:  # Log first few skipped tags
                    logger.debug(f"[StatePreprocessor] Line {line_idx} skipped (tag='{tag}' not interactive): name='{name}', text='{text}'")
                continue
            
            # Use name or text as label (prefer name)
            label = name if name else text
            if not label:
                skipped_no_label += 1
                if line_idx <= 10:  # Log first few skipped labels
                    logger.debug(f"[StatePreprocessor] Line {line_idx} skipped (no label): tag='{tag}', name='{name}', text='{text}'")
                continue
            
            # Create element dict
            elem_dict = {
                "type": element_type,
                "label": label,
                "critical": any(keyword in label.lower() for keyword in self.critical_keywords),
                "disabled": False,  # Linearized format doesn't show disabled state
            }
            
            # For desktop environments, we might have screen coordinates in other fields
            # For now, we'll use the label as the primary identifier
            
            elements.append(elem_dict)
            if len(elements) <= 10:  # Log first few extracted elements
                logger.debug(f"[StatePreprocessor] Extracted element {len(elements)}: {elem_dict}")
        
        # Log statistics
        logger.info(
            f"[StatePreprocessor] Extraction statistics: "
            f"total_lines={len(lines)-1}, "
            f"processed={total_lines_processed}, "
            f"skipped_empty={skipped_empty}, "
            f"skipped_insufficient_parts={skipped_insufficient_parts}, "
            f"skipped_no_element_type={skipped_no_element_type}, "
            f"skipped_no_label={skipped_no_label}, "
            f"extracted={len(elements)}"
        )
        
        if len(elements) == 0 and total_lines_processed > 0:
            # Log sample of tags that were skipped
            sample_tags = []
            for line_idx, line in enumerate(lines[1:11], start=1):  # First 10 data lines
                if line.strip():
                    parts = line.split('\t')
                    if len(parts) >= 4:
                        tag = parts[0].strip().lower()
                        name = parts[1].strip()
                        text = parts[2].strip()
                        sample_tags.append(f"tag='{tag}', name='{name}', text='{text}'")
            if sample_tags:
                logger.warning(
                    f"[StatePreprocessor] No elements extracted. Sample tags from first 10 lines:\n" +
                    "\n".join(f"  Line {i+1}: {tag_info}" for i, tag_info in enumerate(sample_tags))
                )
        
        return elements
    
    def _extract_from_xml(self, axtree_txt: str) -> List[Dict[str, Any]]:
        """
        Extract key interactive elements from OSWorld XML format accessibility tree.
        
        Args:
            axtree_txt: XML string of accessibility tree
            
        Returns:
            List of element dictionaries (filtered and deduplicated)
        """
        elements = []
        seen_elements: set = set()  # Track seen elements to avoid duplicates
        
        try:
            root = ET.fromstring(axtree_txt)
        except ET.ParseError as e:
            logger.debug(f"[StatePreprocessor] XML parse error: {e}")
            return elements
        
        # OSWorld XML namespace mappings (aligned with mm_agents and desktop_env)
        # These are the actual namespaces used in OSWorld accessibility trees
        # Note: OSWorld uses namespace prefixes like "cp:", "st:", "attr:" in XML
        # But in ElementTree, we need to use the full namespace URI
        attr_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/attributes"
        component_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/component"
        state_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/state"
        
        def is_system_container(name: str, parent_name: str) -> bool:
            """Check if element is in a system-level container."""
            name_lower = name.lower() if name else ""
            parent_lower = parent_name.lower() if parent_name else ""
            
            # Check if name or parent name contains system container keywords
            for container in self.system_containers:
                if container in name_lower or container in parent_lower:
                    return True
            return False
        
        def create_element_id(screencoord: str, name: str, text: str, tag: str) -> str:
            """Create unique identifier for element to avoid duplicates."""
            # Use coordinates as primary identifier (most reliable)
            if screencoord:
                return f"{screencoord}|{tag}"
            # Fallback to name + tag
            if name:
                return f"{name}|{tag}"
            # Fallback to text + tag (limited length)
            if text:
                return f"{text[:50]}|{tag}"
            # Last resort: tag only (less reliable for deduplication)
            return f"{tag}|unknown"
        
        def extract_from_node(node, parent_name="", node_index=0, parent_path="", in_system_container=False):
            """Recursively extract elements from XML nodes."""
            # Remove namespace prefix from tag (e.g., '{namespace}button' -> 'button')
            tag = node.tag
            if '}' in tag:
                tag = tag.split('}')[-1]
            
            name = node.get("name", "")
            text = node.text.strip() if node.text else ""
            
            # Check if we're in a system container
            current_in_system = in_system_container or is_system_container(name, parent_name)
            
            # Get class attribute (for additional info, not for type determination)
            class_attr = node.get(f"{{{attr_ns_ubuntu}}}class", "")
            
            # Get state attributes
            enabled = node.get(f"{{{state_ns_ubuntu}}}enabled", "true")
            visible = node.get(f"{{{state_ns_ubuntu}}}visible", "true")
            
            # Get component attributes (position, size)
            # Use the correct namespace URI format for ElementTree
            screencoord = node.get(f"{{{component_ns_ubuntu}}}screencoord", "")
            size = node.get(f"{{{component_ns_ubuntu}}}size", "")
            
            # Build current path for ID generation
            current_path = f"{parent_path}/{tag}[{node_index}]" if parent_path else f"{tag}[{node_index}]"
            
            # In OSWorld, the XML tag name IS the element type/role (from getRoleName())
            # We just need to map it to SafePred's standard types
            element_type = None
            tag_lower = tag.lower()
            
            # Map OSWorld role names (XML tag names) to SafePred standard types
            # OSWorld uses role names like: "push-button", "text", "entry", "link", "tree-item", etc.
            if tag_lower in ['button', 'pushbutton', 'push-button']:
                element_type = 'button'
            elif tag_lower in ['link']:
                element_type = 'link'
            elif tag_lower in ['entry', 'text-field', 'textfield']:
                element_type = 'textbox'
            elif tag_lower in ['combo', 'combobox', 'combo-box']:
                element_type = 'combobox'
            elif tag_lower in ['check', 'checkbox', 'check-box']:
                element_type = 'checkbox'
            elif tag_lower in ['radio', 'radiobutton', 'radio-button']:
                element_type = 'radio'
            elif tag_lower in ['tree-item', 'treeitem', 'tree']:
                # tree-item is an interactive element in tree views (e.g., file browser, bookmarks manager)
                # Treat it as a clickable element similar to button
                element_type = 'tree-item'
            elif tag_lower == 'text':
                # 'text' role might be a textbox if it's interactive
                # Check if it has enabled/visible states (interactive)
                if enabled.lower() == "true" and visible.lower() == "true":
                    element_type = 'textbox'
            elif tag_lower.endswith('button'):
                element_type = 'button'
            elif tag_lower.endswith('link'):
                element_type = 'link'
            elif tag_lower.endswith('textbox') or tag_lower.endswith('text-field'):
                element_type = 'textbox'
            elif tag_lower.endswith('tree-item') or tag_lower.endswith('treeitem'):
                element_type = 'tree-item'
            
            # Only process interactive elements
            # Include tree-item as it's an interactive element in tree views
            if element_type and element_type in ['button', 'link', 'textbox', 'combobox', 'checkbox', 'radio', 'tree-item']:
                # For critical operations, be more lenient with visibility/enabled checks
                # (some critical buttons might be temporarily disabled but still important)
                is_critical_candidate = any(keyword in (name + " " + text).lower() for keyword in self.critical_keywords)
                
                # Skip if not visible or not enabled (unless it's a critical operation candidate)
                if not is_critical_candidate and (visible.lower() != "true" or enabled.lower() != "true"):
                    # Still process children (they might be visible)
                    for idx, child in enumerate(node):
                        extract_from_node(child, name if name else parent_name, node_index=idx, parent_path=current_path, in_system_container=current_in_system)
                    return
                
                # Use name or text as label
                label = name if name else text
                if not label:
                    # Try description attribute
                    description = node.get(f"{{{attr_ns_ubuntu}}}description", "")
                    label = description if description else ""
                
                # Filter system container elements: only keep if they have meaningful content
                if current_in_system:
                    # Skip system container elements without name, text, or meaningful interaction
                    # Include tree-item as it's an interactive element
                    if not label and not (enabled.lower() == "true" and element_type in ['button', 'link', 'tree-item']):
                        # Still process children (they might be meaningful)
                        for idx, child in enumerate(node):
                            extract_from_node(child, name if name else parent_name, node_index=idx, parent_path=current_path, in_system_container=current_in_system)
                        return
                
                # Create unique identifier for deduplication
                element_id = create_element_id(screencoord, name, text, tag)
                if element_id in seen_elements:
                    # Skip duplicate, but still process children
                    for idx, child in enumerate(node):
                        extract_from_node(child, name if name else parent_name, node_index=idx, parent_path=current_path, in_system_container=current_in_system)
                    return
                seen_elements.add(element_id)
                
                # For OSWorld (desktop environment), we use screencoord as the primary identifier
                # bid is kept for compatibility with web environments, but not used for desktop
                bid = None
                element_identifier = None
                
                # Strategy 1: Use screencoord (primary for desktop environments)
                if screencoord:
                    element_identifier = screencoord  # Keep original format: "(x, y)"
                    # Don't set bid for desktop - it's a web concept
                # Strategy 2: Use name + tag (concise identifier for desktop)
                elif name:
                    # Use name + tag as identifier: "Minimize (button)"
                    element_identifier = f"{name} ({tag})"
                    # Generate a short bid for compatibility (optional)
                    bid = f"{name}_{tag}".replace(" ", "_").replace("'", "").replace('"', "")[:50]
                # Strategy 3: Use tag + text (if text is short and meaningful)
                elif text and len(text) < 30 and text.strip():
                    element_identifier = f"{tag}: {text[:30]}"
                    bid = f"{tag}_{text}".replace(" ", "_").replace("'", "").replace('"', "")[:50]
                # Strategy 4: Use attr:id if available (some desktop elements may have IDs)
                elif node.get(f"{{{attr_ns_ubuntu}}}id", ""):
                    element_identifier = node.get(f"{{{attr_ns_ubuntu}}}id", "")
                    bid = element_identifier  # For web compatibility
                # Strategy 5: Use tag + parent name (if parent name is short)
                elif parent_name and len(parent_name) < 20:
                    element_identifier = f"{tag} in {parent_name}"
                    bid = f"{tag}_{parent_name}".replace(" ", "_")[:50]
                # Last resort: Use tag + short index (avoid long paths)
                else:
                    element_identifier = f"{tag}[{node_index}]"
                    bid = f"{tag}_{node_index}"
                
                # Determine if critical
                is_critical = any(keyword in label.lower() for keyword in self.critical_keywords) if label else False
                
                # Build element dict
                elem_dict = {
                    "type": element_type,
                    "label": label,
                    "critical": is_critical,
                    "disabled": enabled.lower() != "true",
                }
                
                # For desktop environments (OSWorld), screencoord is the primary identifier
                if screencoord:
                    elem_dict["screencoord"] = screencoord
                    # Use screencoord as the identifier (for desktop)
                    elem_dict["element_id"] = screencoord
                elif element_identifier:
                    # Use element_identifier as fallback if no screencoord
                    elem_dict["element_id"] = element_identifier
                
                # bid is optional and mainly for web environments (VisualWebArena)
                # For desktop, we don't set bid unless it's from attr:id
                if bid and node.get(f"{{{attr_ns_ubuntu}}}id", ""):
                    elem_dict["bid"] = bid  # Only set if from attr:id (web compatibility)
                
                if size:
                    elem_dict["size"] = size
                if class_attr:
                    elem_dict["class"] = class_attr
                
                # For links, try to extract URL
                if element_type == 'link':
                    # OSWorld links might not have URL in accessibility tree
                    elem_dict["url"] = None
                
                # For form fields, extract value if available
                if element_type in ['textbox', 'combobox']:
                    # Try value namespace first (for textbox values)
                    value_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/value"
                    value = node.get(f"{{{value_ns_ubuntu}}}value", "") or node.get(f"{{{attr_ns_ubuntu}}}value", "")
                    if value:
                        elem_dict["value"] = value
                
                # Extract description attribute (important for element identification)
                description_attr = node.get(f"{{{attr_ns_ubuntu}}}description", "")
                if description_attr and description_attr != label:
                    # Add description if it provides additional context beyond label
                    elem_dict["description"] = description_attr
                
                # For elements without label, try to use description as label
                if not label and description_attr:
                    elem_dict["label"] = description_attr
                    # Re-check if it's critical with the new label
                    if any(keyword in description_attr.lower() for keyword in self.critical_keywords):
                        elem_dict["critical"] = True
                
                elements.append(elem_dict)
            
            # Recursively process children
            for idx, child in enumerate(node):
                extract_from_node(child, name if name else parent_name, node_index=idx, parent_path=current_path, in_system_container=current_in_system)
        
        # Start extraction from root
        extract_from_node(root)
        
        # Sort and limit elements by priority
        # Priority order (higher priority first):
        # 1. Critical operations (Save, Delete, etc.) - most important
        # 2. Elements with screen coordinates (interactive and actionable)
        # 3. Element type priority: button > link > tree-item > textbox > combobox > checkbox > radio
        # 4. Elements with labels (more identifiable)
        # 5. Elements with longer labels (more descriptive)
        if len(elements) > self.max_key_elements:
            # Define element type priority (higher number = higher priority)
            type_priority = {
                'button': 7,
                'link': 6,
                'tree-item': 5,  # Tree items are important for navigation (e.g., bookmarks, file browser)
                'textbox': 4,
                'combobox': 3,
                'checkbox': 2,
                'radio': 1,
            }
            
            elements.sort(key=lambda e: (
                bool(e.get('critical')),  # Critical operations first (True > False)
                bool(e.get('screencoord')),  # Elements with coordinates (interactive)
                type_priority.get(e.get('type', ''), 0),  # Element type priority
                bool(e.get('label')),  # Elements with labels
                -len(e.get('label', ''))  # Longer labels first (negative for descending)
            ), reverse=True)
            
            original_count = len(elements)
            elements = elements[:self.max_key_elements]
            logger.warning(
                f"[StatePreprocessor] Limited key_elements to {self.max_key_elements} (from {original_count} total). "
                f"Critical elements preserved: {sum(1 for e in elements if e.get('critical'))}"
            )
        else:
            # Even if not limiting, sort by priority for consistency
            type_priority = {
                'button': 7,
                'link': 6,
                'tree-item': 5,  # Tree items are important for navigation (e.g., bookmarks, file browser)
                'textbox': 4,
                'combobox': 3,
                'checkbox': 2,
                'radio': 1,
            }
            elements.sort(key=lambda e: (
                bool(e.get('critical')),
                bool(e.get('screencoord')),
                type_priority.get(e.get('type', ''), 0),
                bool(e.get('label')),
                -len(e.get('label', ''))
            ), reverse=True)
        
        return elements
    
    def _extract_url(self, axtree_txt: str) -> Optional[str]:
        """
        Extract URL from Accessibility Tree.
        
        Args:
            axtree_txt: Full Accessibility Tree text
        
        Returns:
            URL string if found, None otherwise
        """
        if not axtree_txt:
            return None
        
        # Look for url='...' pattern in RootWebArea
        url_match = re.search(r"RootWebArea[^,]*url=['\"]([^'\"]+)['\"]", axtree_txt)
        if url_match:
            return url_match.group(1)
        
        return None
    
    def _infer_page_type(self, axtree_txt: str, key_elements: List[Dict[str, Any]]) -> str:
        """
        Infer page type from Accessibility Tree and key elements.
        
        Types:
        - "form": Form page (has Save button, form fields)
        - "list": List page (has table, filter buttons)
        - "detail": Detail page (read-only information)
        - "navigation": Navigation page (mainly links)
        
        Args:
            axtree_txt: Full Accessibility Tree text
            key_elements: List of extracted key elements
        
        Returns:
            Page type string
        """
        if not axtree_txt and not key_elements:
            return "unknown"
        
        # Check for form indicators
        has_save_button = any(
            elem.get("type") == "button" and 
            any(kw in elem.get("label", "").lower() for kw in ["save", "submit", "create"])
            for elem in key_elements
        )
        has_form_fields = any(
            elem.get("type") in ["textbox", "combobox", "checkbox"]
            for elem in key_elements
        )
        
        if has_save_button and has_form_fields:
            return "form"
        
        # Check for list indicators
        has_table = "table" in axtree_txt.lower() or any(
            "table" in str(elem).lower() for elem in key_elements
        )
        has_filter = any(
            "filter" in elem.get("label", "").lower() or
            "search" in elem.get("label", "").lower()
            for elem in key_elements
        )
        
        if has_table or has_filter:
            return "list"
        
        # Check for detail page (read-only, has Cancel button but no Save)
        has_cancel_button = any(
            elem.get("type") == "button" and
            "cancel" in elem.get("label", "").lower()
            for elem in key_elements
        )
        if has_cancel_button and not has_save_button:
            return "detail"
        
        # Default to navigation if mainly links
        if len(key_elements) > 0:
            link_count = sum(1 for elem in key_elements if elem.get("type") == "link")
            if link_count > len(key_elements) * 0.5:
                return "navigation"
        
        return "unknown"
    
    def _simplify_chat_history(self, chat_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Simplify chat history by keeping only recent messages.
        
        Args:
            chat_messages: List of chat message dictionaries
        
        Returns:
            Simplified list containing only recent messages
        """
        if not chat_messages:
            return []
        
        # Keep only the most recent messages
        if len(chat_messages) <= self.max_chat_messages:
            return chat_messages
        
        return chat_messages[-self.max_chat_messages:]
    
    def format_state_for_display(self, compact_state: Dict[str, Any]) -> str:
        """
        Format compact state for display/logging purposes.
        
        Args:
            compact_state: Compact state dictionary
        
        Returns:
            Formatted string representation
        """
        lines = []
        
        if compact_state.get("goal"):
            lines.append(f"Goal: {compact_state['goal']}")
        
        if compact_state.get("policies"):
            lines.append("\nPolicies:")
            for policy in compact_state["policies"]:
                policy_id = policy.get("policy_id", policy.get("id", "N/A"))
                priority = policy.get("priority", policy.get("risk_level", policy.get("severity", "N/A")))
                template = policy.get("policy_template", "")
                lines.append(f"Policy {policy_id} ({priority}): {template}")
        
        if compact_state.get("url"):
            lines.append(f"\nURL: {compact_state['url']}")
        
        if compact_state.get("page_type"):
            lines.append(f"Page Type: {compact_state['page_type']}")
        
        if compact_state.get("key_elements"):
            lines.append("\nKey Elements:")
            for elem in compact_state["key_elements"]:
                elem_str = f"  [{elem.get('bid', 'N/A')}] {elem.get('type', 'unknown')}"
                # Use label, or fallback to description (placeholder/aria-label/name/title)
                display_label = elem.get("label") or elem.get("description") or ""
                if display_label:
                    elem_str += f" '{display_label}'"
                if elem.get("critical"):
                    elem_str += " [CRITICAL]"
                if elem.get("disabled"):
                    elem_str += " [DISABLED]"
                if elem.get("required"):
                    elem_str += " [REQUIRED]"
                if elem.get("value"):
                    elem_str += f" value='{elem['value']}'"
                if elem.get("url"):
                    elem_str += f" url='{elem['url']}'"
                # Add additional attributes for better identification (if label is empty and description is also empty)
                if not display_label:
                    # Show all available attributes (not just one) for better identification
                    attr_parts = []
                    if elem.get("placeholder"):
                        attr_parts.append(f"placeholder='{elem['placeholder']}'")
                    if elem.get("aria_label"):
                        attr_parts.append(f"aria-label='{elem['aria_label']}'")
                    if elem.get("name"):
                        attr_parts.append(f"name='{elem['name']}'")
                    if elem.get("title"):
                        attr_parts.append(f"title='{elem['title']}'")
                    if attr_parts:
                        elem_str += " " + " ".join(attr_parts)
                lines.append(elem_str)
        
        if compact_state.get("chat_history"):
            lines.append("\nChat History:")
            for msg in compact_state["chat_history"]:
                role = msg.get("role", "unknown")
                message = msg.get("message", "")
                lines.append(f"  {role}: {message}")
        
        return "\n".join(lines)

