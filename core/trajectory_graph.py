"""
Trajectory Graph Module for Safety-TS-LMA.

Builds and maintains a directed graph representation of state-action trajectories,
capturing relationships between environment states and actions.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib
import json
from ..utils.logger import get_logger

logger = get_logger("SafePred.TrajectoryGraph")


@dataclass
class StateNode:
    """
    Represents a state node in the trajectory graph.
    
    Attributes:
        state_id: Unique identifier for the state
        state_features: Encoded features of the state (e.g., embeddings)
        raw_state: Raw state representation (e.g., screenshot, DOM)
        risk_score: Local risk score r(v) in [0, 1]
        metadata: Additional metadata (intent, timestamp, etc.)
    """
    
    state_id: str
    state_features: Optional[Any] = None
    raw_state: Optional[Any] = None
    risk_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate state node after initialization."""
        if not (0.0 <= self.risk_score <= 1.0):
            raise ValueError(f"risk_score must be in [0, 1], got {self.risk_score}")


@dataclass
class ActionEdge:
    """
    Represents an action edge in the trajectory graph.
    
    Attributes:
        from_state_id: Source state ID
        to_state_id: Target state ID
        action: Action representation (e.g., click, type, select)
        action_features: Encoded features of the action
        metadata: Additional metadata (timestamp, success, etc.)
    """
    
    from_state_id: str
    to_state_id: str
    action: Any
    action_features: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrajectoryGraph:
    """
    Directed graph representation of state-action trajectories.
    
    Maintains nodes (states) and edges (actions) to model the relationship
    between environment states and agent actions.
    
    Usage:
        graph = TrajectoryGraph()
        graph.add_state(state_id="s0", state_features=..., risk_score=0.1)
        graph.add_edge(from_state="s0", to_state="s1", action="click_button")
    """
    
    def __init__(self):
        """Initialize an empty trajectory graph."""
        self.nodes: Dict[str, StateNode] = {}
        self.edges: Dict[str, List[ActionEdge]] = defaultdict(list)
        self.reverse_edges: Dict[str, List[ActionEdge]] = defaultdict(list)
    
    def add_state(
        self,
        state_id: str,
        state_features: Optional[Any] = None,
        raw_state: Optional[Any] = None,
        risk_score: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StateNode:
        """
        Add a state node to the graph.
        
        Args:
            state_id: Unique identifier for the state
            state_features: Encoded features of the state
            raw_state: Raw state representation
            risk_score: Local risk score in [0, 1]
            metadata: Additional metadata
        
        Returns:
            Created StateNode object
        
        Raises:
            ValueError: If state_id already exists or risk_score is invalid
        """
        if state_id in self.nodes:
            logger.debug(f"State {state_id} already exists, returning existing node")
            return self.nodes[state_id]
        
        node = StateNode(
            state_id=state_id,
            state_features=state_features,
            raw_state=raw_state,
            risk_score=risk_score,
            metadata=metadata or {},
        )
        self.nodes[state_id] = node
        logger.info(f"  → State node added: {state_id[:8]}... (risk: {risk_score:.2f}, total: {len(self.nodes)})")
        return node
    
    def update_state_risk(self, state_id: str, risk_score: float) -> None:
        """
        Update the risk score of an existing state.
        
        Args:
            state_id: State identifier
            risk_score: New risk score in [0, 1]
        
        Raises:
            KeyError: If state_id does not exist
        """
        if state_id not in self.nodes:
            raise KeyError(f"State {state_id} not found in graph")
        self.nodes[state_id].risk_score = risk_score
    
    def add_edge(
        self,
        from_state_id: str,
        to_state_id: str,
        action: Any,
        action_features: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ActionEdge:
        """
        Add an action edge to the graph.
        
        Args:
            from_state_id: Source state ID
            to_state_id: Target state ID
            action: Action representation
            action_features: Encoded features of the action
            metadata: Additional metadata
        
        Returns:
            Created ActionEdge object
        
        Raises:
            KeyError: If from_state_id or to_state_id does not exist
        """
        if from_state_id not in self.nodes:
            raise KeyError(f"Source state {from_state_id} not found")
        if to_state_id not in self.nodes:
            raise KeyError(f"Target state {to_state_id} not found")
        
        edge = ActionEdge(
            from_state_id=from_state_id,
            to_state_id=to_state_id,
            action=action,
            action_features=action_features,
            metadata=metadata or {},
        )
        self.edges[from_state_id].append(edge)
        self.reverse_edges[to_state_id].append(edge)
        action_str = str(action) if action else "None"
        total_edges = sum(len(edges) for edges in self.edges.values())
        logger.info(f"Added edge: {from_state_id} -> {to_state_id} (action: {action_str}, total edges: {total_edges})")
        return edge
    
    def get_state(self, state_id: str) -> Optional[StateNode]:
        """
        Get a state node by ID.
        
        Args:
            state_id: State identifier
        
        Returns:
            StateNode if found, None otherwise
        """
        return self.nodes.get(state_id)
    
    def get_outgoing_edges(self, state_id: str) -> List[ActionEdge]:
        """
        Get all outgoing edges from a state.
        
        Args:
            state_id: State identifier
        
        Returns:
            List of ActionEdge objects
        """
        return self.edges.get(state_id, [])
    
    def get_incoming_edges(self, state_id: str) -> List[ActionEdge]:
        """
        Get all incoming edges to a state.
        
        Args:
            state_id: State identifier
        
        Returns:
            List of ActionEdge objects
        """
        return self.reverse_edges.get(state_id, [])
    
    def get_successors(self, state_id: str) -> List[str]:
        """
        Get all successor state IDs from a state.
        
        Args:
            state_id: State identifier
        
        Returns:
            List of successor state IDs
        """
        return [edge.to_state_id for edge in self.get_outgoing_edges(state_id)]
    
    def get_predecessors(self, state_id: str) -> List[str]:
        """
        Get all predecessor state IDs to a state.
        
        Args:
            state_id: State identifier
        
        Returns:
            List of predecessor state IDs
        """
        return [edge.from_state_id for edge in self.get_incoming_edges(state_id)]
    
    def compute_state_id(self, state_features: Any, raw_state: Any = None) -> str:
        """
        Compute a unique state ID from state features.
        
        This is a helper method for generating consistent state IDs.
        Users can override this or provide custom state IDs.
        
        Args:
            state_features: State features
            raw_state: Raw state representation
        
        Returns:
            Unique state ID (hash-based)
        """
        # Create a hash from state features
        if state_features is not None:
            if isinstance(state_features, (list, tuple)):
                content = json.dumps(state_features, sort_keys=True, default=str)
            elif isinstance(state_features, dict):
                content = json.dumps(state_features, sort_keys=True, default=str)
            else:
                content = str(state_features)
        else:
            content = str(raw_state)
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def add_trajectory(
        self,
        trajectory: List[Tuple[Any, Any]],
        state_encoder=None,
        risk_evaluator=None,
    ) -> None:
        """
        Add a complete trajectory to the graph.
        
        Args:
            trajectory: List of (state, action) tuples: [(s0, a0), (s1, a1), ..., (st, None)]
            state_encoder: Optional function to encode states
            risk_evaluator: Optional function to evaluate risks
        """
        if not trajectory:
            return
        
        # Add all states first
        state_ids = []
        for i, (state, _) in enumerate(trajectory):
            if state_encoder:
                state_features = state_encoder(state)
            else:
                state_features = state
            
            state_id = self.compute_state_id(state_features, state)
            
            # Check if state already exists
            if state_id not in self.nodes:
                risk_score = 0.0
                if risk_evaluator:
                    risk_score = risk_evaluator(state)
                
                self.add_state(
                    state_id=state_id,
                    state_features=state_features,
                    raw_state=state,
                    risk_score=risk_score,
                )
            
            state_ids.append(state_id)
        
        # Add edges
        for i in range(len(trajectory) - 1):
            state, action = trajectory[i]
            next_state, _ = trajectory[i + 1]
            
            from_id = state_ids[i]
            to_id = state_ids[i + 1]
            
            self.add_edge(
                from_state_id=from_id,
                to_state_id=to_id,
                action=action,
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the graph.
        
        Returns:
            Dictionary with graph statistics
        """
        total_edges = sum(len(edges) for edges in self.edges.values())
        avg_out_degree = total_edges / len(self.nodes) if self.nodes else 0
        
        risk_scores = [node.risk_score for node in self.nodes.values()]
        avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0
        max_risk = max(risk_scores) if risk_scores else 0.0
        
        return {
            "num_nodes": len(self.nodes),
            "num_edges": total_edges,
            "avg_out_degree": avg_out_degree,
            "avg_risk_score": avg_risk,
            "max_risk_score": max_risk,
            "min_risk_score": min(risk_scores) if risk_scores else 0.0,
        }
    
    def get_paths_from_state(
        self,
        from_state_id: str,
        max_depth: int = 3,
        max_paths: int = 5,
    ) -> List[List[Dict[str, Any]]]:
        """
        Get paths starting from a given state for risk evaluation.
        
        Args:
            from_state_id: Starting state ID
            max_depth: Maximum depth to explore (default: 3)
            max_paths: Maximum number of paths to return (default: 5)
        
        Returns:
            List of paths, where each path is a list of dicts with keys:
            - 'state_id': State ID
            - 'risk_score': Risk score of the state
            - 'action': Action taken to reach next state
            - 'to_state_id': Next state ID
        """
        if from_state_id not in self.nodes:
            return []
        
        paths = []
        
        def dfs(current_state_id: str, current_path: List[Dict[str, Any]], depth: int):
            if depth >= max_depth or len(paths) >= max_paths:
                if current_path:
                    paths.append(current_path.copy())
                return
            
            # Get outgoing edges
            edges = self.edges.get(current_state_id, [])
            if not edges:
                # Leaf node, add current path
                if current_path:
                    paths.append(current_path.copy())
                return
            
            for edge in edges[:3]:  # Limit branching factor
                to_state_id = edge.to_state_id
                if to_state_id not in self.nodes:
                    continue
                
                to_node = self.nodes[to_state_id]
                
                # Add this transition to path
                path_entry = {
                    'state_id': current_state_id,
                    'risk_score': self.nodes[current_state_id].risk_score,
                    'action': str(edge.action)[:100],
                    'to_state_id': to_state_id,
                    'to_risk_score': to_node.risk_score,
                }
                
                new_path = current_path + [path_entry]
                
                # Continue exploring
                dfs(to_state_id, new_path, depth + 1)
                
                if len(paths) >= max_paths:
                    return
        
        dfs(from_state_id, [], 0)
        return paths[:max_paths]
    
    def get_high_risk_paths(
        self,
        from_state_id: str,
        risk_threshold: float = 0.7,
        max_depth: int = 3,
    ) -> List[List[Dict[str, Any]]]:
        """
        Get paths that lead to high-risk states.
        
        Args:
            from_state_id: Starting state ID
            risk_threshold: Risk threshold for identifying high-risk states
            max_depth: Maximum depth to explore
        
        Returns:
            List of paths that contain high-risk states
        """
        all_paths = self.get_paths_from_state(from_state_id, max_depth=max_depth, max_paths=20)
        high_risk_paths = []
        
        for path in all_paths:
            # Check if any state in the path has high risk
            for step in path:
                if step.get('to_risk_score', 0.0) >= risk_threshold or step.get('risk_score', 0.0) >= risk_threshold:
                    high_risk_paths.append(path)
                    break
        
        return high_risk_paths
    
    def extract_examples(
        self,
        max_examples: int = 5,
        max_risk: Optional[float] = 0.5,
        filter_successful: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Extract state-action-next_state examples from the trajectory graph for few-shot learning.
        
        Args:
            max_examples: Maximum number of examples to extract
            max_risk: Maximum risk score for examples (None = no filter)
            filter_successful: Whether to only include successful transitions (edges with success=True in metadata)
        
        Returns:
            List of examples in format: [{"state": ..., "action": ..., "next_state": ...}, ...]
        """
        examples = []
        
        # Iterate through all edges to extract state-action-next_state pairs
        for from_state_id, edges_list in self.edges.items():
            if from_state_id not in self.nodes:
                continue
            
            from_node = self.nodes[from_state_id]
            
            # Filter by risk score if specified
            if max_risk is not None and from_node.risk_score > max_risk:
                continue
            
            # Check if we have enough examples
            if len(examples) >= max_examples:
                break
            
            for edge in edges_list:
                # Check if we have enough examples
                if len(examples) >= max_examples:
                    break
                
                # Filter by success if specified
                if filter_successful:
                    edge_success = edge.metadata.get("success", True)  # Default to True if not specified
                    if not edge_success:
                        continue
                
                # Get target state node
                to_state_id = edge.to_state_id
                if to_state_id not in self.nodes:
                    continue
                
                to_node = self.nodes[to_state_id]
                
                # Filter target state by risk score if specified
                if max_risk is not None and to_node.risk_score > max_risk:
                    continue
                
                # Extract state representations
                # Prefer raw_state if available, otherwise use state_features or state_id
                from_state = from_node.raw_state if from_node.raw_state is not None else (
                    from_node.state_features if from_node.state_features is not None else from_state_id
                )
                to_state = to_node.raw_state if to_node.raw_state is not None else (
                    to_node.state_features if to_node.state_features is not None else to_state_id
                )
                
                # Create example
                example = {
                    "state": from_state,
                    "action": edge.action,
                    "next_state": to_state,
                }
                examples.append(example)
        
        logger.debug(f"[TrajectoryGraph] Extracted {len(examples)} examples from graph (nodes: {len(self.nodes)}, edges: {sum(len(edges) for edges in self.edges.values())})")
        return examples
    
    def clear(self) -> None:
        """Clear all nodes and edges from the graph."""
        self.nodes.clear()
        self.edges.clear()
        self.reverse_edges.clear()

