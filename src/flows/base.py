#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder - Flow Management System
Enhanced workflow management evolved from PsychoPy's flow system
"""

import uuid
import time
import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class FlowType(Enum):
    """Types of flows"""
    LINEAR = "linear"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"
    LOOP = "loop"
    STATE_MACHINE = "state_machine"
    EVENT_DRIVEN = "event_driven"
    HYBRID = "hybrid"


class FlowStatus(Enum):
    """Flow execution status"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeType(Enum):
    """Types of flow nodes"""
    START = "start"
    END = "end"
    TASK = "task"
    DECISION = "decision"
    PARALLEL_SPLIT = "parallel_split"
    PARALLEL_JOIN = "parallel_join"
    LOOP_START = "loop_start"
    LOOP_END = "loop_end"
    EVENT = "event"
    CUSTOM = "custom"


@dataclass
class FlowNode:
    """Individual node in a flow"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    node_type: NodeType = NodeType.TASK
    task: Optional['BaseTask'] = None
    condition: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    position: Dict[str, float] = field(default_factory=lambda: {"x": 0, "y": 0})
    
    # Connections
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    
    # Execution state
    status: str = "pending"
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None


@dataclass
class FlowConnection:
    """Connection between flow nodes"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_node: str = ""
    target_node: str = ""
    condition: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FlowExecution:
    """Flow execution context"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    flow_id: str = ""
    status: FlowStatus = FlowStatus.IDLE
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    current_nodes: List[str] = field(default_factory=list)
    completed_nodes: List[str] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class BaseFlow(ABC):
    """Base class for workflow management"""
    
    def __init__(
        self,
        name: str,
        flow_type: FlowType = FlowType.LINEAR,
        description: str = "",
        max_execution_time: Optional[float] = None,
        retry_failed_nodes: bool = True,
        parallel_execution: bool = False
    ):
        """
        Initialize flow
        
        Args:
            name: Flow name
            flow_type: Type of flow
            description: Flow description
            max_execution_time: Maximum execution time in seconds
            retry_failed_nodes: Whether to retry failed nodes
            parallel_execution: Whether to enable parallel execution
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.flow_type = flow_type
        self.description = description
        self.max_execution_time = max_execution_time
        self.retry_failed_nodes = retry_failed_nodes
        self.parallel_execution = parallel_execution
        
        # Flow structure
        self.nodes: Dict[str, FlowNode] = {}
        self.connections: Dict[str, FlowConnection] = {}
        self.start_nodes: List[str] = []
        self.end_nodes: List[str] = []
        
        # Execution state
        self.current_execution: Optional[FlowExecution] = None
        self.execution_history: List[FlowExecution] = []
        
        # Callbacks
        self.on_node_start: Optional[Callable] = None
        self.on_node_complete: Optional[Callable] = None
        self.on_flow_complete: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        
        logger.info(f"Flow {self.name} ({self.id}) initialized")
    
    def add_node(
        self,
        name: str,
        node_type: NodeType = NodeType.TASK,
        task: Optional['BaseTask'] = None,
        condition: Optional[Callable] = None,
        position: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> str:
        """Add a node to the flow"""
        
        node = FlowNode(
            name=name,
            node_type=node_type,
            task=task,
            condition=condition,
            position=position or {"x": 0, "y": 0},
            metadata=kwargs
        )
        
        self.nodes[node.id] = node
        
        # Auto-set start/end nodes
        if node_type == NodeType.START:
            self.start_nodes.append(node.id)
        elif node_type == NodeType.END:
            self.end_nodes.append(node.id)
        
        logger.debug(f"Added node {name} ({node.id}) to flow {self.name}")
        return node.id
    
    def add_connection(
        self,
        source_node: str,
        target_node: str,
        condition: Optional[Callable] = None,
        **kwargs
    ) -> str:
        """Add a connection between nodes"""
        
        if source_node not in self.nodes or target_node not in self.nodes:
            raise ValueError("Source and target nodes must exist")
        
        connection = FlowConnection(
            source_node=source_node,
            target_node=target_node,
            condition=condition,
            metadata=kwargs
        )
        
        self.connections[connection.id] = connection
        
        # Update node connections
        self.nodes[source_node].outputs.append(target_node)
        self.nodes[target_node].inputs.append(source_node)
        
        logger.debug(f"Added connection from {source_node} to {target_node}")
        return connection.id
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the flow"""
        
        if node_id not in self.nodes:
            return False
        
        # Remove connections
        connections_to_remove = []
        for conn_id, connection in self.connections.items():
            if connection.source_node == node_id or connection.target_node == node_id:
                connections_to_remove.append(conn_id)
        
        for conn_id in connections_to_remove:
            del self.connections[conn_id]
        
        # Remove from start/end nodes
        if node_id in self.start_nodes:
            self.start_nodes.remove(node_id)
        if node_id in self.end_nodes:
            self.end_nodes.remove(node_id)
        
        # Remove node
        del self.nodes[node_id]
        
        logger.debug(f"Removed node {node_id} from flow {self.name}")
        return True
    
    def validate_flow(self) -> List[str]:
        """Validate flow structure"""
        errors = []
        
        # Check for start nodes
        if not self.start_nodes:
            errors.append("Flow must have at least one start node")
        
        # Check for end nodes
        if not self.end_nodes:
            errors.append("Flow must have at least one end node")
        
        # Check for unreachable nodes
        reachable_nodes = set()
        
        def traverse(node_id: str):
            if node_id in reachable_nodes:
                return
            reachable_nodes.add(node_id)
            
            node = self.nodes[node_id]
            for output_node in node.outputs:
                traverse(output_node)
        
        for start_node in self.start_nodes:
            traverse(start_node)
        
        unreachable = set(self.nodes.keys()) - reachable_nodes
        if unreachable:
            errors.append(f"Unreachable nodes: {list(unreachable)}")
        
        # Check for cycles (simplified check)
        visited = set()
        rec_stack = set()
        
        def has_cycle(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)
            
            node = self.nodes[node_id]
            for output_node in node.outputs:
                if output_node not in visited:
                    if has_cycle(output_node):
                        return True
                elif output_node in rec_stack:
                    return True
            
            rec_stack.remove(node_id)
            return False
        
        for node_id in self.nodes:
            if node_id not in visited:
                if has_cycle(node_id):
                    errors.append("Flow contains cycles")
                    break
        
        return errors
    
    @abstractmethod
    def execute(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the flow"""
        pass
    
    def pause_execution(self) -> bool:
        """Pause flow execution"""
        if self.current_execution and self.current_execution.status == FlowStatus.RUNNING:
            self.current_execution.status = FlowStatus.PAUSED
            logger.info(f"Flow {self.name} execution paused")
            return True
        return False
    
    def resume_execution(self) -> bool:
        """Resume flow execution"""
        if self.current_execution and self.current_execution.status == FlowStatus.PAUSED:
            self.current_execution.status = FlowStatus.RUNNING
            logger.info(f"Flow {self.name} execution resumed")
            return True
        return False
    
    def cancel_execution(self) -> bool:
        """Cancel flow execution"""
        if self.current_execution and self.current_execution.status in [FlowStatus.RUNNING, FlowStatus.PAUSED]:
            self.current_execution.status = FlowStatus.CANCELLED
            self.current_execution.completed_at = time.time()
            logger.info(f"Flow {self.name} execution cancelled")
            return True
        return False
    
    def get_execution_status(self) -> Optional[Dict[str, Any]]:
        """Get current execution status"""
        if not self.current_execution:
            return None
        
        execution = self.current_execution
        current_time = time.time()
        
        return {
            "execution_id": execution.id,
            "flow_id": self.id,
            "flow_name": self.name,
            "status": execution.status.value,
            "started_at": execution.started_at,
            "running_time": current_time - execution.started_at if execution.started_at else 0,
            "current_nodes": execution.current_nodes,
            "completed_nodes": execution.completed_nodes,
            "total_nodes": len(self.nodes),
            "progress": len(execution.completed_nodes) / len(self.nodes) if self.nodes else 0,
            "variables": execution.variables,
            "errors": execution.errors
        }
    
    def get_node_status(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific node"""
        if node_id not in self.nodes:
            return None
        
        node = self.nodes[node_id]
        
        return {
            "node_id": node.id,
            "name": node.name,
            "type": node.node_type.value,
            "status": node.status,
            "started_at": node.started_at,
            "completed_at": node.completed_at,
            "execution_time": (node.completed_at - node.started_at) if node.started_at and node.completed_at else None,
            "has_task": node.task is not None,
            "inputs": len(node.inputs),
            "outputs": len(node.outputs),
            "error": node.error
        }
    
    def export_definition(self) -> Dict[str, Any]:
        """Export flow definition"""
        return {
            "id": self.id,
            "name": self.name,
            "flow_type": self.flow_type.value,
            "description": self.description,
            "nodes": [
                {
                    "id": node.id,
                    "name": node.name,
                    "type": node.node_type.value,
                    "position": node.position,
                    "metadata": node.metadata,
                    "task_id": node.task.id if node.task else None
                }
                for node in self.nodes.values()
            ],
            "connections": [
                {
                    "id": conn.id,
                    "source": conn.source_node,
                    "target": conn.target_node,
                    "metadata": conn.metadata
                }
                for conn in self.connections.values()
            ],
            "start_nodes": self.start_nodes,
            "end_nodes": self.end_nodes,
            "settings": {
                "max_execution_time": self.max_execution_time,
                "retry_failed_nodes": self.retry_failed_nodes,
                "parallel_execution": self.parallel_execution
            }
        }
    
    def import_definition(self, definition: Dict[str, Any]) -> None:
        """Import flow definition"""
        self.name = definition.get("name", self.name)
        self.description = definition.get("description", self.description)
        self.flow_type = FlowType(definition.get("flow_type", self.flow_type.value))
        
        # Import settings
        settings = definition.get("settings", {})
        self.max_execution_time = settings.get("max_execution_time")
        self.retry_failed_nodes = settings.get("retry_failed_nodes", True)
        self.parallel_execution = settings.get("parallel_execution", False)
        
        # Clear existing structure
        self.nodes.clear()
        self.connections.clear()
        self.start_nodes.clear()
        self.end_nodes.clear()
        
        # Import nodes
        for node_def in definition.get("nodes", []):
            node = FlowNode(
                id=node_def["id"],
                name=node_def["name"],
                node_type=NodeType(node_def["type"]),
                position=node_def.get("position", {"x": 0, "y": 0}),
                metadata=node_def.get("metadata", {})
            )
            self.nodes[node.id] = node
        
        # Import connections
        for conn_def in definition.get("connections", []):
            connection = FlowConnection(
                id=conn_def["id"],
                source_node=conn_def["source"],
                target_node=conn_def["target"],
                metadata=conn_def.get("metadata", {})
            )
            self.connections[connection.id] = connection
            
            # Update node connections
            self.nodes[connection.source_node].outputs.append(connection.target_node)
            self.nodes[connection.target_node].inputs.append(connection.source_node)
        
        # Import start/end nodes
        self.start_nodes = definition.get("start_nodes", [])
        self.end_nodes = definition.get("end_nodes", [])
        
        logger.info(f"Imported flow definition for {self.name}")


class LinearFlow(BaseFlow):
    """Linear sequential flow implementation"""
    
    def __init__(self, **kwargs):
        super().__init__(flow_type=FlowType.LINEAR, **kwargs)
    
    def execute(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute linear flow"""
        
        # Validate flow
        errors = self.validate_flow()
        if errors:
            return {
                "success": False,
                "errors": errors,
                "flow_id": self.id
            }
        
        # Create execution context
        execution = FlowExecution(
            flow_id=self.id,
            status=FlowStatus.RUNNING,
            started_at=time.time(),
            variables=inputs or {}
        )
        
        self.current_execution = execution
        
        try:
            # Execute nodes in sequence
            current_nodes = self.start_nodes.copy()
            
            while current_nodes and execution.status == FlowStatus.RUNNING:
                next_nodes = []
                
                for node_id in current_nodes:
                    node = self.nodes[node_id]
                    
                    # Execute node
                    node_result = self._execute_node(node, execution)
                    
                    if node_result["success"]:
                        execution.completed_nodes.append(node_id)
                        execution.results[node_id] = node_result["output"]
                        
                        # Add output nodes to next execution
                        next_nodes.extend(node.outputs)
                    else:
                        execution.errors.append(f"Node {node.name} failed: {node_result.get('error', 'Unknown error')}")
                        if not self.retry_failed_nodes:
                            execution.status = FlowStatus.FAILED
                            break
                
                current_nodes = next_nodes
            
            # Complete execution
            if execution.status == FlowStatus.RUNNING:
                execution.status = FlowStatus.COMPLETED
            
            execution.completed_at = time.time()
            
            # Add to history
            self.execution_history.append(execution)
            
            # Trigger callback
            if self.on_flow_complete:
                self.on_flow_complete(execution)
            
            return {
                "success": execution.status == FlowStatus.COMPLETED,
                "execution_id": execution.id,
                "status": execution.status.value,
                "results": execution.results,
                "errors": execution.errors,
                "execution_time": execution.completed_at - execution.started_at,
                "flow_id": self.id
            }
            
        except Exception as e:
            execution.status = FlowStatus.FAILED
            execution.completed_at = time.time()
            execution.errors.append(str(e))
            
            logger.error(f"Flow execution failed: {str(e)}")
            
            if self.on_error:
                self.on_error(execution, e)
            
            return {
                "success": False,
                "error": str(e),
                "execution_id": execution.id,
                "flow_id": self.id
            }
    
    def _execute_node(self, node: FlowNode, execution: FlowExecution) -> Dict[str, Any]:
        """Execute a single node"""
        
        node.status = "running"
        node.started_at = time.time()
        
        # Trigger callback
        if self.on_node_start:
            self.on_node_start(node, execution)
        
        try:
            if node.node_type == NodeType.TASK and node.task:
                # Execute task
                result = node.task.execute()
                
                if hasattr(result, 'success'):
                    success = result.success
                    output = result.output if hasattr(result, 'output') else result
                    error = result.error_message if hasattr(result, 'error_message') else None
                else:
                    success = result.get("success", True)
                    output = result.get("output", result)
                    error = result.get("error")
                
                node.result = output
                node.error = error
                node.status = "completed" if success else "failed"
                
            elif node.node_type == NodeType.DECISION and node.condition:
                # Evaluate condition
                result = node.condition(execution.variables)
                node.result = result
                node.status = "completed"
                success = True
                output = result
                error = None
                
            else:
                # Default node execution
                node.result = {"executed": True}
                node.status = "completed"
                success = True
                output = node.result
                error = None
            
            node.completed_at = time.time()
            
            # Trigger callback
            if self.on_node_complete:
                self.on_node_complete(node, execution)
            
            return {
                "success": success,
                "output": output,
                "error": error,
                "node_id": node.id
            }
            
        except Exception as e:
            node.status = "failed"
            node.error = str(e)
            node.completed_at = time.time()
            
            logger.error(f"Node {node.name} execution failed: {str(e)}")
            
            return {
                "success": False,
                "output": None,
                "error": str(e),
                "node_id": node.id
            }
