#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder - Base Task Classes
Evolved from PsychoPy's routine system for AI task management
"""

import uuid
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DELEGATED = "delegated"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class TaskType(Enum):
    """Types of tasks"""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    WRITING = "writing"
    CODING = "coding"
    REVIEW = "review"
    COORDINATION = "coordination"
    CUSTOM = "custom"


@dataclass
class TaskContext:
    """Context information for task execution"""
    inputs: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    shared_memory: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Task execution result"""
    success: bool
    output: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    error_message: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskMetrics:
    """Task performance metrics"""
    execution_count: int = 0
    success_rate: float = 0.0
    average_execution_time: float = 0.0
    last_execution_time: float = 0.0
    total_execution_time: float = 0.0
    error_count: int = 0
    last_updated: float = field(default_factory=time.time)


class BaseTask(ABC):
    """
    Base class for AI tasks, evolved from PsychoPy routines
    Represents individual assignments with clear objectives
    """
    
    def __init__(
        self,
        description: str,
        expected_output: str,
        agent: Optional['BaseAgent'] = None,
        tools: Optional[List] = None,
        context: Optional[TaskContext] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        task_type: TaskType = TaskType.CUSTOM,
        max_execution_time: Optional[float] = None,
        retry_count: int = 3,
        callback: Optional[Callable] = None
    ):
        """
        Initialize a base task
        
        Args:
            description: Clear description of what needs to be accomplished
            expected_output: Description of expected output format and content
            agent: Agent assigned to execute this task
            tools: List of tools available for task execution
            context: Additional context and resources for task execution
            priority: Task priority level
            task_type: Type/category of the task
            max_execution_time: Maximum allowed execution time in seconds
            retry_count: Number of retry attempts on failure
            callback: Callback function to execute on task completion
        """
        self.id = str(uuid.uuid4())
        self.description = description
        self.expected_output = expected_output
        self.agent = agent
        self.tools = tools or []
        self.context = context or TaskContext()
        self.priority = priority
        self.task_type = task_type
        self.max_execution_time = max_execution_time
        self.retry_count = retry_count
        self.callback = callback
        
        # State management
        self.status = TaskStatus.PENDING
        self.created_at = time.time()
        self.started_at = None
        self.completed_at = None
        self.current_attempt = 0
        
        # Dependencies and relationships
        self.dependencies = []
        self.dependents = []
        self.parent_task = None
        self.subtasks = []
        
        # Execution tracking
        self.execution_history = []
        self.metrics = TaskMetrics()
        self.result = None
        
        # Collaboration
        self.collaborating_agents = []
        self.shared_resources = {}
        
        logger.info(f"Task {self.id} created: {self.description[:50]}...")
    
    @abstractmethod
    def execute(self) -> TaskResult:
        """
        Execute the task
        
        Returns:
            TaskResult containing execution outcome
        """
        pass
    
    def assign_agent(self, agent: 'BaseAgent') -> None:
        """
        Assign an agent to this task
        
        Args:
            agent: Agent to assign to this task
        """
        self.agent = agent
        self.status = TaskStatus.ASSIGNED
        logger.info(f"Task {self.id} assigned to agent {agent.id}")
    
    def add_dependency(self, task: 'BaseTask') -> None:
        """
        Add a task dependency
        
        Args:
            task: Task that must complete before this task can start
        """
        if task not in self.dependencies:
            self.dependencies.append(task)
            task.dependents.append(self)
            logger.info(f"Task {self.id} now depends on task {task.id}")
    
    def remove_dependency(self, task: 'BaseTask') -> None:
        """
        Remove a task dependency
        
        Args:
            task: Task to remove from dependencies
        """
        if task in self.dependencies:
            self.dependencies.remove(task)
            task.dependents.remove(self)
            logger.info(f"Dependency removed: Task {self.id} no longer depends on task {task.id}")
    
    def can_start(self) -> bool:
        """
        Check if task can start execution
        
        Returns:
            True if all dependencies are completed and agent is available
        """
        # Check dependencies
        for dep in self.dependencies:
            if dep.status != TaskStatus.COMPLETED:
                return False
        
        # Check agent availability
        if self.agent and self.agent.state not in ['idle', 'completed']:
            return False
        
        return True
    
    def start_execution(self) -> None:
        """Start task execution"""
        if not self.can_start():
            raise RuntimeError(f"Task {self.id} cannot start: dependencies not met or agent unavailable")
        
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = time.time()
        self.current_attempt += 1
        
        logger.info(f"Task {self.id} execution started (attempt {self.current_attempt})")
    
    def complete_execution(self, result: TaskResult) -> None:
        """
        Complete task execution
        
        Args:
            result: Task execution result
        """
        self.completed_at = time.time()
        self.result = result
        
        if result.success:
            self.status = TaskStatus.COMPLETED
            logger.info(f"Task {self.id} completed successfully")
        else:
            self.status = TaskStatus.FAILED
            logger.error(f"Task {self.id} failed: {result.error_message}")
        
        # Update metrics
        self._update_metrics(result)
        
        # Execute callback if provided
        if self.callback:
            try:
                self.callback(self, result)
            except Exception as e:
                logger.error(f"Task callback failed: {str(e)}")
        
        # Record execution history
        self.execution_history.append({
            "attempt": self.current_attempt,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result": result,
            "execution_time": self.completed_at - self.started_at
        })
    
    def retry_execution(self) -> bool:
        """
        Retry task execution if retries are available
        
        Returns:
            True if retry is possible, False otherwise
        """
        if self.current_attempt >= self.retry_count:
            logger.warning(f"Task {self.id} has exhausted all retry attempts")
            return False
        
        logger.info(f"Retrying task {self.id} (attempt {self.current_attempt + 1})")
        self.status = TaskStatus.PENDING
        return True
    
    def cancel(self, reason: str = "User cancelled") -> None:
        """
        Cancel task execution
        
        Args:
            reason: Reason for cancellation
        """
        self.status = TaskStatus.CANCELLED
        self.completed_at = time.time()
        
        logger.info(f"Task {self.id} cancelled: {reason}")
    
    def delegate_to(self, agent: 'BaseAgent') -> None:
        """
        Delegate task to another agent
        
        Args:
            agent: Agent to delegate the task to
        """
        old_agent = self.agent
        self.agent = agent
        self.status = TaskStatus.DELEGATED
        
        logger.info(f"Task {self.id} delegated from agent {old_agent.id if old_agent else 'None'} to agent {agent.id}")
    
    def add_subtask(self, subtask: 'BaseTask') -> None:
        """
        Add a subtask
        
        Args:
            subtask: Subtask to add
        """
        subtask.parent_task = self
        self.subtasks.append(subtask)
        logger.info(f"Subtask {subtask.id} added to task {self.id}")
    
    def get_progress(self) -> Dict[str, Any]:
        """
        Get task progress information
        
        Returns:
            Dict containing progress information
        """
        progress = {
            "task_id": self.id,
            "status": self.status.value,
            "progress_percentage": self._calculate_progress_percentage(),
            "execution_time": self._get_execution_time(),
            "dependencies_completed": len([d for d in self.dependencies if d.status == TaskStatus.COMPLETED]),
            "total_dependencies": len(self.dependencies),
            "subtasks_completed": len([s for s in self.subtasks if s.status == TaskStatus.COMPLETED]),
            "total_subtasks": len(self.subtasks)
        }
        
        return progress
    
    def validate_output(self, output: Any) -> bool:
        """
        Validate task output against expected output criteria
        
        Args:
            output: Output to validate
            
        Returns:
            True if output is valid, False otherwise
        """
        # Basic validation - can be overridden by specific task types
        if output is None:
            return False
        
        # Check if output meets basic requirements
        if isinstance(output, str) and len(output.strip()) == 0:
            return False
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive task status"""
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "task_type": self.task_type.value,
            "agent_id": self.agent.id if self.agent else None,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "current_attempt": self.current_attempt,
            "max_attempts": self.retry_count,
            "dependencies": len(self.dependencies),
            "subtasks": len(self.subtasks),
            "can_start": self.can_start(),
            "metrics": self.metrics.__dict__,
            "progress": self.get_progress()
        }
    
    def _update_metrics(self, result: TaskResult) -> None:
        """Update task metrics based on execution result"""
        self.metrics.execution_count += 1
        
        if result.success:
            success_count = self.metrics.execution_count - self.metrics.error_count
            self.metrics.success_rate = success_count / self.metrics.execution_count
        else:
            self.metrics.error_count += 1
            self.metrics.success_rate = (self.metrics.execution_count - self.metrics.error_count) / self.metrics.execution_count
        
        execution_time = self.completed_at - self.started_at
        self.metrics.last_execution_time = execution_time
        self.metrics.total_execution_time += execution_time
        self.metrics.average_execution_time = self.metrics.total_execution_time / self.metrics.execution_count
        self.metrics.last_updated = time.time()
    
    def _calculate_progress_percentage(self) -> float:
        """Calculate task progress percentage"""
        if self.status == TaskStatus.COMPLETED:
            return 100.0
        elif self.status == TaskStatus.FAILED or self.status == TaskStatus.CANCELLED:
            return 0.0
        elif self.status == TaskStatus.IN_PROGRESS:
            # Simple progress calculation based on subtasks
            if self.subtasks:
                completed_subtasks = len([s for s in self.subtasks if s.status == TaskStatus.COMPLETED])
                return (completed_subtasks / len(self.subtasks)) * 100.0
            else:
                return 50.0  # Assume 50% if no subtasks and in progress
        else:
            return 0.0
    
    def _get_execution_time(self) -> Optional[float]:
        """Get current execution time"""
        if self.started_at:
            end_time = self.completed_at or time.time()
            return end_time - self.started_at
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation"""
        return {
            "id": self.id,
            "description": self.description,
            "expected_output": self.expected_output,
            "status": self.status.value,
            "priority": self.priority.value,
            "task_type": self.task_type.value,
            "agent_id": self.agent.id if self.agent else None,
            "tools": [tool.name for tool in self.tools],
            "dependencies": [dep.id for dep in self.dependencies],
            "subtasks": [subtask.id for subtask in self.subtasks],
            "created_at": self.created_at,
            "metrics": self.metrics.__dict__,
            "context": {
                "inputs": self.context.inputs,
                "constraints": self.context.constraints,
                "resources": self.context.resources
            }
        }
