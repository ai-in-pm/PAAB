#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder - Base Crew Classes
Evolved from PsychoPy's experiment system for AI crew management
"""

import uuid
import time
import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class CrewStatus(Enum):
    """Crew execution status"""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    COLLABORATING = "collaborating"
    MONITORING = "monitoring"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class ProcessType(Enum):
    """Crew process types"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    CONSENSUS = "consensus"
    PIPELINE = "pipeline"
    ADAPTIVE = "adaptive"


class CollaborationPattern(Enum):
    """Agent collaboration patterns"""
    INDEPENDENT = "independent"
    PEER_TO_PEER = "peer_to_peer"
    LEADER_FOLLOWER = "leader_follower"
    ROUND_ROBIN = "round_robin"
    BROADCAST = "broadcast"
    MESH = "mesh"


@dataclass
class CrewMemory:
    """Shared memory for crew collaboration"""
    shared_context: Dict[str, Any] = field(default_factory=dict)
    task_results: Dict[str, Any] = field(default_factory=dict)
    agent_communications: List[Dict[str, Any]] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrewMetrics:
    """Crew performance metrics"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_execution_time: float = 0.0
    average_task_time: float = 0.0
    collaboration_efficiency: float = 0.0
    agent_utilization: Dict[str, float] = field(default_factory=dict)
    communication_overhead: float = 0.0
    last_updated: float = field(default_factory=time.time)


class BaseCrew(ABC):
    """
    Base class for AI crews, evolved from PsychoPy experiments
    Manages teams of AI agents working together on complex tasks
    """
    
    def __init__(
        self,
        name: str,
        agents: List['BaseAgent'],
        tasks: List['BaseTask'],
        process_type: ProcessType = ProcessType.SEQUENTIAL,
        collaboration_pattern: CollaborationPattern = CollaborationPattern.PEER_TO_PEER,
        max_execution_time: Optional[float] = None,
        verbose: bool = True,
        memory: Optional[CrewMemory] = None,
        manager_agent: Optional['BaseAgent'] = None,
        planning_enabled: bool = True,
        monitoring_enabled: bool = True
    ):
        """
        Initialize a crew
        
        Args:
            name: Name of the crew
            agents: List of agents in the crew
            tasks: List of tasks to be executed
            process_type: How tasks should be processed
            collaboration_pattern: How agents should collaborate
            max_execution_time: Maximum execution time for the crew
            verbose: Whether to log detailed execution information
            memory: Shared memory system for the crew
            manager_agent: Optional manager agent for hierarchical processes
            planning_enabled: Whether to enable automatic planning
            monitoring_enabled: Whether to enable real-time monitoring
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.agents = agents
        self.tasks = tasks
        self.process_type = process_type
        self.collaboration_pattern = collaboration_pattern
        self.max_execution_time = max_execution_time
        self.verbose = verbose
        self.memory = memory or CrewMemory()
        self.manager_agent = manager_agent
        self.planning_enabled = planning_enabled
        self.monitoring_enabled = monitoring_enabled
        
        # State management
        self.status = CrewStatus.IDLE
        self.created_at = time.time()
        self.started_at = None
        self.completed_at = None
        
        # Execution tracking
        self.current_task_index = 0
        self.execution_plan = None
        self.execution_history = []
        self.metrics = CrewMetrics()
        
        # Results and outputs
        self.results = {}
        self.final_output = None
        self.errors = []
        
        # Callbacks
        self.on_task_complete = None
        self.on_crew_complete = None
        self.on_error = None
        
        # Initialize crew
        self._initialize_crew()
        
        logger.info(f"Crew {self.name} ({self.id}) initialized with {len(self.agents)} agents and {len(self.tasks)} tasks")
    
    def _initialize_crew(self) -> None:
        """Initialize crew components"""
        # Assign crew reference to all agents
        for agent in self.agents:
            agent.crew = self
        
        # Initialize metrics
        self.metrics.total_tasks = len(self.tasks)
        for agent in self.agents:
            self.metrics.agent_utilization[agent.id] = 0.0
    
    @abstractmethod
    def kickoff(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Start crew execution
        
        Args:
            inputs: Initial inputs for the crew
            
        Returns:
            Dict containing execution results
        """
        pass
    
    def plan_execution(self) -> Dict[str, Any]:
        """
        Create an execution plan for the crew
        
        Returns:
            Dict containing the execution plan
        """
        if not self.planning_enabled:
            return {"plan": "Planning disabled", "tasks": self.tasks}
        
        self.status = CrewStatus.PLANNING
        
        try:
            # Analyze task dependencies
            task_dependencies = self._analyze_task_dependencies()
            
            # Assign tasks to agents
            task_assignments = self._assign_tasks_to_agents()
            
            # Create execution sequence
            execution_sequence = self._create_execution_sequence(task_dependencies)
            
            # Estimate execution time
            estimated_time = self._estimate_execution_time()
            
            # Create collaboration plan
            collaboration_plan = self._create_collaboration_plan()
            
            self.execution_plan = {
                "task_dependencies": task_dependencies,
                "task_assignments": task_assignments,
                "execution_sequence": execution_sequence,
                "estimated_time": estimated_time,
                "collaboration_plan": collaboration_plan,
                "process_type": self.process_type.value,
                "created_at": time.time()
            }
            
            logger.info(f"Execution plan created for crew {self.name}")
            return self.execution_plan
            
        except Exception as e:
            logger.error(f"Planning failed for crew {self.name}: {str(e)}")
            self.status = CrewStatus.FAILED
            return {"error": str(e), "success": False}
        finally:
            if self.status == CrewStatus.PLANNING:
                self.status = CrewStatus.IDLE
    
    def execute_sequential(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute tasks sequentially"""
        self.status = CrewStatus.EXECUTING
        self.started_at = time.time()
        
        try:
            results = []
            
            for i, task in enumerate(self.tasks):
                self.current_task_index = i
                
                # Check if task can start
                if not task.can_start():
                    logger.warning(f"Task {task.id} cannot start, skipping")
                    continue
                
                # Execute task
                task_result = self._execute_single_task(task, inputs)
                results.append(task_result)
                
                # Update shared memory
                self.memory.task_results[task.id] = task_result
                
                # Check for early termination
                if not task_result.get("success", False) and task.priority.value >= 3:
                    logger.error(f"Critical task {task.id} failed, stopping execution")
                    break
                
                # Update inputs for next task
                if task_result.get("success", False):
                    inputs = self._merge_inputs(inputs, task_result.get("output", {}))
            
            self.status = CrewStatus.COMPLETED
            self.completed_at = time.time()
            
            return {
                "success": True,
                "results": results,
                "execution_time": self.completed_at - self.started_at,
                "crew_id": self.id
            }
            
        except Exception as e:
            self.status = CrewStatus.FAILED
            logger.error(f"Sequential execution failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def execute_parallel(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute tasks in parallel"""
        self.status = CrewStatus.EXECUTING
        self.started_at = time.time()
        
        try:
            # Group tasks by dependencies
            task_groups = self._group_tasks_by_dependencies()
            results = []
            
            for group in task_groups:
                # Execute tasks in current group in parallel
                group_results = asyncio.run(self._execute_task_group_parallel(group, inputs))
                results.extend(group_results)
                
                # Update inputs with results from this group
                for result in group_results:
                    if result.get("success", False):
                        inputs = self._merge_inputs(inputs, result.get("output", {}))
            
            self.status = CrewStatus.COMPLETED
            self.completed_at = time.time()
            
            return {
                "success": True,
                "results": results,
                "execution_time": self.completed_at - self.started_at,
                "crew_id": self.id
            }
            
        except Exception as e:
            self.status = CrewStatus.FAILED
            logger.error(f"Parallel execution failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def execute_hierarchical(self, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute tasks with hierarchical management"""
        if not self.manager_agent:
            raise ValueError("Hierarchical execution requires a manager agent")
        
        self.status = CrewStatus.EXECUTING
        self.started_at = time.time()
        
        try:
            # Manager creates execution plan
            management_plan = self.manager_agent.create_execution_plan(self.tasks, self.agents)
            
            # Execute according to manager's plan
            results = []
            for step in management_plan["steps"]:
                step_result = self._execute_management_step(step, inputs)
                results.append(step_result)
                
                # Manager reviews and adjusts plan if needed
                if step_result.get("requires_adjustment", False):
                    adjustment = self.manager_agent.adjust_plan(step_result, management_plan)
                    management_plan.update(adjustment)
            
            self.status = CrewStatus.COMPLETED
            self.completed_at = time.time()
            
            return {
                "success": True,
                "results": results,
                "management_plan": management_plan,
                "execution_time": self.completed_at - self.started_at,
                "crew_id": self.id
            }
            
        except Exception as e:
            self.status = CrewStatus.FAILED
            logger.error(f"Hierarchical execution failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def monitor_progress(self) -> Dict[str, Any]:
        """Monitor crew execution progress"""
        if not self.monitoring_enabled:
            return {"monitoring": "disabled"}
        
        progress = {
            "crew_id": self.id,
            "crew_name": self.name,
            "status": self.status.value,
            "current_task_index": self.current_task_index,
            "total_tasks": len(self.tasks),
            "progress_percentage": (self.current_task_index / len(self.tasks)) * 100 if self.tasks else 0,
            "execution_time": time.time() - self.started_at if self.started_at else 0,
            "agent_status": {agent.id: agent.get_status() for agent in self.agents},
            "task_status": {task.id: task.get_status() for task in self.tasks},
            "memory_usage": {
                "shared_context_size": len(self.memory.shared_context),
                "task_results_count": len(self.memory.task_results),
                "communications_count": len(self.memory.agent_communications)
            },
            "metrics": self.metrics.__dict__
        }
        
        return progress
    
    def pause_execution(self) -> None:
        """Pause crew execution"""
        if self.status == CrewStatus.EXECUTING:
            self.status = CrewStatus.PAUSED
            logger.info(f"Crew {self.name} execution paused")
    
    def resume_execution(self) -> None:
        """Resume crew execution"""
        if self.status == CrewStatus.PAUSED:
            self.status = CrewStatus.EXECUTING
            logger.info(f"Crew {self.name} execution resumed")
    
    def cancel_execution(self, reason: str = "User cancelled") -> None:
        """Cancel crew execution"""
        self.status = CrewStatus.CANCELLED
        self.completed_at = time.time()
        
        # Cancel all running tasks
        for task in self.tasks:
            if task.status.value in ["in_progress", "assigned"]:
                task.cancel(reason)
        
        logger.info(f"Crew {self.name} execution cancelled: {reason}")
    
    def add_agent(self, agent: 'BaseAgent') -> None:
        """Add an agent to the crew"""
        if agent not in self.agents:
            self.agents.append(agent)
            agent.crew = self
            self.metrics.agent_utilization[agent.id] = 0.0
            logger.info(f"Agent {agent.id} added to crew {self.name}")
    
    def remove_agent(self, agent: 'BaseAgent') -> None:
        """Remove an agent from the crew"""
        if agent in self.agents:
            self.agents.remove(agent)
            agent.crew = None
            if agent.id in self.metrics.agent_utilization:
                del self.metrics.agent_utilization[agent.id]
            logger.info(f"Agent {agent.id} removed from crew {self.name}")
    
    def add_task(self, task: 'BaseTask') -> None:
        """Add a task to the crew"""
        if task not in self.tasks:
            self.tasks.append(task)
            self.metrics.total_tasks += 1
            logger.info(f"Task {task.id} added to crew {self.name}")
    
    def remove_task(self, task: 'BaseTask') -> None:
        """Remove a task from the crew"""
        if task in self.tasks:
            self.tasks.remove(task)
            self.metrics.total_tasks -= 1
            logger.info(f"Task {task.id} removed from crew {self.name}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive crew status"""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "process_type": self.process_type.value,
            "collaboration_pattern": self.collaboration_pattern.value,
            "agents_count": len(self.agents),
            "tasks_count": len(self.tasks),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "execution_time": self._get_execution_time(),
            "progress": self.monitor_progress(),
            "metrics": self.metrics.__dict__,
            "has_manager": self.manager_agent is not None,
            "planning_enabled": self.planning_enabled,
            "monitoring_enabled": self.monitoring_enabled
        }
    
    def _execute_single_task(self, task: 'BaseTask', inputs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a single task"""
        try:
            # Prepare task context
            if inputs:
                task.context.inputs.update(inputs)
            
            # Start task execution
            task.start_execution()
            
            # Execute task
            result = task.execute()
            
            # Complete task execution
            task.complete_execution(result)
            
            # Update metrics
            self._update_task_metrics(task, result)
            
            return result.__dict__ if hasattr(result, '__dict__') else result
            
        except Exception as e:
            logger.error(f"Task {task.id} execution failed: {str(e)}")
            error_result = {"success": False, "error": str(e), "task_id": task.id}
            task.complete_execution(error_result)
            return error_result
    
    async def _execute_task_group_parallel(self, task_group: List['BaseTask'], inputs: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute a group of tasks in parallel"""
        tasks = []
        for task in task_group:
            if task.can_start():
                tasks.append(asyncio.create_task(self._execute_task_async(task, inputs)))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "task_id": task_group[i].id
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_task_async(self, task: 'BaseTask', inputs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a task asynchronously"""
        return self._execute_single_task(task, inputs)
    
    def _analyze_task_dependencies(self) -> Dict[str, List[str]]:
        """Analyze task dependencies"""
        dependencies = {}
        for task in self.tasks:
            dependencies[task.id] = [dep.id for dep in task.dependencies]
        return dependencies
    
    def _assign_tasks_to_agents(self) -> Dict[str, str]:
        """Assign tasks to agents"""
        assignments = {}
        
        # Simple round-robin assignment if no specific agent assigned
        agent_index = 0
        for task in self.tasks:
            if not task.agent:
                task.assign_agent(self.agents[agent_index % len(self.agents)])
                agent_index += 1
            assignments[task.id] = task.agent.id
        
        return assignments
    
    def _create_execution_sequence(self, dependencies: Dict[str, List[str]]) -> List[str]:
        """Create execution sequence based on dependencies"""
        # Simple topological sort
        sequence = []
        remaining_tasks = set(task.id for task in self.tasks)
        
        while remaining_tasks:
            # Find tasks with no remaining dependencies
            ready_tasks = []
            for task_id in remaining_tasks:
                deps = dependencies.get(task_id, [])
                if all(dep not in remaining_tasks for dep in deps):
                    ready_tasks.append(task_id)
            
            if not ready_tasks:
                # Circular dependency or other issue
                ready_tasks = list(remaining_tasks)  # Force progress
            
            sequence.extend(ready_tasks)
            remaining_tasks -= set(ready_tasks)
        
        return sequence
    
    def _estimate_execution_time(self) -> float:
        """Estimate total execution time"""
        # Simple estimation based on task count and agent availability
        if self.process_type == ProcessType.SEQUENTIAL:
            return len(self.tasks) * 60  # Assume 1 minute per task
        elif self.process_type == ProcessType.PARALLEL:
            return (len(self.tasks) / len(self.agents)) * 60  # Parallel execution
        else:
            return len(self.tasks) * 45  # Other processes
    
    def _create_collaboration_plan(self) -> Dict[str, Any]:
        """Create collaboration plan for agents"""
        return {
            "pattern": self.collaboration_pattern.value,
            "communication_frequency": "as_needed",
            "shared_resources": list(self.memory.shared_context.keys()),
            "coordination_points": ["task_completion", "error_handling", "final_review"]
        }
    
    def _group_tasks_by_dependencies(self) -> List[List['BaseTask']]:
        """Group tasks by their dependency levels"""
        groups = []
        remaining_tasks = self.tasks.copy()
        
        while remaining_tasks:
            current_group = []
            for task in remaining_tasks.copy():
                if task.can_start():
                    current_group.append(task)
                    remaining_tasks.remove(task)
            
            if current_group:
                groups.append(current_group)
            else:
                # Force progress if stuck
                groups.append([remaining_tasks.pop(0)])
        
        return groups
    
    def _execute_management_step(self, step: Dict[str, Any], inputs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a management step in hierarchical process"""
        # Placeholder for hierarchical execution logic
        return {"success": True, "step": step, "requires_adjustment": False}
    
    def _merge_inputs(self, current_inputs: Optional[Dict[str, Any]], new_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge current inputs with new inputs"""
        if current_inputs is None:
            return new_inputs.copy()
        
        merged = current_inputs.copy()
        merged.update(new_inputs)
        return merged
    
    def _update_task_metrics(self, task: 'BaseTask', result: Dict[str, Any]) -> None:
        """Update crew metrics based on task execution"""
        if result.get("success", False):
            self.metrics.completed_tasks += 1
        else:
            self.metrics.failed_tasks += 1
        
        # Update agent utilization
        if task.agent:
            current_utilization = self.metrics.agent_utilization.get(task.agent.id, 0.0)
            self.metrics.agent_utilization[task.agent.id] = current_utilization + 1.0
        
        self.metrics.last_updated = time.time()
    
    def _get_execution_time(self) -> Optional[float]:
        """Get current execution time"""
        if self.started_at:
            end_time = self.completed_at or time.time()
            return end_time - self.started_at
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert crew to dictionary representation"""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "process_type": self.process_type.value,
            "collaboration_pattern": self.collaboration_pattern.value,
            "agents": [agent.to_dict() for agent in self.agents],
            "tasks": [task.to_dict() for task in self.tasks],
            "metrics": self.metrics.__dict__,
            "created_at": self.created_at,
            "execution_plan": self.execution_plan
        }
