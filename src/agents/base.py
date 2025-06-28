#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder - Base Agent Classes
Evolved from PsychoPy's component system for AI agent management
"""

import uuid
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent execution states"""
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    COLLABORATING = "collaborating"
    LEARNING = "learning"
    ERROR = "error"
    COMPLETED = "completed"


class AgentRole(Enum):
    """Predefined agent roles"""
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    CODER = "coder"
    MANAGER = "manager"
    REVIEWER = "reviewer"
    SPECIALIST = "specialist"
    COORDINATOR = "coordinator"


@dataclass
class AgentMemory:
    """Agent memory structure"""
    short_term: Dict[str, Any] = field(default_factory=dict)
    long_term: Dict[str, Any] = field(default_factory=dict)
    episodic: List[Dict[str, Any]] = field(default_factory=list)
    semantic: Dict[str, Any] = field(default_factory=dict)
    working: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMetrics:
    """Agent performance metrics"""
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_execution_time: float = 0.0
    collaboration_score: float = 0.0
    learning_rate: float = 0.0
    efficiency_score: float = 0.0
    last_updated: float = field(default_factory=time.time)


class BaseAgent(ABC):
    """
    Base class for AI agents, evolved from PsychoPy components
    Provides core functionality for autonomous AI agents
    """
    
    def __init__(
        self,
        role: str,
        goal: str,
        backstory: str,
        tools: Optional[List] = None,
        llm: Optional[Any] = None,
        memory: Optional[AgentMemory] = None,
        max_iter: int = 10,
        verbose: bool = True,
        allow_delegation: bool = True,
        step_callback: Optional[Callable] = None
    ):
        """
        Initialize a base agent
        
        Args:
            role: The role of the agent (e.g., "researcher", "analyst")
            goal: The primary goal/objective of the agent
            backstory: Background context for the agent's expertise
            tools: List of tools available to the agent
            llm: Language model instance for the agent
            memory: Memory system for the agent
            max_iter: Maximum iterations for task execution
            verbose: Whether to log detailed execution information
            allow_delegation: Whether agent can delegate tasks to others
            step_callback: Callback function for each execution step
        """
        self.id = str(uuid.uuid4())
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools or []
        self.llm = llm
        self.memory = memory or AgentMemory()
        self.max_iter = max_iter
        self.verbose = verbose
        self.allow_delegation = allow_delegation
        self.step_callback = step_callback
        
        # State management
        self.state = AgentState.IDLE
        self.current_task = None
        self.execution_history = []
        self.metrics = AgentMetrics()
        
        # Collaboration
        self.crew = None
        self.collaborators = []
        
        # Learning
        self.feedback_history = []
        self.adaptation_rules = {}
        
        logger.info(f"Agent {self.id} initialized with role: {self.role}")
    
    @abstractmethod
    def execute_task(self, task: 'Task') -> Dict[str, Any]:
        """
        Execute a given task
        
        Args:
            task: Task object to execute
            
        Returns:
            Dict containing execution results
        """
        pass
    
    def collaborate(self, other_agents: List['BaseAgent'], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collaborate with other agents
        
        Args:
            other_agents: List of agents to collaborate with
            context: Collaboration context and shared information
            
        Returns:
            Dict containing collaboration results
        """
        self.state = AgentState.COLLABORATING
        
        try:
            # Store collaborators
            self.collaborators = other_agents
            
            # Share relevant memory and context
            shared_context = self._prepare_collaboration_context(context)
            
            # Execute collaboration logic
            collaboration_result = self._execute_collaboration(other_agents, shared_context)
            
            # Update memory with collaboration insights
            self._update_collaboration_memory(collaboration_result)
            
            return collaboration_result
            
        except Exception as e:
            logger.error(f"Collaboration failed for agent {self.id}: {str(e)}")
            self.state = AgentState.ERROR
            return {"error": str(e), "success": False}
        finally:
            self.state = AgentState.IDLE
    
    def learn_from_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Learn and adapt from feedback
        
        Args:
            feedback: Feedback data including performance metrics and suggestions
        """
        self.state = AgentState.LEARNING
        
        try:
            # Store feedback
            self.feedback_history.append({
                "timestamp": time.time(),
                "feedback": feedback,
                "context": self.current_task.to_dict() if self.current_task else None
            })
            
            # Extract learning insights
            insights = self._extract_learning_insights(feedback)
            
            # Update adaptation rules
            self._update_adaptation_rules(insights)
            
            # Update metrics
            self._update_metrics_from_feedback(feedback)
            
            logger.info(f"Agent {self.id} learned from feedback: {insights}")
            
        except Exception as e:
            logger.error(f"Learning failed for agent {self.id}: {str(e)}")
        finally:
            self.state = AgentState.IDLE
    
    def delegate_task(self, task: 'Task', target_agent: 'BaseAgent') -> Dict[str, Any]:
        """
        Delegate a task to another agent
        
        Args:
            task: Task to delegate
            target_agent: Agent to delegate the task to
            
        Returns:
            Dict containing delegation results
        """
        if not self.allow_delegation:
            raise ValueError("Agent is not allowed to delegate tasks")
        
        logger.info(f"Agent {self.id} delegating task to agent {target_agent.id}")
        
        # Prepare delegation context
        delegation_context = {
            "delegator": self.id,
            "original_task": task.to_dict(),
            "delegation_reason": "Specialized expertise required",
            "expected_outcome": task.expected_output
        }
        
        # Execute delegation
        result = target_agent.execute_task(task)
        result["delegation_context"] = delegation_context
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics"""
        return {
            "id": self.id,
            "role": self.role,
            "state": self.state.value,
            "current_task": self.current_task.to_dict() if self.current_task else None,
            "metrics": self.metrics.__dict__,
            "memory_size": {
                "short_term": len(self.memory.short_term),
                "long_term": len(self.memory.long_term),
                "episodic": len(self.memory.episodic)
            },
            "tools_count": len(self.tools),
            "collaborators_count": len(self.collaborators)
        }
    
    def _prepare_collaboration_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare context for collaboration"""
        return {
            "agent_id": self.id,
            "role": self.role,
            "relevant_memory": self._get_relevant_memory(context),
            "available_tools": [tool.name for tool in self.tools],
            "context": context
        }
    
    def _execute_collaboration(self, other_agents: List['BaseAgent'], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute collaboration logic - to be overridden by specific agent types"""
        return {
            "success": True,
            "participants": [agent.id for agent in other_agents],
            "context": context,
            "outcome": "Collaboration completed successfully"
        }
    
    def _update_collaboration_memory(self, result: Dict[str, Any]) -> None:
        """Update memory with collaboration results"""
        self.memory.episodic.append({
            "type": "collaboration",
            "timestamp": time.time(),
            "result": result
        })
    
    def _extract_learning_insights(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Extract actionable insights from feedback"""
        return {
            "performance_trend": feedback.get("performance", "stable"),
            "improvement_areas": feedback.get("suggestions", []),
            "strengths": feedback.get("strengths", [])
        }
    
    def _update_adaptation_rules(self, insights: Dict[str, Any]) -> None:
        """Update agent adaptation rules based on insights"""
        for area in insights.get("improvement_areas", []):
            if area not in self.adaptation_rules:
                self.adaptation_rules[area] = {"priority": 1, "actions": []}
            else:
                self.adaptation_rules[area]["priority"] += 1
    
    def _update_metrics_from_feedback(self, feedback: Dict[str, Any]) -> None:
        """Update performance metrics from feedback"""
        if "efficiency" in feedback:
            self.metrics.efficiency_score = feedback["efficiency"]
        if "collaboration_rating" in feedback:
            self.metrics.collaboration_score = feedback["collaboration_rating"]
        self.metrics.last_updated = time.time()
    
    def _get_relevant_memory(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get relevant memory for given context"""
        # Simple implementation - can be enhanced with semantic search
        return {
            "recent_tasks": self.memory.episodic[-5:],  # Last 5 episodes
            "relevant_facts": self.memory.semantic
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary representation"""
        return {
            "id": self.id,
            "role": self.role,
            "goal": self.goal,
            "backstory": self.backstory,
            "state": self.state.value,
            "metrics": self.metrics.__dict__,
            "tools": [tool.name for tool in self.tools],
            "allow_delegation": self.allow_delegation
        }
