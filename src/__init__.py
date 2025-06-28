#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder
A CrewAI-style framework for building AI agent teams, evolved from PsychoPy
"""

__version__ = "1.0.0"
__author__ = "AI in PM"
__email__ = "ai@example.com"
__license__ = "MIT"
__description__ = "PsychoPy AI Agent Builder - CrewAI-style framework for building AI agent teams"

# Core imports for easy access
from .agents.base import BaseAgent, AgentState, AgentRole
from .agents.specialized import ResearcherAgent, AnalystAgent, WriterAgent, CoderAgent

from .tasks.base import BaseTask, TaskStatus, TaskType, TaskPriority
from .crews.base import BaseCrew, CrewStatus, ProcessType, CollaborationPattern

from .tools.base import BaseTool, ToolType, ToolStatus
from .runtime.executor import RuntimeExecutor, ExecutionMode, ExecutionContext

# Version information
def get_version():
    """Get the current version of PAAB"""
    return __version__

# Package metadata
__all__ = [
    # Version
    "__version__",
    "get_version",
    
    # Core classes
    "BaseAgent",
    "BaseTask", 
    "BaseCrew",
    "BaseTool",
    "RuntimeExecutor",
    
    # Specialized agents
    "ResearcherAgent",
    "AnalystAgent", 
    "WriterAgent",
    "CoderAgent",
    
    # Enums
    "AgentState",
    "AgentRole",
    "TaskStatus",
    "TaskType", 
    "TaskPriority",
    "CrewStatus",
    "ProcessType",
    "CollaborationPattern",
    "ToolType",
    "ToolStatus",
    "ExecutionMode",
    
    # Context
    "ExecutionContext",
]
