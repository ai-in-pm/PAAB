#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder - Basic Functionality Tests
Test suite for core functionality
"""

import pytest
import sys
from pathlib import Path
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.base import BaseAgent, AgentState, AgentRole
from agents.specialized import ResearcherAgent, AnalystAgent, WriterAgent, CoderAgent
from tasks.base import BaseTask, TaskStatus, TaskType, TaskPriority
from crews.base import BaseCrew, CrewStatus, ProcessType
from tools.base import BaseTool, ToolType, ToolResult
from runtime.executor import RuntimeExecutor, ExecutionMode


class MockAgent(BaseAgent):
    """Mock agent for testing"""
    
    def execute_task(self, task):
        """Mock task execution"""
        return {
            "success": True,
            "output": f"Mock execution result for task: {task.description[:50]}...",
            "agent_id": self.id,
            "execution_time": 1.0
        }


class MockTool(BaseTool):
    """Mock tool for testing"""
    
    def execute(self, **kwargs):
        """Mock tool execution"""
        return ToolResult(
            success=True,
            output=f"Mock tool result with inputs: {kwargs}",
            execution_time=0.5
        )


class TestBaseAgent:
    """Test BaseAgent functionality"""
    
    def test_agent_creation(self):
        """Test agent creation"""
        agent = MockAgent(
            role="test_agent",
            goal="Test goal",
            backstory="Test backstory"
        )
        
        assert agent.role == "test_agent"
        assert agent.goal == "Test goal"
        assert agent.backstory == "Test backstory"
        assert agent.state == AgentState.IDLE
        assert agent.id is not None
        assert len(agent.id) > 0
    
    def test_agent_status(self):
        """Test agent status retrieval"""
        agent = MockAgent(
            role="test_agent",
            goal="Test goal",
            backstory="Test backstory"
        )
        
        status = agent.get_status()
        
        assert "id" in status
        assert "role" in status
        assert "state" in status
        assert "metrics" in status
        assert status["role"] == "test_agent"
        assert status["state"] == AgentState.IDLE.value
    
    def test_agent_collaboration(self):
        """Test agent collaboration"""
        agent1 = MockAgent(role="agent1", goal="Goal 1", backstory="Backstory 1")
        agent2 = MockAgent(role="agent2", goal="Goal 2", backstory="Backstory 2")
        
        context = {"shared_data": "test_data"}
        result = agent1.collaborate([agent2], context)
        
        assert result["success"] is True
        assert agent2.id in result["participants"]


class TestSpecializedAgents:
    """Test specialized agent classes"""
    
    def test_researcher_agent(self):
        """Test ResearcherAgent"""
        agent = ResearcherAgent(
            goal="Research AI trends",
            backstory="Expert researcher",
            expertise_domains=["AI", "ML"]
        )
        
        assert agent.role == AgentRole.RESEARCHER.value
        assert "AI" in agent.expertise_domains
        assert "ML" in agent.expertise_domains
    
    def test_analyst_agent(self):
        """Test AnalystAgent"""
        agent = AnalystAgent(
            goal="Analyze data",
            backstory="Expert analyst"
        )
        
        assert agent.role == AgentRole.ANALYST.value
        assert "descriptive" in agent.analysis_types
    
    def test_writer_agent(self):
        """Test WriterAgent"""
        agent = WriterAgent(
            goal="Write content",
            backstory="Expert writer"
        )
        
        assert agent.role == AgentRole.WRITER.value
        assert "technical" in agent.writing_styles
    
    def test_coder_agent(self):
        """Test CoderAgent"""
        agent = CoderAgent(
            goal="Write code",
            backstory="Expert developer"
        )
        
        assert agent.role == AgentRole.CODER.value
        assert "python" in agent.programming_languages


class TestBaseTask:
    """Test BaseTask functionality"""
    
    def test_task_creation(self):
        """Test task creation"""
        task = BaseTask(
            description="Test task",
            expected_output="Test output",
            task_type=TaskType.CUSTOM,
            priority=TaskPriority.MEDIUM
        )
        
        assert task.description == "Test task"
        assert task.expected_output == "Test output"
        assert task.task_type == TaskType.CUSTOM
        assert task.priority == TaskPriority.MEDIUM
        assert task.status == TaskStatus.PENDING
        assert task.id is not None
    
    def test_task_assignment(self):
        """Test task assignment to agent"""
        task = BaseTask(
            description="Test task",
            expected_output="Test output"
        )
        
        agent = MockAgent(role="test", goal="test", backstory="test")
        task.assign_agent(agent)
        
        assert task.agent == agent
        assert task.status == TaskStatus.ASSIGNED
    
    def test_task_dependencies(self):
        """Test task dependencies"""
        task1 = BaseTask(description="Task 1", expected_output="Output 1")
        task2 = BaseTask(description="Task 2", expected_output="Output 2")
        
        task2.add_dependency(task1)
        
        assert task1 in task2.dependencies
        assert task2 in task1.dependents
        assert not task2.can_start()  # task1 not completed
        
        # Complete task1
        task1.status = TaskStatus.COMPLETED
        assert task2.can_start()  # Now task2 can start
    
    def test_task_execution_lifecycle(self):
        """Test task execution lifecycle"""
        task = BaseTask(description="Test task", expected_output="Test output")
        agent = MockAgent(role="test", goal="test", backstory="test")
        task.assign_agent(agent)
        
        # Start execution
        task.start_execution()
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.started_at is not None
        
        # Complete execution
        result = ToolResult(success=True, output="Test result")
        task.complete_execution(result)
        assert task.status == TaskStatus.COMPLETED
        assert task.completed_at is not None
        assert task.result == result


class TestBaseCrew:
    """Test BaseCrew functionality"""
    
    def test_crew_creation(self):
        """Test crew creation"""
        agent1 = MockAgent(role="agent1", goal="Goal 1", backstory="Backstory 1")
        agent2 = MockAgent(role="agent2", goal="Goal 2", backstory="Backstory 2")
        
        task1 = BaseTask(description="Task 1", expected_output="Output 1")
        task2 = BaseTask(description="Task 2", expected_output="Output 2")
        
        crew = BaseCrew(
            name="Test Crew",
            agents=[agent1, agent2],
            tasks=[task1, task2],
            process_type=ProcessType.SEQUENTIAL
        )
        
        assert crew.name == "Test Crew"
        assert len(crew.agents) == 2
        assert len(crew.tasks) == 2
        assert crew.process_type == ProcessType.SEQUENTIAL
        assert crew.status == CrewStatus.IDLE
        assert crew.id is not None
    
    def test_crew_planning(self):
        """Test crew execution planning"""
        agent = MockAgent(role="test", goal="test", backstory="test")
        task = BaseTask(description="Test task", expected_output="Test output")
        
        crew = BaseCrew(
            name="Test Crew",
            agents=[agent],
            tasks=[task],
            planning_enabled=True
        )
        
        plan = crew.plan_execution()
        
        assert "task_dependencies" in plan
        assert "task_assignments" in plan
        assert "execution_sequence" in plan
        assert "estimated_time" in plan
    
    def test_crew_agent_management(self):
        """Test crew agent management"""
        agent1 = MockAgent(role="agent1", goal="Goal 1", backstory="Backstory 1")
        agent2 = MockAgent(role="agent2", goal="Goal 2", backstory="Backstory 2")
        
        crew = BaseCrew(name="Test Crew", agents=[agent1], tasks=[])
        
        # Add agent
        crew.add_agent(agent2)
        assert agent2 in crew.agents
        assert agent2.crew == crew
        
        # Remove agent
        crew.remove_agent(agent2)
        assert agent2 not in crew.agents
        assert agent2.crew is None


class TestBaseTool:
    """Test BaseTool functionality"""
    
    def test_tool_creation(self):
        """Test tool creation"""
        tool = MockTool(
            name="test_tool",
            description="Test tool",
            tool_type=ToolType.CUSTOM
        )
        
        assert tool.name == "test_tool"
        assert tool.description == "Test tool"
        assert tool.tool_type == ToolType.CUSTOM
        assert tool.id is not None
    
    def test_tool_execution(self):
        """Test tool execution"""
        tool = MockTool(name="test_tool", description="Test tool")
        
        result = tool.execute_with_validation(test_param="test_value")
        
        assert result.success is True
        assert "test_param" in result.output
        assert result.execution_time > 0
    
    def test_tool_schema(self):
        """Test tool schema generation"""
        from tools.base import ToolParameter
        
        tool = MockTool(
            name="test_tool",
            description="Test tool",
            parameters=[
                ToolParameter(
                    name="param1",
                    type=str,
                    description="Test parameter",
                    required=True
                )
            ]
        )
        
        schema = tool.get_schema()
        
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "test_tool"
        assert "param1" in schema["function"]["parameters"]["properties"]
        assert "param1" in schema["function"]["parameters"]["required"]


class TestRuntimeExecutor:
    """Test RuntimeExecutor functionality"""
    
    def test_executor_creation(self):
        """Test executor creation"""
        executor = RuntimeExecutor(
            max_concurrent_crews=5,
            max_workers=10
        )
        
        assert executor.max_concurrent_crews == 5
        assert executor.max_workers == 10
        assert executor.status.value == "idle"
    
    def test_executor_lifecycle(self):
        """Test executor start/stop lifecycle"""
        executor = RuntimeExecutor()
        
        # Start executor
        executor.start()
        assert executor.status.value == "running"
        
        # Stop executor
        executor.stop()
        assert executor.status.value == "stopped"
    
    def test_executor_metrics(self):
        """Test executor performance metrics"""
        executor = RuntimeExecutor()
        executor.start()
        
        metrics = executor.get_performance_metrics()
        
        assert "total_executions" in metrics
        assert "successful_executions" in metrics
        assert "failed_executions" in metrics
        assert "active_sessions" in metrics
        
        executor.stop()


class TestIntegration:
    """Integration tests"""
    
    def test_simple_crew_workflow(self):
        """Test a simple crew workflow"""
        # Create agents
        researcher = MockAgent(role="researcher", goal="Research", backstory="Expert researcher")
        writer = MockAgent(role="writer", goal="Write", backstory="Expert writer")
        
        # Create tasks
        research_task = BaseTask(
            description="Research AI trends",
            expected_output="Research report",
            task_type=TaskType.RESEARCH
        )
        
        writing_task = BaseTask(
            description="Write summary",
            expected_output="Summary document",
            task_type=TaskType.WRITING
        )
        
        # Assign tasks
        research_task.assign_agent(researcher)
        writing_task.assign_agent(writer)
        
        # Set dependencies
        writing_task.add_dependency(research_task)
        
        # Create crew
        crew = BaseCrew(
            name="Research Crew",
            agents=[researcher, writer],
            tasks=[research_task, writing_task],
            process_type=ProcessType.SEQUENTIAL
        )
        
        # Verify crew setup
        assert len(crew.agents) == 2
        assert len(crew.tasks) == 2
        assert writing_task.dependencies == [research_task]
        
        # Test planning
        plan = crew.plan_execution()
        assert plan is not None
        
        # Verify task assignment
        assignments = plan.get("task_assignments", {})
        assert research_task.id in assignments
        assert writing_task.id in assignments


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
