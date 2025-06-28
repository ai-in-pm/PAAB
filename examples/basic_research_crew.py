#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder - Basic Research Crew Example
Demonstrates how to create and execute a simple research crew
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.specialized import ResearcherAgent, AnalystAgent, WriterAgent
from tasks.base import BaseTask, TaskType, TaskPriority
from crews.base import BaseCrew, ProcessType
from runtime.executor import RuntimeExecutor, ExecutionMode
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_research_crew():
    """Create a research crew with specialized agents"""
    
    # Create specialized agents
    researcher = ResearcherAgent(
        goal="Conduct thorough research on AI agent frameworks and their applications",
        backstory="You are an expert researcher with deep knowledge of AI technologies, "
                 "machine learning frameworks, and software development trends. You excel at "
                 "finding relevant information, analyzing sources, and synthesizing findings.",
        expertise_domains=["artificial intelligence", "machine learning", "software frameworks"]
    )
    
    analyst = AnalystAgent(
        goal="Analyze research findings and extract actionable insights",
        backstory="You are a skilled data analyst with expertise in statistical analysis, "
                 "pattern recognition, and trend identification. You can transform raw research "
                 "data into meaningful insights and recommendations."
    )
    
    writer = WriterAgent(
        goal="Create comprehensive and well-structured reports from research and analysis",
        backstory="You are an experienced technical writer who specializes in creating clear, "
                 "concise, and engaging content. You excel at transforming complex technical "
                 "information into accessible reports and documentation."
    )
    
    # Define research tasks
    research_task = BaseTask(
        description="""
        Research the current landscape of AI agent frameworks, focusing on:
        1. Popular frameworks like CrewAI, AutoGen, LangChain Agents
        2. Key features and capabilities of each framework
        3. Use cases and applications in different industries
        4. Recent developments and trends in the field
        5. Strengths and limitations of existing solutions
        
        Provide comprehensive findings with sources and examples.
        """,
        expected_output="""
        A detailed research report containing:
        - Overview of major AI agent frameworks
        - Comparative analysis of features and capabilities
        - Industry use cases and applications
        - Recent trends and developments
        - Source citations and references
        """,
        task_type=TaskType.RESEARCH,
        priority=TaskPriority.HIGH
    )
    
    analysis_task = BaseTask(
        description="""
        Analyze the research findings to identify:
        1. Market gaps and opportunities
        2. Common patterns and trends across frameworks
        3. Technical strengths and weaknesses
        4. User adoption factors and barriers
        5. Future direction predictions
        
        Provide data-driven insights and recommendations.
        """,
        expected_output="""
        An analytical report containing:
        - Market gap analysis
        - Pattern and trend identification
        - SWOT analysis of major frameworks
        - User adoption insights
        - Strategic recommendations
        - Supporting data and visualizations
        """,
        task_type=TaskType.ANALYSIS,
        priority=TaskPriority.HIGH
    )
    
    writing_task = BaseTask(
        description="""
        Create a comprehensive executive summary report that combines the research 
        findings and analysis into a cohesive document suitable for stakeholders.
        The report should be:
        1. Well-structured with clear sections
        2. Executive-friendly with key insights highlighted
        3. Actionable with specific recommendations
        4. Professional and engaging in tone
        5. Include visual elements where appropriate
        """,
        expected_output="""
        A professional executive summary report including:
        - Executive summary (1-2 pages)
        - Market landscape overview
        - Key findings and insights
        - Strategic recommendations
        - Implementation roadmap
        - Appendices with detailed data
        """,
        task_type=TaskType.WRITING,
        priority=TaskPriority.MEDIUM
    )
    
    # Assign tasks to agents
    research_task.assign_agent(researcher)
    analysis_task.assign_agent(analyst)
    writing_task.assign_agent(writer)
    
    # Set up task dependencies
    analysis_task.add_dependency(research_task)
    writing_task.add_dependency(analysis_task)
    
    # Create the crew
    crew = BaseCrew(
        name="AI Framework Research Crew",
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, writing_task],
        process_type=ProcessType.SEQUENTIAL,
        verbose=True,
        planning_enabled=True,
        monitoring_enabled=True
    )
    
    return crew


def execute_crew_example():
    """Execute the research crew and display results"""
    
    print("🚀 PsychoPy AI Agent Builder - Research Crew Example")
    print("=" * 60)
    
    try:
        # Create the crew
        print("📋 Creating research crew...")
        crew = create_research_crew()
        
        # Display crew information
        print(f"\n👥 Crew: {crew.name}")
        print(f"🔧 Process Type: {crew.process_type.value}")
        print(f"🤖 Agents: {len(crew.agents)}")
        print(f"📝 Tasks: {len(crew.tasks)}")
        
        # Show agent details
        print("\n🤖 Agents:")
        for i, agent in enumerate(crew.agents, 1):
            print(f"  {i}. {agent.role} (ID: {agent.id[:8]}...)")
            print(f"     Goal: {agent.goal[:80]}...")
        
        # Show task details
        print("\n📝 Tasks:")
        for i, task in enumerate(crew.tasks, 1):
            print(f"  {i}. {task.task_type.value.title()} Task (ID: {task.id[:8]}...)")
            print(f"     Description: {task.description[:80]}...")
            print(f"     Agent: {task.agent.role if task.agent else 'Unassigned'}")
            print(f"     Dependencies: {len(task.dependencies)}")
        
        # Create execution plan
        print("\n📊 Creating execution plan...")
        execution_plan = crew.plan_execution()
        
        if execution_plan.get("success", True):
            print("✅ Execution plan created successfully")
            print(f"⏱️  Estimated time: {execution_plan.get('estimated_time', 'Unknown')} seconds")
        else:
            print(f"❌ Planning failed: {execution_plan.get('error', 'Unknown error')}")
            return
        
        # Initialize runtime executor
        print("\n🔧 Initializing runtime executor...")
        executor = RuntimeExecutor(
            max_concurrent_crews=5,
            max_workers=10,
            enable_monitoring=True,
            enable_security=True
        )
        
        executor.start()
        print("✅ Runtime executor started")
        
        # Execute the crew
        print("\n🚀 Executing crew...")
        print("⏳ This may take a few minutes...")
        
        # Prepare input data
        input_data = {
            "research_focus": "AI agent frameworks",
            "target_audience": "technical stakeholders",
            "report_format": "executive summary",
            "deadline": "urgent"
        }
        
        # Execute in asynchronous mode
        result_future = executor.execute_crew(
            crew=crew,
            inputs=input_data,
            mode=ExecutionMode.ASYNCHRONOUS,
            timeout=1800  # 30 minutes
        )
        
        # Monitor progress
        session_id = result_future.result().session_id if hasattr(result_future, 'result') else None
        if session_id:
            print(f"📊 Session ID: {session_id}")
            
            # You could add progress monitoring here
            # status = executor.get_session_status(session_id)
            # print(f"Status: {status}")
        
        # Wait for completion
        print("⏳ Waiting for execution to complete...")
        result = result_future.result()  # This will block until completion
        
        # Display results
        print("\n" + "=" * 60)
        if result.success:
            print("✅ Crew execution completed successfully!")
            print(f"⏱️  Total execution time: {result.execution_time:.2f} seconds")
            
            # Display outputs
            print("\n📋 Execution Results:")
            for key, value in result.outputs.items():
                print(f"  {key}: {str(value)[:100]}...")
            
            # Display metrics
            if result.metrics:
                print("\n📊 Performance Metrics:")
                for key, value in result.metrics.items():
                    print(f"  {key}: {value}")
            
        else:
            print("❌ Crew execution failed!")
            print(f"🔍 Errors: {result.errors}")
            if result.warnings:
                print(f"⚠️  Warnings: {result.warnings}")
        
        # Stop executor
        executor.stop()
        print("\n🔧 Runtime executor stopped")
        
    except KeyboardInterrupt:
        print("\n⚠️  Execution interrupted by user")
        if 'executor' in locals():
            executor.stop()
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        logger.exception("Detailed error information:")
        if 'executor' in locals():
            executor.stop()
    
    print("\n🎉 Example completed!")


def main():
    """Main function"""
    execute_crew_example()


if __name__ == "__main__":
    main()
