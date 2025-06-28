#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder - Advanced Research Pipeline Example
Demonstrates advanced features including memory, learning, and complex workflows
"""

import sys
import asyncio
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.specialized import ResearcherAgent, AnalystAgent, WriterAgent
from agents.learning import ReinforcementLearningSystem, LearningType
from tasks.base import BaseTask, TaskType, TaskPriority
from crews.base import BaseCrew, ProcessType, CollaborationPattern
from flows.base import LinearFlow, FlowNode, NodeType
from memory.base import InMemoryManager, MemoryType, SharedMemory
from runtime.executor import RuntimeExecutor, ExecutionMode
from runtime.monitoring import MonitoringSystem, SystemMonitor, AgentMonitor
from tools.llm_tools import get_default_llm_tools
from integrations.llm_providers import llm_manager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedResearchPipeline:
    """Advanced research pipeline with learning and memory"""
    
    def __init__(self):
        """Initialize the advanced research pipeline"""
        self.agents = {}
        self.tasks = {}
        self.crews = {}
        self.flows = {}
        self.memory_manager = None
        self.shared_memory = None
        self.monitoring_system = None
        self.executor = None
        
        logger.info("Initializing Advanced Research Pipeline")
    
    def setup_memory_system(self):
        """Set up advanced memory and learning systems"""
        logger.info("Setting up memory and learning systems...")
        
        # Create memory manager with embedding support
        self.memory_manager = InMemoryManager(
            max_capacity=50000,
            enable_persistence=True,
            persistence_path="data/memory.pkl"
        )
        
        # Set up embedding function if available
        try:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.memory_manager.set_embedding_function(
                lambda text: embedding_model.encode(text).tolist()
            )
            logger.info("Embedding function configured for memory system")
        except ImportError:
            logger.warning("sentence-transformers not available, using basic similarity")
        
        # Create shared memory for crew collaboration
        self.shared_memory = SharedMemory("research-pipeline-crew")
        
        logger.info("Memory systems configured successfully")
    
    def create_learning_agents(self):
        """Create agents with learning capabilities"""
        logger.info("Creating learning-enabled agents...")
        
        # Advanced Research Agent with reinforcement learning
        researcher = ResearcherAgent(
            goal="Conduct comprehensive research with continuous improvement",
            backstory="""You are an advanced AI researcher with the ability to learn and adapt 
                        from each research task. You continuously improve your research strategies 
                        based on feedback and results.""",
            expertise_domains=["AI", "machine learning", "research methodology", "data analysis"]
        )
        
        # Add learning system
        researcher.learning_system = ReinforcementLearningSystem(
            learning_rate=0.01,
            memory_size=1000,
            enable_transfer=True
        )
        
        # Add memory system
        researcher.memory_manager = self.memory_manager
        
        # Add tools
        for tool in get_default_llm_tools():
            researcher.add_tool(tool)
        
        self.agents['researcher'] = researcher
        
        # Advanced Analyst Agent
        analyst = AnalystAgent(
            goal="Perform deep analysis with pattern recognition and learning",
            backstory="""You are an expert data analyst with advanced pattern recognition 
                        capabilities. You learn from each analysis to improve future insights."""
        )
        
        # Add learning and memory
        analyst.learning_system = ReinforcementLearningSystem(
            learning_rate=0.015,
            memory_size=800
        )
        analyst.memory_manager = self.memory_manager
        
        for tool in get_default_llm_tools():
            analyst.add_tool(tool)
        
        self.agents['analyst'] = analyst
        
        # Advanced Writer Agent
        writer = WriterAgent(
            goal="Create high-quality content with style adaptation",
            backstory="""You are a versatile technical writer who adapts writing style 
                        based on audience feedback and content performance."""
        )
        
        # Add learning and memory
        writer.learning_system = ReinforcementLearningSystem(
            learning_rate=0.02,
            memory_size=600
        )
        writer.memory_manager = self.memory_manager
        
        for tool in get_default_llm_tools():
            writer.add_tool(tool)
        
        self.agents['writer'] = writer
        
        logger.info(f"Created {len(self.agents)} learning-enabled agents")
    
    def create_advanced_tasks(self):
        """Create complex, interconnected tasks"""
        logger.info("Creating advanced task pipeline...")
        
        # Literature Review Task
        literature_task = BaseTask(
            description="""
            Conduct a comprehensive literature review on the specified research topic:
            1. Search for recent academic papers and publications
            2. Identify key researchers and institutions
            3. Analyze research trends and methodologies
            4. Summarize major findings and contributions
            5. Identify research gaps and opportunities
            
            Focus on peer-reviewed sources from the last 3 years.
            """,
            expected_output="""
            Comprehensive literature review including:
            - Annotated bibliography of 20+ key sources
            - Timeline of major developments
            - Key researcher profiles
            - Research trend analysis
            - Identified gaps and opportunities
            """,
            task_type=TaskType.RESEARCH,
            priority=TaskPriority.HIGH,
            max_execution_time=3600  # 1 hour
        )
        
        # Data Collection Task
        data_collection_task = BaseTask(
            description="""
            Collect and organize relevant data for the research topic:
            1. Gather quantitative data from reliable sources
            2. Collect qualitative insights from expert interviews/surveys
            3. Compile industry reports and market data
            4. Organize data in structured formats
            5. Validate data quality and reliability
            """,
            expected_output="""
            Structured data collection including:
            - Quantitative datasets with metadata
            - Qualitative insights summary
            - Industry reports compilation
            - Data quality assessment
            - Recommended data sources for future research
            """,
            task_type=TaskType.RESEARCH,
            priority=TaskPriority.HIGH
        )
        
        # Comparative Analysis Task
        analysis_task = BaseTask(
            description="""
            Perform comprehensive analysis of collected research and data:
            1. Conduct comparative analysis of different approaches/solutions
            2. Identify patterns, trends, and correlations
            3. Perform statistical analysis where applicable
            4. Evaluate strengths and weaknesses of different methods
            5. Generate insights and recommendations
            """,
            expected_output="""
            Detailed analysis report including:
            - Comparative analysis matrix
            - Statistical analysis results
            - Pattern and trend identification
            - SWOT analysis of key approaches
            - Data-driven insights and recommendations
            """,
            task_type=TaskType.ANALYSIS,
            priority=TaskPriority.HIGH
        )
        
        # Synthesis and Recommendations Task
        synthesis_task = BaseTask(
            description="""
            Synthesize all research and analysis into actionable recommendations:
            1. Integrate findings from literature review and data analysis
            2. Develop evidence-based recommendations
            3. Create implementation roadmap
            4. Identify potential risks and mitigation strategies
            5. Suggest future research directions
            """,
            expected_output="""
            Strategic synthesis document including:
            - Executive summary of key findings
            - Evidence-based recommendations
            - Implementation roadmap with timelines
            - Risk assessment and mitigation strategies
            - Future research agenda
            """,
            task_type=TaskType.ANALYSIS,
            priority=TaskPriority.MEDIUM
        )
        
        # Final Report Task
        report_task = BaseTask(
            description="""
            Create a comprehensive final report that combines all research elements:
            1. Write executive summary for stakeholders
            2. Organize findings in logical structure
            3. Include visualizations and supporting materials
            4. Ensure clarity and accessibility for target audience
            5. Provide appendices with detailed data
            """,
            expected_output="""
            Professional research report including:
            - Executive summary (2-3 pages)
            - Detailed findings and analysis (15-20 pages)
            - Recommendations and roadmap (3-5 pages)
            - Supporting visualizations and charts
            - Comprehensive appendices
            """,
            task_type=TaskType.WRITING,
            priority=TaskPriority.MEDIUM
        )
        
        # Assign agents to tasks
        literature_task.assign_agent(self.agents['researcher'])
        data_collection_task.assign_agent(self.agents['researcher'])
        analysis_task.assign_agent(self.agents['analyst'])
        synthesis_task.assign_agent(self.agents['analyst'])
        report_task.assign_agent(self.agents['writer'])
        
        # Set up task dependencies
        analysis_task.add_dependency(literature_task)
        analysis_task.add_dependency(data_collection_task)
        synthesis_task.add_dependency(analysis_task)
        report_task.add_dependency(synthesis_task)
        
        # Store tasks
        self.tasks = {
            'literature': literature_task,
            'data_collection': data_collection_task,
            'analysis': analysis_task,
            'synthesis': synthesis_task,
            'report': report_task
        }
        
        logger.info(f"Created {len(self.tasks)} interconnected tasks")
    
    def create_adaptive_crew(self):
        """Create an adaptive crew with advanced collaboration"""
        logger.info("Creating adaptive research crew...")
        
        crew = BaseCrew(
            name="Advanced Research Pipeline Crew",
            agents=list(self.agents.values()),
            tasks=list(self.tasks.values()),
            process_type=ProcessType.SEQUENTIAL,
            collaboration_pattern=CollaborationPattern.PEER_TO_PEER,
            verbose=True,
            planning_enabled=True,
            monitoring_enabled=True,
            memory_enabled=True
        )
        
        # Set shared memory
        crew.shared_memory = self.shared_memory
        
        # Enable learning from crew performance
        crew.enable_learning = True
        
        self.crews['main'] = crew
        
        logger.info("Adaptive crew created with advanced collaboration features")
    
    def create_workflow(self):
        """Create a complex workflow using flows"""
        logger.info("Creating advanced workflow...")
        
        flow = LinearFlow(
            name="Research Pipeline Flow",
            description="Advanced research pipeline with parallel processing and decision points",
            max_execution_time=7200,  # 2 hours
            retry_failed_nodes=True,
            parallel_execution=True
        )
        
        # Add workflow nodes
        start_node = flow.add_node("Start", NodeType.START)
        
        # Parallel research phase
        lit_node = flow.add_node("Literature Review", NodeType.TASK, task=self.tasks['literature'])
        data_node = flow.add_node("Data Collection", NodeType.TASK, task=self.tasks['data_collection'])
        
        # Analysis phase
        analysis_node = flow.add_node("Analysis", NodeType.TASK, task=self.tasks['analysis'])
        
        # Decision point
        decision_node = flow.add_node(
            "Quality Check", 
            NodeType.DECISION,
            condition=lambda vars: vars.get('analysis_quality', 0) > 0.8
        )
        
        # Synthesis and reporting
        synthesis_node = flow.add_node("Synthesis", NodeType.TASK, task=self.tasks['synthesis'])
        report_node = flow.add_node("Final Report", NodeType.TASK, task=self.tasks['report'])
        
        end_node = flow.add_node("End", NodeType.END)
        
        # Connect nodes
        flow.add_connection(start_node, lit_node)
        flow.add_connection(start_node, data_node)
        flow.add_connection(lit_node, analysis_node)
        flow.add_connection(data_node, analysis_node)
        flow.add_connection(analysis_node, decision_node)
        flow.add_connection(decision_node, synthesis_node)
        flow.add_connection(synthesis_node, report_node)
        flow.add_connection(report_node, end_node)
        
        self.flows['main'] = flow
        
        logger.info("Advanced workflow created with parallel processing and decision points")
    
    def setup_monitoring(self):
        """Set up comprehensive monitoring"""
        logger.info("Setting up monitoring system...")
        
        self.monitoring_system = MonitoringSystem()
        
        # Add system monitor
        system_monitor = SystemMonitor(collection_interval=30.0)
        self.monitoring_system.add_monitor(system_monitor)
        
        # Add agent monitor
        agent_monitor = AgentMonitor(
            agent_manager=self,
            collection_interval=60.0
        )
        self.monitoring_system.add_monitor(agent_monitor)
        
        logger.info("Monitoring system configured")
    
    async def execute_pipeline(self, research_topic: str, target_audience: str = "technical"):
        """Execute the complete research pipeline"""
        logger.info(f"Starting advanced research pipeline for topic: {research_topic}")
        
        try:
            # Start monitoring
            await self.monitoring_system.start()
            
            # Initialize executor
            self.executor = RuntimeExecutor(
                max_concurrent_crews=3,
                max_workers=20,
                enable_monitoring=True,
                enable_security=True
            )
            
            self.executor.start()
            
            # Prepare input data
            input_data = {
                "research_topic": research_topic,
                "target_audience": target_audience,
                "quality_threshold": 0.8,
                "deadline": "2 hours",
                "output_format": "comprehensive_report"
            }
            
            # Store initial context in shared memory
            self.shared_memory.share(
                content={
                    "research_topic": research_topic,
                    "pipeline_start": True,
                    "quality_requirements": "high"
                },
                shared_by="pipeline_manager",
                tags=["context", "initialization"]
            )
            
            # Execute the crew
            logger.info("Executing research crew...")
            result = self.executor.execute_crew(
                crew=self.crews['main'],
                inputs=input_data,
                mode=ExecutionMode.ASYNCHRONOUS,
                timeout=7200  # 2 hours
            )
            
            # Monitor execution progress
            session_id = result.result().session_id if hasattr(result, 'result') else None
            if session_id:
                logger.info(f"Execution session: {session_id}")
                
                # Periodic status updates
                for i in range(12):  # Check every 10 minutes for 2 hours
                    await asyncio.sleep(600)  # 10 minutes
                    status = self.executor.get_session_status(session_id)
                    if status:
                        logger.info(f"Progress: {status['progress']:.1%} - Current: {status.get('current_task', 'Unknown')}")
                        
                        if status['status'] in ['completed', 'failed', 'cancelled']:
                            break
            
            # Wait for completion
            final_result = result.result() if hasattr(result, 'result') else result
            
            # Process results
            if final_result.success:
                logger.info("✅ Research pipeline completed successfully!")
                
                # Store results in memory for future learning
                self.shared_memory.share(
                    content={
                        "pipeline_result": "success",
                        "execution_time": final_result.execution_time,
                        "outputs": final_result.outputs
                    },
                    shared_by="pipeline_manager",
                    tags=["results", "success"]
                )
                
                # Update agent learning systems
                for agent in self.agents.values():
                    if hasattr(agent, 'learning_system'):
                        agent.learning_system.add_experience(
                            state={"topic": research_topic, "audience": target_audience},
                            action={"pipeline": "advanced_research"},
                            result={"success": True, "quality": "high"},
                            reward=1.0,
                            context={"execution_time": final_result.execution_time}
                        )
                
                # Display results
                self._display_results(final_result)
                
            else:
                logger.error("❌ Research pipeline failed!")
                logger.error(f"Errors: {final_result.errors}")
                
                # Learn from failure
                for agent in self.agents.values():
                    if hasattr(agent, 'learning_system'):
                        agent.learning_system.add_experience(
                            state={"topic": research_topic, "audience": target_audience},
                            action={"pipeline": "advanced_research"},
                            result={"success": False, "errors": final_result.errors},
                            reward=-0.5,
                            context={"execution_time": final_result.execution_time}
                        )
            
            return final_result
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise
        
        finally:
            # Cleanup
            if self.executor:
                self.executor.stop()
            
            if self.monitoring_system:
                self.monitoring_system.stop()
            
            logger.info("Pipeline execution completed")
    
    def _display_results(self, result):
        """Display execution results"""
        print("\n" + "="*80)
        print("🎉 ADVANCED RESEARCH PIPELINE RESULTS")
        print("="*80)
        
        print(f"\n⏱️  Execution Time: {result.execution_time:.2f} seconds")
        print(f"📊 Success Rate: {'100%' if result.success else '0%'}")
        
        print("\n📋 Task Outputs:")
        for task_name, output in result.outputs.items():
            print(f"\n🔹 {task_name.title()}:")
            if isinstance(output, dict):
                for key, value in output.items():
                    print(f"   {key}: {str(value)[:100]}...")
            else:
                print(f"   {str(output)[:200]}...")
        
        # Display learning insights
        print("\n🧠 Learning Insights:")
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'learning_system'):
                insights = agent.learning_system.get_learning_insights()
                print(f"\n🤖 {agent_name.title()} Agent:")
                print(f"   Total Experiences: {insights['total_experiences']}")
                print(f"   Success Rate: {insights['success_rate']:.1%}")
                print(f"   Confidence Level: {insights['confidence_level']:.1%}")
                print(f"   Adaptation Score: {insights['adaptation_score']:.1%}")
        
        # Display collaboration stats
        if self.shared_memory:
            collab_stats = self.shared_memory.get_collaboration_stats()
            print(f"\n🤝 Collaboration Statistics:")
            print(f"   Shared Memories: {collab_stats['total_memories']}")
            print(f"   Total Actions: {collab_stats['total_actions']}")
            print(f"   Agent Activity: {len(collab_stats['agent_activity'])} agents")
        
        # Display monitoring metrics
        if self.monitoring_system:
            system_status = self.monitoring_system.get_system_status()
            print(f"\n📊 System Performance:")
            print(f"   System Healthy: {'✅' if system_status['system_healthy'] else '❌'}")
            print(f"   Active Alerts: {system_status['active_alerts']}")
            print(f"   Uptime: {system_status['uptime']:.1f} seconds")
        
        print("\n" + "="*80)
    
    async def run_example(self):
        """Run the complete example"""
        logger.info("🚀 Starting Advanced Research Pipeline Example")
        
        # Setup all components
        self.setup_memory_system()
        self.create_learning_agents()
        self.create_advanced_tasks()
        self.create_adaptive_crew()
        self.create_workflow()
        self.setup_monitoring()
        
        # Execute pipeline
        research_topic = "The Future of AI Agent Frameworks: Trends, Challenges, and Opportunities"
        target_audience = "technical_stakeholders"
        
        result = await self.execute_pipeline(research_topic, target_audience)
        
        logger.info("🎉 Advanced Research Pipeline Example Completed!")
        return result


async def main():
    """Main function"""
    pipeline = AdvancedResearchPipeline()
    await pipeline.run_example()


if __name__ == "__main__":
    asyncio.run(main())
