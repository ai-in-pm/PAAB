#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder - PsychoPy Integration Demo
Demonstrates the integration of PsychoPy experiments with AI agents
"""

import sys
import asyncio
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import with absolute paths to avoid relative import issues
try:
    from psychopy_integration.core import (
        ExperimentType, ExperimentConfig, StroopExperiment, ReactionTimeExperiment
    )
    from agents.psychopy_agents import (
        ExperimentParticipantAgent, ExperimentDesignerAgent, ExperimentAnalystAgent
    )
    from tasks.psychopy_tasks import (
        ExperimentDesignTask, ExperimentExecutionTask, ExperimentAnalysisTask,
        PsychoPyIntegrationTask
    )
    from crews.base import BaseCrew, ProcessType, CollaborationPattern
    from runtime.executor import RuntimeExecutor, ExecutionMode
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    print("Running in simulation mode without full imports")
    IMPORTS_AVAILABLE = False
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PsychoPyIntegrationDemo:
    """Comprehensive demo of PsychoPy integration with AI agents"""
    
    def __init__(self):
        """Initialize the demo"""
        self.agents = {}
        self.experiments = {}
        self.results = {}
        
        logger.info("Initializing PsychoPy Integration Demo")
    
    def create_specialized_agents(self):
        """Create specialized agents for psychological experiments"""
        logger.info("Creating specialized agents...")
        
        # Experiment Designer Agent
        designer = ExperimentDesignerAgent(
            goal="Design rigorous psychological experiments",
            backstory="You are an expert experimental psychologist with 15 years of experience in cognitive research"
        )
        self.agents['designer'] = designer
        
        # Experiment Analyst Agent
        analyst = ExperimentAnalystAgent(
            goal="Analyze experimental data and provide insights",
            backstory="You are a statistical expert specializing in psychological research analysis"
        )
        self.agents['analyst'] = analyst
        
        # Participant Agents with different cognitive profiles
        participants = []
        
        # Optimal performer
        optimal_agent = ExperimentParticipantAgent(
            goal="Perform optimally in psychological experiments",
            backstory="You are an AI agent designed for optimal performance",
            cognitive_profile={
                "base_reaction_time": 0.4,
                "reaction_time_variability": 0.05,
                "accuracy_rate": 0.95,
                "attention_span": 600.0,
                "fatigue_rate": 0.0005,
                "learning_rate": 0.02
            },
            response_strategy="optimal"
        )
        participants.append(optimal_agent)
        
        # Human-like performer
        human_like_agent = ExperimentParticipantAgent(
            goal="Simulate human-like performance in experiments",
            backstory="You are an AI agent that mimics human cognitive performance",
            cognitive_profile={
                "base_reaction_time": 0.7,
                "reaction_time_variability": 0.15,
                "accuracy_rate": 0.82,
                "attention_span": 300.0,
                "fatigue_rate": 0.002,
                "learning_rate": 0.015
            },
            response_strategy="human-like"
        )
        participants.append(human_like_agent)
        
        # Variable performer (simulates individual differences)
        variable_agent = ExperimentParticipantAgent(
            goal="Show variable performance patterns",
            backstory="You are an AI agent with variable cognitive performance",
            cognitive_profile={
                "base_reaction_time": 0.9,
                "reaction_time_variability": 0.25,
                "accuracy_rate": 0.75,
                "attention_span": 200.0,
                "fatigue_rate": 0.003,
                "learning_rate": 0.01
            },
            response_strategy="human-like"
        )
        participants.append(variable_agent)
        
        self.agents['participants'] = participants
        
        logger.info(f"Created {len(participants)} participant agents with different cognitive profiles")
    
    async def demo_stroop_experiment(self):
        """Demonstrate Stroop color-word interference experiment"""
        logger.info("🎨 STROOP EXPERIMENT DEMO")
        print("-" * 40)
        
        # Create experiment configuration
        config = ExperimentConfig(
            name="AI_Stroop_Experiment",
            experiment_type=ExperimentType.ATTENTION,
            description="Stroop color-word interference task with AI agents",
            num_trials=60,
            duration=300.0,
            randomize_trials=True,
            window_size=(800, 600),
            background_color="gray",
            text_color="white",
            font_size=48
        )
        
        # Create and run experiment
        experiment = StroopExperiment(config)
        
        print(f"📋 Experiment: {config.name}")
        print(f"🎯 Type: {config.experiment_type.value}")
        print(f"📊 Trials: {config.num_trials}")
        
        # Run with each participant agent
        all_results = []
        for i, participant in enumerate(self.agents['participants']):
            print(f"\n🤖 Participant {i+1}: {participant.response_strategy} strategy")
            
            # Reset agent state
            participant.reset_state()
            
            # Participate in experiment
            result = participant.participate_in_experiment(experiment)
            result.participant_id = f"agent_{i+1}_{participant.response_strategy}"
            
            all_results.append(result)
            
            # Show results
            accuracy = result.performance_metrics.get("accuracy", 0)
            mean_rt = result.performance_metrics.get("mean_reaction_time", 0)
            print(f"   📈 Accuracy: {accuracy:.1%}")
            print(f"   ⏱️  Mean RT: {mean_rt:.3f}s")
            print(f"   🎯 Trials: {len(result.responses)}")
        
        self.results['stroop'] = all_results
        return all_results
    
    async def demo_reaction_time_experiment(self):
        """Demonstrate simple reaction time experiment"""
        logger.info("⚡ REACTION TIME EXPERIMENT DEMO")
        print("-" * 40)
        
        # Create experiment configuration
        config = ExperimentConfig(
            name="AI_ReactionTime_Experiment",
            experiment_type=ExperimentType.REACTION_TIME,
            description="Simple reaction time task with AI agents",
            num_trials=40,
            duration=200.0,
            randomize_trials=True
        )
        
        # Create and run experiment
        experiment = ReactionTimeExperiment(config)
        
        print(f"📋 Experiment: {config.name}")
        print(f"🎯 Type: {config.experiment_type.value}")
        print(f"📊 Trials: {config.num_trials}")
        
        # Run with participant agents
        all_results = []
        for i, participant in enumerate(self.agents['participants']):
            print(f"\n🤖 Participant {i+1}: {participant.response_strategy} strategy")
            
            # Reset agent state
            participant.reset_state()
            
            # Participate in experiment
            result = participant.participate_in_experiment(experiment)
            result.participant_id = f"rt_agent_{i+1}_{participant.response_strategy}"
            
            all_results.append(result)
            
            # Show results
            accuracy = result.performance_metrics.get("accuracy", 0)
            mean_rt = result.performance_metrics.get("mean_reaction_time", 0)
            print(f"   📈 Accuracy: {accuracy:.1%}")
            print(f"   ⚡ Mean RT: {mean_rt:.3f}s")
            print(f"   🎯 Trials: {len(result.responses)}")
        
        self.results['reaction_time'] = all_results
        return all_results
    
    async def demo_task_integration(self):
        """Demonstrate task-based experiment integration"""
        logger.info("🔧 TASK INTEGRATION DEMO")
        print("-" * 40)
        
        # Create comprehensive integration task
        integration_task = PsychoPyIntegrationTask(
            research_question="How do different AI cognitive profiles affect performance in attention tasks?",
            experiment_type=ExperimentType.ATTENTION,
            participant_agents=self.agents['participants']
        )
        
        print(f"📋 Research Question: {integration_task.research_question}")
        print(f"🎯 Experiment Type: {integration_task.experiment_type.value}")
        print(f"🤖 Participants: {len(integration_task.participant_agents)}")
        
        # Execute the complete pipeline
        print("\n🚀 Executing experimental pipeline...")
        result = await integration_task.execute()
        
        if result["success"]:
            output = result["output"]
            print("✅ Pipeline completed successfully!")
            print(f"⏱️  Total execution time: {result['execution_time']:.2f}s")
            
            # Show summary
            summary = output["pipeline_summary"]
            print(f"\n📊 PIPELINE SUMMARY:")
            print(f"   👥 Participants: {summary['total_participants']}")
            print(f"   📝 Total Trials: {summary['total_trials']}")
            print(f"   📈 Overall Accuracy: {summary['overall_accuracy']:.1%}")
            print(f"   ⏱️  Mean RT: {summary['mean_reaction_time']:.3f}s")
            
            # Show insights from analysis
            insights = output["analysis"]["insights"]
            if insights:
                print(f"\n💡 KEY INSIGHTS:")
                for insight in insights:
                    print(f"   • {insight}")
            
            self.results['integration'] = output
            return output
        else:
            print(f"❌ Pipeline failed: {result.get('error')}")
            return None
    
    async def demo_crew_based_experiment(self):
        """Demonstrate crew-based experimental workflow"""
        logger.info("👥 CREW-BASED EXPERIMENT DEMO")
        print("-" * 40)
        
        # Create experimental crew
        crew_agents = [
            self.agents['designer'],
            self.agents['analyst']
        ] + self.agents['participants'][:2]  # Include 2 participants
        
        # Create tasks for the crew
        design_task = ExperimentDesignTask(
            research_question="What are the cognitive mechanisms underlying attention and interference?",
            experiment_type=ExperimentType.COGNITIVE_TASK
        )
        
        execution_task = ExperimentExecutionTask(
            experiment_config=ExperimentConfig(
                name="Crew_Cognitive_Experiment",
                experiment_type=ExperimentType.COGNITIVE_TASK,
                num_trials=50
            ),
            participant_agents=self.agents['participants'][:2]
        )
        
        analysis_task = ExperimentAnalysisTask(
            experiment_results=[],  # Will be populated after execution
            analysis_type="comprehensive"
        )
        
        # Assign tasks to agents
        design_task.assign_agent(self.agents['designer'])
        analysis_task.assign_agent(self.agents['analyst'])
        
        # Set up dependencies
        execution_task.add_dependency(design_task)
        analysis_task.add_dependency(execution_task)
        
        # Create crew
        from crews.base import BaseCrew, ProcessType
        
        crew = BaseCrew(
            name="Experimental Psychology Crew",
            agents=crew_agents,
            tasks=[design_task, execution_task, analysis_task],
            process_type=ProcessType.SEQUENTIAL,
            verbose=True
        )
        
        print(f"👥 Crew: {crew.name}")
        print(f"🤖 Agents: {len(crew.agents)}")
        print(f"📝 Tasks: {len(crew.tasks)}")
        
        # Execute crew (simplified simulation)
        print("\n🚀 Executing experimental crew...")
        
        # Simulate crew execution
        print("   🎨 Designer creating experiment...")
        design_result = await design_task.execute(self.agents['designer'])
        
        if design_result["success"]:
            print("   ✅ Experiment design completed")
            
            # Update execution task with design
            execution_task.experiment_config = design_result["output"]["config"]
            
            print("   🧪 Executing experiment with participants...")
            execution_result = await execution_task.execute()
            
            if execution_result["success"]:
                print("   ✅ Experiment execution completed")
                
                # Update analysis task with results
                analysis_task.experiment_results = execution_result["output"]["results"]
                
                print("   📊 Analyzing results...")
                analysis_result = await analysis_task.execute(self.agents['analyst'])
                
                if analysis_result["success"]:
                    print("   ✅ Analysis completed")
                    
                    # Show crew results
                    print(f"\n📋 CREW RESULTS:")
                    print(f"   🎯 Experiment: {design_result['output']['config'].name}")
                    print(f"   👥 Participants: {len(execution_result['output']['results'])}")
                    print(f"   📊 Accuracy: {execution_result['output']['summary']['overall_accuracy']:.1%}")
                    print(f"   ⏱️  Mean RT: {execution_result['output']['summary']['mean_reaction_time']:.3f}s")
                    
                    return {
                        "design": design_result,
                        "execution": execution_result,
                        "analysis": analysis_result
                    }
        
        print("❌ Crew execution failed")
        return None
    
    def display_cognitive_profiles(self):
        """Display cognitive profiles of participant agents"""
        print("\n🧠 COGNITIVE PROFILES")
        print("-" * 40)
        
        for i, participant in enumerate(self.agents['participants']):
            profile = participant.cognitive_profile
            print(f"\n🤖 Agent {i+1} ({participant.response_strategy}):")
            print(f"   ⚡ Base RT: {profile['base_reaction_time']:.3f}s")
            print(f"   📊 RT Variability: {profile['reaction_time_variability']:.3f}")
            print(f"   🎯 Accuracy Rate: {profile['accuracy_rate']:.1%}")
            print(f"   🧠 Attention Span: {profile['attention_span']:.0f}s")
            print(f"   😴 Fatigue Rate: {profile['fatigue_rate']:.4f}")
            print(f"   📈 Learning Rate: {profile['learning_rate']:.3f}")
    
    def display_learning_progress(self):
        """Display learning progress of agents"""
        print("\n📈 LEARNING PROGRESS")
        print("-" * 40)
        
        for i, participant in enumerate(self.agents['participants']):
            cognitive_state = participant.get_cognitive_state()
            print(f"\n🤖 Agent {i+1} ({participant.response_strategy}):")
            print(f"   🧪 Trials Completed: {cognitive_state['trials_completed']}")
            print(f"   🔬 Experiments: {cognitive_state['experiments_completed']}")
            print(f"   🎯 Current Accuracy: {cognitive_state['current_accuracy']:.1%}")
            print(f"   ⏱️  Current Mean RT: {cognitive_state['current_mean_rt']:.3f}s")
            print(f"   😴 Fatigue Level: {cognitive_state['fatigue_level']:.3f}")
            
            if cognitive_state['learned_patterns']:
                print(f"   🧠 Learned Patterns:")
                for pattern, strength in cognitive_state['learned_patterns'].items():
                    print(f"      • {pattern}: {strength:.2f}")
    
    async def run_complete_demo(self):
        """Run the complete demonstration"""
        print("🎉 PSYCHOPY AI AGENT BUILDER - PSYCHOPY INTEGRATION DEMO")
        print("=" * 70)
        print("🧠 Psychological experiments meet AI agent intelligence")
        print("🔬 Demonstrating the fusion of experimental psychology and AI")
        print("=" * 70)
        
        # Setup
        self.create_specialized_agents()
        self.display_cognitive_profiles()
        
        # Run experiments
        await self.demo_stroop_experiment()
        await self.demo_reaction_time_experiment()
        
        # Show learning progress
        self.display_learning_progress()
        
        # Advanced demos
        await self.demo_task_integration()
        await self.demo_crew_based_experiment()
        
        print("\n" + "=" * 70)
        print("🎊 PSYCHOPY INTEGRATION DEMO COMPLETED!")
        print("=" * 70)
        
        print("\n🌟 Key Features Demonstrated:")
        print("   ✅ PsychoPy Experiment Integration")
        print("   ✅ AI Agents as Experiment Participants")
        print("   ✅ Cognitive Profile Simulation")
        print("   ✅ Learning and Adaptation")
        print("   ✅ Experimental Design Automation")
        print("   ✅ Statistical Analysis Integration")
        print("   ✅ Crew-based Experimental Workflows")
        
        print("\n🔬 Experimental Psychology Features:")
        print("   🎨 Stroop Color-Word Interference")
        print("   ⚡ Simple Reaction Time Tasks")
        print("   🧠 Cognitive Load Manipulation")
        print("   📊 Performance Metrics Analysis")
        print("   📈 Learning Curve Tracking")
        print("   😴 Fatigue Effect Simulation")
        
        print("\n🤖 AI Agent Capabilities:")
        print("   🎯 Multiple Response Strategies")
        print("   🧠 Realistic Cognitive Profiles")
        print("   📈 Adaptive Learning")
        print("   🔄 Individual Differences")
        print("   👥 Collaborative Experimentation")
        
        print("\n💡 Applications:")
        print("   🔬 Cognitive Psychology Research")
        print("   🧪 Experimental Design Validation")
        print("   🤖 AI Behavior Testing")
        print("   📊 Statistical Power Analysis")
        print("   🎓 Educational Simulations")
        print("   🏥 Clinical Assessment Tools")
        
        return True


async def main():
    """Main function"""
    demo = PsychoPyIntegrationDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())
