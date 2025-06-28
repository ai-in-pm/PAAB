#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder - PsychoPy Integration Tasks
Tasks that integrate psychological experiments with AI agents
"""

import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

from .base import BaseTask, TaskType, TaskPriority, TaskStatus
from ..psychopy_integration.core import (
    ExperimentType, ExperimentConfig, create_experiment,
    StroopExperiment, ReactionTimeExperiment
)
from ..agents.psychopy_agents import (
    ExperimentParticipantAgent, ExperimentDesignerAgent, ExperimentAnalystAgent
)

logger = logging.getLogger(__name__)


class ExperimentDesignTask(BaseTask):
    """Task for designing psychological experiments"""
    
    def __init__(
        self,
        research_question: str,
        experiment_type: ExperimentType,
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize experiment design task
        
        Args:
            research_question: The research question to address
            experiment_type: Type of experiment to design
            constraints: Design constraints (time, resources, etc.)
        """
        description = f"Design a {experiment_type.value} experiment to address: {research_question}"
        expected_output = "Complete experimental design with methodology, trial structure, and analysis plan"
        
        super().__init__(
            description=description,
            expected_output=expected_output,
            task_type=TaskType.DESIGN,
            priority=TaskPriority.HIGH,
            **kwargs
        )
        
        self.research_question = research_question
        self.experiment_type = experiment_type
        self.constraints = constraints or {}
        self.design_output = None
    
    async def execute(self, agent=None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute experiment design task"""
        
        if not isinstance(agent, ExperimentDesignerAgent):
            logger.warning("Task should be assigned to ExperimentDesignerAgent")
        
        try:
            self.status = TaskStatus.RUNNING
            start_time = time.time()
            
            logger.info(f"Designing experiment: {self.research_question}")
            
            # Design the experiment
            if agent:
                design = agent.design_experiment(
                    experiment_type=self.experiment_type,
                    research_question=self.research_question,
                    constraints=self.constraints
                )
            else:
                # Fallback design
                design = self._create_default_design()
            
            # Create experiment configuration
            config = self._create_experiment_config(design)
            
            # Store results
            self.design_output = {
                "design": design,
                "config": config,
                "research_question": self.research_question,
                "experiment_type": self.experiment_type.value
            }
            
            execution_time = time.time() - start_time
            
            self.status = TaskStatus.COMPLETED
            
            return {
                "success": True,
                "output": self.design_output,
                "execution_time": execution_time,
                "message": f"Experiment design completed for {self.experiment_type.value}"
            }
            
        except Exception as e:
            self.status = TaskStatus.FAILED
            logger.error(f"Experiment design task failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    def _create_default_design(self) -> Dict[str, Any]:
        """Create a default experimental design"""
        return {
            "experiment_type": self.experiment_type,
            "research_question": self.research_question,
            "hypothesis": f"There will be significant effects in the {self.experiment_type.value} task",
            "independent_variables": ["condition"],
            "dependent_variables": ["accuracy", "reaction_time"],
            "design_type": "within_subjects",
            "sample_size": 30,
            "trial_structure": {
                "num_trials": 80,
                "trial_duration": 2.5,
                "iti": 1.0,
                "blocks": 4
            }
        }
    
    def _create_experiment_config(self, design: Dict[str, Any]) -> ExperimentConfig:
        """Create PsychoPy experiment configuration from design"""
        
        trial_structure = design.get("trial_structure", {})
        
        return ExperimentConfig(
            name=f"{self.experiment_type.value}_experiment",
            experiment_type=self.experiment_type,
            description=self.research_question,
            num_trials=trial_structure.get("num_trials", 80),
            duration=trial_structure.get("num_trials", 80) * trial_structure.get("trial_duration", 2.5),
            randomize_trials=True,
            data_filename=f"data/{self.experiment_type.value}_{int(time.time())}"
        )


class ExperimentExecutionTask(BaseTask):
    """Task for executing psychological experiments with AI agents"""
    
    def __init__(
        self,
        experiment_config: ExperimentConfig,
        participant_agents: List[ExperimentParticipantAgent],
        num_sessions: int = 1,
        **kwargs
    ):
        """
        Initialize experiment execution task
        
        Args:
            experiment_config: Configuration for the experiment
            participant_agents: AI agents to participate in the experiment
            num_sessions: Number of experimental sessions to run
        """
        description = f"Execute {experiment_config.name} with {len(participant_agents)} AI participants"
        expected_output = "Experimental data and results from all participants"
        
        super().__init__(
            description=description,
            expected_output=expected_output,
            task_type=TaskType.EXECUTION,
            priority=TaskPriority.HIGH,
            **kwargs
        )
        
        self.experiment_config = experiment_config
        self.participant_agents = participant_agents
        self.num_sessions = num_sessions
        self.results = []
    
    async def execute(self, agent=None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the experiment with AI participants"""
        
        try:
            self.status = TaskStatus.RUNNING
            start_time = time.time()
            
            logger.info(f"Executing experiment: {self.experiment_config.name}")
            
            all_results = []
            
            # Run sessions
            for session in range(self.num_sessions):
                logger.info(f"Running session {session + 1}/{self.num_sessions}")
                
                session_results = []
                
                # Run experiment with each participant agent
                for i, participant in enumerate(self.participant_agents):
                    logger.info(f"Participant {i + 1}: {participant.role}")
                    
                    # Create experiment instance
                    experiment = create_experiment(
                        self.experiment_config.experiment_type,
                        self.experiment_config
                    )
                    
                    # Reset agent state
                    participant.reset_state()
                    
                    # Run experiment
                    result = participant.participate_in_experiment(experiment)
                    result.session_id = f"session_{session + 1}"
                    result.participant_id = f"agent_{i + 1}_{participant.role}"
                    
                    session_results.append(result)
                
                all_results.extend(session_results)
            
            self.results = all_results
            execution_time = time.time() - start_time
            
            # Calculate summary statistics
            summary = self._calculate_summary_statistics(all_results)
            
            self.status = TaskStatus.COMPLETED
            
            return {
                "success": True,
                "output": {
                    "results": all_results,
                    "summary": summary,
                    "num_participants": len(self.participant_agents),
                    "num_sessions": self.num_sessions,
                    "total_trials": sum(len(r.responses) for r in all_results)
                },
                "execution_time": execution_time,
                "message": f"Experiment completed with {len(all_results)} participants"
            }
            
        except Exception as e:
            self.status = TaskStatus.FAILED
            logger.error(f"Experiment execution task failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    def _calculate_summary_statistics(self, results) -> Dict[str, Any]:
        """Calculate summary statistics across all participants"""
        
        if not results:
            return {}
        
        # Aggregate all responses
        all_responses = []
        for result in results:
            all_responses.extend(result.responses)
        
        if not all_responses:
            return {}
        
        # Calculate statistics
        accuracies = [r.accuracy for r in all_responses if r.accuracy is not None]
        rts = [r.reaction_time for r in all_responses if r.reaction_time is not None]
        
        summary = {
            "total_responses": len(all_responses),
            "overall_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0,
            "mean_reaction_time": sum(rts) / len(rts) if rts else 0,
            "participant_count": len(results),
            "completion_rate": len([r for r in results if r.responses]) / len(results)
        }
        
        # Per-participant statistics
        participant_stats = []
        for result in results:
            if result.responses:
                participant_accuracies = [r.accuracy for r in result.responses if r.accuracy is not None]
                participant_rts = [r.reaction_time for r in result.responses if r.reaction_time is not None]
                
                participant_stats.append({
                    "participant_id": result.participant_id,
                    "accuracy": sum(participant_accuracies) / len(participant_accuracies) if participant_accuracies else 0,
                    "mean_rt": sum(participant_rts) / len(participant_rts) if participant_rts else 0,
                    "trial_count": len(result.responses)
                })
        
        summary["participant_statistics"] = participant_stats
        
        return summary


class ExperimentAnalysisTask(BaseTask):
    """Task for analyzing experimental results"""
    
    def __init__(
        self,
        experiment_results: List,
        analysis_type: str = "comprehensive",
        **kwargs
    ):
        """
        Initialize experiment analysis task
        
        Args:
            experiment_results: Results from experiment execution
            analysis_type: Type of analysis to perform
        """
        description = f"Analyze experimental results from {len(experiment_results)} participants"
        expected_output = "Statistical analysis report with insights and recommendations"
        
        super().__init__(
            description=description,
            expected_output=expected_output,
            task_type=TaskType.ANALYSIS,
            priority=TaskPriority.MEDIUM,
            **kwargs
        )
        
        self.experiment_results = experiment_results
        self.analysis_type = analysis_type
        self.analysis_output = None
    
    async def execute(self, agent=None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute experiment analysis"""
        
        if not isinstance(agent, ExperimentAnalystAgent):
            logger.warning("Task should be assigned to ExperimentAnalystAgent")
        
        try:
            self.status = TaskStatus.RUNNING
            start_time = time.time()
            
            logger.info(f"Analyzing results from {len(self.experiment_results)} participants")
            
            # Perform analysis for each participant
            individual_analyses = []
            for result in self.experiment_results:
                if agent:
                    analysis = agent.analyze_results(result)
                else:
                    analysis = self._basic_analysis(result)
                
                individual_analyses.append({
                    "participant_id": result.participant_id,
                    "analysis": analysis
                })
            
            # Aggregate analysis
            aggregate_analysis = self._aggregate_analyses(individual_analyses)
            
            # Generate insights
            insights = self._generate_insights(aggregate_analysis)
            
            self.analysis_output = {
                "individual_analyses": individual_analyses,
                "aggregate_analysis": aggregate_analysis,
                "insights": insights,
                "analysis_type": self.analysis_type
            }
            
            execution_time = time.time() - start_time
            
            self.status = TaskStatus.COMPLETED
            
            return {
                "success": True,
                "output": self.analysis_output,
                "execution_time": execution_time,
                "message": f"Analysis completed for {len(self.experiment_results)} participants"
            }
            
        except Exception as e:
            self.status = TaskStatus.FAILED
            logger.error(f"Experiment analysis task failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    def _basic_analysis(self, result) -> Dict[str, Any]:
        """Basic analysis when no analyst agent is available"""
        
        if not result.responses:
            return {"error": "No responses to analyze"}
        
        accuracies = [r.accuracy for r in result.responses if r.accuracy is not None]
        rts = [r.reaction_time for r in result.responses if r.reaction_time is not None]
        
        return {
            "basic_statistics": {
                "total_trials": len(result.responses),
                "accuracy_rate": sum(accuracies) / len(accuracies) if accuracies else 0,
                "mean_rt": sum(rts) / len(rts) if rts else 0,
                "error_rate": 1 - (sum(accuracies) / len(accuracies)) if accuracies else 1
            }
        }
    
    def _aggregate_analyses(self, individual_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate individual participant analyses"""
        
        # Extract basic statistics
        all_accuracies = []
        all_rts = []
        
        for analysis in individual_analyses:
            basic_stats = analysis["analysis"].get("basic_statistics", {})
            if "accuracy_rate" in basic_stats:
                all_accuracies.append(basic_stats["accuracy_rate"])
            if "mean_rt" in basic_stats:
                all_rts.append(basic_stats["mean_rt"])
        
        aggregate = {
            "group_statistics": {
                "n_participants": len(individual_analyses),
                "mean_accuracy": sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0,
                "mean_rt": sum(all_rts) / len(all_rts) if all_rts else 0,
                "accuracy_range": [min(all_accuracies), max(all_accuracies)] if all_accuracies else [0, 0],
                "rt_range": [min(all_rts), max(all_rts)] if all_rts else [0, 0]
            }
        }
        
        return aggregate
    
    def _generate_insights(self, aggregate_analysis: Dict[str, Any]) -> List[str]:
        """Generate insights from aggregate analysis"""
        
        insights = []
        
        group_stats = aggregate_analysis.get("group_statistics", {})
        mean_accuracy = group_stats.get("mean_accuracy", 0)
        mean_rt = group_stats.get("mean_rt", 0)
        
        # Accuracy insights
        if mean_accuracy > 0.9:
            insights.append("High accuracy suggests the task was well within participants' capabilities")
        elif mean_accuracy < 0.7:
            insights.append("Lower accuracy indicates the task was challenging or instructions unclear")
        
        # Reaction time insights
        if mean_rt > 1.5:
            insights.append("Longer reaction times suggest high cognitive load or decision complexity")
        elif mean_rt < 0.4:
            insights.append("Very fast responses may indicate automatic processing or guessing")
        
        # Variability insights
        accuracy_range = group_stats.get("accuracy_range", [0, 0])
        if accuracy_range[1] - accuracy_range[0] > 0.3:
            insights.append("High variability in accuracy suggests individual differences in strategy or ability")
        
        return insights


class PsychoPyIntegrationTask(BaseTask):
    """Comprehensive task that integrates design, execution, and analysis"""
    
    def __init__(
        self,
        research_question: str,
        experiment_type: ExperimentType,
        participant_agents: List[ExperimentParticipantAgent],
        **kwargs
    ):
        """
        Initialize comprehensive PsychoPy integration task
        
        Args:
            research_question: Research question to address
            experiment_type: Type of experiment
            participant_agents: AI agents to participate
        """
        description = f"Complete experimental pipeline: design, execute, and analyze {experiment_type.value} experiment"
        expected_output = "Full experimental report with design, data, and analysis"
        
        super().__init__(
            description=description,
            expected_output=expected_output,
            task_type=TaskType.RESEARCH,
            priority=TaskPriority.HIGH,
            **kwargs
        )
        
        self.research_question = research_question
        self.experiment_type = experiment_type
        self.participant_agents = participant_agents
        self.pipeline_output = None
    
    async def execute(self, agent=None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute the complete experimental pipeline"""
        
        try:
            self.status = TaskStatus.RUNNING
            start_time = time.time()
            
            logger.info(f"Starting experimental pipeline: {self.research_question}")
            
            # Phase 1: Design
            design_task = ExperimentDesignTask(
                research_question=self.research_question,
                experiment_type=self.experiment_type
            )
            
            design_result = await design_task.execute()
            if not design_result["success"]:
                raise Exception(f"Design phase failed: {design_result.get('error')}")
            
            experiment_config = design_result["output"]["config"]
            
            # Phase 2: Execution
            execution_task = ExperimentExecutionTask(
                experiment_config=experiment_config,
                participant_agents=self.participant_agents
            )
            
            execution_result = await execution_task.execute()
            if not execution_result["success"]:
                raise Exception(f"Execution phase failed: {execution_result.get('error')}")
            
            experiment_results = execution_result["output"]["results"]
            
            # Phase 3: Analysis
            analysis_task = ExperimentAnalysisTask(
                experiment_results=experiment_results
            )
            
            analysis_result = await analysis_task.execute()
            if not analysis_result["success"]:
                raise Exception(f"Analysis phase failed: {analysis_result.get('error')}")
            
            # Compile final output
            self.pipeline_output = {
                "research_question": self.research_question,
                "experiment_type": self.experiment_type.value,
                "design": design_result["output"],
                "execution": execution_result["output"],
                "analysis": analysis_result["output"],
                "pipeline_summary": {
                    "total_participants": len(self.participant_agents),
                    "total_trials": execution_result["output"]["total_trials"],
                    "overall_accuracy": execution_result["output"]["summary"]["overall_accuracy"],
                    "mean_reaction_time": execution_result["output"]["summary"]["mean_reaction_time"]
                }
            }
            
            execution_time = time.time() - start_time
            
            self.status = TaskStatus.COMPLETED
            
            return {
                "success": True,
                "output": self.pipeline_output,
                "execution_time": execution_time,
                "message": "Complete experimental pipeline executed successfully"
            }
            
        except Exception as e:
            self.status = TaskStatus.FAILED
            logger.error(f"PsychoPy integration task failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
