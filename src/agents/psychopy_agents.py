#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder - PsychoPy Specialized Agents
AI agents that can participate in psychological experiments
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging

from .base import BaseAgent, AgentState
from ..psychopy_integration.core import (
    ExperimentType, StimulusType, ResponseType, 
    Trial, Response, Stimulus, ExperimentResults
)

logger = logging.getLogger(__name__)


class ExperimentParticipantAgent(BaseAgent):
    """Agent that can participate in psychological experiments"""
    
    def __init__(
        self,
        goal: str = "Participate in psychological experiments and provide responses",
        backstory: str = "You are an AI agent designed to participate in psychological experiments",
        cognitive_profile: Optional[Dict[str, float]] = None,
        response_strategy: str = "optimal",
        learning_enabled: bool = True,
        **kwargs
    ):
        """
        Initialize experiment participant agent
        
        Args:
            goal: Agent's goal
            backstory: Agent's backstory
            cognitive_profile: Cognitive characteristics (reaction time, accuracy, etc.)
            response_strategy: How the agent responds ('optimal', 'human-like', 'random')
            learning_enabled: Whether the agent learns from experiments
        """
        super().__init__(
            role="experiment_participant",
            goal=goal,
            backstory=backstory,
            **kwargs
        )
        
        # Cognitive profile
        self.cognitive_profile = cognitive_profile or {
            "base_reaction_time": 0.5,  # seconds
            "reaction_time_variability": 0.1,
            "accuracy_rate": 0.85,
            "attention_span": 300.0,  # seconds
            "fatigue_rate": 0.001,  # per trial
            "learning_rate": 0.01
        }
        
        self.response_strategy = response_strategy
        self.learning_enabled = learning_enabled
        
        # Experiment state
        self.current_experiment = None
        self.trial_history = []
        self.performance_history = []
        self.fatigue_level = 0.0
        self.learned_patterns = {}
        
        logger.info(f"Initialized experiment participant agent with {response_strategy} strategy")
    
    def participate_in_experiment(self, experiment) -> ExperimentResults:
        """Participate in a psychological experiment"""
        logger.info(f"Agent participating in experiment: {experiment.config.name}")
        
        self.current_experiment = experiment
        self.fatigue_level = 0.0
        
        # Add self to experiment
        experiment.add_ai_agent(self)
        
        # Run experiment
        results = experiment.run_experiment()
        
        # Learn from results if enabled
        if self.learning_enabled:
            self.learn_from_experiment(results)
        
        return results
    
    def respond_to_trial(self, trial: Trial) -> Response:
        """Generate response to an experimental trial"""
        
        # Update fatigue
        self.fatigue_level += self.cognitive_profile["fatigue_rate"]
        
        # Determine response based on strategy
        if self.response_strategy == "optimal":
            response = self._optimal_response(trial)
        elif self.response_strategy == "human-like":
            response = self._human_like_response(trial)
        elif self.response_strategy == "random":
            response = self._random_response(trial)
        else:
            response = self._optimal_response(trial)
        
        # Store trial in history
        self.trial_history.append({
            "trial": trial,
            "response": response,
            "fatigue_level": self.fatigue_level,
            "timestamp": time.time()
        })
        
        return response
    
    def _optimal_response(self, trial: Trial) -> Response:
        """Generate optimal response (perfect performance)"""
        
        # Simulate processing time
        base_rt = self.cognitive_profile["base_reaction_time"]
        rt_variability = self.cognitive_profile["reaction_time_variability"]
        fatigue_effect = self.fatigue_level * 0.1
        
        reaction_time = np.random.normal(
            base_rt + fatigue_effect,
            rt_variability
        )
        reaction_time = max(0.1, reaction_time)  # Minimum RT
        
        # Simulate response time
        time.sleep(min(reaction_time, 0.1))  # Don't actually wait full time
        
        # Perfect accuracy for optimal strategy
        accuracy = True
        response_value = trial.expected_response
        
        return Response(
            trial_id=trial.id,
            response_type=ResponseType.KEYBOARD,
            value=response_value,
            reaction_time=reaction_time,
            accuracy=accuracy,
            confidence=1.0
        )
    
    def _human_like_response(self, trial: Trial) -> Response:
        """Generate human-like response with realistic errors"""
        
        # Calculate reaction time with human-like variability
        base_rt = self.cognitive_profile["base_reaction_time"]
        rt_variability = self.cognitive_profile["reaction_time_variability"]
        fatigue_effect = self.fatigue_level * 0.2
        
        reaction_time = np.random.normal(
            base_rt + fatigue_effect,
            rt_variability
        )
        reaction_time = max(0.15, reaction_time)  # Human minimum RT
        
        # Calculate accuracy with fatigue and learning effects
        base_accuracy = self.cognitive_profile["accuracy_rate"]
        fatigue_penalty = self.fatigue_level * 0.1
        
        # Check for learned patterns
        pattern_bonus = 0.0
        if trial.trial_type in self.learned_patterns:
            pattern_bonus = self.learned_patterns[trial.trial_type] * 0.1
        
        accuracy_prob = base_accuracy - fatigue_penalty + pattern_bonus
        accuracy_prob = max(0.1, min(0.95, accuracy_prob))  # Bounds
        
        accuracy = np.random.random() < accuracy_prob
        
        # Determine response
        if accuracy:
            response_value = trial.expected_response
        else:
            # Generate error response
            response_value = self._generate_error_response(trial)
        
        # Simulate response time
        time.sleep(min(reaction_time, 0.1))
        
        return Response(
            trial_id=trial.id,
            response_type=ResponseType.KEYBOARD,
            value=response_value,
            reaction_time=reaction_time,
            accuracy=accuracy,
            confidence=accuracy_prob
        )
    
    def _random_response(self, trial: Trial) -> Response:
        """Generate random response"""
        
        # Random reaction time
        reaction_time = np.random.uniform(0.2, 2.0)
        
        # Random response from possible options
        if hasattr(trial, 'response_options'):
            response_value = np.random.choice(trial.response_options)
        else:
            # Default random responses
            response_value = np.random.choice(['r', 'g', 'b', 'y', 'space'])
        
        accuracy = response_value == trial.expected_response
        
        time.sleep(min(reaction_time, 0.1))
        
        return Response(
            trial_id=trial.id,
            response_type=ResponseType.KEYBOARD,
            value=response_value,
            reaction_time=reaction_time,
            accuracy=accuracy,
            confidence=0.5
        )
    
    def _generate_error_response(self, trial: Trial) -> str:
        """Generate a realistic error response"""
        
        # For Stroop-like tasks, common errors are related responses
        if trial.trial_type in ["congruent", "incongruent"]:
            # Color confusion errors
            color_confusions = {
                'r': ['g', 'y'],  # red confused with green or yellow
                'g': ['r', 'b'],  # green confused with red or blue
                'b': ['g', 'y'],  # blue confused with green or yellow
                'y': ['r', 'b']   # yellow confused with red or blue
            }
            
            if trial.expected_response in color_confusions:
                return np.random.choice(color_confusions[trial.expected_response])
        
        # Default random error
        possible_responses = ['r', 'g', 'b', 'y', 'space']
        possible_responses.remove(trial.expected_response)
        return np.random.choice(possible_responses)
    
    def learn_from_experiment(self, results: ExperimentResults) -> None:
        """Learn from experiment results"""
        
        if not results.responses:
            return
        
        # Analyze performance by trial type
        trial_types = {}
        for response in results.responses:
            # Find corresponding trial
            trial = next((t["trial"] for t in self.trial_history if t["trial"].id == response.trial_id), None)
            if trial:
                trial_type = trial.trial_type
                if trial_type not in trial_types:
                    trial_types[trial_type] = {"correct": 0, "total": 0}
                
                trial_types[trial_type]["total"] += 1
                if response.accuracy:
                    trial_types[trial_type]["correct"] += 1
        
        # Update learned patterns
        learning_rate = self.cognitive_profile["learning_rate"]
        for trial_type, stats in trial_types.items():
            accuracy = stats["correct"] / stats["total"]
            
            if trial_type not in self.learned_patterns:
                self.learned_patterns[trial_type] = 0.0
            
            # Update pattern strength based on performance
            if accuracy > 0.8:  # Good performance
                self.learned_patterns[trial_type] += learning_rate
            elif accuracy < 0.5:  # Poor performance
                self.learned_patterns[trial_type] -= learning_rate * 0.5
            
            # Keep within bounds
            self.learned_patterns[trial_type] = max(0.0, min(1.0, self.learned_patterns[trial_type]))
        
        # Store performance history
        self.performance_history.append({
            "experiment": results.experiment_id,
            "accuracy": results.performance_metrics.get("accuracy", 0),
            "mean_rt": results.performance_metrics.get("mean_reaction_time", 0),
            "trial_types": trial_types,
            "timestamp": time.time()
        })
        
        logger.info(f"Agent learned from experiment: {len(self.learned_patterns)} patterns")
    
    def get_cognitive_state(self) -> Dict[str, Any]:
        """Get current cognitive state"""
        return {
            "fatigue_level": self.fatigue_level,
            "learned_patterns": self.learned_patterns.copy(),
            "trials_completed": len(self.trial_history),
            "experiments_completed": len(self.performance_history),
            "current_accuracy": self._calculate_recent_accuracy(),
            "current_mean_rt": self._calculate_recent_rt()
        }
    
    def _calculate_recent_accuracy(self, window: int = 20) -> float:
        """Calculate accuracy over recent trials"""
        if not self.trial_history:
            return 0.0
        
        recent_trials = self.trial_history[-window:]
        correct = sum(1 for t in recent_trials if t["response"].accuracy)
        return correct / len(recent_trials)
    
    def _calculate_recent_rt(self, window: int = 20) -> float:
        """Calculate mean reaction time over recent trials"""
        if not self.trial_history:
            return 0.0
        
        recent_trials = self.trial_history[-window:]
        rts = [t["response"].reaction_time for t in recent_trials if t["response"].reaction_time]
        return np.mean(rts) if rts else 0.0
    
    def reset_state(self) -> None:
        """Reset agent state for new experiment"""
        self.fatigue_level = 0.0
        self.trial_history = []
        self.current_experiment = None
        logger.info("Agent state reset for new experiment")


class ExperimentDesignerAgent(BaseAgent):
    """Agent that designs psychological experiments"""
    
    def __init__(
        self,
        goal: str = "Design and optimize psychological experiments",
        backstory: str = "You are an expert experimental psychologist who designs rigorous experiments",
        **kwargs
    ):
        super().__init__(
            role="experiment_designer",
            goal=goal,
            backstory=backstory,
            **kwargs
        )
        
        self.experiment_templates = {}
        self.design_principles = {
            "counterbalancing": True,
            "randomization": True,
            "control_conditions": True,
            "adequate_power": True,
            "ethical_considerations": True
        }
    
    def design_experiment(
        self,
        experiment_type: ExperimentType,
        research_question: str,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Design a psychological experiment"""
        
        logger.info(f"Designing {experiment_type.value} experiment")
        
        # Basic experiment structure
        design = {
            "experiment_type": experiment_type,
            "research_question": research_question,
            "hypothesis": self._generate_hypothesis(research_question),
            "independent_variables": self._identify_ivs(experiment_type),
            "dependent_variables": self._identify_dvs(experiment_type),
            "design_type": self._determine_design_type(experiment_type),
            "sample_size": self._calculate_sample_size(experiment_type),
            "trial_structure": self._design_trial_structure(experiment_type),
            "counterbalancing": self._design_counterbalancing(experiment_type),
            "controls": self._identify_controls(experiment_type)
        }
        
        # Apply constraints if provided
        if constraints:
            design = self._apply_constraints(design, constraints)
        
        return design
    
    def _generate_hypothesis(self, research_question: str) -> str:
        """Generate hypothesis from research question"""
        # Simple hypothesis generation (could be enhanced with NLP)
        return f"Based on the research question '{research_question}', we hypothesize that there will be significant differences in the measured outcomes."
    
    def _identify_ivs(self, experiment_type: ExperimentType) -> List[str]:
        """Identify independent variables for experiment type"""
        iv_mapping = {
            ExperimentType.ATTENTION: ["stimulus_type", "distractor_presence"],
            ExperimentType.MEMORY: ["encoding_condition", "retention_interval"],
            ExperimentType.REACTION_TIME: ["stimulus_modality", "response_complexity"],
            ExperimentType.COGNITIVE_TASK: ["task_difficulty", "cognitive_load"]
        }
        return iv_mapping.get(experiment_type, ["condition"])
    
    def _identify_dvs(self, experiment_type: ExperimentType) -> List[str]:
        """Identify dependent variables for experiment type"""
        dv_mapping = {
            ExperimentType.ATTENTION: ["accuracy", "reaction_time", "error_rate"],
            ExperimentType.MEMORY: ["recall_accuracy", "recognition_accuracy"],
            ExperimentType.REACTION_TIME: ["reaction_time", "accuracy"],
            ExperimentType.COGNITIVE_TASK: ["performance_score", "completion_time"]
        }
        return dv_mapping.get(experiment_type, ["accuracy", "reaction_time"])
    
    def _determine_design_type(self, experiment_type: ExperimentType) -> str:
        """Determine appropriate experimental design"""
        # Simple mapping - could be more sophisticated
        return "within_subjects"  # Most common for cognitive experiments
    
    def _calculate_sample_size(self, experiment_type: ExperimentType) -> int:
        """Calculate appropriate sample size"""
        # Basic power analysis (simplified)
        base_sizes = {
            ExperimentType.ATTENTION: 30,
            ExperimentType.MEMORY: 25,
            ExperimentType.REACTION_TIME: 20,
            ExperimentType.COGNITIVE_TASK: 35
        }
        return base_sizes.get(experiment_type, 30)
    
    def _design_trial_structure(self, experiment_type: ExperimentType) -> Dict[str, Any]:
        """Design trial structure for experiment"""
        structures = {
            ExperimentType.ATTENTION: {
                "num_trials": 100,
                "trial_duration": 3.0,
                "iti": 1.0,
                "blocks": 4
            },
            ExperimentType.REACTION_TIME: {
                "num_trials": 80,
                "trial_duration": 2.0,
                "iti": 1.5,
                "blocks": 2
            }
        }
        return structures.get(experiment_type, {
            "num_trials": 60,
            "trial_duration": 2.5,
            "iti": 1.0,
            "blocks": 3
        })
    
    def _design_counterbalancing(self, experiment_type: ExperimentType) -> Dict[str, Any]:
        """Design counterbalancing scheme"""
        return {
            "method": "latin_square",
            "factors": ["stimulus_order", "response_mapping"],
            "complete_counterbalancing": True
        }
    
    def _identify_controls(self, experiment_type: ExperimentType) -> List[str]:
        """Identify necessary experimental controls"""
        return [
            "practice_trials",
            "attention_checks",
            "fatigue_breaks",
            "randomized_presentation"
        ]
    
    def _apply_constraints(self, design: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Apply constraints to experimental design"""
        
        if "max_duration" in constraints:
            # Adjust trial numbers to fit time constraint
            max_duration = constraints["max_duration"]
            trial_duration = design["trial_structure"]["trial_duration"]
            iti = design["trial_structure"]["iti"]
            max_trials = int(max_duration / (trial_duration + iti))
            design["trial_structure"]["num_trials"] = min(
                design["trial_structure"]["num_trials"],
                max_trials
            )
        
        if "min_trials" in constraints:
            design["trial_structure"]["num_trials"] = max(
                design["trial_structure"]["num_trials"],
                constraints["min_trials"]
            )
        
        return design


class ExperimentAnalystAgent(BaseAgent):
    """Agent that analyzes experimental results"""
    
    def __init__(
        self,
        goal: str = "Analyze psychological experiment data and provide insights",
        backstory: str = "You are an expert in experimental psychology and statistical analysis",
        **kwargs
    ):
        super().__init__(
            role="experiment_analyst",
            goal=goal,
            backstory=backstory,
            **kwargs
        )
    
    def analyze_results(self, results: ExperimentResults) -> Dict[str, Any]:
        """Analyze experimental results"""
        
        logger.info(f"Analyzing results from experiment: {results.experiment_id}")
        
        analysis = {
            "basic_statistics": self._calculate_basic_stats(results),
            "performance_analysis": self._analyze_performance(results),
            "temporal_analysis": self._analyze_temporal_patterns(results),
            "error_analysis": self._analyze_errors(results),
            "recommendations": self._generate_recommendations(results)
        }
        
        return analysis
    
    def _calculate_basic_stats(self, results: ExperimentResults) -> Dict[str, Any]:
        """Calculate basic descriptive statistics"""
        
        if not results.responses:
            return {}
        
        # Accuracy statistics
        accuracies = [r.accuracy for r in results.responses if r.accuracy is not None]
        accuracy_rate = sum(accuracies) / len(accuracies) if accuracies else 0
        
        # Reaction time statistics
        rts = [r.reaction_time for r in results.responses if r.reaction_time is not None]
        
        stats = {
            "total_trials": len(results.responses),
            "accuracy_rate": accuracy_rate,
            "error_rate": 1 - accuracy_rate,
            "mean_rt": np.mean(rts) if rts else 0,
            "median_rt": np.median(rts) if rts else 0,
            "std_rt": np.std(rts) if rts else 0,
            "min_rt": np.min(rts) if rts else 0,
            "max_rt": np.max(rts) if rts else 0
        }
        
        return stats
    
    def _analyze_performance(self, results: ExperimentResults) -> Dict[str, Any]:
        """Analyze performance patterns"""
        
        # Performance over time (learning/fatigue effects)
        trial_blocks = self._create_trial_blocks(results.responses, block_size=20)
        
        block_performance = []
        for i, block in enumerate(trial_blocks):
            block_accuracy = sum(r.accuracy for r in block if r.accuracy is not None) / len(block)
            block_rt = np.mean([r.reaction_time for r in block if r.reaction_time is not None])
            
            block_performance.append({
                "block": i + 1,
                "accuracy": block_accuracy,
                "mean_rt": block_rt,
                "trial_count": len(block)
            })
        
        return {
            "block_performance": block_performance,
            "learning_trend": self._detect_learning_trend(block_performance),
            "fatigue_trend": self._detect_fatigue_trend(block_performance)
        }
    
    def _analyze_temporal_patterns(self, results: ExperimentResults) -> Dict[str, Any]:
        """Analyze temporal patterns in responses"""
        
        rts = [r.reaction_time for r in results.responses if r.reaction_time is not None]
        
        if not rts:
            return {}
        
        # Detect outliers
        q1, q3 = np.percentile(rts, [25, 75])
        iqr = q3 - q1
        outlier_threshold = q3 + 1.5 * iqr
        outliers = [rt for rt in rts if rt > outlier_threshold]
        
        return {
            "rt_distribution": {
                "quartiles": [np.percentile(rts, q) for q in [25, 50, 75]],
                "outliers": len(outliers),
                "outlier_percentage": len(outliers) / len(rts) * 100
            },
            "temporal_consistency": np.std(rts) / np.mean(rts) if np.mean(rts) > 0 else 0
        }
    
    def _analyze_errors(self, results: ExperimentResults) -> Dict[str, Any]:
        """Analyze error patterns"""
        
        errors = [r for r in results.responses if r.accuracy is False]
        
        if not errors:
            return {"error_count": 0}
        
        # Error types (if available in metadata)
        error_types = {}
        for error in errors:
            error_type = error.metadata.get("error_type", "unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "error_count": len(errors),
            "error_rate": len(errors) / len(results.responses),
            "error_types": error_types,
            "error_rt_pattern": np.mean([e.reaction_time for e in errors if e.reaction_time])
        }
    
    def _generate_recommendations(self, results: ExperimentResults) -> List[str]:
        """Generate recommendations based on analysis"""
        
        recommendations = []
        
        # Check accuracy
        accuracy = results.performance_metrics.get("accuracy", 0)
        if accuracy < 0.7:
            recommendations.append("Consider simplifying the task or providing more practice trials")
        elif accuracy > 0.95:
            recommendations.append("Task may be too easy - consider increasing difficulty")
        
        # Check reaction times
        mean_rt = results.performance_metrics.get("mean_reaction_time", 0)
        if mean_rt > 2.0:
            recommendations.append("Long reaction times suggest high cognitive load")
        elif mean_rt < 0.3:
            recommendations.append("Very fast responses may indicate guessing")
        
        # Check trial count
        if len(results.responses) < 50:
            recommendations.append("Consider increasing number of trials for more reliable data")
        
        return recommendations
    
    def _create_trial_blocks(self, responses: List[Response], block_size: int = 20) -> List[List[Response]]:
        """Create blocks of trials for analysis"""
        blocks = []
        for i in range(0, len(responses), block_size):
            blocks.append(responses[i:i + block_size])
        return blocks
    
    def _detect_learning_trend(self, block_performance: List[Dict[str, Any]]) -> str:
        """Detect learning trend across blocks"""
        if len(block_performance) < 2:
            return "insufficient_data"
        
        accuracies = [block["accuracy"] for block in block_performance]
        
        # Simple trend detection
        if accuracies[-1] > accuracies[0] + 0.1:
            return "improving"
        elif accuracies[-1] < accuracies[0] - 0.1:
            return "declining"
        else:
            return "stable"
    
    def _detect_fatigue_trend(self, block_performance: List[Dict[str, Any]]) -> str:
        """Detect fatigue trend across blocks"""
        if len(block_performance) < 2:
            return "insufficient_data"
        
        rts = [block["mean_rt"] for block in block_performance if block["mean_rt"]]
        
        if not rts:
            return "no_data"
        
        # Fatigue typically shows as increasing RT
        if rts[-1] > rts[0] + 0.2:
            return "fatigue_detected"
        else:
            return "no_fatigue"
