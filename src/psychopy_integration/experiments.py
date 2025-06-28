#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder - Comprehensive Experiment Library
Complete collection of psychological experiments and paradigms
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging

from .core import (
    PsychoPyExperiment, ExperimentConfig, ExperimentType, 
    Stimulus, Trial, Response, StimulusType, ResponseType
)

logger = logging.getLogger(__name__)


class NBackExperiment(PsychoPyExperiment):
    """N-Back working memory experiment"""
    
    def __init__(self, config: ExperimentConfig, n_level: int = 2):
        """Initialize N-Back experiment"""
        super().__init__(config)
        self.n_level = n_level
        self.stimuli_pool = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        self.target_probability = 0.3
    
    def create_trials(self) -> List[Trial]:
        """Create N-Back trials"""
        trials = []
        stimulus_sequence = []
        
        # Generate stimulus sequence
        for i in range(self.config.num_trials):
            if i >= self.n_level and np.random.random() < self.target_probability:
                # Target trial - repeat stimulus from n positions back
                stimulus = stimulus_sequence[i - self.n_level]
                is_target = True
            else:
                # Non-target trial
                available_stimuli = self.stimuli_pool.copy()
                if i >= self.n_level:
                    # Remove n-back stimulus to ensure non-target
                    n_back_stimulus = stimulus_sequence[i - self.n_level]
                    if n_back_stimulus in available_stimuli:
                        available_stimuli.remove(n_back_stimulus)
                
                stimulus = np.random.choice(available_stimuli)
                is_target = False
            
            stimulus_sequence.append(stimulus)
            
            # Create trial
            trial_stimulus = Stimulus(
                id=f"nback_stimulus_{i+1}",
                stimulus_type=StimulusType.TEXT,
                content=stimulus,
                duration=1.5,
                properties={"font_size": 0.15, "color": "white"}
            )
            
            trial = Trial(
                id=f"nback_trial_{i+1}",
                trial_number=i + 1,
                stimuli=[trial_stimulus],
                expected_response="space" if is_target else None,
                trial_type="target" if is_target else "non_target",
                conditions={
                    "n_level": self.n_level,
                    "stimulus": stimulus,
                    "is_target": is_target,
                    "position": i + 1
                }
            )
            
            trials.append(trial)
        
        return trials
    
    def present_stimulus(self, stimulus: Stimulus) -> None:
        """Present N-Back stimulus"""
        logger.debug(f"Presenting N-Back stimulus: {stimulus.content}")
    
    def collect_response(self, trial: Trial) -> Response:
        """Collect N-Back response"""
        # Simulate response collection
        start_time = time.time()
        
        # Simulate response based on trial type
        is_target = trial.conditions["is_target"]
        
        # Simulate hit rate and false alarm rate
        if is_target:
            # Hit rate ~80%
            response_given = np.random.random() < 0.8
        else:
            # False alarm rate ~20%
            response_given = np.random.random() < 0.2
        
        rt = np.random.normal(0.6, 0.15) if response_given else 2.0  # Timeout
        rt = max(0.2, rt)
        
        response_value = "space" if response_given else None
        accuracy = (response_given and is_target) or (not response_given and not is_target)
        
        return Response(
            trial_id=trial.id,
            response_type=ResponseType.KEYBOARD,
            value=response_value,
            reaction_time=rt,
            accuracy=accuracy
        )


class FlankersExperiment(PsychoPyExperiment):
    """Eriksen Flankers attention experiment"""
    
    def __init__(self, config: ExperimentConfig):
        """Initialize Flankers experiment"""
        super().__init__(config)
        self.target_arrows = ['<', '>']
        self.flanker_types = ['congruent', 'incongruent', 'neutral']
    
    def create_trials(self) -> List[Trial]:
        """Create Flankers trials"""
        trials = []
        
        for i in range(self.config.num_trials):
            target = np.random.choice(self.target_arrows)
            flanker_type = np.random.choice(self.flanker_types)
            
            if flanker_type == 'congruent':
                flankers = target
                stimulus_text = f"{flankers}{flankers}{target}{flankers}{flankers}"
            elif flanker_type == 'incongruent':
                flankers = '<' if target == '>' else '>'
                stimulus_text = f"{flankers}{flankers}{target}{flankers}{flankers}"
            else:  # neutral
                flankers = '-'
                stimulus_text = f"{flankers}{flankers}{target}{flankers}{flankers}"
            
            # Create stimulus
            stimulus = Stimulus(
                id=f"flankers_stimulus_{i+1}",
                stimulus_type=StimulusType.TEXT,
                content=stimulus_text,
                duration=2.0,
                properties={
                    "font_size": 0.1,
                    "color": "white",
                    "font": "Courier"
                }
            )
            
            # Create trial
            trial = Trial(
                id=f"flankers_trial_{i+1}",
                trial_number=i + 1,
                stimuli=[stimulus],
                expected_response="left" if target == '<' else "right",
                trial_type=flanker_type,
                conditions={
                    "target": target,
                    "flanker_type": flanker_type,
                    "stimulus_text": stimulus_text
                }
            )
            
            trials.append(trial)
        
        return trials
    
    def present_stimulus(self, stimulus: Stimulus) -> None:
        """Present Flankers stimulus"""
        logger.debug(f"Presenting Flankers: {stimulus.content}")
    
    def collect_response(self, trial: Trial) -> Response:
        """Collect Flankers response"""
        flanker_type = trial.conditions["flanker_type"]
        expected = trial.expected_response
        
        # Simulate flanker interference effects
        if flanker_type == 'congruent':
            base_rt = 0.45
            accuracy_prob = 0.95
        elif flanker_type == 'incongruent':
            base_rt = 0.55  # Slower due to interference
            accuracy_prob = 0.85  # Less accurate
        else:  # neutral
            base_rt = 0.50
            accuracy_prob = 0.90
        
        rt = np.random.normal(base_rt, 0.1)
        rt = max(0.2, rt)
        
        correct = np.random.random() < accuracy_prob
        response_value = expected if correct else ("left" if expected == "right" else "right")
        
        return Response(
            trial_id=trial.id,
            response_type=ResponseType.KEYBOARD,
            value=response_value,
            reaction_time=rt,
            accuracy=correct
        )


class VisualSearchExperiment(PsychoPyExperiment):
    """Visual search experiment"""
    
    def __init__(self, config: ExperimentConfig):
        """Initialize Visual Search experiment"""
        super().__init__(config)
        self.set_sizes = [4, 8, 12, 16]
        self.target_present_prob = 0.5
        self.target_color = "red"
        self.distractor_color = "blue"
    
    def create_trials(self) -> List[Trial]:
        """Create Visual Search trials"""
        trials = []
        
        for i in range(self.config.num_trials):
            set_size = np.random.choice(self.set_sizes)
            target_present = np.random.random() < self.target_present_prob
            
            # Generate item positions
            positions = self._generate_positions(set_size)
            
            # Create stimulus
            stimulus = Stimulus(
                id=f"visual_search_stimulus_{i+1}",
                stimulus_type=StimulusType.CUSTOM,
                content="visual_search_array",
                duration=5.0,
                properties={
                    "set_size": set_size,
                    "target_present": target_present,
                    "positions": positions,
                    "target_color": self.target_color,
                    "distractor_color": self.distractor_color
                }
            )
            
            # Create trial
            trial = Trial(
                id=f"visual_search_trial_{i+1}",
                trial_number=i + 1,
                stimuli=[stimulus],
                expected_response="present" if target_present else "absent",
                trial_type="target_present" if target_present else "target_absent",
                conditions={
                    "set_size": set_size,
                    "target_present": target_present
                }
            )
            
            trials.append(trial)
        
        return trials
    
    def _generate_positions(self, set_size: int) -> List[Tuple[float, float]]:
        """Generate random positions for search items"""
        positions = []
        min_distance = 0.15  # Minimum distance between items
        
        for _ in range(set_size):
            attempts = 0
            while attempts < 100:  # Prevent infinite loop
                x = np.random.uniform(-0.8, 0.8)
                y = np.random.uniform(-0.6, 0.6)
                
                # Check distance from existing positions
                valid = True
                for existing_x, existing_y in positions:
                    distance = np.sqrt((x - existing_x)**2 + (y - existing_y)**2)
                    if distance < min_distance:
                        valid = False
                        break
                
                if valid:
                    positions.append((x, y))
                    break
                
                attempts += 1
        
        return positions
    
    def present_stimulus(self, stimulus: Stimulus) -> None:
        """Present Visual Search array"""
        logger.debug(f"Presenting visual search array: set size {stimulus.properties['set_size']}")
    
    def collect_response(self, trial: Trial) -> Response:
        """Collect Visual Search response"""
        set_size = trial.conditions["set_size"]
        target_present = trial.conditions["target_present"]
        
        # Simulate search time based on set size and target presence
        if target_present:
            # Target present: parallel search for color
            base_rt = 0.5 + set_size * 0.01  # Shallow slope
        else:
            # Target absent: serial search
            base_rt = 0.5 + set_size * 0.03  # Steeper slope
        
        rt = np.random.normal(base_rt, 0.15)
        rt = max(0.3, rt)
        
        # Accuracy decreases with set size
        accuracy_prob = 0.95 - (set_size - 4) * 0.02
        correct = np.random.random() < accuracy_prob
        
        expected = trial.expected_response
        response_value = expected if correct else ("present" if expected == "absent" else "absent")
        
        return Response(
            trial_id=trial.id,
            response_type=ResponseType.KEYBOARD,
            value=response_value,
            reaction_time=rt,
            accuracy=correct
        )


class ChangeBlindnessExperiment(PsychoPyExperiment):
    """Change blindness experiment"""
    
    def __init__(self, config: ExperimentConfig):
        """Initialize Change Blindness experiment"""
        super().__init__(config)
        self.change_probability = 0.5
        self.change_types = ['color', 'position', 'size', 'orientation']
    
    def create_trials(self) -> List[Trial]:
        """Create Change Blindness trials"""
        trials = []
        
        for i in range(self.config.num_trials):
            change_occurs = np.random.random() < self.change_probability
            change_type = np.random.choice(self.change_types) if change_occurs else None
            
            # Create stimulus
            stimulus = Stimulus(
                id=f"change_blindness_stimulus_{i+1}",
                stimulus_type=StimulusType.CUSTOM,
                content="change_detection_array",
                duration=10.0,
                properties={
                    "change_occurs": change_occurs,
                    "change_type": change_type,
                    "num_objects": 6,
                    "presentation_time": 0.5,
                    "blank_duration": 0.1
                }
            )
            
            # Create trial
            trial = Trial(
                id=f"change_blindness_trial_{i+1}",
                trial_number=i + 1,
                stimuli=[stimulus],
                expected_response="change" if change_occurs else "no_change",
                trial_type="change" if change_occurs else "no_change",
                conditions={
                    "change_occurs": change_occurs,
                    "change_type": change_type
                }
            )
            
            trials.append(trial)
        
        return trials
    
    def present_stimulus(self, stimulus: Stimulus) -> None:
        """Present Change Blindness stimulus"""
        logger.debug(f"Presenting change detection array")
    
    def collect_response(self, trial: Trial) -> Response:
        """Collect Change Blindness response"""
        change_occurs = trial.conditions["change_occurs"]
        
        # Simulate change blindness effects
        if change_occurs:
            # Hit rate varies by change type
            change_type = trial.conditions["change_type"]
            hit_rates = {'color': 0.8, 'position': 0.6, 'size': 0.7, 'orientation': 0.5}
            hit_rate = hit_rates.get(change_type, 0.6)
            detected = np.random.random() < hit_rate
        else:
            # False alarm rate
            detected = np.random.random() < 0.2
        
        rt = np.random.normal(2.5, 1.0)  # Longer RTs for change detection
        rt = max(0.5, rt)
        
        response_value = "change" if detected else "no_change"
        accuracy = (detected and change_occurs) or (not detected and not change_occurs)
        
        return Response(
            trial_id=trial.id,
            response_type=ResponseType.KEYBOARD,
            value=response_value,
            reaction_time=rt,
            accuracy=accuracy
        )


class PrimingExperiment(PsychoPyExperiment):
    """Semantic priming experiment"""
    
    def __init__(self, config: ExperimentConfig):
        """Initialize Priming experiment"""
        super().__init__(config)
        
        # Word pairs for priming
        self.related_pairs = [
            ("doctor", "nurse"), ("cat", "dog"), ("bread", "butter"),
            ("table", "chair"), ("sun", "moon"), ("hot", "cold")
        ]
        self.unrelated_pairs = [
            ("doctor", "tree"), ("cat", "book"), ("bread", "car"),
            ("table", "music"), ("sun", "pencil"), ("hot", "flower")
        ]
    
    def create_trials(self) -> List[Trial]:
        """Create Priming trials"""
        trials = []
        
        # Half related, half unrelated
        num_related = self.config.num_trials // 2
        num_unrelated = self.config.num_trials - num_related
        
        trial_types = ['related'] * num_related + ['unrelated'] * num_unrelated
        np.random.shuffle(trial_types)
        
        for i, trial_type in enumerate(trial_types):
            if trial_type == 'related':
                prime, target = self.related_pairs[i % len(self.related_pairs)]
            else:
                prime, target = self.unrelated_pairs[i % len(self.unrelated_pairs)]
            
            # Create prime stimulus
            prime_stimulus = Stimulus(
                id=f"prime_{i+1}",
                stimulus_type=StimulusType.TEXT,
                content=prime.upper(),
                duration=0.2,
                properties={"font_size": 0.08, "color": "white"}
            )
            
            # Create target stimulus
            target_stimulus = Stimulus(
                id=f"target_{i+1}",
                stimulus_type=StimulusType.TEXT,
                content=target.upper(),
                duration=2.0,
                properties={"font_size": 0.1, "color": "white"}
            )
            
            # Create trial
            trial = Trial(
                id=f"priming_trial_{i+1}",
                trial_number=i + 1,
                stimuli=[prime_stimulus, target_stimulus],
                expected_response="word",  # Lexical decision task
                trial_type=trial_type,
                conditions={
                    "prime": prime,
                    "target": target,
                    "relatedness": trial_type
                }
            )
            
            trials.append(trial)
        
        return trials
    
    def present_stimulus(self, stimulus: Stimulus) -> None:
        """Present Priming stimulus"""
        logger.debug(f"Presenting priming stimulus: {stimulus.content}")
    
    def collect_response(self, trial: Trial) -> Response:
        """Collect Priming response"""
        relatedness = trial.conditions["relatedness"]
        
        # Simulate priming effects
        if relatedness == 'related':
            base_rt = 0.55  # Faster for related pairs
        else:
            base_rt = 0.65  # Slower for unrelated pairs
        
        rt = np.random.normal(base_rt, 0.1)
        rt = max(0.3, rt)
        
        # High accuracy for word recognition
        accuracy = np.random.random() < 0.95
        
        return Response(
            trial_id=trial.id,
            response_type=ResponseType.KEYBOARD,
            value="word",
            reaction_time=rt,
            accuracy=accuracy
        )


# Experiment factory with all paradigms
EXPERIMENT_REGISTRY = {
    'stroop': 'StroopExperiment',
    'reaction_time': 'ReactionTimeExperiment', 
    'nback': 'NBackExperiment',
    'flankers': 'FlankersExperiment',
    'visual_search': 'VisualSearchExperiment',
    'change_blindness': 'ChangeBlindnessExperiment',
    'priming': 'PrimingExperiment'
}


def create_experiment_by_name(experiment_name: str, config: ExperimentConfig) -> PsychoPyExperiment:
    """Create experiment by name"""
    
    experiment_classes = {
        'nback': NBackExperiment,
        'flankers': FlankersExperiment,
        'visual_search': VisualSearchExperiment,
        'change_blindness': ChangeBlindnessExperiment,
        'priming': PrimingExperiment
    }
    
    if experiment_name in experiment_classes:
        return experiment_classes[experiment_name](config)
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")


def get_available_experiments() -> List[str]:
    """Get list of available experiments"""
    return list(EXPERIMENT_REGISTRY.keys())


def get_experiment_description(experiment_name: str) -> str:
    """Get description of experiment"""
    descriptions = {
        'stroop': 'Color-word interference task measuring selective attention',
        'reaction_time': 'Simple reaction time measurement',
        'nback': 'Working memory task with n-back paradigm',
        'flankers': 'Eriksen flankers task measuring attention and interference',
        'visual_search': 'Visual search task with varying set sizes',
        'change_blindness': 'Change detection task measuring visual attention',
        'priming': 'Semantic priming task measuring word processing'
    }
    
    return descriptions.get(experiment_name, "No description available")
