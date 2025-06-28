#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder - PsychoPy Core Integration
Integrates PsychoPy's experimental psychology features with AI agents
"""

import time
import json
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# Try to import PsychoPy components (graceful fallback if not available)
try:
    from psychopy import visual, core, event, data, sound, gui, clock
    from psychopy.hardware import keyboard
    PSYCHOPY_AVAILABLE = True
    logger.info("PsychoPy successfully imported")
except ImportError:
    PSYCHOPY_AVAILABLE = False
    logger.warning("PsychoPy not available - using simulation mode")
    # Create mock classes for development
    class MockPsychoPy:
        class visual:
            class Window: pass
            class TextStim: pass
            class ImageStim: pass
            class Rect: pass
            class Circle: pass
        class core:
            @staticmethod
            def wait(duration): time.sleep(duration)
            @staticmethod
            def getTime(): return time.time()
            @staticmethod
            def quit(): pass
        class event:
            @staticmethod
            def getKeys(): return []
            @staticmethod
            def clearEvents(): pass
        class data:
            class TrialHandler: pass
            class ExperimentHandler: pass
        class sound:
            class Sound: pass
        class gui:
            class Dlg: pass
        class clock:
            class Clock: pass
    
    visual = MockPsychoPy.visual
    core = MockPsychoPy.core
    event = MockPsychoPy.event
    data = MockPsychoPy.data
    sound = MockPsychoPy.sound
    gui = MockPsychoPy.gui
    clock = MockPsychoPy.clock


class ExperimentType(Enum):
    """Types of psychological experiments"""
    COGNITIVE_TASK = "cognitive_task"
    PERCEPTION = "perception"
    ATTENTION = "attention"
    MEMORY = "memory"
    LEARNING = "learning"
    DECISION_MAKING = "decision_making"
    REACTION_TIME = "reaction_time"
    PSYCHOPHYSICS = "psychophysics"
    SOCIAL_COGNITION = "social_cognition"
    CUSTOM = "custom"


class StimulusType(Enum):
    """Types of stimuli"""
    TEXT = "text"
    IMAGE = "image"
    SOUND = "sound"
    VIDEO = "video"
    SHAPE = "shape"
    GRATING = "grating"
    NOISE = "noise"
    CUSTOM = "custom"


class ResponseType(Enum):
    """Types of responses"""
    KEYBOARD = "keyboard"
    MOUSE = "mouse"
    VOICE = "voice"
    EYE_TRACKING = "eye_tracking"
    PHYSIOLOGICAL = "physiological"
    CUSTOM = "custom"


@dataclass
class ExperimentConfig:
    """Configuration for psychological experiments"""
    name: str
    experiment_type: ExperimentType
    description: str = ""
    duration: float = 300.0  # seconds
    num_trials: int = 100
    randomize_trials: bool = True
    collect_demographics: bool = True
    window_size: Tuple[int, int] = (1024, 768)
    fullscreen: bool = False
    background_color: str = "gray"
    text_color: str = "white"
    font_size: int = 24
    response_keys: List[str] = field(default_factory=lambda: ['space', 'escape'])
    data_filename: str = ""
    save_log: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Stimulus:
    """Represents a stimulus in an experiment"""
    id: str
    stimulus_type: StimulusType
    content: Any  # Text, image path, sound file, etc.
    position: Tuple[float, float] = (0, 0)
    size: Optional[Tuple[float, float]] = None
    duration: float = 1.0
    onset_time: float = 0.0
    properties: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trial:
    """Represents a single trial in an experiment"""
    id: str
    trial_number: int
    stimuli: List[Stimulus]
    expected_response: Optional[str] = None
    trial_type: str = "standard"
    iti: float = 1.0  # Inter-trial interval
    timeout: float = 5.0
    feedback: bool = True
    conditions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Response:
    """Represents a participant response"""
    trial_id: str
    response_type: ResponseType
    value: Any  # Key pressed, mouse position, etc.
    reaction_time: float
    accuracy: Optional[bool] = None
    confidence: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResults:
    """Results from an experiment session"""
    experiment_id: str
    participant_id: str
    session_id: str
    start_time: float
    end_time: float
    responses: List[Response]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    demographics: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class PsychoPyExperiment(ABC):
    """Base class for PsychoPy experiments integrated with AI agents"""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.window = None
        self.clock = None
        self.keyboard = None
        self.trials = []
        self.current_trial = 0
        self.responses = []
        self.experiment_handler = None
        self.is_running = False
        
        # AI Agent integration
        self.ai_agents = []
        self.agent_responses = []
        self.adaptive_difficulty = False
        self.real_time_analysis = False
        
        logger.info(f"Initialized experiment: {config.name}")
    
    def setup_window(self) -> None:
        """Set up the PsychoPy window"""
        if PSYCHOPY_AVAILABLE:
            self.window = visual.Window(
                size=self.config.window_size,
                fullscr=self.config.fullscreen,
                color=self.config.background_color,
                units='pix'
            )
            self.clock = core.Clock()
            self.keyboard = keyboard.Keyboard()
        else:
            logger.info("PsychoPy not available - using simulation mode")
            self.window = "MockWindow"
            self.clock = "MockClock"
            self.keyboard = "MockKeyboard"
    
    def setup_experiment_handler(self) -> None:
        """Set up PsychoPy experiment handler for data collection"""
        if PSYCHOPY_AVAILABLE:
            self.experiment_handler = data.ExperimentHandler(
                name=self.config.name,
                version='1.0',
                extraInfo={'participant': 'AI_Agent_Test'},
                runtimeInfo=None,
                originPath=None,
                savePickle=True,
                saveWideText=True,
                dataFileName=self.config.data_filename or f"data/{self.config.name}"
            )
    
    @abstractmethod
    def create_trials(self) -> List[Trial]:
        """Create the trial sequence for the experiment"""
        pass
    
    @abstractmethod
    def present_stimulus(self, stimulus: Stimulus) -> None:
        """Present a stimulus to the participant/agent"""
        pass
    
    @abstractmethod
    def collect_response(self, trial: Trial) -> Response:
        """Collect response from participant/agent"""
        pass
    
    def run_trial(self, trial: Trial) -> Response:
        """Run a single trial"""
        logger.debug(f"Running trial {trial.trial_number}")
        
        # Present stimuli
        for stimulus in trial.stimuli:
            self.present_stimulus(stimulus)
        
        # Collect response
        response = self.collect_response(trial)
        
        # Store response
        self.responses.append(response)
        
        # Add to experiment handler if available
        if self.experiment_handler and PSYCHOPY_AVAILABLE:
            self.experiment_handler.addData('trial_id', trial.id)
            self.experiment_handler.addData('response', response.value)
            self.experiment_handler.addData('rt', response.reaction_time)
            self.experiment_handler.addData('accuracy', response.accuracy)
            self.experiment_handler.nextEntry()
        
        # Inter-trial interval
        if PSYCHOPY_AVAILABLE:
            core.wait(trial.iti)
        else:
            time.sleep(trial.iti)
        
        return response
    
    def run_experiment(self) -> ExperimentResults:
        """Run the complete experiment"""
        logger.info(f"Starting experiment: {self.config.name}")
        
        start_time = time.time()
        self.is_running = True
        
        try:
            # Setup
            self.setup_window()
            self.setup_experiment_handler()
            
            # Create trials
            self.trials = self.create_trials()
            if self.config.randomize_trials:
                np.random.shuffle(self.trials)
            
            # Show instructions
            self.show_instructions()
            
            # Run trials
            for trial in self.trials:
                if not self.is_running:
                    break
                
                response = self.run_trial(trial)
                
                # Real-time AI analysis if enabled
                if self.real_time_analysis and self.ai_agents:
                    self.analyze_response_with_ai(trial, response)
                
                # Adaptive difficulty if enabled
                if self.adaptive_difficulty:
                    self.adjust_difficulty(response)
            
            # Show completion message
            self.show_completion()
            
        except Exception as e:
            logger.error(f"Experiment error: {str(e)}")
            raise
        
        finally:
            end_time = time.time()
            self.is_running = False
            
            # Cleanup
            if self.window and PSYCHOPY_AVAILABLE:
                self.window.close()
            
            if PSYCHOPY_AVAILABLE:
                core.quit()
        
        # Create results
        results = ExperimentResults(
            experiment_id=self.config.name,
            participant_id="AI_Agent",
            session_id=f"session_{int(start_time)}",
            start_time=start_time,
            end_time=end_time,
            responses=self.responses,
            performance_metrics=self.calculate_performance_metrics()
        )
        
        logger.info(f"Experiment completed: {len(self.responses)} responses collected")
        return results
    
    def show_instructions(self) -> None:
        """Show experiment instructions"""
        if PSYCHOPY_AVAILABLE and self.window:
            instructions = visual.TextStim(
                self.window,
                text="Welcome to the experiment!\n\nPress SPACE to begin.",
                color=self.config.text_color,
                height=self.config.font_size
            )
            instructions.draw()
            self.window.flip()
            
            # Wait for space key
            event.waitKeys(keyList=['space'])
        else:
            logger.info("Showing instructions (simulation mode)")
            time.sleep(1)
    
    def show_completion(self) -> None:
        """Show experiment completion message"""
        if PSYCHOPY_AVAILABLE and self.window:
            completion = visual.TextStim(
                self.window,
                text="Experiment completed!\n\nThank you for participating.",
                color=self.config.text_color,
                height=self.config.font_size
            )
            completion.draw()
            self.window.flip()
            core.wait(2)
        else:
            logger.info("Experiment completed (simulation mode)")
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics from responses"""
        if not self.responses:
            return {}
        
        # Basic metrics
        total_responses = len(self.responses)
        correct_responses = sum(1 for r in self.responses if r.accuracy)
        accuracy = correct_responses / total_responses if total_responses > 0 else 0
        
        reaction_times = [r.reaction_time for r in self.responses if r.reaction_time is not None]
        mean_rt = np.mean(reaction_times) if reaction_times else 0
        std_rt = np.std(reaction_times) if reaction_times else 0
        
        return {
            "accuracy": accuracy,
            "mean_reaction_time": mean_rt,
            "std_reaction_time": std_rt,
            "total_trials": total_responses,
            "correct_trials": correct_responses
        }
    
    def add_ai_agent(self, agent) -> None:
        """Add an AI agent to participate in the experiment"""
        self.ai_agents.append(agent)
        logger.info(f"Added AI agent to experiment: {agent.role}")
    
    def analyze_response_with_ai(self, trial: Trial, response: Response) -> None:
        """Analyze response using AI agents"""
        for agent in self.ai_agents:
            try:
                # Create analysis task for the agent
                analysis_data = {
                    "trial": trial,
                    "response": response,
                    "experiment_context": self.config.name
                }
                
                # Store agent analysis
                self.agent_responses.append({
                    "agent_id": agent.id,
                    "trial_id": trial.id,
                    "analysis": analysis_data,
                    "timestamp": time.time()
                })
                
            except Exception as e:
                logger.error(f"AI analysis error: {str(e)}")
    
    def adjust_difficulty(self, response: Response) -> None:
        """Adjust experiment difficulty based on performance"""
        # Simple adaptive algorithm
        if response.accuracy:
            # Increase difficulty slightly
            self.config.timeout *= 0.95
        else:
            # Decrease difficulty slightly
            self.config.timeout *= 1.05
        
        # Keep within reasonable bounds
        self.config.timeout = max(0.5, min(10.0, self.config.timeout))
    
    def stop_experiment(self) -> None:
        """Stop the experiment"""
        self.is_running = False
        logger.info("Experiment stopped by user")


class StroopExperiment(PsychoPyExperiment):
    """Classic Stroop color-word interference experiment"""
    
    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        self.colors = ['red', 'green', 'blue', 'yellow']
        self.color_keys = {'red': 'r', 'green': 'g', 'blue': 'b', 'yellow': 'y'}
    
    def create_trials(self) -> List[Trial]:
        """Create Stroop trials"""
        trials = []
        
        for i in range(self.config.num_trials):
            # Random word and color
            word = np.random.choice(self.colors)
            color = np.random.choice(self.colors)
            
            # Determine trial type
            trial_type = "congruent" if word == color else "incongruent"
            
            # Create stimulus
            stimulus = Stimulus(
                id=f"stroop_stimulus_{i}",
                stimulus_type=StimulusType.TEXT,
                content=word.upper(),
                properties={
                    "color": color,
                    "font_size": self.config.font_size
                }
            )
            
            # Create trial
            trial = Trial(
                id=f"stroop_trial_{i}",
                trial_number=i + 1,
                stimuli=[stimulus],
                expected_response=self.color_keys[color],
                trial_type=trial_type,
                conditions={"word": word, "color": color, "congruency": trial_type}
            )
            
            trials.append(trial)
        
        return trials
    
    def present_stimulus(self, stimulus: Stimulus) -> None:
        """Present Stroop stimulus"""
        if PSYCHOPY_AVAILABLE and self.window:
            text_stim = visual.TextStim(
                self.window,
                text=stimulus.content,
                color=stimulus.properties["color"],
                height=stimulus.properties["font_size"],
                pos=stimulus.position
            )
            text_stim.draw()
            self.window.flip()
        else:
            logger.debug(f"Presenting stimulus: {stimulus.content} in {stimulus.properties['color']}")
    
    def collect_response(self, trial: Trial) -> Response:
        """Collect Stroop response"""
        start_time = time.time()
        
        if PSYCHOPY_AVAILABLE:
            keys = event.waitKeys(
                maxWait=trial.timeout,
                keyList=list(self.color_keys.values()) + ['escape'],
                timeStamped=self.clock
            )
            
            if keys:
                key, rt = keys[0]
                if key == 'escape':
                    self.stop_experiment()
                    return None
                
                # Check accuracy
                accuracy = key == trial.expected_response
                
                return Response(
                    trial_id=trial.id,
                    response_type=ResponseType.KEYBOARD,
                    value=key,
                    reaction_time=rt,
                    accuracy=accuracy
                )
        else:
            # Simulation mode
            time.sleep(0.5)  # Simulate response time
            simulated_key = np.random.choice(list(self.color_keys.values()))
            rt = np.random.normal(0.8, 0.2)  # Simulate realistic RT
            accuracy = simulated_key == trial.expected_response
            
            return Response(
                trial_id=trial.id,
                response_type=ResponseType.KEYBOARD,
                value=simulated_key,
                reaction_time=rt,
                accuracy=accuracy
            )
        
        # No response (timeout)
        return Response(
            trial_id=trial.id,
            response_type=ResponseType.KEYBOARD,
            value=None,
            reaction_time=trial.timeout,
            accuracy=False
        )


class ReactionTimeExperiment(PsychoPyExperiment):
    """Simple reaction time experiment"""
    
    def create_trials(self) -> List[Trial]:
        """Create reaction time trials"""
        trials = []
        
        for i in range(self.config.num_trials):
            # Random delay before stimulus
            delay = np.random.uniform(1.0, 3.0)
            
            # Create stimulus
            stimulus = Stimulus(
                id=f"rt_stimulus_{i}",
                stimulus_type=StimulusType.SHAPE,
                content="circle",
                onset_time=delay,
                properties={"color": "white", "size": 50}
            )
            
            # Create trial
            trial = Trial(
                id=f"rt_trial_{i}",
                trial_number=i + 1,
                stimuli=[stimulus],
                expected_response="space",
                trial_type="simple_rt",
                conditions={"delay": delay}
            )
            
            trials.append(trial)
        
        return trials
    
    def present_stimulus(self, stimulus: Stimulus) -> None:
        """Present reaction time stimulus"""
        # Wait for onset time
        if PSYCHOPY_AVAILABLE:
            core.wait(stimulus.onset_time)
        else:
            time.sleep(stimulus.onset_time)
        
        if PSYCHOPY_AVAILABLE and self.window:
            if stimulus.content == "circle":
                circle = visual.Circle(
                    self.window,
                    radius=stimulus.properties["size"],
                    fillColor=stimulus.properties["color"],
                    pos=stimulus.position
                )
                circle.draw()
                self.window.flip()
        else:
            logger.debug(f"Presenting {stimulus.content} after {stimulus.onset_time}s delay")
    
    def collect_response(self, trial: Trial) -> Response:
        """Collect reaction time response"""
        start_time = time.time()
        
        if PSYCHOPY_AVAILABLE:
            keys = event.waitKeys(
                maxWait=trial.timeout,
                keyList=['space', 'escape'],
                timeStamped=self.clock
            )
            
            if keys:
                key, rt = keys[0]
                if key == 'escape':
                    self.stop_experiment()
                    return None
                
                return Response(
                    trial_id=trial.id,
                    response_type=ResponseType.KEYBOARD,
                    value=key,
                    reaction_time=rt,
                    accuracy=True  # Any response is correct for simple RT
                )
        else:
            # Simulation mode
            rt = np.random.normal(0.3, 0.1)  # Simulate realistic RT
            time.sleep(rt)
            
            return Response(
                trial_id=trial.id,
                response_type=ResponseType.KEYBOARD,
                value="space",
                reaction_time=rt,
                accuracy=True
            )
        
        # No response (timeout)
        return Response(
            trial_id=trial.id,
            response_type=ResponseType.KEYBOARD,
            value=None,
            reaction_time=trial.timeout,
            accuracy=False
        )


# Import additional modules
from .stimuli import StimulusFactory, StimulusManager, TextStimulus, ImageStimulus, GratingStimulus
from .hardware import HardwareManager, KeyboardDevice, MouseDevice, EyeTrackerDevice
from .data_collection import DataCollector, DataAnalyzer, TrialData, SessionData


class AdvancedExperiment(PsychoPyExperiment):
    """Advanced experiment with full PsychoPy integration"""

    def __init__(self, config: ExperimentConfig):
        """Initialize advanced experiment"""
        super().__init__(config)

        # Advanced components
        self.stimulus_manager = StimulusManager(self.window)
        self.hardware_manager = HardwareManager()
        self.data_collector = DataCollector(config.name)

        # Setup hardware
        self.setup_hardware()

        # Experiment state
        self.current_block = 0
        self.blocks = []
        self.practice_mode = False

    def setup_hardware(self) -> None:
        """Setup hardware devices"""
        # Add keyboard
        keyboard_device = KeyboardDevice()
        self.hardware_manager.add_device(keyboard_device)

        # Add mouse
        mouse_device = MouseDevice(win=self.window)
        self.hardware_manager.add_device(mouse_device)

        # Connect all devices
        self.hardware_manager.connect_all_devices()

    def create_stimulus_set(self) -> None:
        """Create comprehensive stimulus set"""
        # Text stimuli
        instruction_text = TextStimulus(
            win=self.window,
            text="Welcome to the experiment",
            height=0.08,
            color="white"
        )
        self.stimulus_manager.add_stimulus("instructions", instruction_text)

        # Fixation cross
        fixation = TextStimulus(
            win=self.window,
            text="+",
            height=0.1,
            color="white"
        )
        self.stimulus_manager.add_stimulus("fixation", fixation)

        # Feedback stimuli
        correct_feedback = TextStimulus(
            win=self.window,
            text="Correct!",
            height=0.06,
            color="green"
        )
        self.stimulus_manager.add_stimulus("correct", correct_feedback)

        incorrect_feedback = TextStimulus(
            win=self.window,
            text="Incorrect",
            height=0.06,
            color="red"
        )
        self.stimulus_manager.add_stimulus("incorrect", incorrect_feedback)

    def run_practice_block(self, num_trials: int = 10) -> None:
        """Run practice block"""
        self.practice_mode = True
        logger.info(f"Starting practice block with {num_trials} trials")

        # Show practice instructions
        practice_instructions = TextStimulus(
            win=self.window,
            text="Practice Block\n\nPress SPACE when ready",
            height=0.06,
            color="yellow"
        )
        practice_instructions.draw()
        if self.window and PSYCHOPY_AVAILABLE:
            self.window.flip()
            event.waitKeys(keyList=['space'])

        # Run practice trials
        practice_trials = self.create_trials()[:num_trials]
        for trial in practice_trials:
            self.run_trial(trial)

        self.practice_mode = False
        logger.info("Practice block completed")

    def run_experiment_blocks(self) -> None:
        """Run experiment in blocks with breaks"""
        trials_per_block = self.config.num_trials // 4  # 4 blocks
        all_trials = self.create_trials()

        for block_num in range(4):
            self.current_block = block_num + 1
            start_idx = block_num * trials_per_block
            end_idx = start_idx + trials_per_block
            block_trials = all_trials[start_idx:end_idx]

            logger.info(f"Starting block {self.current_block}/4")

            # Block instructions
            block_text = f"Block {self.current_block} of 4\n\nPress SPACE to begin"
            block_instructions = TextStimulus(
                win=self.window,
                text=block_text,
                height=0.06,
                color="white"
            )
            block_instructions.draw()
            if self.window and PSYCHOPY_AVAILABLE:
                self.window.flip()
                event.waitKeys(keyList=['space'])

            # Run block trials
            for trial in block_trials:
                self.run_trial(trial)

            # Break between blocks (except last)
            if block_num < 3:
                self.show_break_screen()

    def show_break_screen(self) -> None:
        """Show break screen between blocks"""
        break_text = f"Break Time!\n\nBlock {self.current_block} completed.\n\nPress SPACE when ready to continue"
        break_screen = TextStimulus(
            win=self.window,
            text=break_text,
            height=0.06,
            color="cyan"
        )
        break_screen.draw()
        if self.window and PSYCHOPY_AVAILABLE:
            self.window.flip()
            event.waitKeys(keyList=['space'])

    def show_trial_feedback(self, response: Response) -> None:
        """Show feedback for trial response"""
        if not self.practice_mode:
            return  # Only show feedback in practice

        if response.accuracy:
            feedback_stim = self.stimulus_manager.get_stimulus("correct")
        else:
            feedback_stim = self.stimulus_manager.get_stimulus("incorrect")

        if feedback_stim:
            feedback_stim.draw()
            if self.window and PSYCHOPY_AVAILABLE:
                self.window.flip()
                core.wait(0.5)  # Show feedback for 500ms


class ComprehensiveStroopExperiment(AdvancedExperiment):
    """Comprehensive Stroop experiment with all features"""

    def __init__(self, config: ExperimentConfig):
        """Initialize comprehensive Stroop experiment"""
        super().__init__(config)

        # Stroop-specific setup
        self.colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']
        self.color_keys = {
            'red': 'r', 'green': 'g', 'blue': 'b',
            'yellow': 'y', 'purple': 'p', 'orange': 'o'
        }
        self.congruency_conditions = ['congruent', 'incongruent', 'neutral']

        # Create comprehensive stimulus set
        self.create_stroop_stimuli()

    def create_stroop_stimuli(self) -> None:
        """Create Stroop-specific stimuli"""
        self.create_stimulus_set()  # Base stimuli

        # Create color word stimuli for each condition
        for color in self.colors:
            for word in self.colors + ['XXXX']:  # Include neutral condition
                stimulus_name = f"stroop_{word}_{color}"

                stroop_stim = TextStimulus(
                    win=self.window,
                    text=word.upper(),
                    height=0.12,
                    color=color,
                    bold=True
                )
                self.stimulus_manager.add_stimulus(stimulus_name, stroop_stim)

    def create_trials(self) -> List[Trial]:
        """Create comprehensive Stroop trials"""
        trials = []
        trial_id = 0

        # Calculate trials per condition
        trials_per_condition = self.config.num_trials // len(self.congruency_conditions)

        for condition in self.congruency_conditions:
            for i in range(trials_per_condition):
                trial_id += 1

                if condition == 'congruent':
                    color = np.random.choice(self.colors)
                    word = color
                elif condition == 'incongruent':
                    color = np.random.choice(self.colors)
                    word = np.random.choice([c for c in self.colors if c != color])
                else:  # neutral
                    color = np.random.choice(self.colors)
                    word = 'XXXX'

                # Create stimulus
                stimulus = Stimulus(
                    id=f"stroop_stimulus_{trial_id}",
                    stimulus_type=StimulusType.TEXT,
                    content=word.upper(),
                    properties={
                        "color": color,
                        "word": word,
                        "font_size": 0.12,
                        "bold": True
                    }
                )

                # Create trial
                trial = Trial(
                    id=f"stroop_trial_{trial_id}",
                    trial_number=trial_id,
                    stimuli=[stimulus],
                    expected_response=self.color_keys[color],
                    trial_type=condition,
                    conditions={
                        "word": word,
                        "color": color,
                        "congruency": condition,
                        "block": (trial_id - 1) // (trials_per_condition // 4) + 1
                    }
                )

                trials.append(trial)

        return trials

    def present_stimulus(self, stimulus: Stimulus) -> None:
        """Present Stroop stimulus with timing control"""
        # Show fixation
        fixation = self.stimulus_manager.get_stimulus("fixation")
        if fixation:
            fixation.draw()
            if self.window and PSYCHOPY_AVAILABLE:
                self.window.flip()
                core.wait(0.5)  # 500ms fixation

        # Present Stroop stimulus
        word = stimulus.properties["word"]
        color = stimulus.properties["color"]
        stimulus_name = f"stroop_{word}_{color}"

        stroop_stim = self.stimulus_manager.get_stimulus(stimulus_name)
        if stroop_stim:
            stroop_stim.draw()
            if self.window and PSYCHOPY_AVAILABLE:
                self.window.flip()
        else:
            logger.warning(f"Stimulus not found: {stimulus_name}")


# Factory function for creating experiments
def create_experiment(experiment_type: ExperimentType, config: ExperimentConfig) -> PsychoPyExperiment:
    """Factory function to create experiments"""

    experiment_classes = {
        ExperimentType.ATTENTION: ComprehensiveStroopExperiment,
        ExperimentType.REACTION_TIME: ReactionTimeExperiment,
        ExperimentType.COGNITIVE_TASK: ComprehensiveStroopExperiment,
        ExperimentType.PERCEPTION: AdvancedExperiment,
        ExperimentType.MEMORY: AdvancedExperiment,
        ExperimentType.LEARNING: AdvancedExperiment,
        ExperimentType.DECISION_MAKING: AdvancedExperiment,
        ExperimentType.PSYCHOPHYSICS: AdvancedExperiment,
        ExperimentType.SOCIAL_COGNITION: AdvancedExperiment,
    }

    experiment_class = experiment_classes.get(experiment_type, AdvancedExperiment)
    return experiment_class(config)
