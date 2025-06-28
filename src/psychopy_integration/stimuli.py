#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder - Comprehensive Stimulus System
Full implementation of PsychoPy stimulus types and presentation
"""

import numpy as np
import time
import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# Try to import PsychoPy components
try:
    from psychopy import visual, sound, core, event, data, gui, monitors
    from psychopy.visual import filters
    from psychopy.tools import colorspacetools, mathtools
    PSYCHOPY_AVAILABLE = True
except ImportError:
    PSYCHOPY_AVAILABLE = False
    logger.warning("PsychoPy not available - using simulation mode")


class StimulusCategory(Enum):
    """Categories of stimuli"""
    VISUAL = "visual"
    AUDITORY = "auditory"
    TACTILE = "tactile"
    MULTIMODAL = "multimodal"


class VisualStimulusType(Enum):
    """Types of visual stimuli"""
    TEXT = "text"
    IMAGE = "image"
    SHAPE = "shape"
    GRATING = "grating"
    GABOR = "gabor"
    NOISE = "noise"
    MOVIE = "movie"
    POLYGON = "polygon"
    CIRCLE = "circle"
    RECT = "rect"
    LINE = "line"
    DOT = "dot"
    APERTURE = "aperture"
    RADIAL = "radial"
    ENVELOPE = "envelope"


class AuditoryStimulusType(Enum):
    """Types of auditory stimuli"""
    TONE = "tone"
    NOISE = "noise"
    SPEECH = "speech"
    MUSIC = "music"
    SOUND_FILE = "sound_file"
    MICROPHONE = "microphone"


@dataclass
class StimulusProperties:
    """Base properties for all stimuli"""
    name: str = ""
    pos: Tuple[float, float] = (0, 0)
    size: Union[float, Tuple[float, float]] = 1.0
    ori: float = 0.0  # orientation in degrees
    opacity: float = 1.0
    contrast: float = 1.0
    depth: float = 0.0
    interpolate: bool = True
    autoLog: bool = True
    autoDraw: bool = False
    units: str = "norm"  # 'norm', 'pix', 'deg', 'cm', 'height'


@dataclass
class VisualProperties(StimulusProperties):
    """Properties specific to visual stimuli"""
    color: Union[str, Tuple[float, float, float]] = "white"
    colorSpace: str = "rgb"
    fillColor: Union[str, Tuple[float, float, float]] = None
    lineColor: Union[str, Tuple[float, float, float]] = None
    lineWidth: float = 1.0
    vertices: Optional[List[Tuple[float, float]]] = None
    closeShape: bool = True


@dataclass
class TextProperties(VisualProperties):
    """Properties for text stimuli"""
    text: str = ""
    font: str = "Arial"
    height: float = 0.1
    bold: bool = False
    italic: bool = False
    alignHoriz: str = "center"  # 'left', 'center', 'right'
    alignVert: str = "center"  # 'top', 'center', 'bottom'
    wrapWidth: Optional[float] = None
    flipHoriz: bool = False
    flipVert: bool = False
    languageStyle: str = "LTR"  # 'LTR', 'RTL', 'Arabic'


@dataclass
class ImageProperties(VisualProperties):
    """Properties for image stimuli"""
    image: str = ""  # path to image file
    mask: Optional[str] = None
    texRes: int = 128
    flipHoriz: bool = False
    flipVert: bool = False


@dataclass
class GratingProperties(VisualProperties):
    """Properties for grating stimuli"""
    tex: str = "sin"  # 'sin', 'sqr', 'saw', 'tri', or numpy array
    mask: str = "none"  # 'none', 'gauss', 'circle', 'cross', 'raisedCos'
    sf: float = 1.0  # spatial frequency
    phase: float = 0.0
    texRes: int = 256
    blendmode: str = "avg"


@dataclass
class NoiseProperties(VisualProperties):
    """Properties for noise stimuli"""
    noiseType: str = "white"  # 'white', 'uniform', 'normal', 'binary'
    noiseFractalPower: float = 0.0
    noiseFilterLower: float = 1.0
    noiseFilterUpper: float = 8.0
    noiseFilterOrder: float = 0.0
    noiseClip: float = 3.0
    imageComponent: str = "Phase"  # 'Phase', 'Amp'


@dataclass
class AudioProperties:
    """Properties for audio stimuli"""
    value: Union[str, float, np.ndarray] = 440.0  # frequency or file path
    secs: float = 1.0
    octave: int = 4
    sampleRate: int = 44100
    bits: int = 16
    name: str = ""
    autoLog: bool = True
    loops: int = 0
    stereo: bool = True
    volume: float = 1.0
    startTime: float = 0.0
    stopTime: Optional[float] = None


class BaseStimulus(ABC):
    """Base class for all stimuli"""
    
    def __init__(self, win=None, properties: StimulusProperties = None):
        """
        Initialize base stimulus
        
        Args:
            win: PsychoPy window object
            properties: Stimulus properties
        """
        self.win = win
        self.properties = properties or StimulusProperties()
        self.stimulus_obj = None
        self.is_drawn = False
        self.creation_time = time.time()
        self.draw_times = []
        
    @abstractmethod
    def create(self) -> Any:
        """Create the actual PsychoPy stimulus object"""
        pass
    
    def draw(self) -> None:
        """Draw the stimulus"""
        if self.stimulus_obj and PSYCHOPY_AVAILABLE:
            self.stimulus_obj.draw()
            self.is_drawn = True
            self.draw_times.append(time.time())
        else:
            logger.debug(f"Drawing {self.__class__.__name__} (simulation mode)")
    
    def setAutoDraw(self, value: bool) -> None:
        """Set auto-draw property"""
        if self.stimulus_obj and PSYCHOPY_AVAILABLE:
            self.stimulus_obj.setAutoDraw(value)
        self.properties.autoDraw = value
    
    def setPos(self, pos: Tuple[float, float]) -> None:
        """Set position"""
        if self.stimulus_obj and PSYCHOPY_AVAILABLE:
            self.stimulus_obj.setPos(pos)
        self.properties.pos = pos
    
    def setSize(self, size: Union[float, Tuple[float, float]]) -> None:
        """Set size"""
        if self.stimulus_obj and PSYCHOPY_AVAILABLE:
            self.stimulus_obj.setSize(size)
        self.properties.size = size
    
    def setOri(self, ori: float) -> None:
        """Set orientation"""
        if self.stimulus_obj and PSYCHOPY_AVAILABLE:
            self.stimulus_obj.setOri(ori)
        self.properties.ori = ori
    
    def setOpacity(self, opacity: float) -> None:
        """Set opacity"""
        if self.stimulus_obj and PSYCHOPY_AVAILABLE:
            self.stimulus_obj.setOpacity(opacity)
        self.properties.opacity = opacity


class TextStimulus(BaseStimulus):
    """Text stimulus implementation"""
    
    def __init__(self, win=None, properties: TextProperties = None, **kwargs):
        """Initialize text stimulus"""
        if properties is None:
            properties = TextProperties(**kwargs)
        super().__init__(win, properties)
        self.create()
    
    def create(self) -> Any:
        """Create PsychoPy TextStim object"""
        if PSYCHOPY_AVAILABLE and self.win:
            self.stimulus_obj = visual.TextStim(
                win=self.win,
                text=self.properties.text,
                font=self.properties.font,
                pos=self.properties.pos,
                height=self.properties.height,
                color=self.properties.color,
                colorSpace=self.properties.colorSpace,
                opacity=self.properties.opacity,
                contrast=self.properties.contrast,
                units=self.properties.units,
                ori=self.properties.ori,
                bold=self.properties.bold,
                italic=self.properties.italic,
                alignHoriz=self.properties.alignHoriz,
                alignVert=self.properties.alignVert,
                wrapWidth=self.properties.wrapWidth,
                flipHoriz=self.properties.flipHoriz,
                flipVert=self.properties.flipVert,
                languageStyle=self.properties.languageStyle,
                name=self.properties.name,
                autoLog=self.properties.autoLog,
                autoDraw=self.properties.autoDraw,
                depth=self.properties.depth
            )
        return self.stimulus_obj
    
    def setText(self, text: str) -> None:
        """Set text content"""
        if self.stimulus_obj and PSYCHOPY_AVAILABLE:
            self.stimulus_obj.setText(text)
        self.properties.text = text
    
    def setColor(self, color: Union[str, Tuple[float, float, float]]) -> None:
        """Set text color"""
        if self.stimulus_obj and PSYCHOPY_AVAILABLE:
            self.stimulus_obj.setColor(color)
        self.properties.color = color
    
    def setHeight(self, height: float) -> None:
        """Set text height"""
        if self.stimulus_obj and PSYCHOPY_AVAILABLE:
            self.stimulus_obj.setHeight(height)
        self.properties.height = height


class ImageStimulus(BaseStimulus):
    """Image stimulus implementation"""
    
    def __init__(self, win=None, properties: ImageProperties = None, **kwargs):
        """Initialize image stimulus"""
        if properties is None:
            properties = ImageProperties(**kwargs)
        super().__init__(win, properties)
        self.create()
    
    def create(self) -> Any:
        """Create PsychoPy ImageStim object"""
        if PSYCHOPY_AVAILABLE and self.win:
            self.stimulus_obj = visual.ImageStim(
                win=self.win,
                image=self.properties.image,
                mask=self.properties.mask,
                pos=self.properties.pos,
                size=self.properties.size,
                color=self.properties.color,
                colorSpace=self.properties.colorSpace,
                opacity=self.properties.opacity,
                contrast=self.properties.contrast,
                units=self.properties.units,
                ori=self.properties.ori,
                texRes=self.properties.texRes,
                flipHoriz=self.properties.flipHoriz,
                flipVert=self.properties.flipVert,
                name=self.properties.name,
                autoLog=self.properties.autoLog,
                autoDraw=self.properties.autoDraw,
                depth=self.properties.depth,
                interpolate=self.properties.interpolate
            )
        return self.stimulus_obj
    
    def setImage(self, image: str) -> None:
        """Set image file"""
        if self.stimulus_obj and PSYCHOPY_AVAILABLE:
            self.stimulus_obj.setImage(image)
        self.properties.image = image


class GratingStimulus(BaseStimulus):
    """Grating stimulus implementation"""
    
    def __init__(self, win=None, properties: GratingProperties = None, **kwargs):
        """Initialize grating stimulus"""
        if properties is None:
            properties = GratingProperties(**kwargs)
        super().__init__(win, properties)
        self.create()
    
    def create(self) -> Any:
        """Create PsychoPy GratingStim object"""
        if PSYCHOPY_AVAILABLE and self.win:
            self.stimulus_obj = visual.GratingStim(
                win=self.win,
                tex=self.properties.tex,
                mask=self.properties.mask,
                pos=self.properties.pos,
                size=self.properties.size,
                sf=self.properties.sf,
                ori=self.properties.ori,
                phase=self.properties.phase,
                color=self.properties.color,
                colorSpace=self.properties.colorSpace,
                opacity=self.properties.opacity,
                contrast=self.properties.contrast,
                units=self.properties.units,
                texRes=self.properties.texRes,
                blendmode=self.properties.blendmode,
                name=self.properties.name,
                autoLog=self.properties.autoLog,
                autoDraw=self.properties.autoDraw,
                depth=self.properties.depth,
                interpolate=self.properties.interpolate
            )
        return self.stimulus_obj
    
    def setSF(self, sf: float) -> None:
        """Set spatial frequency"""
        if self.stimulus_obj and PSYCHOPY_AVAILABLE:
            self.stimulus_obj.setSF(sf)
        self.properties.sf = sf
    
    def setPhase(self, phase: float) -> None:
        """Set phase"""
        if self.stimulus_obj and PSYCHOPY_AVAILABLE:
            self.stimulus_obj.setPhase(phase)
        self.properties.phase = phase


class NoiseStimulus(BaseStimulus):
    """Noise stimulus implementation"""
    
    def __init__(self, win=None, properties: NoiseProperties = None, **kwargs):
        """Initialize noise stimulus"""
        if properties is None:
            properties = NoiseProperties(**kwargs)
        super().__init__(win, properties)
        self.create()
    
    def create(self) -> Any:
        """Create PsychoPy NoiseStim object"""
        if PSYCHOPY_AVAILABLE and self.win:
            self.stimulus_obj = visual.NoiseStim(
                win=self.win,
                mask=self.properties.mask,
                pos=self.properties.pos,
                size=self.properties.size,
                color=self.properties.color,
                colorSpace=self.properties.colorSpace,
                opacity=self.properties.opacity,
                contrast=self.properties.contrast,
                units=self.properties.units,
                ori=self.properties.ori,
                noiseType=self.properties.noiseType,
                noiseFractalPower=self.properties.noiseFractalPower,
                noiseFilterLower=self.properties.noiseFilterLower,
                noiseFilterUpper=self.properties.noiseFilterUpper,
                noiseFilterOrder=self.properties.noiseFilterOrder,
                noiseClip=self.properties.noiseClip,
                imageComponent=self.properties.imageComponent,
                name=self.properties.name,
                autoLog=self.properties.autoLog,
                autoDraw=self.properties.autoDraw,
                depth=self.properties.depth,
                interpolate=self.properties.interpolate
            )
        return self.stimulus_obj
    
    def updateNoise(self) -> None:
        """Update noise pattern"""
        if self.stimulus_obj and PSYCHOPY_AVAILABLE:
            self.stimulus_obj.updateNoise()


class ShapeStimulus(BaseStimulus):
    """Shape stimulus implementation"""
    
    def __init__(self, win=None, properties: VisualProperties = None, shape_type: str = "circle", **kwargs):
        """Initialize shape stimulus"""
        if properties is None:
            properties = VisualProperties(**kwargs)
        self.shape_type = shape_type
        super().__init__(win, properties)
        self.create()
    
    def create(self) -> Any:
        """Create PsychoPy shape object"""
        if PSYCHOPY_AVAILABLE and self.win:
            if self.shape_type == "circle":
                self.stimulus_obj = visual.Circle(
                    win=self.win,
                    radius=self.properties.size,
                    pos=self.properties.pos,
                    fillColor=self.properties.fillColor or self.properties.color,
                    lineColor=self.properties.lineColor,
                    lineWidth=self.properties.lineWidth,
                    opacity=self.properties.opacity,
                    contrast=self.properties.contrast,
                    units=self.properties.units,
                    name=self.properties.name,
                    autoLog=self.properties.autoLog,
                    autoDraw=self.properties.autoDraw,
                    depth=self.properties.depth
                )
            elif self.shape_type == "rect":
                self.stimulus_obj = visual.Rect(
                    win=self.win,
                    width=self.properties.size[0] if isinstance(self.properties.size, tuple) else self.properties.size,
                    height=self.properties.size[1] if isinstance(self.properties.size, tuple) else self.properties.size,
                    pos=self.properties.pos,
                    fillColor=self.properties.fillColor or self.properties.color,
                    lineColor=self.properties.lineColor,
                    lineWidth=self.properties.lineWidth,
                    opacity=self.properties.opacity,
                    contrast=self.properties.contrast,
                    units=self.properties.units,
                    ori=self.properties.ori,
                    name=self.properties.name,
                    autoLog=self.properties.autoLog,
                    autoDraw=self.properties.autoDraw,
                    depth=self.properties.depth
                )
            elif self.shape_type == "polygon":
                self.stimulus_obj = visual.Polygon(
                    win=self.win,
                    edges=len(self.properties.vertices) if self.properties.vertices else 6,
                    radius=self.properties.size,
                    pos=self.properties.pos,
                    fillColor=self.properties.fillColor or self.properties.color,
                    lineColor=self.properties.lineColor,
                    lineWidth=self.properties.lineWidth,
                    opacity=self.properties.opacity,
                    contrast=self.properties.contrast,
                    units=self.properties.units,
                    ori=self.properties.ori,
                    closeShape=self.properties.closeShape,
                    name=self.properties.name,
                    autoLog=self.properties.autoLog,
                    autoDraw=self.properties.autoDraw,
                    depth=self.properties.depth
                )
        return self.stimulus_obj


class AudioStimulus:
    """Audio stimulus implementation"""
    
    def __init__(self, properties: AudioProperties = None, **kwargs):
        """Initialize audio stimulus"""
        if properties is None:
            properties = AudioProperties(**kwargs)
        self.properties = properties
        self.stimulus_obj = None
        self.is_playing = False
        self.creation_time = time.time()
        self.play_times = []
        self.create()
    
    def create(self) -> Any:
        """Create PsychoPy Sound object"""
        if PSYCHOPY_AVAILABLE:
            self.stimulus_obj = sound.Sound(
                value=self.properties.value,
                secs=self.properties.secs,
                octave=self.properties.octave,
                sampleRate=self.properties.sampleRate,
                bits=self.properties.bits,
                name=self.properties.name,
                autoLog=self.properties.autoLog,
                loops=self.properties.loops,
                stereo=self.properties.stereo
            )
            if hasattr(self.stimulus_obj, 'setVolume'):
                self.stimulus_obj.setVolume(self.properties.volume)
        return self.stimulus_obj
    
    def play(self, when: float = 0) -> None:
        """Play the sound"""
        if self.stimulus_obj and PSYCHOPY_AVAILABLE:
            self.stimulus_obj.play(when=when)
            self.is_playing = True
            self.play_times.append(time.time())
        else:
            logger.debug(f"Playing audio stimulus (simulation mode)")
    
    def stop(self) -> None:
        """Stop the sound"""
        if self.stimulus_obj and PSYCHOPY_AVAILABLE:
            self.stimulus_obj.stop()
        self.is_playing = False
    
    def setVolume(self, volume: float) -> None:
        """Set volume"""
        if self.stimulus_obj and PSYCHOPY_AVAILABLE and hasattr(self.stimulus_obj, 'setVolume'):
            self.stimulus_obj.setVolume(volume)
        self.properties.volume = volume


class StimulusManager:
    """Manager for stimulus collections and sequences"""

    def __init__(self, win=None):
        """Initialize stimulus manager"""
        self.win = win
        self.stimuli = {}
        self.sequences = {}
        self.current_sequence = None

    def add_stimulus(self, name: str, stimulus: BaseStimulus) -> None:
        """Add stimulus to collection"""
        self.stimuli[name] = stimulus

    def get_stimulus(self, name: str) -> Optional[BaseStimulus]:
        """Get stimulus by name"""
        return self.stimuli.get(name)

    def remove_stimulus(self, name: str) -> None:
        """Remove stimulus from collection"""
        if name in self.stimuli:
            del self.stimuli[name]

    def create_sequence(self, name: str, stimulus_list: List[str]) -> None:
        """Create stimulus sequence"""
        self.sequences[name] = stimulus_list

    def play_sequence(self, name: str, timing: Optional[List[float]] = None) -> None:
        """Play stimulus sequence"""
        if name not in self.sequences:
            return

        sequence = self.sequences[name]
        timing = timing or [1.0] * len(sequence)

        for i, stim_name in enumerate(sequence):
            if stim_name in self.stimuli:
                self.stimuli[stim_name].draw()
                if PSYCHOPY_AVAILABLE and self.win:
                    self.win.flip()
                    if PSYCHOPY_AVAILABLE:
                        core.wait(timing[i])
                    else:
                        time.sleep(timing[i])


class StimulusFactory:
    """Factory for creating stimuli"""

    @staticmethod
    def create_stimulus(stimulus_type: str, win=None, **kwargs) -> BaseStimulus:
        """Create stimulus of specified type"""

        stimulus_classes = {
            "text": TextStimulus,
            "image": ImageStimulus,
            "grating": GratingStimulus,
            "noise": NoiseStimulus,
            "circle": lambda w, **k: ShapeStimulus(w, shape_type="circle", **k),
            "rect": lambda w, **k: ShapeStimulus(w, shape_type="rect", **k),
            "polygon": lambda w, **k: ShapeStimulus(w, shape_type="polygon", **k),
        }

        if stimulus_type in stimulus_classes:
            return stimulus_classes[stimulus_type](win, **kwargs)
        else:
            raise ValueError(f"Unknown stimulus type: {stimulus_type}")

    @staticmethod
    def create_audio_stimulus(**kwargs) -> AudioStimulus:
        """Create audio stimulus"""
        return AudioStimulus(**kwargs)
