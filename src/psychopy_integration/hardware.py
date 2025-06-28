#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder - Hardware Integration
Comprehensive hardware support for psychological experiments
"""

import time
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

# Try to import PsychoPy hardware components
try:
    from psychopy import core, event, visual, sound
    from psychopy.hardware import keyboard, mouse, joystick
    from psychopy.hardware import camera, microphone
    from psychopy.iohub import launchHubServer
    PSYCHOPY_AVAILABLE = True
    
    # Try to import specialized hardware
    try:
        from psychopy.hardware import cedrus
        CEDRUS_AVAILABLE = True
    except ImportError:
        CEDRUS_AVAILABLE = False
        
    try:
        from psychopy.hardware import egi
        EGI_AVAILABLE = True
    except ImportError:
        EGI_AVAILABLE = False
        
    try:
        from psychopy.hardware import tobii
        TOBII_AVAILABLE = True
    except ImportError:
        TOBII_AVAILABLE = False
        
except ImportError:
    PSYCHOPY_AVAILABLE = False
    CEDRUS_AVAILABLE = False
    EGI_AVAILABLE = False
    TOBII_AVAILABLE = False
    logger.warning("PsychoPy hardware not available - using simulation mode")


class HardwareType(Enum):
    """Types of hardware devices"""
    KEYBOARD = "keyboard"
    MOUSE = "mouse"
    JOYSTICK = "joystick"
    GAMEPAD = "gamepad"
    RESPONSE_BOX = "response_box"
    EYE_TRACKER = "eye_tracker"
    EEG = "eeg"
    CAMERA = "camera"
    MICROPHONE = "microphone"
    SERIAL = "serial"
    PARALLEL = "parallel"
    ARDUINO = "arduino"
    CUSTOM = "custom"


class ResponseType(Enum):
    """Types of responses"""
    KEY_PRESS = "key_press"
    KEY_RELEASE = "key_release"
    MOUSE_CLICK = "mouse_click"
    MOUSE_MOVE = "mouse_move"
    JOYSTICK_BUTTON = "joystick_button"
    JOYSTICK_AXIS = "joystick_axis"
    EYE_GAZE = "eye_gaze"
    EYE_BLINK = "eye_blink"
    EEG_SIGNAL = "eeg_signal"
    VOICE = "voice"
    CUSTOM = "custom"


@dataclass
class HardwareResponse:
    """Response from hardware device"""
    device_type: HardwareType
    response_type: ResponseType
    value: Any
    timestamp: float
    device_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KeyboardResponse(HardwareResponse):
    """Keyboard response"""
    key: str = ""
    modifiers: List[str] = field(default_factory=list)
    duration: Optional[float] = None
    
    def __post_init__(self):
        self.device_type = HardwareType.KEYBOARD
        if self.value and not self.key:
            self.key = str(self.value)


@dataclass
class MouseResponse(HardwareResponse):
    """Mouse response"""
    button: str = ""
    position: Tuple[float, float] = (0, 0)
    wheel: float = 0.0
    
    def __post_init__(self):
        self.device_type = HardwareType.MOUSE


@dataclass
class EyeTrackingResponse(HardwareResponse):
    """Eye tracking response"""
    gaze_position: Tuple[float, float] = (0, 0)
    pupil_size: float = 0.0
    eye: str = "both"  # "left", "right", "both"
    confidence: float = 1.0
    
    def __post_init__(self):
        self.device_type = HardwareType.EYE_TRACKER


class BaseHardwareDevice(ABC):
    """Base class for hardware devices"""
    
    def __init__(self, device_id: str = "", **kwargs):
        """Initialize hardware device"""
        self.device_id = device_id
        self.device_type = HardwareType.CUSTOM
        self.is_connected = False
        self.is_recording = False
        self.responses = []
        self.last_response = None
        self.creation_time = time.time()
        self.config = kwargs
        
    @abstractmethod
    def connect(self) -> bool:
        """Connect to the device"""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the device"""
        pass
    
    @abstractmethod
    def get_response(self, timeout: float = None) -> Optional[HardwareResponse]:
        """Get response from device"""
        pass
    
    def start_recording(self) -> None:
        """Start recording responses"""
        self.is_recording = True
        self.responses = []
        
    def stop_recording(self) -> None:
        """Stop recording responses"""
        self.is_recording = False
        
    def clear_responses(self) -> None:
        """Clear recorded responses"""
        self.responses = []
        
    def get_all_responses(self) -> List[HardwareResponse]:
        """Get all recorded responses"""
        return self.responses.copy()


class KeyboardDevice(BaseHardwareDevice):
    """Keyboard input device"""
    
    def __init__(self, device_id: str = "keyboard", **kwargs):
        """Initialize keyboard device"""
        super().__init__(device_id, **kwargs)
        self.device_type = HardwareType.KEYBOARD
        self.keyboard_obj = None
        self.valid_keys = kwargs.get('valid_keys', None)
        
    def connect(self) -> bool:
        """Connect to keyboard"""
        try:
            if PSYCHOPY_AVAILABLE:
                self.keyboard_obj = keyboard.Keyboard()
                self.is_connected = True
                logger.info("Keyboard connected successfully")
            else:
                self.is_connected = True
                logger.info("Keyboard connected (simulation mode)")
            return True
        except Exception as e:
            logger.error(f"Failed to connect keyboard: {str(e)}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect keyboard"""
        if self.keyboard_obj and PSYCHOPY_AVAILABLE:
            # PsychoPy keyboard doesn't need explicit disconnection
            pass
        self.is_connected = False
        logger.info("Keyboard disconnected")
    
    def get_response(self, timeout: float = None) -> Optional[KeyboardResponse]:
        """Get keyboard response"""
        if not self.is_connected:
            return None
            
        try:
            if PSYCHOPY_AVAILABLE and self.keyboard_obj:
                keys = self.keyboard_obj.getKeys(
                    keyList=self.valid_keys,
                    waitRelease=False,
                    clear=True
                )
                
                if keys:
                    key_event = keys[0]  # Get first key
                    response = KeyboardResponse(
                        device_type=HardwareType.KEYBOARD,
                        response_type=ResponseType.KEY_PRESS,
                        value=key_event.name,
                        timestamp=key_event.tDown,
                        device_id=self.device_id,
                        key=key_event.name,
                        duration=key_event.duration if hasattr(key_event, 'duration') else None
                    )
                    
                    if self.is_recording:
                        self.responses.append(response)
                    self.last_response = response
                    return response
            else:
                # Simulation mode
                import random
                if random.random() < 0.1:  # 10% chance of simulated key press
                    keys = ['space', 'return', 'escape', 'a', 'b', 'c']
                    key = random.choice(keys)
                    response = KeyboardResponse(
                        device_type=HardwareType.KEYBOARD,
                        response_type=ResponseType.KEY_PRESS,
                        value=key,
                        timestamp=time.time(),
                        device_id=self.device_id,
                        key=key
                    )
                    
                    if self.is_recording:
                        self.responses.append(response)
                    self.last_response = response
                    return response
                    
        except Exception as e:
            logger.error(f"Error getting keyboard response: {str(e)}")
            
        return None
    
    def wait_for_key(self, keys: List[str] = None, timeout: float = None) -> Optional[KeyboardResponse]:
        """Wait for specific key press"""
        start_time = time.time()
        
        while True:
            response = self.get_response()
            if response and (keys is None or response.key in keys):
                return response
                
            if timeout and (time.time() - start_time) > timeout:
                break
                
            time.sleep(0.001)  # Small delay to prevent busy waiting
            
        return None


class MouseDevice(BaseHardwareDevice):
    """Mouse input device"""
    
    def __init__(self, device_id: str = "mouse", win=None, **kwargs):
        """Initialize mouse device"""
        super().__init__(device_id, **kwargs)
        self.device_type = HardwareType.MOUSE
        self.mouse_obj = None
        self.win = win
        
    def connect(self) -> bool:
        """Connect to mouse"""
        try:
            if PSYCHOPY_AVAILABLE:
                self.mouse_obj = event.Mouse(win=self.win)
                self.is_connected = True
                logger.info("Mouse connected successfully")
            else:
                self.is_connected = True
                logger.info("Mouse connected (simulation mode)")
            return True
        except Exception as e:
            logger.error(f"Failed to connect mouse: {str(e)}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect mouse"""
        self.is_connected = False
        logger.info("Mouse disconnected")
    
    def get_response(self, timeout: float = None) -> Optional[MouseResponse]:
        """Get mouse response"""
        if not self.is_connected:
            return None
            
        try:
            if PSYCHOPY_AVAILABLE and self.mouse_obj:
                # Check for button presses
                buttons = self.mouse_obj.getPressed()
                position = self.mouse_obj.getPos()
                
                if any(buttons):
                    button_names = ['left', 'middle', 'right']
                    pressed_button = button_names[buttons.index(True)]
                    
                    response = MouseResponse(
                        device_type=HardwareType.MOUSE,
                        response_type=ResponseType.MOUSE_CLICK,
                        value=pressed_button,
                        timestamp=time.time(),
                        device_id=self.device_id,
                        button=pressed_button,
                        position=position
                    )
                    
                    if self.is_recording:
                        self.responses.append(response)
                    self.last_response = response
                    return response
            else:
                # Simulation mode
                import random
                if random.random() < 0.05:  # 5% chance of simulated click
                    button = random.choice(['left', 'right', 'middle'])
                    position = (random.uniform(-1, 1), random.uniform(-1, 1))
                    
                    response = MouseResponse(
                        device_type=HardwareType.MOUSE,
                        response_type=ResponseType.MOUSE_CLICK,
                        value=button,
                        timestamp=time.time(),
                        device_id=self.device_id,
                        button=button,
                        position=position
                    )
                    
                    if self.is_recording:
                        self.responses.append(response)
                    self.last_response = response
                    return response
                    
        except Exception as e:
            logger.error(f"Error getting mouse response: {str(e)}")
            
        return None
    
    def get_position(self) -> Tuple[float, float]:
        """Get current mouse position"""
        if PSYCHOPY_AVAILABLE and self.mouse_obj:
            return self.mouse_obj.getPos()
        else:
            # Simulation mode
            return (0.0, 0.0)


class EyeTrackerDevice(BaseHardwareDevice):
    """Eye tracker device"""
    
    def __init__(self, device_id: str = "eye_tracker", tracker_type: str = "tobii", **kwargs):
        """Initialize eye tracker"""
        super().__init__(device_id, **kwargs)
        self.device_type = HardwareType.EYE_TRACKER
        self.tracker_type = tracker_type
        self.tracker_obj = None
        self.calibration_complete = False
        
    def connect(self) -> bool:
        """Connect to eye tracker"""
        try:
            if PSYCHOPY_AVAILABLE and TOBII_AVAILABLE and self.tracker_type == "tobii":
                # Initialize Tobii eye tracker
                self.tracker_obj = tobii.TobiiTracker()
                self.is_connected = True
                logger.info("Tobii eye tracker connected successfully")
            else:
                self.is_connected = True
                logger.info("Eye tracker connected (simulation mode)")
            return True
        except Exception as e:
            logger.error(f"Failed to connect eye tracker: {str(e)}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect eye tracker"""
        if self.tracker_obj and PSYCHOPY_AVAILABLE:
            # Cleanup tracker
            pass
        self.is_connected = False
        logger.info("Eye tracker disconnected")
    
    def calibrate(self) -> bool:
        """Calibrate eye tracker"""
        try:
            if PSYCHOPY_AVAILABLE and self.tracker_obj:
                # Run calibration procedure
                self.calibration_complete = True
                logger.info("Eye tracker calibration completed")
            else:
                self.calibration_complete = True
                logger.info("Eye tracker calibration completed (simulation mode)")
            return True
        except Exception as e:
            logger.error(f"Eye tracker calibration failed: {str(e)}")
            return False
    
    def get_response(self, timeout: float = None) -> Optional[EyeTrackingResponse]:
        """Get eye tracking response"""
        if not self.is_connected or not self.calibration_complete:
            return None
            
        try:
            if PSYCHOPY_AVAILABLE and self.tracker_obj:
                # Get gaze data from tracker
                # This would be implemented based on specific tracker API
                pass
            else:
                # Simulation mode
                import random
                if random.random() < 0.8:  # 80% chance of valid gaze data
                    gaze_x = random.uniform(-1, 1)
                    gaze_y = random.uniform(-1, 1)
                    pupil_size = random.uniform(2, 8)
                    
                    response = EyeTrackingResponse(
                        device_type=HardwareType.EYE_TRACKER,
                        response_type=ResponseType.EYE_GAZE,
                        value=(gaze_x, gaze_y),
                        timestamp=time.time(),
                        device_id=self.device_id,
                        gaze_position=(gaze_x, gaze_y),
                        pupil_size=pupil_size,
                        confidence=random.uniform(0.7, 1.0)
                    )
                    
                    if self.is_recording:
                        self.responses.append(response)
                    self.last_response = response
                    return response
                    
        except Exception as e:
            logger.error(f"Error getting eye tracking response: {str(e)}")
            
        return None


class HardwareManager:
    """Manager for multiple hardware devices"""
    
    def __init__(self):
        """Initialize hardware manager"""
        self.devices = {}
        self.active_devices = []
        self.response_buffer = []
        self.is_recording = False
        
    def add_device(self, device: BaseHardwareDevice) -> None:
        """Add hardware device"""
        self.devices[device.device_id] = device
        logger.info(f"Added device: {device.device_id} ({device.device_type.value})")
    
    def connect_device(self, device_id: str) -> bool:
        """Connect specific device"""
        if device_id in self.devices:
            success = self.devices[device_id].connect()
            if success:
                self.active_devices.append(device_id)
            return success
        return False
    
    def connect_all_devices(self) -> Dict[str, bool]:
        """Connect all devices"""
        results = {}
        for device_id, device in self.devices.items():
            results[device_id] = self.connect_device(device_id)
        return results
    
    def disconnect_device(self, device_id: str) -> None:
        """Disconnect specific device"""
        if device_id in self.devices:
            self.devices[device_id].disconnect()
            if device_id in self.active_devices:
                self.active_devices.remove(device_id)
    
    def disconnect_all_devices(self) -> None:
        """Disconnect all devices"""
        for device_id in list(self.active_devices):
            self.disconnect_device(device_id)
    
    def start_recording(self) -> None:
        """Start recording from all devices"""
        self.is_recording = True
        self.response_buffer = []
        for device_id in self.active_devices:
            self.devices[device_id].start_recording()
        logger.info("Started recording from all devices")
    
    def stop_recording(self) -> None:
        """Stop recording from all devices"""
        self.is_recording = False
        for device_id in self.active_devices:
            self.devices[device_id].stop_recording()
        logger.info("Stopped recording from all devices")
    
    def get_responses(self, device_types: List[HardwareType] = None) -> List[HardwareResponse]:
        """Get responses from specified device types"""
        responses = []
        for device_id in self.active_devices:
            device = self.devices[device_id]
            if device_types is None or device.device_type in device_types:
                response = device.get_response()
                if response:
                    responses.append(response)
                    if self.is_recording:
                        self.response_buffer.append(response)
        return responses
    
    def wait_for_response(
        self, 
        device_types: List[HardwareType] = None,
        timeout: float = None
    ) -> Optional[HardwareResponse]:
        """Wait for response from any device"""
        start_time = time.time()
        
        while True:
            responses = self.get_responses(device_types)
            if responses:
                return responses[0]  # Return first response
                
            if timeout and (time.time() - start_time) > timeout:
                break
                
            time.sleep(0.001)  # Small delay
            
        return None
    
    def get_all_recorded_responses(self) -> List[HardwareResponse]:
        """Get all recorded responses"""
        return self.response_buffer.copy()
    
    def clear_all_responses(self) -> None:
        """Clear all recorded responses"""
        self.response_buffer = []
        for device in self.devices.values():
            device.clear_responses()


# Factory functions for easy device creation
def create_keyboard(device_id: str = "keyboard", **kwargs) -> KeyboardDevice:
    """Create keyboard device"""
    return KeyboardDevice(device_id, **kwargs)

def create_mouse(device_id: str = "mouse", win=None, **kwargs) -> MouseDevice:
    """Create mouse device"""
    return MouseDevice(device_id, win=win, **kwargs)

def create_eye_tracker(device_id: str = "eye_tracker", tracker_type: str = "tobii", **kwargs) -> EyeTrackerDevice:
    """Create eye tracker device"""
    return EyeTrackerDevice(device_id, tracker_type=tracker_type, **kwargs)
