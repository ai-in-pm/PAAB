#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder - Data Collection and Analysis
Comprehensive data handling for psychological experiments
"""

import os
import json
import csv
import pickle
import time
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Try to import PsychoPy data components
try:
    from psychopy import data, core, logging as psychopy_logging
    from psychopy.tools import filetools
    PSYCHOPY_AVAILABLE = True
except ImportError:
    PSYCHOPY_AVAILABLE = False
    logger.warning("PsychoPy data tools not available - using basic implementation")


class DataFormat(Enum):
    """Data output formats"""
    CSV = "csv"
    JSON = "json"
    PICKLE = "pickle"
    EXCEL = "excel"
    HDF5 = "hdf5"
    MATLAB = "matlab"
    PSYCHOPY = "psychopy"


class AnalysisType(Enum):
    """Types of statistical analysis"""
    DESCRIPTIVE = "descriptive"
    INFERENTIAL = "inferential"
    CORRELATION = "correlation"
    REGRESSION = "regression"
    ANOVA = "anova"
    TTEST = "ttest"
    NONPARAMETRIC = "nonparametric"
    TIME_SERIES = "time_series"
    CUSTOM = "custom"


@dataclass
class TrialData:
    """Data from a single trial"""
    trial_id: str
    trial_number: int
    participant_id: str
    experiment_id: str
    condition: str
    start_time: float
    end_time: float
    duration: float
    responses: List[Dict[str, Any]] = field(default_factory=list)
    stimuli: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_response(self, response_data: Dict[str, Any]) -> None:
        """Add response to trial"""
        response_data['timestamp'] = response_data.get('timestamp', time.time())
        self.responses.append(response_data)
    
    def add_stimulus(self, stimulus_data: Dict[str, Any]) -> None:
        """Add stimulus to trial"""
        stimulus_data['timestamp'] = stimulus_data.get('timestamp', time.time())
        self.stimuli.append(stimulus_data)
    
    def add_event(self, event_data: Dict[str, Any]) -> None:
        """Add event to trial"""
        event_data['timestamp'] = event_data.get('timestamp', time.time())
        self.events.append(event_data)
    
    def get_primary_response(self) -> Optional[Dict[str, Any]]:
        """Get the primary response for this trial"""
        if self.responses:
            return self.responses[0]  # First response is typically primary
        return None
    
    def get_reaction_time(self) -> Optional[float]:
        """Get reaction time for primary response"""
        primary_response = self.get_primary_response()
        if primary_response and 'reaction_time' in primary_response:
            return primary_response['reaction_time']
        elif primary_response and 'timestamp' in primary_response:
            return primary_response['timestamp'] - self.start_time
        return None
    
    def get_accuracy(self) -> Optional[bool]:
        """Get accuracy for primary response"""
        primary_response = self.get_primary_response()
        if primary_response and 'accuracy' in primary_response:
            return primary_response['accuracy']
        return None


@dataclass
class SessionData:
    """Data from an experimental session"""
    session_id: str
    participant_id: str
    experiment_id: str
    start_time: float
    end_time: Optional[float] = None
    trials: List[TrialData] = field(default_factory=list)
    participant_info: Dict[str, Any] = field(default_factory=dict)
    experiment_info: Dict[str, Any] = field(default_factory=dict)
    hardware_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_trial(self, trial: TrialData) -> None:
        """Add trial to session"""
        self.trials.append(trial)
    
    def get_trial_count(self) -> int:
        """Get number of trials"""
        return len(self.trials)
    
    def get_session_duration(self) -> float:
        """Get total session duration"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    def get_accuracy_rate(self) -> float:
        """Get overall accuracy rate"""
        accurate_trials = sum(1 for trial in self.trials if trial.get_accuracy())
        total_trials = len([trial for trial in self.trials if trial.get_accuracy() is not None])
        return accurate_trials / total_trials if total_trials > 0 else 0.0
    
    def get_mean_reaction_time(self) -> float:
        """Get mean reaction time"""
        reaction_times = [trial.get_reaction_time() for trial in self.trials]
        valid_rts = [rt for rt in reaction_times if rt is not None]
        return np.mean(valid_rts) if valid_rts else 0.0


class DataCollector:
    """Collects and manages experimental data"""
    
    def __init__(self, experiment_id: str, data_dir: str = "data"):
        """
        Initialize data collector
        
        Args:
            experiment_id: Unique experiment identifier
            data_dir: Directory for data storage
        """
        self.experiment_id = experiment_id
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.current_session = None
        self.current_trial = None
        self.sessions = {}
        
        # PsychoPy data handler
        self.psychopy_handler = None
        if PSYCHOPY_AVAILABLE:
            self.setup_psychopy_handler()
    
    def setup_psychopy_handler(self) -> None:
        """Setup PsychoPy data handler"""
        try:
            self.psychopy_handler = data.ExperimentHandler(
                name=self.experiment_id,
                version='1.0',
                extraInfo={},
                runtimeInfo=None,
                originPath=None,
                savePickle=True,
                saveWideText=True,
                dataFileName=str(self.data_dir / self.experiment_id)
            )
            logger.info("PsychoPy data handler initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PsychoPy data handler: {str(e)}")
    
    def start_session(
        self, 
        participant_id: str, 
        participant_info: Dict[str, Any] = None
    ) -> SessionData:
        """Start a new experimental session"""
        session_id = f"{participant_id}_{int(time.time())}"
        
        self.current_session = SessionData(
            session_id=session_id,
            participant_id=participant_id,
            experiment_id=self.experiment_id,
            start_time=time.time(),
            participant_info=participant_info or {},
            experiment_info={'experiment_id': self.experiment_id}
        )
        
        self.sessions[session_id] = self.current_session
        
        # Add to PsychoPy handler
        if self.psychopy_handler:
            self.psychopy_handler.addData('session_id', session_id)
            self.psychopy_handler.addData('participant_id', participant_id)
        
        logger.info(f"Started session: {session_id}")
        return self.current_session
    
    def end_session(self) -> None:
        """End current experimental session"""
        if self.current_session:
            self.current_session.end_time = time.time()
            logger.info(f"Ended session: {self.current_session.session_id}")
            self.current_session = None
    
    def start_trial(
        self, 
        trial_id: str, 
        trial_number: int, 
        condition: str = "",
        metadata: Dict[str, Any] = None
    ) -> TrialData:
        """Start a new trial"""
        if not self.current_session:
            raise ValueError("No active session. Call start_session() first.")
        
        self.current_trial = TrialData(
            trial_id=trial_id,
            trial_number=trial_number,
            participant_id=self.current_session.participant_id,
            experiment_id=self.experiment_id,
            condition=condition,
            start_time=time.time(),
            end_time=0,
            duration=0,
            metadata=metadata or {}
        )
        
        # Add to PsychoPy handler
        if self.psychopy_handler:
            self.psychopy_handler.addData('trial_id', trial_id)
            self.psychopy_handler.addData('trial_number', trial_number)
            self.psychopy_handler.addData('condition', condition)
        
        logger.debug(f"Started trial: {trial_id}")
        return self.current_trial
    
    def end_trial(self) -> None:
        """End current trial"""
        if self.current_trial:
            self.current_trial.end_time = time.time()
            self.current_trial.duration = self.current_trial.end_time - self.current_trial.start_time
            
            # Add to session
            if self.current_session:
                self.current_session.add_trial(self.current_trial)
            
            # Add to PsychoPy handler
            if self.psychopy_handler:
                self.psychopy_handler.nextEntry()
            
            logger.debug(f"Ended trial: {self.current_trial.trial_id}")
            self.current_trial = None
    
    def add_response(self, response_data: Dict[str, Any]) -> None:
        """Add response to current trial"""
        if self.current_trial:
            self.current_trial.add_response(response_data)
            
            # Add to PsychoPy handler
            if self.psychopy_handler:
                for key, value in response_data.items():
                    self.psychopy_handler.addData(key, value)
    
    def add_stimulus(self, stimulus_data: Dict[str, Any]) -> None:
        """Add stimulus to current trial"""
        if self.current_trial:
            self.current_trial.add_stimulus(stimulus_data)
    
    def add_event(self, event_data: Dict[str, Any]) -> None:
        """Add event to current trial"""
        if self.current_trial:
            self.current_trial.add_event(event_data)
    
    def save_session(
        self, 
        session_id: str = None, 
        formats: List[DataFormat] = None
    ) -> Dict[str, str]:
        """Save session data in specified formats"""
        if session_id is None and self.current_session:
            session_id = self.current_session.session_id
        
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        formats = formats or [DataFormat.CSV, DataFormat.JSON]
        
        saved_files = {}
        
        for format_type in formats:
            try:
                filename = self._save_session_format(session, format_type)
                saved_files[format_type.value] = filename
                logger.info(f"Saved session data: {filename}")
            except Exception as e:
                logger.error(f"Failed to save {format_type.value}: {str(e)}")
        
        return saved_files
    
    def _save_session_format(self, session: SessionData, format_type: DataFormat) -> str:
        """Save session in specific format"""
        base_filename = f"{session.session_id}_{session.experiment_id}"
        
        if format_type == DataFormat.CSV:
            filename = self.data_dir / f"{base_filename}.csv"
            self._save_csv(session, filename)
        elif format_type == DataFormat.JSON:
            filename = self.data_dir / f"{base_filename}.json"
            self._save_json(session, filename)
        elif format_type == DataFormat.PICKLE:
            filename = self.data_dir / f"{base_filename}.pkl"
            self._save_pickle(session, filename)
        elif format_type == DataFormat.EXCEL:
            filename = self.data_dir / f"{base_filename}.xlsx"
            self._save_excel(session, filename)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        return str(filename)
    
    def _save_csv(self, session: SessionData, filename: Path) -> None:
        """Save session as CSV"""
        rows = []
        for trial in session.trials:
            base_row = {
                'session_id': session.session_id,
                'participant_id': session.participant_id,
                'experiment_id': session.experiment_id,
                'trial_id': trial.trial_id,
                'trial_number': trial.trial_number,
                'condition': trial.condition,
                'start_time': trial.start_time,
                'end_time': trial.end_time,
                'duration': trial.duration,
                'reaction_time': trial.get_reaction_time(),
                'accuracy': trial.get_accuracy()
            }
            
            # Add response data
            primary_response = trial.get_primary_response()
            if primary_response:
                for key, value in primary_response.items():
                    base_row[f'response_{key}'] = value
            
            # Add metadata
            for key, value in trial.metadata.items():
                base_row[f'meta_{key}'] = value
            
            rows.append(base_row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
    
    def _save_json(self, session: SessionData, filename: Path) -> None:
        """Save session as JSON"""
        session_dict = asdict(session)
        with open(filename, 'w') as f:
            json.dump(session_dict, f, indent=2, default=str)
    
    def _save_pickle(self, session: SessionData, filename: Path) -> None:
        """Save session as pickle"""
        with open(filename, 'wb') as f:
            pickle.dump(session, f)
    
    def _save_excel(self, session: SessionData, filename: Path) -> None:
        """Save session as Excel"""
        # Create DataFrame similar to CSV
        rows = []
        for trial in session.trials:
            base_row = {
                'session_id': session.session_id,
                'participant_id': session.participant_id,
                'experiment_id': session.experiment_id,
                'trial_id': trial.trial_id,
                'trial_number': trial.trial_number,
                'condition': trial.condition,
                'start_time': trial.start_time,
                'end_time': trial.end_time,
                'duration': trial.duration,
                'reaction_time': trial.get_reaction_time(),
                'accuracy': trial.get_accuracy()
            }
            rows.append(base_row)
        
        df = pd.DataFrame(rows)
        df.to_excel(filename, index=False)


class DataAnalyzer:
    """Analyzes experimental data"""
    
    def __init__(self):
        """Initialize data analyzer"""
        self.analysis_cache = {}
    
    def load_session_data(self, filepath: str) -> SessionData:
        """Load session data from file"""
        filepath = Path(filepath)
        
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                data_dict = json.load(f)
            return self._dict_to_session(data_dict)
        elif filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def _dict_to_session(self, data_dict: Dict[str, Any]) -> SessionData:
        """Convert dictionary to SessionData object"""
        # Convert trial dictionaries to TrialData objects
        trials = []
        for trial_dict in data_dict.get('trials', []):
            trial = TrialData(**trial_dict)
            trials.append(trial)
        
        data_dict['trials'] = trials
        return SessionData(**data_dict)
    
    def analyze_session(self, session: SessionData) -> Dict[str, Any]:
        """Perform comprehensive analysis of session data"""
        analysis = {
            'basic_stats': self._basic_statistics(session),
            'performance_metrics': self._performance_metrics(session),
            'temporal_analysis': self._temporal_analysis(session),
            'condition_analysis': self._condition_analysis(session)
        }
        
        return analysis
    
    def _basic_statistics(self, session: SessionData) -> Dict[str, Any]:
        """Calculate basic descriptive statistics"""
        reaction_times = [trial.get_reaction_time() for trial in session.trials]
        valid_rts = [rt for rt in reaction_times if rt is not None]
        
        accuracies = [trial.get_accuracy() for trial in session.trials]
        valid_accuracies = [acc for acc in accuracies if acc is not None]
        
        return {
            'total_trials': len(session.trials),
            'valid_rt_trials': len(valid_rts),
            'mean_rt': np.mean(valid_rts) if valid_rts else 0,
            'median_rt': np.median(valid_rts) if valid_rts else 0,
            'std_rt': np.std(valid_rts) if valid_rts else 0,
            'min_rt': np.min(valid_rts) if valid_rts else 0,
            'max_rt': np.max(valid_rts) if valid_rts else 0,
            'accuracy_rate': np.mean(valid_accuracies) if valid_accuracies else 0,
            'error_rate': 1 - np.mean(valid_accuracies) if valid_accuracies else 1,
            'session_duration': session.get_session_duration()
        }
    
    def _performance_metrics(self, session: SessionData) -> Dict[str, Any]:
        """Calculate performance metrics"""
        # Performance over time (learning/fatigue effects)
        block_size = max(1, len(session.trials) // 5)  # 5 blocks
        blocks = []
        
        for i in range(0, len(session.trials), block_size):
            block_trials = session.trials[i:i + block_size]
            block_rts = [trial.get_reaction_time() for trial in block_trials]
            block_accs = [trial.get_accuracy() for trial in block_trials]
            
            valid_rts = [rt for rt in block_rts if rt is not None]
            valid_accs = [acc for acc in block_accs if acc is not None]
            
            blocks.append({
                'block_number': len(blocks) + 1,
                'trial_range': (i + 1, min(i + block_size, len(session.trials))),
                'mean_rt': np.mean(valid_rts) if valid_rts else 0,
                'accuracy': np.mean(valid_accs) if valid_accs else 0,
                'trial_count': len(block_trials)
            })
        
        return {
            'block_analysis': blocks,
            'learning_trend': self._detect_trend([b['accuracy'] for b in blocks]),
            'fatigue_trend': self._detect_trend([b['mean_rt'] for b in blocks])
        }
    
    def _temporal_analysis(self, session: SessionData) -> Dict[str, Any]:
        """Analyze temporal patterns"""
        reaction_times = [trial.get_reaction_time() for trial in session.trials]
        valid_rts = [rt for rt in reaction_times if rt is not None]
        
        if not valid_rts:
            return {}
        
        # Outlier detection
        q1, q3 = np.percentile(valid_rts, [25, 75])
        iqr = q3 - q1
        outlier_threshold = q3 + 1.5 * iqr
        outliers = [rt for rt in valid_rts if rt > outlier_threshold]
        
        return {
            'rt_quartiles': [np.percentile(valid_rts, q) for q in [25, 50, 75]],
            'outlier_count': len(outliers),
            'outlier_percentage': len(outliers) / len(valid_rts) * 100,
            'coefficient_of_variation': np.std(valid_rts) / np.mean(valid_rts)
        }
    
    def _condition_analysis(self, session: SessionData) -> Dict[str, Any]:
        """Analyze performance by condition"""
        conditions = {}
        
        for trial in session.trials:
            condition = trial.condition or 'default'
            if condition not in conditions:
                conditions[condition] = {'rts': [], 'accuracies': []}
            
            rt = trial.get_reaction_time()
            acc = trial.get_accuracy()
            
            if rt is not None:
                conditions[condition]['rts'].append(rt)
            if acc is not None:
                conditions[condition]['accuracies'].append(acc)
        
        condition_stats = {}
        for condition, data in conditions.items():
            condition_stats[condition] = {
                'trial_count': len(data['rts']) + len(data['accuracies']),
                'mean_rt': np.mean(data['rts']) if data['rts'] else 0,
                'accuracy': np.mean(data['accuracies']) if data['accuracies'] else 0
            }
        
        return condition_stats
    
    def _detect_trend(self, values: List[float]) -> str:
        """Detect trend in values"""
        if len(values) < 2:
            return 'insufficient_data'
        
        # Simple linear trend detection
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'
