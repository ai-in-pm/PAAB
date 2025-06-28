#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder - PsychoPy Experiments Page
Interface for creating and running psychological experiments
"""

import streamlit as st
import time
import json
from typing import Dict, Any, List

# Page configuration
st.set_page_config(
    page_title="PsychoPy Experiments",
    page_icon="🧠",
    layout="wide"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'experiments' not in st.session_state:
        st.session_state.experiments = []
    if 'experiment_results' not in st.session_state:
        st.session_state.experiment_results = []

def show_experiment_library():
    """Show available experiment paradigms"""
    st.header("🧪 Experiment Library")
    
    # Experiment categories
    experiment_categories = {
        "Attention & Executive Control": [
            {"name": "Stroop Color-Word", "description": "Classic interference task", "trials": 60, "duration": "8 min"},
            {"name": "Eriksen Flankers", "description": "Attention and conflict monitoring", "trials": 100, "duration": "10 min"},
            {"name": "Attention Network Test", "description": "Alerting, orienting, executive attention", "trials": 144, "duration": "20 min"},
            {"name": "Task Switching", "description": "Cognitive flexibility assessment", "trials": 80, "duration": "12 min"}
        ],
        "Memory & Learning": [
            {"name": "N-Back Task", "description": "Working memory assessment", "trials": 80, "duration": "15 min"},
            {"name": "Serial Position", "description": "Memory curve analysis", "trials": 40, "duration": "10 min"},
            {"name": "Recognition Memory", "description": "Old/new recognition task", "trials": 120, "duration": "18 min"},
            {"name": "Paired Associates", "description": "Associative learning task", "trials": 60, "duration": "12 min"}
        ],
        "Perception & Psychophysics": [
            {"name": "Visual Search", "description": "Feature and conjunction search", "trials": 80, "duration": "12 min"},
            {"name": "Change Blindness", "description": "Change detection paradigm", "trials": 40, "duration": "8 min"},
            {"name": "Motion Detection", "description": "Threshold measurement", "trials": 100, "duration": "15 min"},
            {"name": "Contrast Sensitivity", "description": "Psychometric function", "trials": 120, "duration": "20 min"}
        ],
        "Language & Cognition": [
            {"name": "Semantic Priming", "description": "Word processing and priming", "trials": 100, "duration": "12 min"},
            {"name": "Lexical Decision", "description": "Word/nonword classification", "trials": 80, "duration": "10 min"},
            {"name": "Reading Comprehension", "description": "Text processing assessment", "trials": 30, "duration": "25 min"},
            {"name": "Sentence Processing", "description": "Syntactic complexity effects", "trials": 60, "duration": "15 min"}
        ]
    }
    
    # Display experiments by category
    for category, experiments in experiment_categories.items():
        with st.expander(f"📋 {category}", expanded=False):
            cols = st.columns(2)
            for i, exp in enumerate(experiments):
                with cols[i % 2]:
                    with st.container():
                        st.subheader(f"🧪 {exp['name']}")
                        st.write(exp['description'])
                        st.write(f"**Trials:** {exp['trials']} | **Duration:** {exp['duration']}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"Create {exp['name']}", key=f"create_{exp['name']}"):
                                create_experiment(exp)
                        with col2:
                            if st.button(f"Preview {exp['name']}", key=f"preview_{exp['name']}"):
                                preview_experiment(exp)

def create_experiment(experiment_template: Dict[str, Any]):
    """Create a new experiment from template"""
    st.success(f"✅ Created experiment: {experiment_template['name']}")
    
    # Add to session state
    experiment_config = {
        "id": f"exp_{len(st.session_state.experiments) + 1}",
        "name": experiment_template['name'],
        "description": experiment_template['description'],
        "trials": experiment_template['trials'],
        "duration": experiment_template['duration'],
        "created_at": time.time(),
        "status": "configured"
    }
    
    st.session_state.experiments.append(experiment_config)
    
    # Show configuration options
    with st.expander("⚙️ Experiment Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Basic Settings**")
            num_trials = st.number_input("Number of Trials", value=experiment_template['trials'], min_value=10, max_value=500)
            randomize = st.checkbox("Randomize Trial Order", value=True)
            practice_trials = st.number_input("Practice Trials", value=10, min_value=0, max_value=50)
        
        with col2:
            st.write("**Timing Settings**")
            stimulus_duration = st.number_input("Stimulus Duration (ms)", value=1500, min_value=100, max_value=10000)
            iti = st.number_input("Inter-trial Interval (ms)", value=1000, min_value=100, max_value=5000)
            timeout = st.number_input("Response Timeout (ms)", value=3000, min_value=500, max_value=10000)

def preview_experiment(experiment_template: Dict[str, Any]):
    """Preview experiment details"""
    st.info(f"🔍 Previewing: {experiment_template['name']}")
    
    with st.expander("📋 Experiment Details", expanded=True):
        st.write(f"**Description:** {experiment_template['description']}")
        st.write(f"**Number of Trials:** {experiment_template['trials']}")
        st.write(f"**Estimated Duration:** {experiment_template['duration']}")
        
        # Show sample trial sequence
        st.write("**Sample Trial Sequence:**")
        if experiment_template['name'] == "Stroop Color-Word":
            st.write("1. Fixation cross (500ms)")
            st.write("2. Color word stimulus (until response)")
            st.write("3. Feedback (500ms)")
            st.write("4. Inter-trial interval (1000ms)")
        elif experiment_template['name'] == "N-Back Task":
            st.write("1. Stimulus presentation (500ms)")
            st.write("2. Response window (2000ms)")
            st.write("3. Feedback (300ms)")
            st.write("4. Inter-trial interval (1000ms)")
        else:
            st.write("1. Fixation (500ms)")
            st.write("2. Stimulus presentation")
            st.write("3. Response collection")
            st.write("4. Inter-trial interval")

def show_ai_participants():
    """Show AI participant configuration"""
    st.header("🤖 AI Participants")
    
    # Predefined cognitive profiles
    cognitive_profiles = {
        "Optimal Performer": {
            "description": "AI agent optimized for perfect performance",
            "reaction_time": "Very fast (300-500ms)",
            "accuracy": "Near perfect (95-98%)",
            "fatigue": "Minimal",
            "learning": "Rapid adaptation"
        },
        "Human-like Performer": {
            "description": "AI agent simulating typical human performance",
            "reaction_time": "Moderate (600-900ms)",
            "accuracy": "Good (80-85%)",
            "fatigue": "Gradual decline",
            "learning": "Steady improvement"
        },
        "Impaired Performer": {
            "description": "AI agent simulating cognitive limitations",
            "reaction_time": "Slow (1000-1500ms)",
            "accuracy": "Variable (60-75%)",
            "fatigue": "Rapid onset",
            "learning": "Slow adaptation"
        },
        "Variable Performer": {
            "description": "AI agent with high individual differences",
            "reaction_time": "Highly variable (400-1200ms)",
            "accuracy": "Inconsistent (50-90%)",
            "fatigue": "Unpredictable",
            "learning": "Erratic progress"
        }
    }
    
    # Display profiles
    cols = st.columns(2)
    for i, (profile_name, profile_data) in enumerate(cognitive_profiles.items()):
        with cols[i % 2]:
            with st.container():
                st.subheader(f"🧠 {profile_name}")
                st.write(profile_data['description'])
                
                with st.expander("Profile Details"):
                    st.write(f"**Reaction Time:** {profile_data['reaction_time']}")
                    st.write(f"**Accuracy:** {profile_data['accuracy']}")
                    st.write(f"**Fatigue:** {profile_data['fatigue']}")
                    st.write(f"**Learning:** {profile_data['learning']}")
                
                if st.button(f"Select {profile_name}", key=f"select_{profile_name}"):
                    st.success(f"✅ Selected: {profile_name}")

def show_experiment_execution():
    """Show experiment execution interface"""
    st.header("🚀 Experiment Execution")
    
    if not st.session_state.experiments:
        st.info("📝 No experiments configured. Create an experiment first!")
        return
    
    # Select experiment to run
    experiment_names = [exp['name'] for exp in st.session_state.experiments]
    selected_exp = st.selectbox("Select Experiment", experiment_names)
    
    if selected_exp:
        experiment = next(exp for exp in st.session_state.experiments if exp['name'] == selected_exp)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Experiment Details**")
            st.write(f"Name: {experiment['name']}")
            st.write(f"Trials: {experiment['trials']}")
            st.write(f"Duration: {experiment['duration']}")
            st.write(f"Status: {experiment['status']}")
        
        with col2:
            st.write("**Execution Options**")
            num_participants = st.number_input("Number of AI Participants", value=3, min_value=1, max_value=10)
            execution_mode = st.selectbox("Execution Mode", ["Sequential", "Parallel", "Batch"])
            save_data = st.checkbox("Save Data", value=True)
        
        # Run experiment button
        if st.button("🚀 Run Experiment", type="primary"):
            run_experiment_simulation(experiment, num_participants, execution_mode)

def run_experiment_simulation(experiment: Dict[str, Any], num_participants: int, execution_mode: str):
    """Simulate experiment execution"""
    st.success(f"🚀 Starting experiment: {experiment['name']}")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Simulate execution
    for i in range(num_participants):
        progress = (i + 1) / num_participants
        progress_bar.progress(progress)
        status_text.text(f"Running participant {i + 1}/{num_participants}...")
        time.sleep(0.5)  # Simulate processing time
    
    # Generate simulated results
    results = {
        "experiment_id": experiment['id'],
        "experiment_name": experiment['name'],
        "participants": num_participants,
        "execution_mode": execution_mode,
        "overall_accuracy": 0.82,
        "mean_reaction_time": 0.687,
        "completion_time": time.time(),
        "status": "completed"
    }
    
    st.session_state.experiment_results.append(results)
    
    status_text.text("✅ Experiment completed!")
    
    # Show results summary
    with st.expander("📊 Results Summary", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall Accuracy", f"{results['overall_accuracy']:.1%}")
        
        with col2:
            st.metric("Mean RT", f"{results['mean_reaction_time']:.3f}s")
        
        with col3:
            st.metric("Participants", results['participants'])

def show_results_dashboard():
    """Show experiment results dashboard"""
    st.header("📊 Results Dashboard")
    
    if not st.session_state.experiment_results:
        st.info("📈 No experiment results yet. Run an experiment to see results!")
        return
    
    # Results summary
    total_experiments = len(st.session_state.experiment_results)
    total_participants = sum(result['participants'] for result in st.session_state.experiment_results)
    avg_accuracy = sum(result['overall_accuracy'] for result in st.session_state.experiment_results) / total_experiments
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Experiments", total_experiments)
    
    with col2:
        st.metric("Total Participants", total_participants)
    
    with col3:
        st.metric("Average Accuracy", f"{avg_accuracy:.1%}")
    
    # Recent results
    st.subheader("📋 Recent Results")
    for result in st.session_state.experiment_results[-5:]:
        with st.expander(f"🧪 {result['experiment_name']} - {result['participants']} participants"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Accuracy:** {result['overall_accuracy']:.1%}")
                st.write(f"**Mean RT:** {result['mean_reaction_time']:.3f}s")
            
            with col2:
                st.write(f"**Participants:** {result['participants']}")
                st.write(f"**Status:** {result['status']}")

def main():
    """Main function"""
    st.title("🧠 PsychoPy Experiments")
    st.markdown("*Psychological experiments powered by AI agents*")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar navigation
    with st.sidebar:
        st.header("🧭 Navigation")
        page = st.radio(
            "Select Page",
            ["📚 Experiment Library", "🤖 AI Participants", "🚀 Run Experiments", "📊 Results Dashboard"]
        )
    
    # Show selected page
    if page == "📚 Experiment Library":
        show_experiment_library()
    elif page == "🤖 AI Participants":
        show_ai_participants()
    elif page == "🚀 Run Experiments":
        show_experiment_execution()
    elif page == "📊 Results Dashboard":
        show_results_dashboard()

if __name__ == "__main__":
    main()
