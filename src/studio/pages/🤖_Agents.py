#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder - Enhanced Agents Page
Comprehensive agent creation and management interface with PsychoPy integration
"""

import streamlit as st
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from agents.specialized import ResearcherAgent, AnalystAgent, WriterAgent, CoderAgent
    from agents.psychopy_agents import ExperimentParticipantAgent, ExperimentDesignerAgent, ExperimentAnalystAgent
    AGENTS_AVAILABLE = True
except ImportError:
    AGENTS_AVAILABLE = False
    st.warning("⚠️ Agent modules not fully available - using simulation mode")

st.set_page_config(
    page_title="PAAB - Agents",
    page_icon="🤖",
    layout="wide"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'agents' not in st.session_state:
        st.session_state.agents = []
    if 'agent_templates' not in st.session_state:
        st.session_state.agent_templates = get_default_templates()
    if 'cognitive_profiles' not in st.session_state:
        st.session_state.cognitive_profiles = get_cognitive_profiles()

def get_default_templates() -> Dict[str, Any]:
    """Get default agent templates"""
    return {
        "research_team": {
            "name": "Research Team",
            "description": "Complete research team with researcher, analyst, and writer",
            "agents": [
                {
                    "name": "Senior Researcher",
                    "role": "researcher",
                    "goal": "Conduct comprehensive research on assigned topics",
                    "backstory": "You are a senior researcher with 15 years of experience in academic and industry research.",
                    "expertise_domains": ["Research", "AI/ML", "Data Science"]
                },
                {
                    "name": "Data Analyst",
                    "role": "analyst",
                    "goal": "Analyze data and extract meaningful insights",
                    "backstory": "You are an expert data analyst with strong statistical and analytical skills.",
                    "expertise_domains": ["Data Science", "Analysis", "AI/ML"]
                },
                {
                    "name": "Technical Writer",
                    "role": "writer",
                    "goal": "Create clear and comprehensive reports",
                    "backstory": "You are a technical writer specializing in research documentation.",
                    "expertise_domains": ["Writing", "Research"]
                }
            ]
        },
        "psychopy_team": {
            "name": "PsychoPy Experiment Team",
            "description": "Specialized team for psychological experiments",
            "agents": [
                {
                    "name": "Experiment Designer",
                    "role": "experiment_designer",
                    "goal": "Design rigorous psychological experiments",
                    "backstory": "You are an expert experimental psychologist with 15 years of experience in cognitive research.",
                    "expertise_domains": ["Psychology", "Experimental Design", "Statistics"]
                },
                {
                    "name": "Experiment Analyst",
                    "role": "experiment_analyst",
                    "goal": "Analyze experimental data and provide insights",
                    "backstory": "You are a statistical expert specializing in psychological research analysis.",
                    "expertise_domains": ["Statistics", "Psychology", "Data Analysis"]
                },
                {
                    "name": "Cognitive Agent",
                    "role": "experiment_participant",
                    "goal": "Participate in psychological experiments with human-like behavior",
                    "backstory": "You are an AI agent designed to simulate human cognitive performance.",
                    "expertise_domains": ["Cognitive Modeling", "Psychology"]
                }
            ]
        }
    }

def get_cognitive_profiles() -> Dict[str, Any]:
    """Get cognitive profiles for experiment participants"""
    return {
        "optimal": {
            "name": "Optimal Performer",
            "description": "AI agent optimized for perfect performance",
            "base_reaction_time": 0.4,
            "reaction_time_variability": 0.05,
            "accuracy_rate": 0.95,
            "attention_span": 600.0,
            "fatigue_rate": 0.0005,
            "learning_rate": 0.02
        },
        "human_like": {
            "name": "Human-like Performer",
            "description": "AI agent simulating typical human performance",
            "base_reaction_time": 0.7,
            "reaction_time_variability": 0.15,
            "accuracy_rate": 0.82,
            "attention_span": 300.0,
            "fatigue_rate": 0.002,
            "learning_rate": 0.015
        },
        "impaired": {
            "name": "Impaired Performer",
            "description": "AI agent simulating cognitive limitations",
            "base_reaction_time": 1.2,
            "reaction_time_variability": 0.25,
            "accuracy_rate": 0.65,
            "attention_span": 150.0,
            "fatigue_rate": 0.005,
            "learning_rate": 0.008
        },
        "variable": {
            "name": "Variable Performer",
            "description": "AI agent with high individual differences",
            "base_reaction_time": 0.9,
            "reaction_time_variability": 0.3,
            "accuracy_rate": 0.75,
            "attention_span": 200.0,
            "fatigue_rate": 0.003,
            "learning_rate": 0.01
        }
    }

def main():
    """Main agents page"""

    st.title("🤖 Enhanced Agent Management")
    st.markdown("*Create sophisticated AI agents with PsychoPy integration*")

    # Initialize session state
    initialize_session_state()

    # Sidebar navigation
    with st.sidebar:
        st.header("🧭 Navigation")
        if st.button("🏠 Dashboard"):
            st.switch_page("main.py")
        if st.button("📝 Tasks"):
            st.switch_page("pages/📝_Tasks.py")
        if st.button("👥 Crews"):
            st.switch_page("pages/👥_Crews.py")
        if st.button("🚀 Execution"):
            st.switch_page("pages/🚀_Execution.py")
        if st.button("🧠 PsychoPy Experiments"):
            st.switch_page("pages/🧠_PsychoPy_Experiments.py")

        # Agent statistics
        st.header("📊 Agent Statistics")
        total_agents = len(st.session_state.agents)
        st.metric("Total Agents", total_agents)

        if total_agents > 0:
            roles = [agent['role'] for agent in st.session_state.agents]
            role_counts = {role: roles.count(role) for role in set(roles)}

            st.write("**Agents by Role:**")
            for role, count in role_counts.items():
                st.write(f"• {role.title()}: {count}")

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "➕ Create Agent",
        "📋 Manage Agents",
        "🧠 Cognitive Profiles",
        "📚 Templates",
        "🔧 Advanced Tools"
    ])
    
    with tab1:
        show_create_agent_interface()

    with tab2:
        show_manage_agents_interface()

    with tab3:
        show_cognitive_profiles_interface()

    with tab4:
        show_templates_interface()

    with tab5:
        show_advanced_tools_interface()

def show_create_agent_interface():
    """Show enhanced agent creation interface"""
    st.header("🚀 Create New Agent")

    # Agent type selection
    agent_type = st.selectbox(
        "🎯 Agent Type",
        ["Standard Agent", "PsychoPy Experiment Participant", "PsychoPy Experiment Designer", "PsychoPy Experiment Analyst"],
        help="Choose the type of agent to create"
    )

    if agent_type == "Standard Agent":
        show_standard_agent_form()
    elif agent_type == "PsychoPy Experiment Participant":
        show_experiment_participant_form()
    elif agent_type == "PsychoPy Experiment Designer":
        show_experiment_designer_form()
    elif agent_type == "PsychoPy Experiment Analyst":
        show_experiment_analyst_form()

def show_standard_agent_form():
    """Show standard agent creation form"""
    with st.form("create_standard_agent_form"):
        st.subheader("🤖 Standard Agent Configuration")

        col1, col2 = st.columns(2)

        with col1:
            agent_name = st.text_input("Agent Name", placeholder="e.g., Research Assistant")
            agent_role = st.selectbox(
                "Agent Role",
                ["researcher", "analyst", "writer", "coder", "custom"],
                help="Select the primary role for this agent"
            )

            if agent_role == "custom":
                custom_role = st.text_input("Custom Role", placeholder="e.g., data_scientist")
                agent_role = custom_role if custom_role else "custom"

            # Personality traits
            st.write("**🧠 Personality Traits**")
            creativity = st.slider("Creativity", 0.0, 1.0, 0.7, help="How creative should the agent be?")
            analytical = st.slider("Analytical", 0.0, 1.0, 0.8, help="How analytical should the agent be?")
            collaborative = st.slider("Collaborative", 0.0, 1.0, 0.6, help="How collaborative should the agent be?")

        with col2:
            agent_goal = st.text_area(
                "Agent Goal",
                placeholder="e.g., Research and analyze market trends in AI technology",
                help="What should this agent accomplish?",
                height=100
            )

            agent_backstory = st.text_area(
                "Agent Backstory",
                placeholder="e.g., You are an expert market researcher with 10 years of experience...",
                help="Background and expertise of the agent",
                height=100
            )

            # Communication style
            st.write("**💬 Communication Style**")
            communication_style = st.selectbox(
                "Style",
                ["Professional", "Casual", "Academic", "Technical", "Creative"],
                help="How should the agent communicate?"
            )

            verbosity = st.selectbox(
                "Verbosity",
                ["Concise", "Moderate", "Detailed", "Comprehensive"],
                help="How detailed should responses be?"
            )

        # Advanced settings
        with st.expander("🔧 Advanced Settings", expanded=False):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("**⚙️ Execution Settings**")
                max_iterations = st.number_input("Max Iterations", min_value=1, max_value=100, value=25)
                timeout = st.number_input("Timeout (seconds)", min_value=10, max_value=3600, value=300)
                retry_attempts = st.number_input("Retry Attempts", min_value=0, max_value=10, value=3)

            with col2:
                st.write("**🧠 Cognitive Settings**")
                memory_enabled = st.checkbox("Enable Memory", value=True)
                learning_enabled = st.checkbox("Enable Learning", value=True)
                adaptation_enabled = st.checkbox("Enable Adaptation", value=False)

                memory_size = st.number_input("Memory Size", min_value=100, max_value=10000, value=1000)

            with col3:
                st.write("**🤝 Collaboration Settings**")
                collaboration_enabled = st.checkbox("Enable Collaboration", value=True)
                leadership_style = st.selectbox("Leadership Style", ["Democratic", "Autocratic", "Laissez-faire"])
                conflict_resolution = st.selectbox("Conflict Resolution", ["Compromise", "Compete", "Accommodate", "Avoid"])

        # Expertise and skills
        with st.expander("🎓 Expertise & Skills", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                expertise_domains = st.multiselect(
                    "Expertise Domains",
                    ["AI/ML", "Data Science", "Research", "Writing", "Analysis", "Coding",
                     "Business", "Marketing", "Psychology", "Statistics", "Design", "Education"],
                    help="Areas of expertise for this agent"
                )

                programming_languages = st.multiselect(
                    "Programming Languages",
                    ["Python", "R", "JavaScript", "Java", "C++", "SQL", "MATLAB", "Julia"],
                    help="Programming languages the agent knows"
                )

            with col2:
                tools_and_frameworks = st.multiselect(
                    "Tools & Frameworks",
                    ["TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy",
                     "PsychoPy", "SPSS", "Tableau", "Power BI", "Git"],
                    help="Tools and frameworks the agent can use"
                )

                soft_skills = st.multiselect(
                    "Soft Skills",
                    ["Critical Thinking", "Problem Solving", "Communication", "Leadership",
                     "Time Management", "Adaptability", "Creativity", "Attention to Detail"],
                    help="Soft skills the agent possesses"
                )

        # Performance metrics
        with st.expander("📊 Performance Metrics", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                quality_threshold = st.slider("Quality Threshold", 0.0, 1.0, 0.8, help="Minimum quality standard")
                speed_preference = st.slider("Speed vs Quality", 0.0, 1.0, 0.5, help="0=Quality focused, 1=Speed focused")

            with col2:
                error_tolerance = st.slider("Error Tolerance", 0.0, 1.0, 0.1, help="Acceptable error rate")
                innovation_level = st.slider("Innovation Level", 0.0, 1.0, 0.6, help="How innovative should solutions be?")

        submitted = st.form_submit_button("🚀 Create Standard Agent", use_container_width=True)

        if submitted:
            create_standard_agent(
                agent_name, agent_role, agent_goal, agent_backstory,
                creativity, analytical, collaborative, communication_style, verbosity,
                max_iterations, timeout, retry_attempts, memory_enabled, learning_enabled,
                adaptation_enabled, memory_size, collaboration_enabled, leadership_style,
                conflict_resolution, expertise_domains, programming_languages,
                tools_and_frameworks, soft_skills, quality_threshold, speed_preference,
                error_tolerance, innovation_level
            )

def show_experiment_participant_form():
    """Show experiment participant agent creation form"""
    with st.form("create_participant_agent_form"):
        st.subheader("🧠 Experiment Participant Agent")

        col1, col2 = st.columns(2)

        with col1:
            agent_name = st.text_input("Agent Name", placeholder="e.g., Cognitive Participant")

            # Cognitive profile selection
            profile_type = st.selectbox(
                "Cognitive Profile",
                ["optimal", "human_like", "impaired", "variable", "custom"],
                help="Select a predefined cognitive profile"
            )

            if profile_type != "custom":
                profile = st.session_state.cognitive_profiles[profile_type]
                st.info(f"**{profile['name']}**: {profile['description']}")

            # Response strategy
            response_strategy = st.selectbox(
                "Response Strategy",
                ["optimal", "human-like", "random", "strategic"],
                help="How should the agent respond to experimental tasks?"
            )

            learning_enabled = st.checkbox("Enable Learning", value=True, help="Should the agent learn from experience?")

        with col2:
            agent_goal = st.text_area(
                "Agent Goal",
                value="Participate in psychological experiments and provide realistic responses",
                help="What should this agent accomplish?"
            )

            agent_backstory = st.text_area(
                "Agent Backstory",
                value="You are an AI agent designed to participate in psychological experiments with human-like cognitive performance.",
                help="Background of the agent"
            )

        # Cognitive parameters
        with st.expander("🧠 Cognitive Parameters", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("**⚡ Reaction Time**")
                if profile_type == "custom":
                    base_rt = st.number_input("Base RT (seconds)", 0.1, 3.0, 0.7, 0.1)
                    rt_variability = st.number_input("RT Variability", 0.01, 0.5, 0.15, 0.01)
                else:
                    profile = st.session_state.cognitive_profiles[profile_type]
                    base_rt = st.number_input("Base RT (seconds)", 0.1, 3.0, profile["base_reaction_time"], 0.1)
                    rt_variability = st.number_input("RT Variability", 0.01, 0.5, profile["reaction_time_variability"], 0.01)

            with col2:
                st.write("**🎯 Performance**")
                if profile_type == "custom":
                    accuracy_rate = st.slider("Accuracy Rate", 0.0, 1.0, 0.8, 0.01)
                    attention_span = st.number_input("Attention Span (sec)", 60, 1800, 300)
                else:
                    profile = st.session_state.cognitive_profiles[profile_type]
                    accuracy_rate = st.slider("Accuracy Rate", 0.0, 1.0, profile["accuracy_rate"], 0.01)
                    attention_span = st.number_input("Attention Span (sec)", 60, 1800, int(profile["attention_span"]))

            with col3:
                st.write("**📈 Learning & Fatigue**")
                if profile_type == "custom":
                    fatigue_rate = st.number_input("Fatigue Rate", 0.0, 0.01, 0.002, 0.001)
                    learning_rate = st.number_input("Learning Rate", 0.0, 0.1, 0.015, 0.001)
                else:
                    profile = st.session_state.cognitive_profiles[profile_type]
                    fatigue_rate = st.number_input("Fatigue Rate", 0.0, 0.01, profile["fatigue_rate"], 0.001)
                    learning_rate = st.number_input("Learning Rate", 0.0, 0.1, profile["learning_rate"], 0.001)

        # Behavioral parameters
        with st.expander("🎭 Behavioral Parameters", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**🎲 Response Patterns**")
                error_patterns = st.multiselect(
                    "Error Types",
                    ["Random", "Systematic", "Fatigue-related", "Attention lapses"],
                    default=["Random", "Fatigue-related"]
                )

                speed_accuracy_tradeoff = st.slider(
                    "Speed-Accuracy Tradeoff",
                    0.0, 1.0, 0.5,
                    help="0=Accuracy focused, 1=Speed focused"
                )

            with col2:
                st.write("**🧠 Cognitive Biases**")
                cognitive_biases = st.multiselect(
                    "Simulated Biases",
                    ["Confirmation bias", "Anchoring", "Availability heuristic", "Recency effect"],
                    help="Cognitive biases to simulate"
                )

                individual_differences = st.slider(
                    "Individual Differences",
                    0.0, 1.0, 0.3,
                    help="How much should performance vary?"
                )

        submitted = st.form_submit_button("🧠 Create Experiment Participant", use_container_width=True)

        if submitted:
            create_experiment_participant_agent(
                agent_name, agent_goal, agent_backstory, profile_type,
                response_strategy, learning_enabled, base_rt, rt_variability,
                accuracy_rate, attention_span, fatigue_rate, learning_rate,
                error_patterns, speed_accuracy_tradeoff, cognitive_biases,
                individual_differences
            )

def show_experiment_designer_form():
    """Show experiment designer agent creation form"""
    with st.form("create_designer_agent_form"):
        st.subheader("🔬 Experiment Designer Agent")

        col1, col2 = st.columns(2)

        with col1:
            agent_name = st.text_input("Agent Name", placeholder="e.g., Experiment Designer")

            # Specialization areas
            specializations = st.multiselect(
                "Specialization Areas",
                ["Cognitive Psychology", "Social Psychology", "Developmental Psychology",
                 "Neuropsychology", "Psychophysics", "Human Factors"],
                default=["Cognitive Psychology"],
                help="Areas of experimental expertise"
            )

            # Design philosophy
            design_philosophy = st.selectbox(
                "Design Philosophy",
                ["Rigorous & Conservative", "Innovative & Exploratory", "Balanced", "Rapid Prototyping"],
                help="Approach to experimental design"
            )

        with col2:
            agent_goal = st.text_area(
                "Agent Goal",
                value="Design rigorous and innovative psychological experiments",
                help="What should this agent accomplish?"
            )

            agent_backstory = st.text_area(
                "Agent Backstory",
                value="You are an expert experimental psychologist with extensive experience in designing and conducting psychological research.",
                help="Background of the agent"
            )

        # Design parameters
        with st.expander("🔬 Design Parameters", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("**📊 Statistical Power**")
                min_power = st.slider("Minimum Power", 0.7, 0.99, 0.8)
                alpha_level = st.selectbox("Alpha Level", [0.05, 0.01, 0.001], index=0)
                effect_size_sensitivity = st.selectbox("Effect Size", ["Small", "Medium", "Large"], index=1)

            with col2:
                st.write("**⚖️ Design Principles**")
                counterbalancing = st.checkbox("Counterbalancing", value=True)
                randomization = st.checkbox("Randomization", value=True)
                control_conditions = st.checkbox("Control Conditions", value=True)
                blinding = st.checkbox("Blinding", value=False)

            with col3:
                st.write("**🎯 Optimization**")
                optimize_for = st.selectbox("Optimize For", ["Power", "Efficiency", "Validity", "Reliability"])
                complexity_preference = st.slider("Design Complexity", 0.0, 1.0, 0.5)
                innovation_level = st.slider("Innovation Level", 0.0, 1.0, 0.6)

        submitted = st.form_submit_button("🔬 Create Experiment Designer", use_container_width=True)

        if submitted:
            create_experiment_designer_agent(
                agent_name, agent_goal, agent_backstory, specializations,
                design_philosophy, min_power, alpha_level, effect_size_sensitivity,
                counterbalancing, randomization, control_conditions, blinding,
                optimize_for, complexity_preference, innovation_level
            )

def show_experiment_analyst_form():
    """Show experiment analyst agent creation form"""
    with st.form("create_analyst_agent_form"):
        st.subheader("📊 Experiment Analyst Agent")

        col1, col2 = st.columns(2)

        with col1:
            agent_name = st.text_input("Agent Name", placeholder="e.g., Statistical Analyst")

            # Statistical expertise
            statistical_methods = st.multiselect(
                "Statistical Methods",
                ["Descriptive Statistics", "T-tests", "ANOVA", "Regression", "Non-parametric",
                 "Bayesian", "Machine Learning", "Time Series", "Meta-analysis"],
                default=["Descriptive Statistics", "T-tests", "ANOVA"],
                help="Statistical methods the agent can use"
            )

            # Analysis style
            analysis_style = st.selectbox(
                "Analysis Style",
                ["Conservative", "Exploratory", "Comprehensive", "Focused"],
                help="Approach to data analysis"
            )

        with col2:
            agent_goal = st.text_area(
                "Agent Goal",
                value="Analyze experimental data and provide comprehensive statistical insights",
                help="What should this agent accomplish?"
            )

            agent_backstory = st.text_area(
                "Agent Backstory",
                value="You are a statistical expert specializing in psychological research analysis with deep knowledge of experimental design and data interpretation.",
                help="Background of the agent"
            )

        # Analysis parameters
        with st.expander("📈 Analysis Parameters", expanded=True):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("**🔍 Analysis Depth**")
                detail_level = st.selectbox("Detail Level", ["Basic", "Standard", "Comprehensive", "Exhaustive"])
                visualization_style = st.selectbox("Visualization", ["Minimal", "Standard", "Rich", "Interactive"])

            with col2:
                st.write("**⚠️ Quality Control**")
                outlier_detection = st.checkbox("Outlier Detection", value=True)
                assumption_checking = st.checkbox("Assumption Checking", value=True)
                effect_size_reporting = st.checkbox("Effect Size Reporting", value=True)

            with col3:
                st.write("**📝 Reporting**")
                report_style = st.selectbox("Report Style", ["Academic", "Business", "Technical", "Executive"])
                include_recommendations = st.checkbox("Include Recommendations", value=True)

        submitted = st.form_submit_button("📊 Create Experiment Analyst", use_container_width=True)

        if submitted:
            create_experiment_analyst_agent(
                agent_name, agent_goal, agent_backstory, statistical_methods,
                analysis_style, detail_level, visualization_style, outlier_detection,
                assumption_checking, effect_size_reporting, report_style,
                include_recommendations
            )

def create_standard_agent(agent_name, agent_role, agent_goal, agent_backstory, *args):
    """Create a standard agent"""
    try:
        if AGENTS_AVAILABLE:
            # Create agent based on role
            agent_classes = {
                'researcher': 'ResearcherAgent',
                'analyst': 'AnalystAgent',
                'writer': 'WriterAgent',
                'coder': 'CoderAgent'
            }

            agent_class_name = agent_classes.get(agent_role, 'BaseAgent')

            # Create agent info for session state
            agent_info = {
                'id': f"agent_{len(st.session_state.agents) + 1}",
                'name': agent_name,
                'role': agent_role,
                'goal': agent_goal,
                'backstory': agent_backstory,
                'type': 'standard',
                'created_at': time.time(),
                'status': 'active'
            }

            st.session_state.agents.append(agent_info)
            st.success(f"✅ Created standard agent: {agent_name}")
        else:
            # Simulation mode
            agent_info = {
                'id': f"agent_{len(st.session_state.agents) + 1}",
                'name': agent_name,
                'role': agent_role,
                'goal': agent_goal,
                'backstory': agent_backstory,
                'type': 'standard',
                'created_at': time.time(),
                'status': 'active'
            }

            st.session_state.agents.append(agent_info)
            st.success(f"✅ Created standard agent: {agent_name} (simulation mode)")

    except Exception as e:
        st.error(f"❌ Error creating agent: {str(e)}")

def create_experiment_participant_agent(agent_name, agent_goal, agent_backstory, profile_type, *args):
    """Create an experiment participant agent"""
    try:
        # Create cognitive profile
        if profile_type in st.session_state.cognitive_profiles:
            cognitive_profile = st.session_state.cognitive_profiles[profile_type].copy()
        else:
            cognitive_profile = {
                "base_reaction_time": args[5] if len(args) > 5 else 0.7,
                "reaction_time_variability": args[6] if len(args) > 6 else 0.15,
                "accuracy_rate": args[7] if len(args) > 7 else 0.8,
                "attention_span": args[8] if len(args) > 8 else 300.0,
                "fatigue_rate": args[9] if len(args) > 9 else 0.002,
                "learning_rate": args[10] if len(args) > 10 else 0.015
            }

        agent_info = {
            'id': f"agent_{len(st.session_state.agents) + 1}",
            'name': agent_name,
            'role': 'experiment_participant',
            'goal': agent_goal,
            'backstory': agent_backstory,
            'type': 'psychopy_participant',
            'cognitive_profile': cognitive_profile,
            'profile_type': profile_type,
            'response_strategy': args[1] if len(args) > 1 else 'human-like',
            'learning_enabled': args[2] if len(args) > 2 else True,
            'created_at': time.time(),
            'status': 'active'
        }

        st.session_state.agents.append(agent_info)
        st.success(f"🧠 Created experiment participant: {agent_name}")

    except Exception as e:
        st.error(f"❌ Error creating participant agent: {str(e)}")

def create_experiment_designer_agent(agent_name, agent_goal, agent_backstory, specializations, *args):
    """Create an experiment designer agent"""
    try:
        agent_info = {
            'id': f"agent_{len(st.session_state.agents) + 1}",
            'name': agent_name,
            'role': 'experiment_designer',
            'goal': agent_goal,
            'backstory': agent_backstory,
            'type': 'psychopy_designer',
            'specializations': specializations,
            'design_philosophy': args[0] if len(args) > 0 else 'Balanced',
            'created_at': time.time(),
            'status': 'active'
        }

        st.session_state.agents.append(agent_info)
        st.success(f"🔬 Created experiment designer: {agent_name}")

    except Exception as e:
        st.error(f"❌ Error creating designer agent: {str(e)}")

def create_experiment_analyst_agent(agent_name, agent_goal, agent_backstory, statistical_methods, *args):
    """Create an experiment analyst agent"""
    try:
        agent_info = {
            'id': f"agent_{len(st.session_state.agents) + 1}",
            'name': agent_name,
            'role': 'experiment_analyst',
            'goal': agent_goal,
            'backstory': agent_backstory,
            'type': 'psychopy_analyst',
            'statistical_methods': statistical_methods,
            'analysis_style': args[0] if len(args) > 0 else 'Standard',
            'created_at': time.time(),
            'status': 'active'
        }

        st.session_state.agents.append(agent_info)
        st.success(f"📊 Created experiment analyst: {agent_name}")

    except Exception as e:
        st.error(f"❌ Error creating analyst agent: {str(e)}")

def show_manage_agents_interface():
    """Show agent management interface"""
    st.header("📋 Manage Agents")

    if not st.session_state.agents:
        st.info("👥 No agents created yet. Create your first agent in the 'Create Agent' tab!")
        return

    # Agent filters
    col1, col2, col3 = st.columns(3)

    with col1:
        role_filter = st.selectbox(
            "Filter by Role",
            ["All"] + list(set(agent['role'] for agent in st.session_state.agents))
        )

    with col2:
        type_filter = st.selectbox(
            "Filter by Type",
            ["All"] + list(set(agent['type'] for agent in st.session_state.agents))
        )

    with col3:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "active", "inactive", "archived"]
        )

    # Filter agents
    filtered_agents = st.session_state.agents
    if role_filter != "All":
        filtered_agents = [a for a in filtered_agents if a['role'] == role_filter]
    if type_filter != "All":
        filtered_agents = [a for a in filtered_agents if a['type'] == type_filter]
    if status_filter != "All":
        filtered_agents = [a for a in filtered_agents if a['status'] == status_filter]

    # Display agents
    for i, agent in enumerate(filtered_agents):
        with st.expander(f"🤖 {agent['name']} ({agent['role']})", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**ID:** {agent['id']}")
                st.write(f"**Role:** {agent['role']}")
                st.write(f"**Type:** {agent['type']}")
                st.write(f"**Status:** {agent['status']}")
                st.write(f"**Created:** {time.ctime(agent['created_at'])}")

            with col2:
                st.write(f"**Goal:** {agent['goal']}")
                st.write(f"**Backstory:** {agent['backstory'][:100]}...")

                # Show type-specific info
                if agent['type'] == 'psychopy_participant':
                    st.write(f"**Profile:** {agent.get('profile_type', 'Unknown')}")
                    st.write(f"**Strategy:** {agent.get('response_strategy', 'Unknown')}")
                elif agent['type'] == 'psychopy_designer':
                    st.write(f"**Specializations:** {', '.join(agent.get('specializations', []))}")
                elif agent['type'] == 'psychopy_analyst':
                    st.write(f"**Methods:** {', '.join(agent.get('statistical_methods', []))}")

            # Action buttons
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                if st.button(f"Edit {agent['name']}", key=f"edit_{agent['id']}"):
                    st.info("Edit functionality coming soon!")

            with col2:
                if st.button(f"Clone {agent['name']}", key=f"clone_{agent['id']}"):
                    clone_agent(agent)

            with col3:
                new_status = "inactive" if agent['status'] == "active" else "active"
                if st.button(f"{'Deactivate' if agent['status'] == 'active' else 'Activate'}", key=f"toggle_{agent['id']}"):
                    toggle_agent_status(agent['id'], new_status)

            with col4:
                if st.button(f"Delete", key=f"delete_{agent['id']}", type="secondary"):
                    if st.button(f"Confirm Delete", key=f"confirm_delete_{agent['id']}", type="primary"):
                        delete_agent(agent['id'])

def clone_agent(agent):
    """Clone an existing agent"""
    cloned_agent = agent.copy()
    cloned_agent['id'] = f"agent_{len(st.session_state.agents) + 1}"
    cloned_agent['name'] = f"{agent['name']} (Copy)"
    cloned_agent['created_at'] = time.time()

    st.session_state.agents.append(cloned_agent)
    st.success(f"✅ Cloned agent: {cloned_agent['name']}")
    st.rerun()

def toggle_agent_status(agent_id, new_status):
    """Toggle agent status"""
    for agent in st.session_state.agents:
        if agent['id'] == agent_id:
            agent['status'] = new_status
            st.success(f"✅ Agent status updated to: {new_status}")
            st.rerun()
            break

def delete_agent(agent_id):
    """Delete an agent"""
    st.session_state.agents = [a for a in st.session_state.agents if a['id'] != agent_id]
    st.success("✅ Agent deleted successfully")
    st.rerun()

def show_cognitive_profiles_interface():
    """Show cognitive profiles management interface"""
    st.header("🧠 Cognitive Profiles")

    # Display existing profiles
    st.subheader("📋 Available Profiles")

    for profile_key, profile in st.session_state.cognitive_profiles.items():
        with st.expander(f"🧠 {profile['name']}", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Description:** {profile['description']}")
                st.write(f"**Base RT:** {profile['base_reaction_time']:.3f}s")
                st.write(f"**RT Variability:** {profile['reaction_time_variability']:.3f}")
                st.write(f"**Accuracy:** {profile['accuracy_rate']:.1%}")

            with col2:
                st.write(f"**Attention Span:** {profile['attention_span']:.0f}s")
                st.write(f"**Fatigue Rate:** {profile['fatigue_rate']:.4f}")
                st.write(f"**Learning Rate:** {profile['learning_rate']:.3f}")

                # Visualization
                st.write("**Performance Characteristics:**")
                performance_data = {
                    "Speed": 1 - (profile['base_reaction_time'] - 0.3) / 1.0,
                    "Accuracy": profile['accuracy_rate'],
                    "Consistency": 1 - profile['reaction_time_variability'],
                    "Endurance": 1 - profile['fatigue_rate'] * 1000,
                    "Adaptability": profile['learning_rate'] * 50
                }

                for metric, value in performance_data.items():
                    st.progress(max(0, min(1, value)), text=f"{metric}: {value:.1%}")

    # Create new profile
    st.subheader("➕ Create Custom Profile")

    with st.form("create_cognitive_profile"):
        col1, col2 = st.columns(2)

        with col1:
            profile_name = st.text_input("Profile Name", placeholder="e.g., Expert Performer")
            profile_description = st.text_area("Description", placeholder="Describe this cognitive profile...")

            base_rt = st.number_input("Base Reaction Time (s)", 0.1, 3.0, 0.7, 0.1)
            rt_variability = st.number_input("RT Variability", 0.01, 0.5, 0.15, 0.01)
            accuracy_rate = st.slider("Accuracy Rate", 0.0, 1.0, 0.8, 0.01)

        with col2:
            attention_span = st.number_input("Attention Span (s)", 60, 1800, 300)
            fatigue_rate = st.number_input("Fatigue Rate", 0.0, 0.01, 0.002, 0.001)
            learning_rate = st.number_input("Learning Rate", 0.0, 0.1, 0.015, 0.001)

            # Preview performance
            st.write("**Performance Preview:**")
            preview_data = {
                "Speed": 1 - (base_rt - 0.3) / 1.0,
                "Accuracy": accuracy_rate,
                "Consistency": 1 - rt_variability,
                "Endurance": 1 - fatigue_rate * 1000,
                "Adaptability": learning_rate * 50
            }

            for metric, value in preview_data.items():
                st.progress(max(0, min(1, value)), text=f"{metric}: {value:.1%}")

        if st.form_submit_button("🧠 Create Profile", use_container_width=True):
            if profile_name:
                profile_key = profile_name.lower().replace(" ", "_")
                new_profile = {
                    "name": profile_name,
                    "description": profile_description,
                    "base_reaction_time": base_rt,
                    "reaction_time_variability": rt_variability,
                    "accuracy_rate": accuracy_rate,
                    "attention_span": attention_span,
                    "fatigue_rate": fatigue_rate,
                    "learning_rate": learning_rate
                }

                st.session_state.cognitive_profiles[profile_key] = new_profile
                st.success(f"✅ Created cognitive profile: {profile_name}")
                st.rerun()
            else:
                st.error("❌ Please provide a profile name")

def show_templates_interface():
    """Show agent templates interface"""
    st.header("📚 Agent Templates")

    # Display available templates
    for template_key, template in st.session_state.agent_templates.items():
        with st.expander(f"📋 {template['name']}", expanded=False):
            st.write(f"**Description:** {template['description']}")
            st.write(f"**Number of Agents:** {len(template['agents'])}")

            # Show agents in template
            st.write("**Included Agents:**")
            for agent in template['agents']:
                st.write(f"• **{agent['name']}** ({agent['role']}) - {agent['goal']}")

            col1, col2 = st.columns(2)

            with col1:
                if st.button(f"Use Template", key=f"use_{template_key}"):
                    use_template(template)

            with col2:
                if st.button(f"Customize Template", key=f"customize_{template_key}"):
                    st.info("Template customization coming soon!")

    # Create custom template
    st.subheader("➕ Create Custom Template")

    with st.form("create_template"):
        template_name = st.text_input("Template Name", placeholder="e.g., My Research Team")
        template_description = st.text_area("Description", placeholder="Describe this template...")

        # Select agents to include
        if st.session_state.agents:
            selected_agents = st.multiselect(
                "Select Agents to Include",
                options=[f"{agent['name']} ({agent['role']})" for agent in st.session_state.agents],
                help="Choose existing agents to include in this template"
            )
        else:
            st.info("Create some agents first to include them in templates")
            selected_agents = []

        if st.form_submit_button("📚 Create Template", use_container_width=True):
            if template_name and selected_agents:
                # Create template from selected agents
                template_agents = []
                for agent_desc in selected_agents:
                    agent_name = agent_desc.split(" (")[0]
                    agent = next((a for a in st.session_state.agents if a['name'] == agent_name), None)
                    if agent:
                        template_agents.append({
                            "name": agent['name'],
                            "role": agent['role'],
                            "goal": agent['goal'],
                            "backstory": agent['backstory'],
                            "expertise_domains": agent.get('expertise_domains', [])
                        })

                template_key = template_name.lower().replace(" ", "_")
                new_template = {
                    "name": template_name,
                    "description": template_description,
                    "agents": template_agents
                }

                st.session_state.agent_templates[template_key] = new_template
                st.success(f"✅ Created template: {template_name}")
                st.rerun()
            else:
                st.error("❌ Please provide template name and select agents")

def use_template(template):
    """Use an agent template to create agents"""
    try:
        created_count = 0

        for agent_template in template['agents']:
            agent_info = {
                'id': f"agent_{len(st.session_state.agents) + 1}",
                'name': agent_template['name'],
                'role': agent_template['role'],
                'goal': agent_template['goal'],
                'backstory': agent_template['backstory'],
                'type': 'standard',
                'expertise_domains': agent_template.get('expertise_domains', []),
                'created_at': time.time(),
                'status': 'active'
            }

            st.session_state.agents.append(agent_info)
            created_count += 1

        st.success(f"✅ Created {created_count} agents from template: {template['name']}")
        st.rerun()

    except Exception as e:
        st.error(f"❌ Error using template: {str(e)}")

def show_advanced_tools_interface():
    """Show advanced tools interface"""
    st.header("🔧 Advanced Tools")

    # Agent analytics
    st.subheader("📊 Agent Analytics")

    if st.session_state.agents:
        # Agent distribution
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Agents by Role:**")
            roles = [agent['role'] for agent in st.session_state.agents]
            role_counts = {role: roles.count(role) for role in set(roles)}

            for role, count in role_counts.items():
                percentage = count / len(st.session_state.agents) * 100
                st.write(f"• {role.title()}: {count} ({percentage:.1f}%)")

        with col2:
            st.write("**Agents by Type:**")
            types = [agent['type'] for agent in st.session_state.agents]
            type_counts = {type_: types.count(type_) for type_ in set(types)}

            for type_, count in type_counts.items():
                percentage = count / len(st.session_state.agents) * 100
                st.write(f"• {type_.title()}: {count} ({percentage:.1f}%)")

        # Performance simulation
        st.subheader("🎯 Performance Simulation")

        if st.button("🚀 Run Performance Simulation"):
            run_performance_simulation()

    else:
        st.info("📈 Create some agents to see analytics and run simulations")

    # Bulk operations
    st.subheader("⚡ Bulk Operations")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📥 Export Agents", use_container_width=True):
            export_agents()

    with col2:
        uploaded_file = st.file_uploader("📤 Import Agents", type=['json'])
        if uploaded_file and st.button("Import", use_container_width=True):
            import_agents(uploaded_file)

    with col3:
        if st.button("🗑️ Clear All Agents", use_container_width=True, type="secondary"):
            if st.button("⚠️ Confirm Clear All", use_container_width=True, type="primary"):
                clear_all_agents()

def run_performance_simulation():
    """Run performance simulation for agents"""
    st.info("🔄 Running performance simulation...")

    # Simulate performance for each agent type
    results = {}

    for agent in st.session_state.agents:
        if agent['type'] == 'psychopy_participant':
            # Simulate cognitive performance
            profile = agent.get('cognitive_profile', {})
            base_rt = profile.get('base_reaction_time', 0.7)
            accuracy = profile.get('accuracy_rate', 0.8)

            # Add some randomness
            simulated_rt = base_rt + np.random.normal(0, 0.1)
            simulated_accuracy = accuracy + np.random.normal(0, 0.05)
            simulated_accuracy = max(0, min(1, simulated_accuracy))

            results[agent['name']] = {
                'reaction_time': simulated_rt,
                'accuracy': simulated_accuracy,
                'efficiency': simulated_accuracy / simulated_rt
            }

    if results:
        st.success("✅ Simulation completed!")

        # Display results
        st.write("**Simulation Results:**")
        for agent_name, metrics in results.items():
            with st.expander(f"🤖 {agent_name}"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Reaction Time", f"{metrics['reaction_time']:.3f}s")

                with col2:
                    st.metric("Accuracy", f"{metrics['accuracy']:.1%}")

                with col3:
                    st.metric("Efficiency", f"{metrics['efficiency']:.2f}")
    else:
        st.warning("⚠️ No experiment participant agents found for simulation")

def export_agents():
    """Export agents to JSON"""
    if st.session_state.agents:
        agents_data = {
            'agents': st.session_state.agents,
            'cognitive_profiles': st.session_state.cognitive_profiles,
            'templates': st.session_state.agent_templates,
            'export_time': time.time()
        }

        json_data = json.dumps(agents_data, indent=2, default=str)

        st.download_button(
            label="📥 Download Agents JSON",
            data=json_data,
            file_name=f"psychopy_agents_{int(time.time())}.json",
            mime="application/json"
        )

        st.success("✅ Agents exported successfully!")
    else:
        st.warning("⚠️ No agents to export")

def import_agents(uploaded_file):
    """Import agents from JSON"""
    try:
        agents_data = json.load(uploaded_file)

        if 'agents' in agents_data:
            st.session_state.agents.extend(agents_data['agents'])

        if 'cognitive_profiles' in agents_data:
            st.session_state.cognitive_profiles.update(agents_data['cognitive_profiles'])

        if 'templates' in agents_data:
            st.session_state.agent_templates.update(agents_data['templates'])

        st.success("✅ Agents imported successfully!")
        st.rerun()

    except Exception as e:
        st.error(f"❌ Error importing agents: {str(e)}")

def clear_all_agents():
    """Clear all agents"""
    st.session_state.agents = []
    st.success("✅ All agents cleared!")
    st.rerun()

if __name__ == "__main__":
    main()
