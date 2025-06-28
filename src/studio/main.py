#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder - Visual Studio Interface
Streamlit-based visual interface for building AI agent crews
"""

import streamlit as st
import sys
from pathlib import Path
import json
import yaml
from typing import Dict, Any, List, Optional
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.specialized import ResearcherAgent, AnalystAgent, WriterAgent, CoderAgent
from tasks.base import BaseTask, TaskType, TaskPriority
from crews.base import BaseCrew, ProcessType, CollaborationPattern
from runtime.executor import RuntimeExecutor, ExecutionMode

# Configure Streamlit page
st.set_page_config(
    page_title="PsychoPy AI Agent Builder Studio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'agents' not in st.session_state:
    st.session_state.agents = []
if 'tasks' not in st.session_state:
    st.session_state.tasks = []
if 'crews' not in st.session_state:
    st.session_state.crews = []
if 'executor' not in st.session_state:
    st.session_state.executor = None
if 'execution_results' not in st.session_state:
    st.session_state.execution_results = []


def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🤖 PsychoPy AI Agent Builder Studio")
    st.markdown("**Build powerful AI agent teams with visual drag-and-drop interface**")
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Dashboard", "🤖 Agents", "📝 Tasks", "👥 Crews", "🚀 Execution", "📊 Monitoring", "⚙️ Settings"]
    )
    
    # Route to appropriate page
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🤖 Agents":
        show_agents_page()
    elif page == "📝 Tasks":
        show_tasks_page()
    elif page == "👥 Crews":
        show_crews_page()
    elif page == "🚀 Execution":
        show_execution_page()
    elif page == "📊 Monitoring":
        show_monitoring_page()
    elif page == "⚙️ Settings":
        show_settings_page()


def show_dashboard():
    """Show the main dashboard"""
    
    st.header("📊 Dashboard")
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Agents", len(st.session_state.agents))
    
    with col2:
        st.metric("Tasks", len(st.session_state.tasks))
    
    with col3:
        st.metric("Crews", len(st.session_state.crews))
    
    with col4:
        st.metric("Executions", len(st.session_state.execution_results))
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("➕ Create Agent", use_container_width=True):
            st.switch_page("pages/🤖_Agents.py")

    with col2:
        if st.button("📝 Create Task", use_container_width=True):
            st.switch_page("pages/📝_Tasks.py")

    with col3:
        if st.button("👥 Create Crew", use_container_width=True):
            st.switch_page("pages/👥_Crews.py")

    # PsychoPy Integration Section
    st.subheader("🧠 PsychoPy Integration")
    st.markdown("*Psychological experiments meet AI agent intelligence*")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🧪 PsychoPy Experiments", use_container_width=True):
            st.switch_page("pages/🧠_PsychoPy_Experiments.py")

    with col2:
        if st.button("🔬 Design Experiment", use_container_width=True):
            st.info("🧪 Create psychological experiments with AI agents as participants")

    with col3:
        if st.button("📊 Analyze Results", use_container_width=True):
            st.info("📈 Statistical analysis of experimental data")

    # Recent activity
    st.subheader("📈 Recent Activity")
    
    if st.session_state.execution_results:
        for result in st.session_state.execution_results[-5:]:
            with st.expander(f"Execution: {result.get('crew_name', 'Unknown')} - {result.get('status', 'Unknown')}"):
                st.json(result)
    else:
        st.info("No recent executions. Create a crew and run it to see activity here.")
    
    # Getting started guide
    if not st.session_state.agents and not st.session_state.tasks and not st.session_state.crews:
        st.subheader("🎯 Getting Started")
        st.markdown("""
        Welcome to PsychoPy AI Agent Builder Studio! Here's how to get started:
        
        1. **Create Agents** 🤖 - Define specialized AI agents with specific roles and capabilities
        2. **Define Tasks** 📝 - Create tasks that describe what needs to be accomplished
        3. **Build Crews** 👥 - Combine agents and tasks into collaborative teams
        4. **Execute** 🚀 - Run your crews and monitor their performance
        
        Click on the navigation menu to begin building your first AI agent team!
        """)


def show_agents_page():
    """Show the agents management page"""
    
    st.header("🤖 Agent Management")
    
    # Create new agent section
    with st.expander("➕ Create New Agent", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            agent_name = st.text_input("Agent Name", placeholder="e.g., Research Specialist")
            agent_role = st.selectbox("Agent Role", ["researcher", "analyst", "writer", "coder"])
            agent_goal = st.text_area("Agent Goal", placeholder="Describe what this agent should accomplish...")
        
        with col2:
            agent_backstory = st.text_area("Agent Backstory", placeholder="Describe the agent's background and expertise...")
            agent_domains = st.text_input("Expertise Domains (comma-separated)", placeholder="e.g., AI, machine learning, data science")
        
        if st.button("Create Agent", type="primary"):
            if agent_name and agent_goal and agent_backstory:
                # Create agent based on role
                agent_classes = {
                    'researcher': ResearcherAgent,
                    'analyst': AnalystAgent,
                    'writer': WriterAgent,
                    'coder': CoderAgent
                }
                
                agent_class = agent_classes[agent_role]
                domains = [d.strip() for d in agent_domains.split(",")] if agent_domains else []
                
                agent = agent_class(
                    goal=agent_goal,
                    backstory=agent_backstory,
                    expertise_domains=domains if agent_role == 'researcher' else None
                )
                
                # Store agent info
                agent_info = {
                    'name': agent_name,
                    'role': agent_role,
                    'goal': agent_goal,
                    'backstory': agent_backstory,
                    'domains': domains,
                    'id': agent.id,
                    'object': agent
                }
                
                st.session_state.agents.append(agent_info)
                st.success(f"✅ Agent '{agent_name}' created successfully!")
                st.rerun()
            else:
                st.error("Please fill in all required fields.")
    
    # Display existing agents
    st.subheader("📋 Existing Agents")
    
    if st.session_state.agents:
        for i, agent_info in enumerate(st.session_state.agents):
            with st.expander(f"{agent_info['name']} ({agent_info['role']})"):
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.write(f"**Goal:** {agent_info['goal']}")
                    st.write(f"**Backstory:** {agent_info['backstory']}")
                
                with col2:
                    st.write(f"**Role:** {agent_info['role']}")
                    st.write(f"**ID:** {agent_info['id'][:8]}...")
                    if agent_info.get('domains'):
                        st.write(f"**Domains:** {', '.join(agent_info['domains'])}")
                
                with col3:
                    if st.button(f"🗑️ Delete", key=f"delete_agent_{i}"):
                        st.session_state.agents.pop(i)
                        st.rerun()
    else:
        st.info("No agents created yet. Create your first agent above!")


def show_tasks_page():
    """Show the tasks management page"""
    
    st.header("📝 Task Management")
    
    # Create new task section
    with st.expander("➕ Create New Task", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            task_description = st.text_area("Task Description", placeholder="Describe what needs to be accomplished...")
            task_expected_output = st.text_area("Expected Output", placeholder="Describe the expected result...")
        
        with col2:
            task_type = st.selectbox("Task Type", ["research", "analysis", "writing", "coding", "custom"])
            task_priority = st.selectbox("Priority", ["low", "medium", "high", "critical"])
            assigned_agent = st.selectbox("Assign to Agent", ["None"] + [a['name'] for a in st.session_state.agents])
        
        if st.button("Create Task", type="primary"):
            if task_description and task_expected_output:
                # Map string values to enums
                task_type_map = {
                    'research': TaskType.RESEARCH,
                    'analysis': TaskType.ANALYSIS,
                    'writing': TaskType.WRITING,
                    'coding': TaskType.CODING,
                    'custom': TaskType.CUSTOM
                }
                
                priority_map = {
                    'low': TaskPriority.LOW,
                    'medium': TaskPriority.MEDIUM,
                    'high': TaskPriority.HIGH,
                    'critical': TaskPriority.CRITICAL
                }
                
                task = BaseTask(
                    description=task_description,
                    expected_output=task_expected_output,
                    task_type=task_type_map[task_type],
                    priority=priority_map[task_priority]
                )
                
                # Assign agent if selected
                if assigned_agent != "None":
                    agent_info = next(a for a in st.session_state.agents if a['name'] == assigned_agent)
                    task.assign_agent(agent_info['object'])
                
                # Store task info
                task_info = {
                    'description': task_description,
                    'expected_output': task_expected_output,
                    'type': task_type,
                    'priority': task_priority,
                    'assigned_agent': assigned_agent if assigned_agent != "None" else None,
                    'id': task.id,
                    'object': task
                }
                
                st.session_state.tasks.append(task_info)
                st.success(f"✅ Task created successfully!")
                st.rerun()
            else:
                st.error("Please fill in all required fields.")
    
    # Display existing tasks
    st.subheader("📋 Existing Tasks")
    
    if st.session_state.tasks:
        for i, task_info in enumerate(st.session_state.tasks):
            with st.expander(f"Task {i+1}: {task_info['type'].title()} ({task_info['priority']} priority)"):
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.write(f"**Description:** {task_info['description']}")
                    st.write(f"**Expected Output:** {task_info['expected_output']}")
                
                with col2:
                    st.write(f"**Type:** {task_info['type']}")
                    st.write(f"**Priority:** {task_info['priority']}")
                    st.write(f"**Assigned Agent:** {task_info['assigned_agent'] or 'Unassigned'}")
                    st.write(f"**ID:** {task_info['id'][:8]}...")
                
                with col3:
                    if st.button(f"🗑️ Delete", key=f"delete_task_{i}"):
                        st.session_state.tasks.pop(i)
                        st.rerun()
    else:
        st.info("No tasks created yet. Create your first task above!")


def show_crews_page():
    """Show the crews management page"""
    
    st.header("👥 Crew Management")
    
    # Create new crew section
    with st.expander("➕ Create New Crew", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            crew_name = st.text_input("Crew Name", placeholder="e.g., Research Team")
            crew_process = st.selectbox("Process Type", ["sequential", "parallel", "hierarchical"])
            crew_collaboration = st.selectbox("Collaboration Pattern", ["peer_to_peer", "leader_follower", "round_robin"])
        
        with col2:
            selected_agents = st.multiselect("Select Agents", [a['name'] for a in st.session_state.agents])
            selected_tasks = st.multiselect("Select Tasks", [f"Task {i+1}: {t['type']}" for i, t in enumerate(st.session_state.tasks)])
        
        if st.button("Create Crew", type="primary"):
            if crew_name and selected_agents and selected_tasks:
                # Get selected agent objects
                crew_agents = []
                for agent_name in selected_agents:
                    agent_info = next(a for a in st.session_state.agents if a['name'] == agent_name)
                    crew_agents.append(agent_info['object'])
                
                # Get selected task objects
                crew_tasks = []
                for task_desc in selected_tasks:
                    task_index = int(task_desc.split(":")[0].split()[1]) - 1
                    crew_tasks.append(st.session_state.tasks[task_index]['object'])
                
                # Map string values to enums
                process_map = {
                    'sequential': ProcessType.SEQUENTIAL,
                    'parallel': ProcessType.PARALLEL,
                    'hierarchical': ProcessType.HIERARCHICAL
                }
                
                collaboration_map = {
                    'peer_to_peer': CollaborationPattern.PEER_TO_PEER,
                    'leader_follower': CollaborationPattern.LEADER_FOLLOWER,
                    'round_robin': CollaborationPattern.ROUND_ROBIN
                }
                
                crew = BaseCrew(
                    name=crew_name,
                    agents=crew_agents,
                    tasks=crew_tasks,
                    process_type=process_map[crew_process],
                    collaboration_pattern=collaboration_map[crew_collaboration]
                )
                
                # Store crew info
                crew_info = {
                    'name': crew_name,
                    'process_type': crew_process,
                    'collaboration_pattern': crew_collaboration,
                    'agents': selected_agents,
                    'tasks': selected_tasks,
                    'id': crew.id,
                    'object': crew
                }
                
                st.session_state.crews.append(crew_info)
                st.success(f"✅ Crew '{crew_name}' created successfully!")
                st.rerun()
            else:
                st.error("Please fill in all required fields and select at least one agent and task.")
    
    # Display existing crews
    st.subheader("📋 Existing Crews")
    
    if st.session_state.crews:
        for i, crew_info in enumerate(st.session_state.crews):
            with st.expander(f"{crew_info['name']} ({crew_info['process_type']})"):
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.write(f"**Process Type:** {crew_info['process_type']}")
                    st.write(f"**Collaboration:** {crew_info['collaboration_pattern']}")
                    st.write(f"**Agents:** {', '.join(crew_info['agents'])}")
                
                with col2:
                    st.write(f"**Tasks:** {len(crew_info['tasks'])}")
                    for task in crew_info['tasks']:
                        st.write(f"  • {task}")
                    st.write(f"**ID:** {crew_info['id'][:8]}...")
                
                with col3:
                    if st.button(f"🚀 Execute", key=f"execute_crew_{i}"):
                        st.session_state.selected_crew = crew_info
                        st.switch_page("🚀 Execution")
                    
                    if st.button(f"🗑️ Delete", key=f"delete_crew_{i}"):
                        st.session_state.crews.pop(i)
                        st.rerun()
    else:
        st.info("No crews created yet. Create your first crew above!")


def show_execution_page():
    """Show the execution page"""
    
    st.header("🚀 Crew Execution")
    
    if not st.session_state.crews:
        st.warning("No crews available for execution. Create a crew first!")
        return
    
    # Select crew to execute
    selected_crew_name = st.selectbox("Select Crew to Execute", [c['name'] for c in st.session_state.crews])
    crew_info = next(c for c in st.session_state.crews if c['name'] == selected_crew_name)
    
    # Execution configuration
    col1, col2 = st.columns(2)
    
    with col1:
        execution_mode = st.selectbox("Execution Mode", ["asynchronous", "synchronous", "parallel"])
        timeout = st.number_input("Timeout (seconds)", min_value=60, max_value=3600, value=600)
    
    with col2:
        input_data = st.text_area("Input Data (JSON)", placeholder='{"key": "value"}', height=100)
    
    # Execute button
    if st.button("🚀 Execute Crew", type="primary"):
        try:
            # Parse input data
            inputs = json.loads(input_data) if input_data.strip() else {}
            
            # Initialize executor if not exists
            if st.session_state.executor is None:
                st.session_state.executor = RuntimeExecutor()
                st.session_state.executor.start()
            
            # Execute crew
            mode_map = {
                'synchronous': ExecutionMode.SYNCHRONOUS,
                'asynchronous': ExecutionMode.ASYNCHRONOUS,
                'parallel': ExecutionMode.PARALLEL
            }
            
            with st.spinner("Executing crew..."):
                result = st.session_state.executor.execute_crew(
                    crew=crew_info['object'],
                    inputs=inputs,
                    mode=mode_map[execution_mode],
                    timeout=timeout
                )
                
                # Handle async result
                if execution_mode != 'synchronous':
                    result = result.result()  # Wait for completion
                
                # Store result
                result_info = {
                    'crew_name': crew_info['name'],
                    'execution_mode': execution_mode,
                    'success': result.success,
                    'execution_time': result.execution_time,
                    'outputs': result.outputs,
                    'errors': result.errors,
                    'timestamp': time.time()
                }
                
                st.session_state.execution_results.append(result_info)
                
                # Display result
                if result.success:
                    st.success(f"✅ Crew executed successfully in {result.execution_time:.2f} seconds!")
                    
                    with st.expander("📋 Execution Results", expanded=True):
                        st.json(result.outputs)
                else:
                    st.error("❌ Crew execution failed!")
                    st.error(f"Errors: {result.errors}")
        
        except json.JSONDecodeError:
            st.error("Invalid JSON in input data!")
        except Exception as e:
            st.error(f"Execution error: {str(e)}")
    
    # Display recent executions
    st.subheader("📈 Recent Executions")
    
    if st.session_state.execution_results:
        for result in reversed(st.session_state.execution_results[-10:]):
            status_icon = "✅" if result['success'] else "❌"
            with st.expander(f"{status_icon} {result['crew_name']} - {time.ctime(result['timestamp'])}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Mode:** {result['execution_mode']}")
                    st.write(f"**Success:** {result['success']}")
                    st.write(f"**Execution Time:** {result['execution_time']:.2f}s")
                
                with col2:
                    if result['errors']:
                        st.write(f"**Errors:** {result['errors']}")
                    
                    if result['outputs']:
                        st.write("**Outputs:**")
                        st.json(result['outputs'])
    else:
        st.info("No executions yet. Execute a crew to see results here!")


def show_monitoring_page():
    """Show the monitoring page"""
    
    st.header("📊 Monitoring & Analytics")
    
    if st.session_state.executor:
        # Get performance metrics
        metrics = st.session_state.executor.get_performance_metrics()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Executions", metrics.get('total_executions', 0))
        
        with col2:
            st.metric("Successful", metrics.get('successful_executions', 0))
        
        with col3:
            st.metric("Failed", metrics.get('failed_executions', 0))
        
        with col4:
            st.metric("Active Sessions", metrics.get('active_sessions', 0))
        
        # Performance charts
        st.subheader("📈 Performance Trends")
        
        if st.session_state.execution_results:
            import pandas as pd
            import plotly.express as px
            
            # Create DataFrame from execution results
            df = pd.DataFrame(st.session_state.execution_results)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Execution time trend
            fig = px.line(df, x='timestamp', y='execution_time', 
                         title='Execution Time Trend',
                         labels={'execution_time': 'Execution Time (seconds)'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Success rate
            success_rate = df['success'].mean() * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        else:
            st.info("No execution data available for monitoring.")
    
    else:
        st.info("Runtime executor not initialized. Execute a crew to start monitoring.")


def show_settings_page():
    """Show the settings page"""
    
    st.header("⚙️ Settings")
    
    # Runtime settings
    st.subheader("🔧 Runtime Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_concurrent_crews = st.number_input("Max Concurrent Crews", min_value=1, max_value=50, value=10)
        max_workers = st.number_input("Max Workers", min_value=1, max_value=100, value=50)
    
    with col2:
        enable_monitoring = st.checkbox("Enable Monitoring", value=True)
        enable_security = st.checkbox("Enable Security", value=True)
    
    # Export/Import
    st.subheader("📁 Export/Import")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📤 Export Configuration"):
            config = {
                'agents': [
                    {k: v for k, v in agent.items() if k != 'object'}
                    for agent in st.session_state.agents
                ],
                'tasks': [
                    {k: v for k, v in task.items() if k != 'object'}
                    for task in st.session_state.tasks
                ],
                'crews': [
                    {k: v for k, v in crew.items() if k != 'object'}
                    for crew in st.session_state.crews
                ]
            }
            
            st.download_button(
                label="Download Configuration",
                data=yaml.dump(config, default_flow_style=False),
                file_name="paab_config.yaml",
                mime="text/yaml"
            )
    
    with col2:
        uploaded_file = st.file_uploader("📥 Import Configuration", type=['yaml', 'yml'])
        if uploaded_file:
            try:
                config = yaml.safe_load(uploaded_file)
                st.success("Configuration loaded successfully!")
                st.json(config)
            except Exception as e:
                st.error(f"Error loading configuration: {str(e)}")
    
    # Clear data
    st.subheader("🗑️ Data Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Clear Agents", type="secondary"):
            st.session_state.agents = []
            st.rerun()
    
    with col2:
        if st.button("Clear Tasks", type="secondary"):
            st.session_state.tasks = []
            st.rerun()
    
    with col3:
        if st.button("Clear All Data", type="secondary"):
            st.session_state.agents = []
            st.session_state.tasks = []
            st.session_state.crews = []
            st.session_state.execution_results = []
            st.rerun()


def launch_studio():
    """Launch the studio interface"""
    main()


if __name__ == "__main__":
    main()
