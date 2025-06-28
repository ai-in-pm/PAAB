#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder - Crews Page
Crew creation and management interface
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from crews.base import BaseCrew, ProcessType, CollaborationPattern

st.set_page_config(
    page_title="PAAB - Crews",
    page_icon="👥",
    layout="wide"
)

def main():
    """Main crews page"""
    
    st.title("👥 Crew Management")
    
    # Initialize session state
    if 'crews' not in st.session_state:
        st.session_state.crews = []
    if 'agents' not in st.session_state:
        st.session_state.agents = []
    if 'tasks' not in st.session_state:
        st.session_state.tasks = []
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        if st.button("🏠 Dashboard"):
            st.switch_page("main.py")
        if st.button("🤖 Agents"):
            st.switch_page("pages/🤖_Agents.py")
        if st.button("📝 Tasks"):
            st.switch_page("pages/📝_Tasks.py")
        if st.button("🚀 Execution"):
            st.switch_page("pages/🚀_Execution.py")
    
    # Main content
    tab1, tab2 = st.tabs(["➕ Create Crew", "📋 Manage Crews"])
    
    with tab1:
        st.header("Create New Crew")
        
        if not st.session_state.agents:
            st.warning("⚠️ No agents available. Please create agents first before creating a crew.")
            if st.button("🤖 Go to Agents Page"):
                st.switch_page("pages/🤖_Agents.py")
            return
        
        if not st.session_state.tasks:
            st.warning("⚠️ No tasks available. Please create tasks first before creating a crew.")
            if st.button("📝 Go to Tasks Page"):
                st.switch_page("pages/📝_Tasks.py")
            return
        
        # Crew creation form
        with st.form("create_crew_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                crew_name = st.text_input("Crew Name", placeholder="e.g., Research Team Alpha")
                crew_description = st.text_area(
                    "Crew Description",
                    placeholder="Describe the purpose and goals of this crew...",
                    help="What is this crew designed to accomplish?"
                )
                
            with col2:
                process_type = st.selectbox(
                    "Process Type",
                    [p.value for p in ProcessType],
                    help="How should tasks be executed?"
                )
                
                collaboration_pattern = st.selectbox(
                    "Collaboration Pattern",
                    [c.value for c in CollaborationPattern],
                    help="How should agents collaborate?"
                )
            
            # Agent selection
            st.subheader("👥 Select Agents")
            selected_agents = st.multiselect(
                "Crew Members",
                [f"{agent['name']} ({agent['role']})" for agent in st.session_state.agents],
                help="Select agents to include in this crew"
            )
            
            # Task selection
            st.subheader("📝 Select Tasks")
            selected_tasks = st.multiselect(
                "Crew Tasks",
                [f"{task['name']} ({task['task_type']})" for task in st.session_state.tasks],
                help="Select tasks for this crew to execute"
            )
            
            # Advanced settings
            with st.expander("🔧 Advanced Settings"):
                verbose = st.checkbox("Verbose Logging", value=True, help="Enable detailed logging")
                planning_enabled = st.checkbox("Planning Enabled", value=True, help="Enable automatic planning")
                monitoring_enabled = st.checkbox("Monitoring Enabled", value=True, help="Enable performance monitoring")
                memory_enabled = st.checkbox("Memory Enabled", value=True, help="Enable crew memory")
                
                max_execution_time = st.number_input("Max Execution Time (seconds)", min_value=300, max_value=14400, value=3600)
                max_iterations = st.number_input("Max Iterations", min_value=1, max_value=100, value=25)
            
            submitted = st.form_submit_button("🚀 Create Crew", use_container_width=True)
            
            if submitted:
                if crew_name and selected_agents and selected_tasks:
                    try:
                        # Get selected agent objects
                        crew_agents = []
                        for agent_name in selected_agents:
                            agent_name_clean = agent_name.split(" (")[0]
                            for agent in st.session_state.agents:
                                if agent['name'] == agent_name_clean:
                                    crew_agents.append(agent['agent_object'])
                                    break
                        
                        # Get selected task objects
                        crew_tasks = []
                        for task_name in selected_tasks:
                            task_name_clean = task_name.split(" (")[0]
                            for task in st.session_state.tasks:
                                if task['name'] == task_name_clean:
                                    crew_tasks.append(task['task_object'])
                                    break
                        
                        # Create crew
                        crew = BaseCrew(
                            name=crew_name,
                            agents=crew_agents,
                            tasks=crew_tasks,
                            process_type=ProcessType(process_type),
                            collaboration_pattern=CollaborationPattern(collaboration_pattern),
                            verbose=verbose,
                            planning_enabled=planning_enabled,
                            monitoring_enabled=monitoring_enabled,
                            memory_enabled=memory_enabled
                        )
                        
                        # Store crew info
                        crew_info = {
                            'id': crew.id,
                            'name': crew_name,
                            'description': crew_description,
                            'process_type': process_type,
                            'collaboration_pattern': collaboration_pattern,
                            'agents': selected_agents,
                            'tasks': selected_tasks,
                            'verbose': verbose,
                            'planning_enabled': planning_enabled,
                            'monitoring_enabled': monitoring_enabled,
                            'memory_enabled': memory_enabled,
                            'max_execution_time': max_execution_time,
                            'max_iterations': max_iterations,
                            'created_at': crew.created_at,
                            'crew_object': crew
                        }
                        
                        st.session_state.crews.append(crew_info)
                        st.success(f"✅ Crew '{crew_name}' created successfully!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Error creating crew: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
                else:
                    st.error("❌ Please fill in all required fields and select at least one agent and one task")
    
    with tab2:
        st.header("Manage Existing Crews")
        
        if st.session_state.crews:
            # Crew list
            for i, crew_info in enumerate(st.session_state.crews):
                with st.expander(f"👥 {crew_info['name']} ({crew_info['process_type']})"):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.write(f"**Description:** {crew_info['description']}")
                        st.write(f"**Process Type:** {crew_info['process_type']}")
                        st.write(f"**Collaboration:** {crew_info['collaboration_pattern']}")
                        
                    with col2:
                        st.write(f"**ID:** {crew_info['id'][:8]}...")
                        st.write(f"**Created:** {crew_info['created_at']}")
                        st.write(f"**Agents:** {len(crew_info['agents'])}")
                        st.write(f"**Tasks:** {len(crew_info['tasks'])}")
                    
                    with col3:
                        if st.button(f"🚀 Execute", key=f"execute_crew_{i}"):
                            st.session_state.selected_crew = crew_info
                            st.switch_page("pages/🚀_Execution.py")
                        
                        if st.button(f"✏️ Edit", key=f"edit_crew_{i}"):
                            st.info("Edit functionality coming soon!")
                        
                        if st.button(f"🗑️ Delete", key=f"delete_crew_{i}"):
                            st.session_state.crews.pop(i)
                            st.rerun()
                    
                    # Show crew details
                    with st.expander("📋 Crew Details"):
                        st.write("**Agents:**")
                        for agent in crew_info['agents']:
                            st.write(f"  • {agent}")
                        
                        st.write("**Tasks:**")
                        for task in crew_info['tasks']:
                            st.write(f"  • {task}")
                        
                        st.write("**Settings:**")
                        st.write(f"  • Verbose: {crew_info['verbose']}")
                        st.write(f"  • Planning: {crew_info['planning_enabled']}")
                        st.write(f"  • Monitoring: {crew_info['monitoring_enabled']}")
                        st.write(f"  • Memory: {crew_info['memory_enabled']}")
            
            # Bulk actions
            st.subheader("🔧 Bulk Actions")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📤 Export All Crews"):
                    st.info("Export functionality coming soon!")
            
            with col2:
                if st.button("📥 Import Crews"):
                    st.info("Import functionality coming soon!")
            
            with col3:
                if st.button("🗑️ Clear All Crews"):
                    if st.button("⚠️ Confirm Delete All"):
                        st.session_state.crews = []
                        st.rerun()
        else:
            st.info("No crews created yet. Use the 'Create Crew' tab to get started!")
            
            # Quick start templates
            st.subheader("🚀 Quick Start Templates")
            
            if st.session_state.agents and st.session_state.tasks:
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📊 Research Crew Template", use_container_width=True):
                        # Create a research crew with available agents and tasks
                        research_agents = [agent for agent in st.session_state.agents if agent['role'] in ['researcher', 'analyst', 'writer']]
                        research_tasks = [task for task in st.session_state.tasks if task['task_type'] in ['research', 'analysis', 'writing']]
                        
                        if research_agents and research_tasks:
                            crew_agents = [agent['agent_object'] for agent in research_agents[:3]]  # Take first 3
                            crew_tasks = [task['task_object'] for task in research_tasks[:3]]  # Take first 3
                            
                            crew = BaseCrew(
                                name="Research Crew Template",
                                agents=crew_agents,
                                tasks=crew_tasks,
                                process_type=ProcessType.SEQUENTIAL,
                                collaboration_pattern=CollaborationPattern.PEER_TO_PEER,
                                verbose=True,
                                planning_enabled=True,
                                monitoring_enabled=True,
                                memory_enabled=True
                            )
                            
                            crew_info = {
                                'id': crew.id,
                                'name': "Research Crew Template",
                                'description': "Template crew for research workflows",
                                'process_type': ProcessType.SEQUENTIAL.value,
                                'collaboration_pattern': CollaborationPattern.PEER_TO_PEER.value,
                                'agents': [f"{agent['name']} ({agent['role']})" for agent in research_agents[:3]],
                                'tasks': [f"{task['name']} ({task['task_type']})" for task in research_tasks[:3]],
                                'verbose': True,
                                'planning_enabled': True,
                                'monitoring_enabled': True,
                                'memory_enabled': True,
                                'max_execution_time': 3600,
                                'max_iterations': 25,
                                'created_at': crew.created_at,
                                'crew_object': crew
                            }
                            
                            st.session_state.crews.append(crew_info)
                            st.success("✅ Research crew template created!")
                            st.rerun()
                        else:
                            st.warning("Need research-related agents and tasks to create template")
                
                with col2:
                    if st.button("💻 Development Crew Template", use_container_width=True):
                        # Create a development crew with available agents and tasks
                        dev_agents = [agent for agent in st.session_state.agents if agent['role'] in ['coder', 'analyst', 'writer']]
                        dev_tasks = [task for task in st.session_state.tasks if task['task_type'] in ['coding', 'analysis', 'review']]
                        
                        if dev_agents and dev_tasks:
                            crew_agents = [agent['agent_object'] for agent in dev_agents[:3]]  # Take first 3
                            crew_tasks = [task['task_object'] for task in dev_tasks[:3]]  # Take first 3
                            
                            crew = BaseCrew(
                                name="Development Crew Template",
                                agents=crew_agents,
                                tasks=crew_tasks,
                                process_type=ProcessType.SEQUENTIAL,
                                collaboration_pattern=CollaborationPattern.LEADER_FOLLOWER,
                                verbose=True,
                                planning_enabled=True,
                                monitoring_enabled=True,
                                memory_enabled=True
                            )
                            
                            crew_info = {
                                'id': crew.id,
                                'name': "Development Crew Template",
                                'description': "Template crew for development workflows",
                                'process_type': ProcessType.SEQUENTIAL.value,
                                'collaboration_pattern': CollaborationPattern.LEADER_FOLLOWER.value,
                                'agents': [f"{agent['name']} ({agent['role']})" for agent in dev_agents[:3]],
                                'tasks': [f"{task['name']} ({task['task_type']})" for task in dev_tasks[:3]],
                                'verbose': True,
                                'planning_enabled': True,
                                'monitoring_enabled': True,
                                'memory_enabled': True,
                                'max_execution_time': 3600,
                                'max_iterations': 25,
                                'created_at': crew.created_at,
                                'crew_object': crew
                            }
                            
                            st.session_state.crews.append(crew_info)
                            st.success("✅ Development crew template created!")
                            st.rerun()
                        else:
                            st.warning("Need development-related agents and tasks to create template")
            else:
                st.info("Create some agents and tasks first to use crew templates!")

if __name__ == "__main__":
    main()
