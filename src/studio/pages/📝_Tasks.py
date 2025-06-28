#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder - Tasks Page
Task creation and management interface
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tasks.base import BaseTask, TaskType, TaskPriority

st.set_page_config(
    page_title="PAAB - Tasks",
    page_icon="📝",
    layout="wide"
)

def main():
    """Main tasks page"""
    
    st.title("📝 Task Management")
    
    # Initialize session state
    if 'tasks' not in st.session_state:
        st.session_state.tasks = []
    if 'agents' not in st.session_state:
        st.session_state.agents = []
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        if st.button("🏠 Dashboard"):
            st.switch_page("main.py")
        if st.button("🤖 Agents"):
            st.switch_page("pages/🤖_Agents.py")
        if st.button("👥 Crews"):
            st.switch_page("pages/👥_Crews.py")
        if st.button("🚀 Execution"):
            st.switch_page("pages/🚀_Execution.py")
    
    # Main content
    tab1, tab2 = st.tabs(["➕ Create Task", "📋 Manage Tasks"])
    
    with tab1:
        st.header("Create New Task")
        
        # Task creation form
        with st.form("create_task_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                task_name = st.text_input("Task Name", placeholder="e.g., Market Research Analysis")
                task_type = st.selectbox(
                    "Task Type",
                    [t.value for t in TaskType],
                    help="Select the type of task"
                )
                task_priority = st.selectbox(
                    "Priority",
                    [p.value for p in TaskPriority],
                    help="Task priority level"
                )
                
            with col2:
                task_description = st.text_area(
                    "Task Description",
                    placeholder="Describe what needs to be accomplished...",
                    help="Detailed description of the task"
                )
                
                expected_output = st.text_area(
                    "Expected Output",
                    placeholder="Describe the expected deliverable...",
                    help="What should the task produce?"
                )
            
            # Agent assignment
            if st.session_state.agents:
                assigned_agent = st.selectbox(
                    "Assign to Agent",
                    ["None"] + [f"{agent['name']} ({agent['role']})" for agent in st.session_state.agents],
                    help="Assign this task to a specific agent"
                )
            else:
                st.info("No agents available. Create agents first to assign tasks.")
                assigned_agent = "None"
            
            # Advanced settings
            with st.expander("🔧 Advanced Settings"):
                max_execution_time = st.number_input("Max Execution Time (seconds)", min_value=60, max_value=7200, value=1800)
                retry_count = st.number_input("Retry Count", min_value=0, max_value=5, value=3)
                
                # Dependencies
                if st.session_state.tasks:
                    dependencies = st.multiselect(
                        "Task Dependencies",
                        [f"{task['name']} ({task['id'][:8]})" for task in st.session_state.tasks],
                        help="Tasks that must complete before this task can start"
                    )
                else:
                    dependencies = []
                
                # Context and metadata
                context = st.text_area("Additional Context", placeholder="Any additional context or requirements...")
                tags = st.text_input("Tags", placeholder="research, analysis, urgent", help="Comma-separated tags")
            
            submitted = st.form_submit_button("🚀 Create Task", use_container_width=True)
            
            if submitted:
                if task_name and task_description and expected_output:
                    try:
                        # Create task
                        task = BaseTask(
                            description=task_description,
                            expected_output=expected_output,
                            task_type=TaskType(task_type),
                            priority=TaskPriority(task_priority),
                            max_execution_time=max_execution_time
                        )
                        
                        # Find assigned agent
                        assigned_agent_obj = None
                        if assigned_agent != "None":
                            agent_name = assigned_agent.split(" (")[0]
                            for agent in st.session_state.agents:
                                if agent['name'] == agent_name:
                                    assigned_agent_obj = agent['agent_object']
                                    task.assign_agent(assigned_agent_obj)
                                    break
                        
                        # Add dependencies
                        for dep in dependencies:
                            dep_id = dep.split("(")[1].split(")")[0]
                            for existing_task in st.session_state.tasks:
                                if existing_task['id'].startswith(dep_id):
                                    task.add_dependency(existing_task['task_object'])
                                    break
                        
                        # Store task info
                        task_info = {
                            'id': task.id,
                            'name': task_name,
                            'description': task_description,
                            'expected_output': expected_output,
                            'task_type': task_type,
                            'priority': task_priority,
                            'assigned_agent': assigned_agent if assigned_agent != "None" else None,
                            'max_execution_time': max_execution_time,
                            'retry_count': retry_count,
                            'dependencies': dependencies,
                            'context': context,
                            'tags': [tag.strip() for tag in tags.split(",") if tag.strip()],
                            'created_at': task.created_at,
                            'task_object': task
                        }
                        
                        st.session_state.tasks.append(task_info)
                        st.success(f"✅ Task '{task_name}' created successfully!")
                        st.balloons()
                        
                    except Exception as e:
                        st.error(f"❌ Error creating task: {str(e)}")
                else:
                    st.error("❌ Please fill in all required fields")
    
    with tab2:
        st.header("Manage Existing Tasks")
        
        if st.session_state.tasks:
            # Filter and sort options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_type = st.selectbox("Filter by Type", ["All"] + [t.value for t in TaskType])
            with col2:
                filter_priority = st.selectbox("Filter by Priority", ["All"] + [p.value for p in TaskPriority])
            with col3:
                sort_by = st.selectbox("Sort by", ["Created Date", "Priority", "Name", "Type"])
            
            # Apply filters
            filtered_tasks = st.session_state.tasks
            
            if filter_type != "All":
                filtered_tasks = [t for t in filtered_tasks if t['task_type'] == filter_type]
            
            if filter_priority != "All":
                filtered_tasks = [t for t in filtered_tasks if t['priority'] == filter_priority]
            
            # Sort tasks
            if sort_by == "Priority":
                priority_order = {p.value: i for i, p in enumerate(TaskPriority)}
                filtered_tasks.sort(key=lambda x: priority_order.get(x['priority'], 999))
            elif sort_by == "Name":
                filtered_tasks.sort(key=lambda x: x['name'])
            elif sort_by == "Type":
                filtered_tasks.sort(key=lambda x: x['task_type'])
            else:  # Created Date
                filtered_tasks.sort(key=lambda x: x['created_at'], reverse=True)
            
            # Task list
            for i, task_info in enumerate(filtered_tasks):
                # Find original index for operations
                original_index = st.session_state.tasks.index(task_info)
                
                with st.expander(f"📝 {task_info['name']} ({task_info['priority']})"):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.write(f"**Type:** {task_info['task_type']}")
                        st.write(f"**Description:** {task_info['description'][:100]}...")
                        st.write(f"**Expected Output:** {task_info['expected_output'][:100]}...")
                        
                    with col2:
                        st.write(f"**ID:** {task_info['id'][:8]}...")
                        st.write(f"**Created:** {task_info['created_at']}")
                        st.write(f"**Assigned Agent:** {task_info['assigned_agent'] or 'Unassigned'}")
                        if task_info['tags']:
                            st.write(f"**Tags:** {', '.join(task_info['tags'])}")
                    
                    with col3:
                        if st.button(f"✏️ Edit", key=f"edit_task_{original_index}"):
                            st.info("Edit functionality coming soon!")
                        
                        if st.button(f"🗑️ Delete", key=f"delete_task_{original_index}"):
                            st.session_state.tasks.pop(original_index)
                            st.rerun()
                        
                        if st.button(f"🚀 Execute", key=f"execute_task_{original_index}"):
                            st.info("Task execution functionality coming soon!")
            
            # Bulk actions
            st.subheader("🔧 Bulk Actions")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📤 Export All Tasks"):
                    st.info("Export functionality coming soon!")
            
            with col2:
                if st.button("📥 Import Tasks"):
                    st.info("Import functionality coming soon!")
            
            with col3:
                if st.button("🗑️ Clear All Tasks"):
                    if st.button("⚠️ Confirm Delete All"):
                        st.session_state.tasks = []
                        st.rerun()
        else:
            st.info("No tasks created yet. Use the 'Create Task' tab to get started!")
            
            # Quick start templates
            st.subheader("🚀 Quick Start Templates")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Research Pipeline Template", use_container_width=True):
                    # Create research pipeline tasks
                    templates = [
                        {
                            'name': 'Literature Review',
                            'description': 'Conduct comprehensive literature review on the research topic',
                            'expected_output': 'Annotated bibliography with key findings and research gaps',
                            'task_type': TaskType.RESEARCH.value,
                            'priority': TaskPriority.HIGH.value
                        },
                        {
                            'name': 'Data Analysis',
                            'description': 'Analyze collected data and identify patterns and trends',
                            'expected_output': 'Statistical analysis report with visualizations',
                            'task_type': TaskType.ANALYSIS.value,
                            'priority': TaskPriority.HIGH.value
                        },
                        {
                            'name': 'Report Writing',
                            'description': 'Compile research findings into comprehensive report',
                            'expected_output': 'Professional research report with recommendations',
                            'task_type': TaskType.WRITING.value,
                            'priority': TaskPriority.MEDIUM.value
                        }
                    ]
                    
                    for template in templates:
                        task = BaseTask(
                            description=template['description'],
                            expected_output=template['expected_output'],
                            task_type=TaskType(template['task_type']),
                            priority=TaskPriority(template['priority'])
                        )
                        
                        task_info = {
                            'id': task.id,
                            'name': template['name'],
                            'description': template['description'],
                            'expected_output': template['expected_output'],
                            'task_type': template['task_type'],
                            'priority': template['priority'],
                            'assigned_agent': None,
                            'max_execution_time': 1800,
                            'retry_count': 3,
                            'dependencies': [],
                            'context': '',
                            'tags': ['research', 'template'],
                            'created_at': task.created_at,
                            'task_object': task
                        }
                        
                        st.session_state.tasks.append(task_info)
                    
                    st.success("✅ Research pipeline template created!")
                    st.rerun()
            
            with col2:
                if st.button("💻 Development Pipeline Template", use_container_width=True):
                    # Create development pipeline tasks
                    templates = [
                        {
                            'name': 'Requirements Analysis',
                            'description': 'Analyze and document software requirements',
                            'expected_output': 'Detailed requirements specification document',
                            'task_type': TaskType.ANALYSIS.value,
                            'priority': TaskPriority.HIGH.value
                        },
                        {
                            'name': 'Code Development',
                            'description': 'Implement the software solution based on requirements',
                            'expected_output': 'Working code with proper documentation',
                            'task_type': TaskType.CODING.value,
                            'priority': TaskPriority.HIGH.value
                        },
                        {
                            'name': 'Code Review',
                            'description': 'Review code for quality, security, and best practices',
                            'expected_output': 'Code review report with recommendations',
                            'task_type': TaskType.REVIEW.value,
                            'priority': TaskPriority.MEDIUM.value
                        }
                    ]
                    
                    for template in templates:
                        task = BaseTask(
                            description=template['description'],
                            expected_output=template['expected_output'],
                            task_type=TaskType(template['task_type']),
                            priority=TaskPriority(template['priority'])
                        )
                        
                        task_info = {
                            'id': task.id,
                            'name': template['name'],
                            'description': template['description'],
                            'expected_output': template['expected_output'],
                            'task_type': template['task_type'],
                            'priority': template['priority'],
                            'assigned_agent': None,
                            'max_execution_time': 1800,
                            'retry_count': 3,
                            'dependencies': [],
                            'context': '',
                            'tags': ['development', 'template'],
                            'created_at': task.created_at,
                            'task_object': task
                        }
                        
                        st.session_state.tasks.append(task_info)
                    
                    st.success("✅ Development pipeline template created!")
                    st.rerun()

if __name__ == "__main__":
    main()
