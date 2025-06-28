#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder - Execution Page
Crew execution and monitoring interface
"""

import streamlit as st
import sys
import time
import asyncio
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from runtime.executor import RuntimeExecutor, ExecutionMode

st.set_page_config(
    page_title="PAAB - Execution",
    page_icon="🚀",
    layout="wide"
)

def main():
    """Main execution page"""
    
    st.title("🚀 Crew Execution & Monitoring")
    
    # Initialize session state
    if 'crews' not in st.session_state:
        st.session_state.crews = []
    if 'execution_results' not in st.session_state:
        st.session_state.execution_results = []
    if 'selected_crew' not in st.session_state:
        st.session_state.selected_crew = None
    if 'executor' not in st.session_state:
        st.session_state.executor = None
    
    # Sidebar navigation
    with st.sidebar:
        st.header("Navigation")
        if st.button("🏠 Dashboard"):
            st.switch_page("main.py")
        if st.button("🤖 Agents"):
            st.switch_page("pages/🤖_Agents.py")
        if st.button("📝 Tasks"):
            st.switch_page("pages/📝_Tasks.py")
        if st.button("👥 Crews"):
            st.switch_page("pages/👥_Crews.py")
        
        # Execution controls
        st.header("⚡ Quick Controls")
        if st.button("🛑 Stop All Executions"):
            if st.session_state.executor:
                st.session_state.executor.stop()
                st.success("All executions stopped!")
        
        if st.button("🔄 Refresh Status"):
            st.rerun()
    
    # Main content
    if not st.session_state.crews:
        st.warning("⚠️ No crews available. Please create crews first before executing.")
        if st.button("👥 Go to Crews Page"):
            st.switch_page("pages/👥_Crews.py")
        return
    
    tab1, tab2, tab3 = st.tabs(["🚀 Execute Crew", "📊 Monitor Execution", "📈 Results & Analytics"])
    
    with tab1:
        st.header("Execute Crew")
        
        # Crew selection
        if st.session_state.selected_crew:
            selected_crew_name = st.session_state.selected_crew['name']
            st.info(f"Selected crew: **{selected_crew_name}**")
            
            if st.button("🔄 Change Crew"):
                st.session_state.selected_crew = None
                st.rerun()
        else:
            crew_options = [f"{crew['name']} ({crew['id'][:8]})" for crew in st.session_state.crews]
            selected_crew_option = st.selectbox("Select Crew to Execute", crew_options)
            
            if selected_crew_option:
                crew_name = selected_crew_option.split(" (")[0]
                for crew in st.session_state.crews:
                    if crew['name'] == crew_name:
                        st.session_state.selected_crew = crew
                        break
        
        if st.session_state.selected_crew:
            crew_info = st.session_state.selected_crew
            
            # Show crew details
            with st.expander("📋 Crew Details"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Name:** {crew_info['name']}")
                    st.write(f"**Process Type:** {crew_info['process_type']}")
                    st.write(f"**Collaboration:** {crew_info['collaboration_pattern']}")
                    
                with col2:
                    st.write(f"**Agents:** {len(crew_info['agents'])}")
                    st.write(f"**Tasks:** {len(crew_info['tasks'])}")
                    st.write(f"**Created:** {crew_info['created_at']}")
                
                st.write("**Agents:**")
                for agent in crew_info['agents']:
                    st.write(f"  • {agent}")
                
                st.write("**Tasks:**")
                for task in crew_info['tasks']:
                    st.write(f"  • {task}")
            
            # Execution configuration
            st.subheader("⚙️ Execution Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                execution_mode = st.selectbox(
                    "Execution Mode",
                    [mode.value for mode in ExecutionMode],
                    help="Choose how to execute the crew"
                )
                
                timeout = st.number_input(
                    "Timeout (seconds)",
                    min_value=60,
                    max_value=14400,
                    value=3600,
                    help="Maximum execution time"
                )
            
            with col2:
                # Input parameters
                st.write("**Input Parameters:**")
                input_params = {}
                
                # Common input fields
                topic = st.text_input("Topic/Subject", placeholder="e.g., AI agent frameworks")
                if topic:
                    input_params['topic'] = topic
                
                deadline = st.selectbox("Deadline", ["normal", "urgent", "flexible"])
                input_params['deadline'] = deadline
                
                quality_level = st.selectbox("Quality Level", ["standard", "high", "premium"])
                input_params['quality_level'] = quality_level
                
                # Custom inputs
                custom_inputs = st.text_area(
                    "Custom Inputs (JSON format)",
                    placeholder='{"key": "value", "another_key": "another_value"}',
                    help="Additional parameters in JSON format"
                )
                
                if custom_inputs:
                    try:
                        import json
                        custom_data = json.loads(custom_inputs)
                        input_params.update(custom_data)
                    except json.JSONDecodeError:
                        st.error("Invalid JSON format in custom inputs")
            
            # Execution button
            st.subheader("🚀 Execute")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🚀 Start Execution", use_container_width=True, type="primary"):
                    try:
                        # Initialize executor if not exists
                        if not st.session_state.executor:
                            st.session_state.executor = RuntimeExecutor(
                                max_concurrent_crews=5,
                                max_workers=20,
                                enable_monitoring=True
                            )
                            st.session_state.executor.start()
                        
                        # Execute crew
                        with st.spinner("Starting crew execution..."):
                            result = st.session_state.executor.execute_crew(
                                crew=crew_info['crew_object'],
                                inputs=input_params,
                                mode=ExecutionMode(execution_mode),
                                timeout=timeout
                            )
                            
                            # Store execution info
                            execution_info = {
                                'crew_name': crew_info['name'],
                                'crew_id': crew_info['id'],
                                'execution_mode': execution_mode,
                                'inputs': input_params,
                                'timeout': timeout,
                                'started_at': time.time(),
                                'status': 'running',
                                'result': result
                            }
                            
                            st.session_state.execution_results.append(execution_info)
                            st.success("✅ Crew execution started!")
                            st.balloons()
                    
                    except Exception as e:
                        st.error(f"❌ Error starting execution: {str(e)}")
            
            with col2:
                if st.button("🧪 Test Run", use_container_width=True):
                    st.info("Test run functionality coming soon!")
    
    with tab2:
        st.header("Monitor Active Executions")
        
        if st.session_state.execution_results:
            # Filter active executions
            active_executions = [exec for exec in st.session_state.execution_results if exec['status'] == 'running']
            
            if active_executions:
                for i, execution in enumerate(active_executions):
                    with st.expander(f"🚀 {execution['crew_name']} - {execution['execution_mode']}"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write(f"**Crew:** {execution['crew_name']}")
                            st.write(f"**Mode:** {execution['execution_mode']}")
                            st.write(f"**Started:** {time.ctime(execution['started_at'])}")
                        
                        with col2:
                            elapsed_time = time.time() - execution['started_at']
                            st.write(f"**Elapsed:** {elapsed_time:.1f}s")
                            st.write(f"**Timeout:** {execution['timeout']}s")
                            
                            # Progress bar (simulated)
                            progress = min(elapsed_time / execution['timeout'], 1.0)
                            st.progress(progress)
                        
                        with col3:
                            if st.button(f"🛑 Stop", key=f"stop_exec_{i}"):
                                execution['status'] = 'stopped'
                                st.rerun()
                            
                            if st.button(f"📊 Details", key=f"details_exec_{i}"):
                                st.info("Detailed monitoring coming soon!")
                        
                        # Show inputs
                        if execution['inputs']:
                            st.write("**Inputs:**")
                            for key, value in execution['inputs'].items():
                                st.write(f"  • {key}: {value}")
            else:
                st.info("No active executions")
        else:
            st.info("No executions started yet")
        
        # System status
        st.subheader("🖥️ System Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Executions", len([e for e in st.session_state.execution_results if e['status'] == 'running']))
        
        with col2:
            st.metric("Total Executions", len(st.session_state.execution_results))
        
        with col3:
            completed = len([e for e in st.session_state.execution_results if e['status'] == 'completed'])
            st.metric("Completed", completed)
        
        with col4:
            failed = len([e for e in st.session_state.execution_results if e['status'] == 'failed'])
            st.metric("Failed", failed)
    
    with tab3:
        st.header("Results & Analytics")
        
        if st.session_state.execution_results:
            # Execution history
            st.subheader("📋 Execution History")
            
            for i, execution in enumerate(reversed(st.session_state.execution_results)):
                status_icon = {
                    'running': '🔄',
                    'completed': '✅',
                    'failed': '❌',
                    'stopped': '🛑'
                }.get(execution['status'], '❓')
                
                with st.expander(f"{status_icon} {execution['crew_name']} - {execution['status'].title()}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Crew:** {execution['crew_name']}")
                        st.write(f"**Mode:** {execution['execution_mode']}")
                        st.write(f"**Status:** {execution['status']}")
                        st.write(f"**Started:** {time.ctime(execution['started_at'])}")
                    
                    with col2:
                        if execution['status'] in ['completed', 'failed', 'stopped']:
                            elapsed = time.time() - execution['started_at']
                            st.write(f"**Duration:** {elapsed:.1f}s")
                        
                        if execution['inputs']:
                            st.write("**Inputs:**")
                            for key, value in execution['inputs'].items():
                                st.write(f"  • {key}: {value}")
                    
                    # Show results if completed
                    if execution['status'] == 'completed' and hasattr(execution.get('result'), 'outputs'):
                        st.write("**Results:**")
                        st.json(execution['result'].outputs)
            
            # Analytics
            st.subheader("📊 Analytics")
            
            # Execution statistics
            total_executions = len(st.session_state.execution_results)
            completed_executions = len([e for e in st.session_state.execution_results if e['status'] == 'completed'])
            failed_executions = len([e for e in st.session_state.execution_results if e['status'] == 'failed'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                success_rate = (completed_executions / total_executions * 100) if total_executions > 0 else 0
                st.metric("Success Rate", f"{success_rate:.1f}%")
            
            with col2:
                avg_duration = 0
                completed = [e for e in st.session_state.execution_results if e['status'] == 'completed']
                if completed:
                    durations = [time.time() - e['started_at'] for e in completed]
                    avg_duration = sum(durations) / len(durations)
                st.metric("Avg Duration", f"{avg_duration:.1f}s")
            
            with col3:
                most_used_crew = "N/A"
                if st.session_state.execution_results:
                    crew_counts = {}
                    for exec in st.session_state.execution_results:
                        crew_name = exec['crew_name']
                        crew_counts[crew_name] = crew_counts.get(crew_name, 0) + 1
                    most_used_crew = max(crew_counts, key=crew_counts.get)
                st.metric("Most Used Crew", most_used_crew)
            
            # Clear history
            if st.button("🗑️ Clear Execution History"):
                if st.button("⚠️ Confirm Clear History"):
                    st.session_state.execution_results = []
                    st.rerun()
        else:
            st.info("No execution results yet. Execute some crews to see analytics!")

if __name__ == "__main__":
    main()
