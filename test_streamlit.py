#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple Streamlit test app
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

st.set_page_config(
    page_title="PAAB Test",
    page_icon="🧪",
    layout="wide"
)

def main():
    st.title("🧪 PsychoPy AI Agent Builder - Test App")
    st.write("If you can see this, Streamlit is working!")
    
    st.header("System Information")
    st.write(f"Python version: {sys.version}")
    st.write(f"Streamlit version: {st.__version__}")
    
    st.header("Test Imports")
    
    try:
        from agents.base import BaseAgent
        st.success("✅ Base agent import successful")
    except Exception as e:
        st.error(f"❌ Base agent import failed: {str(e)}")
    
    try:
        from tasks.base import BaseTask
        st.success("✅ Base task import successful")
    except Exception as e:
        st.error(f"❌ Base task import failed: {str(e)}")
    
    try:
        from crews.base import BaseCrew
        st.success("✅ Base crew import successful")
    except Exception as e:
        st.error(f"❌ Base crew import failed: {str(e)}")
    
    try:
        from agents.specialized import ResearcherAgent
        agent = ResearcherAgent(
            goal="Test agent",
            backstory="Test backstory"
        )
        st.success(f"✅ Created test agent: {agent.role}")
    except Exception as e:
        st.error(f"❌ Agent creation failed: {str(e)}")
    
    st.header("Next Steps")
    st.info("If all tests pass, you can run the full studio with: `python -m streamlit run src/studio/main.py`")

if __name__ == "__main__":
    main()
