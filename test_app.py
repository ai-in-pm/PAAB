#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple test script to verify PAAB functionality
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test basic imports"""
    print("🧪 Testing PAAB imports...")
    
    try:
        # Test basic imports
        from agents.base import BaseAgent, AgentState
        print("✅ Base agent import successful")
        
        from tasks.base import BaseTask, TaskType
        print("✅ Base task import successful")
        
        from crews.base import BaseCrew, ProcessType
        print("✅ Base crew import successful")
        
        from tools.base import BaseTool
        print("✅ Base tool import successful")
        
        print("🎉 All core imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {str(e)}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    print("\n🧪 Testing basic functionality...")

    try:
        from agents.specialized import ResearcherAgent
        from tasks.base import TaskType, TaskPriority
        from crews.base import ProcessType

        # Test agent creation
        agent = ResearcherAgent(
            goal="Test the system",
            backstory="A test agent for verification"
        )
        print(f"✅ Created agent: {agent.role}")

        # Test enums
        task_type = TaskType.RESEARCH
        priority = TaskPriority.HIGH
        process_type = ProcessType.SEQUENTIAL
        print(f"✅ Enums working: {task_type}, {priority}, {process_type}")

        print("🎉 Basic functionality test passed!")
        return True

    except Exception as e:
        print(f"❌ Functionality test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_streamlit_app():
    """Test if we can import the Streamlit app"""
    print("\n🧪 Testing Streamlit app import...")
    
    try:
        import streamlit as st
        print("✅ Streamlit import successful")
        
        # Try to import our studio main
        from studio.main import main as studio_main
        print("✅ Studio main import successful")
        
        print("🎉 Streamlit app test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Streamlit app test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 PsychoPy AI Agent Builder - Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_streamlit_app
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! PAAB is ready to run.")
        print("\n🚀 To start the studio, run:")
        print("   python -m streamlit run src/studio/main.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
