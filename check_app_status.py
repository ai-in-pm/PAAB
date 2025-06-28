#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Check if the PAAB Studio is running and accessible
"""

import requests
import time
import sys

def check_streamlit_status(url="http://localhost:8501", timeout=5):
    """Check if Streamlit app is running"""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return True, f"✅ App is running at {url}"
        else:
            return False, f"❌ App returned status code: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, f"❌ Cannot connect to {url} - app may not be running"
    except requests.exceptions.Timeout:
        return False, f"❌ Timeout connecting to {url}"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

def main():
    """Main status checker"""
    print("🔍 Checking PsychoPy AI Agent Builder Studio Status...")
    print("-" * 50)
    
    # Check if app is running
    is_running, message = check_streamlit_status()
    print(message)
    
    if is_running:
        print("\n🎉 Studio is running successfully!")
        print("🌐 Open your browser to: http://localhost:8501")
        print("📚 Features available:")
        print("   • Visual Agent Builder")
        print("   • Task Designer")
        print("   • Crew Orchestrator")
        print("   • Execution Monitor")
        print("   • Performance Analytics")
        
        # Try to get more info about the app
        try:
            response = requests.get("http://localhost:8501/_stcore/health", timeout=3)
            if response.status_code == 200:
                print("✅ Streamlit health check passed")
        except:
            print("⚠️  Could not verify Streamlit health endpoint")
            
    else:
        print("\n❌ Studio is not running or not accessible")
        print("💡 Try running: python -m streamlit run src/studio/main.py")
    
    return is_running

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
