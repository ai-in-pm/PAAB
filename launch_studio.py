#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple launcher for PsychoPy AI Agent Builder Studio
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    """Launch the PAAB Studio"""
    print("🚀 Launching PsychoPy AI Agent Builder Studio...")
    
    # Add src to Python path
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    
    # Set environment variable for the app
    os.environ["PYTHONPATH"] = str(src_path)
    
    # Studio file path
    studio_file = src_path / "studio" / "main.py"
    
    if not studio_file.exists():
        print(f"❌ Studio file not found: {studio_file}")
        return False
    
    print(f"📁 Studio file: {studio_file}")
    print("🌐 Starting Streamlit server...")
    print("📍 URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Launch Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(studio_file),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ]
        
        # Run the command
        result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
        return result.returncode == 0
        
    except KeyboardInterrupt:
        print("\n🛑 Studio stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error launching studio: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
