#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder Studio Launcher
Simple launcher that handles dependencies and starts the studio
"""

import sys
import subprocess
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['streamlit', 'plotly', 'pandas', 'pydantic']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is available")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies(packages):
    """Install missing dependencies"""
    if packages:
        print(f"\n📦 Installing missing packages: {', '.join(packages)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages)
            print("✅ Dependencies installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    return True

def main():
    """Main launcher function"""
    print("🚀 PsychoPy AI Agent Builder Studio Launcher")
    print("=" * 50)
    
    # Add src to Python path
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    os.environ["PYTHONPATH"] = str(src_path)
    
    print("📁 Project directory:", Path(__file__).parent)
    print("🐍 Python version:", sys.version)
    print("📦 Checking dependencies...")
    
    # Check and install dependencies
    missing = check_dependencies()
    if missing:
        if not install_dependencies(missing):
            print("❌ Failed to install dependencies. Please install manually:")
            print(f"   pip install {' '.join(missing)}")
            return False
    
    # Verify studio file exists
    studio_file = src_path / "studio" / "main.py"
    if not studio_file.exists():
        print(f"❌ Studio file not found: {studio_file}")
        return False
    
    print("✅ All dependencies satisfied!")
    print("🌐 Starting Streamlit studio...")
    print("📍 URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop")
    print("-" * 50)
    
    try:
        # Launch Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(studio_file),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ]
        
        subprocess.run(cmd, cwd=str(Path(__file__).parent))
        
    except KeyboardInterrupt:
        print("\n🛑 Studio stopped by user")
        return True
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install with: pip install streamlit")
        return False
    except Exception as e:
        print(f"❌ Error launching studio: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
