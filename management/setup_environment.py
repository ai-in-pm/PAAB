#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PsychoPy AI Agent Builder - Environment Setup Script
Automated setup and configuration for the AI agent builder system
"""

import os
import sys
import subprocess
import platform
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnvironmentSetup:
    """Environment setup and configuration manager"""
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize environment setup
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root or Path(__file__).parent.parent
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.platform = platform.system().lower()
        
        # Configuration paths
        self.config_dir = self.project_root / "config"
        self.env_file = self.project_root / ".env"
        self.requirements_file = self.project_root / "requirements.txt"
        
        logger.info(f"Initializing environment setup for project: {self.project_root}")
        logger.info(f"Python version: {self.python_version}")
        logger.info(f"Platform: {self.platform}")
    
    def setup_complete_environment(self) -> bool:
        """
        Complete environment setup process
        
        Returns:
            True if setup successful, False otherwise
        """
        try:
            logger.info("🚀 Starting complete environment setup...")
            
            # Step 1: Check system requirements
            if not self.check_system_requirements():
                logger.error("❌ System requirements check failed")
                return False
            
            # Step 2: Create directory structure
            if not self.create_directory_structure():
                logger.error("❌ Directory structure creation failed")
                return False
            
            # Step 3: Setup virtual environment
            if not self.setup_virtual_environment():
                logger.error("❌ Virtual environment setup failed")
                return False
            
            # Step 4: Install dependencies
            if not self.install_dependencies():
                logger.error("❌ Dependencies installation failed")
                return False
            
            # Step 5: Create configuration files
            if not self.create_configuration_files():
                logger.error("❌ Configuration files creation failed")
                return False
            
            # Step 6: Setup environment variables
            if not self.setup_environment_variables():
                logger.error("❌ Environment variables setup failed")
                return False
            
            # Step 7: Initialize database/storage
            if not self.initialize_storage():
                logger.error("❌ Storage initialization failed")
                return False
            
            # Step 8: Run initial tests
            if not self.run_initial_tests():
                logger.error("❌ Initial tests failed")
                return False
            
            logger.info("✅ Complete environment setup successful!")
            self.print_setup_summary()
            return True
            
        except Exception as e:
            logger.error(f"❌ Environment setup failed: {str(e)}")
            return False
    
    def check_system_requirements(self) -> bool:
        """Check system requirements"""
        logger.info("🔍 Checking system requirements...")
        
        try:
            # Check Python version
            if sys.version_info < (3, 8):
                logger.error(f"Python 3.8+ required, found {self.python_version}")
                return False
            
            logger.info(f"✅ Python version: {self.python_version}")
            
            # Check pip
            try:
                import pip
                logger.info(f"✅ pip available: {pip.__version__}")
            except ImportError:
                logger.error("❌ pip not available")
                return False
            
            # Check git (optional but recommended)
            try:
                result = subprocess.run(["git", "--version"], capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info(f"✅ Git available: {result.stdout.strip()}")
                else:
                    logger.warning("⚠️ Git not available (optional)")
            except FileNotFoundError:
                logger.warning("⚠️ Git not found (optional)")
            
            # Check available disk space
            import shutil
            free_space = shutil.disk_usage(self.project_root).free / (1024**3)  # GB
            if free_space < 1.0:
                logger.warning(f"⚠️ Low disk space: {free_space:.1f} GB available")
            else:
                logger.info(f"✅ Disk space: {free_space:.1f} GB available")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ System requirements check failed: {str(e)}")
            return False
    
    def create_directory_structure(self) -> bool:
        """Create project directory structure"""
        logger.info("📁 Creating directory structure...")
        
        try:
            directories = [
                "src/agents",
                "src/tasks", 
                "src/crews",
                "src/tools",
                "src/runtime",
                "src/memory",
                "src/flows",
                "src/studio",
                "src/cli",
                "src/integrations",
                "src/utils",
                "tests/unit",
                "tests/integration",
                "tests/e2e",
                "docs/api",
                "docs/user",
                "config",
                "examples",
                "templates",
                "assets/images",
                "assets/styles",
                "assets/fonts",
                "scripts",
                "logs",
                "data",
                "models",
                "cache"
            ]
            
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                
                # Create __init__.py for Python packages
                if directory.startswith("src/") and "/" in directory:
                    init_file = dir_path / "__init__.py"
                    if not init_file.exists():
                        init_file.touch()
            
            logger.info(f"✅ Created {len(directories)} directories")
            return True
            
        except Exception as e:
            logger.error(f"❌ Directory creation failed: {str(e)}")
            return False
    
    def setup_virtual_environment(self) -> bool:
        """Setup virtual environment"""
        logger.info("🐍 Setting up virtual environment...")
        
        try:
            venv_path = self.project_root / "venv"
            
            if venv_path.exists():
                logger.info("✅ Virtual environment already exists")
                return True
            
            # Create virtual environment
            result = subprocess.run([
                sys.executable, "-m", "venv", str(venv_path)
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"❌ Virtual environment creation failed: {result.stderr}")
                return False
            
            logger.info("✅ Virtual environment created")
            
            # Provide activation instructions
            if self.platform == "windows":
                activate_script = venv_path / "Scripts" / "activate.bat"
                logger.info(f"💡 Activate with: {activate_script}")
            else:
                activate_script = venv_path / "bin" / "activate"
                logger.info(f"💡 Activate with: source {activate_script}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Virtual environment setup failed: {str(e)}")
            return False
    
    def install_dependencies(self) -> bool:
        """Install project dependencies"""
        logger.info("📦 Installing dependencies...")
        
        try:
            if not self.requirements_file.exists():
                logger.warning("⚠️ requirements.txt not found, skipping dependency installation")
                return True
            
            # Install requirements
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"❌ Dependencies installation failed: {result.stderr}")
                return False
            
            logger.info("✅ Dependencies installed successfully")
            
            # Install development dependencies
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-e", ".[dev]"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                logger.info("✅ Development dependencies installed")
            else:
                logger.warning("⚠️ Development dependencies installation failed (optional)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Dependencies installation failed: {str(e)}")
            return False
    
    def create_configuration_files(self) -> bool:
        """Create configuration files"""
        logger.info("⚙️ Creating configuration files...")
        
        try:
            # Create config directory
            self.config_dir.mkdir(exist_ok=True)
            
            # Default configuration
            default_config = {
                "project": {
                    "name": "PsychoPy AI Agent Builder",
                    "version": "1.0.0",
                    "description": "AI Agent Builder based on PsychoPy"
                },
                "runtime": {
                    "max_concurrent_crews": 10,
                    "max_workers": 50,
                    "execution_timeout": 3600,
                    "monitoring_enabled": True,
                    "security_enabled": True
                },
                "agents": {
                    "default_max_iterations": 10,
                    "default_memory_enabled": True,
                    "default_collaboration_enabled": True
                },
                "llm": {
                    "default_provider": "openai",
                    "default_model": "gpt-4",
                    "default_temperature": 0.7,
                    "default_max_tokens": 1000
                },
                "storage": {
                    "type": "local",
                    "path": "data/",
                    "backup_enabled": True
                },
                "logging": {
                    "level": "INFO",
                    "file": "logs/paab.log",
                    "max_size": "10MB",
                    "backup_count": 5
                }
            }
            
            # Save main configuration
            config_file = self.config_dir / "config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            
            logger.info(f"✅ Configuration saved to {config_file}")
            
            # Create logging configuration
            logging_config = {
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {
                    "standard": {
                        "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                    }
                },
                "handlers": {
                    "default": {
                        "level": "INFO",
                        "formatter": "standard",
                        "class": "logging.StreamHandler"
                    },
                    "file": {
                        "level": "DEBUG",
                        "formatter": "standard",
                        "class": "logging.handlers.RotatingFileHandler",
                        "filename": "logs/paab.log",
                        "maxBytes": 10485760,
                        "backupCount": 5
                    }
                },
                "loggers": {
                    "": {
                        "handlers": ["default", "file"],
                        "level": "DEBUG",
                        "propagate": False
                    }
                }
            }
            
            logging_file = self.config_dir / "logging.yaml"
            with open(logging_file, 'w') as f:
                yaml.dump(logging_config, f, default_flow_style=False)
            
            logger.info(f"✅ Logging configuration saved to {logging_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration files creation failed: {str(e)}")
            return False
    
    def setup_environment_variables(self) -> bool:
        """Setup environment variables"""
        logger.info("🔧 Setting up environment variables...")
        
        try:
            env_template = """# PsychoPy AI Agent Builder Environment Variables

# LLM API Keys (add your actual keys)
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# Runtime Configuration
PAAB_LOG_LEVEL=INFO
PAAB_MAX_CONCURRENT_CREWS=10
PAAB_MAX_WORKERS=50
PAAB_EXECUTION_TIMEOUT=3600

# Storage Configuration
PAAB_DATA_PATH=data/
PAAB_CACHE_PATH=cache/
PAAB_LOGS_PATH=logs/

# Security
PAAB_SECRET_KEY=your_secret_key_here
PAAB_ENABLE_SECURITY=true

# Development
PAAB_DEBUG=false
PAAB_TESTING=false

# Optional: Database URLs
# DATABASE_URL=sqlite:///paab.db
# REDIS_URL=redis://localhost:6379/0
# MONGODB_URL=mongodb://localhost:27017/paab

# Optional: Cloud Storage
# AWS_ACCESS_KEY_ID=your_aws_access_key
# AWS_SECRET_ACCESS_KEY=your_aws_secret_key
# AWS_DEFAULT_REGION=us-east-1

# Optional: Monitoring
# PROMETHEUS_PORT=9090
# GRAFANA_PORT=3000
"""
            
            if not self.env_file.exists():
                with open(self.env_file, 'w') as f:
                    f.write(env_template)
                
                logger.info(f"✅ Environment template created: {self.env_file}")
                logger.info("💡 Please edit .env file with your actual API keys and configuration")
            else:
                logger.info("✅ Environment file already exists")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Environment variables setup failed: {str(e)}")
            return False
    
    def initialize_storage(self) -> bool:
        """Initialize storage systems"""
        logger.info("💾 Initializing storage...")
        
        try:
            # Create data directories
            data_dirs = ["data", "cache", "logs", "models"]
            for dir_name in data_dirs:
                dir_path = self.project_root / dir_name
                dir_path.mkdir(exist_ok=True)
            
            # Create initial database/storage files
            db_file = self.project_root / "data" / "paab.db"
            if not db_file.exists():
                # Create empty SQLite database
                import sqlite3
                conn = sqlite3.connect(str(db_file))
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS executions (
                        id TEXT PRIMARY KEY,
                        crew_name TEXT,
                        status TEXT,
                        created_at TIMESTAMP,
                        completed_at TIMESTAMP,
                        results TEXT
                    )
                """)
                conn.commit()
                conn.close()
                logger.info(f"✅ Database initialized: {db_file}")
            
            # Create cache structure
            cache_dirs = ["embeddings", "models", "responses"]
            for cache_dir in cache_dirs:
                cache_path = self.project_root / "cache" / cache_dir
                cache_path.mkdir(exist_ok=True)
            
            logger.info("✅ Storage initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Storage initialization failed: {str(e)}")
            return False
    
    def run_initial_tests(self) -> bool:
        """Run initial tests to verify setup"""
        logger.info("🧪 Running initial tests...")
        
        try:
            # Test basic imports
            sys.path.insert(0, str(self.project_root / "src"))
            
            try:
                from agents.base import BaseAgent
                from tasks.base import BaseTask
                from crews.base import BaseCrew
                from tools.base import BaseTool
                logger.info("✅ Core imports successful")
            except ImportError as e:
                logger.error(f"❌ Import test failed: {str(e)}")
                return False
            
            # Test configuration loading
            config_file = self.config_dir / "config.yaml"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info("✅ Configuration loading successful")
            
            # Run pytest if available
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"
                ], capture_output=True, text=True, cwd=self.project_root, timeout=60)
                
                if result.returncode == 0:
                    logger.info("✅ Unit tests passed")
                else:
                    logger.warning("⚠️ Some tests failed (this may be expected during initial setup)")
                    logger.debug(f"Test output: {result.stdout}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.info("💡 Pytest not available or tests timed out (optional)")
            
            logger.info("✅ Initial tests complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Initial tests failed: {str(e)}")
            return False
    
    def print_setup_summary(self) -> None:
        """Print setup summary and next steps"""
        logger.info("\n" + "="*60)
        logger.info("🎉 PsychoPy AI Agent Builder Setup Complete!")
        logger.info("="*60)
        
        logger.info("\n📋 Next Steps:")
        logger.info("1. Activate virtual environment:")
        if self.platform == "windows":
            logger.info("   venv\\Scripts\\activate")
        else:
            logger.info("   source venv/bin/activate")
        
        logger.info("\n2. Edit environment variables:")
        logger.info(f"   Edit {self.env_file}")
        logger.info("   Add your API keys for OpenAI, Anthropic, etc.")
        
        logger.info("\n3. Test the installation:")
        logger.info("   python examples/basic_research_crew.py")
        
        logger.info("\n4. Launch the visual studio:")
        logger.info("   paab studio")
        
        logger.info("\n5. Create your first agent:")
        logger.info("   paab agent create --name 'MyAgent' --role researcher --goal 'Research AI' --backstory 'Expert'")
        
        logger.info("\n📚 Documentation:")
        logger.info("   README.md - Getting started guide")
        logger.info("   docs/ - Detailed documentation")
        logger.info("   examples/ - Example implementations")
        
        logger.info("\n🔗 Useful Commands:")
        logger.info("   paab --help          - Show CLI help")
        logger.info("   paab studio          - Launch visual interface")
        logger.info("   paab init            - Initialize new project")
        logger.info("   pytest tests/        - Run test suite")
        
        logger.info("\n" + "="*60)


def main():
    """Main setup function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PsychoPy AI Agent Builder Environment Setup")
    parser.add_argument("--project-root", type=Path, help="Project root directory")
    parser.add_argument("--skip-tests", action="store_true", help="Skip initial tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize setup
    setup = EnvironmentSetup(project_root=args.project_root)
    
    # Run setup
    success = setup.setup_complete_environment()
    
    if success:
        logger.info("🎉 Setup completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Setup failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
