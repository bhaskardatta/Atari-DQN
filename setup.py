#!/usr/bin/env python3
# filepath: /Users/bhaskar/Desktop/atari/setup.py
"""
Setup script for the adaptive RL project.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {cmd}")
        print(f"   Error: {e.stderr}")
        return False


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def create_virtual_environment():
    """Create and activate virtual environment."""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    if not run_command(f"{sys.executable} -m venv venv", 
                      "Creating virtual environment"):
        return False
    
    print("ℹ️  Virtual environment created at ./venv")
    print("ℹ️  To activate: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)")
    return True


def install_requirements():
    """Install Python requirements."""
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    # Determine pip command based on OS
    if os.name == 'nt':  # Windows
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Unix-like
        pip_cmd = "venv/bin/pip"
    
    # Check if virtual environment exists
    if not Path("venv").exists():
        print("⚠️  Virtual environment not found, using system pip")
        pip_cmd = "pip"
    
    return run_command(f"{pip_cmd} install -r requirements.txt", 
                      "Installing Python requirements")


def setup_directories():
    """Create necessary directories."""
    directories = [
        "results",
        "results/models",
        "results/logs", 
        "results/videos",
        "results/plots",
        "test_results",
        "test_results/models",
        "test_results/logs",
        "test_results/videos",
        "test_results/plots"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Created necessary directories")
    return True


def verify_installation():
    """Verify that key packages are installed."""
    try:
        import torch
        import numpy
        import gym
        import matplotlib
        import pandas
        import yaml
        import cv2
        print("✅ All key packages are available")
        
        # Check PyTorch GPU availability
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️  CUDA not available (CPU training only)")
        
        return True
    except ImportError as e:
        print(f"❌ Package verification failed: {e}")
        return False


def main():
    print("🚀 Setting up Adaptive RL for Atari Breakout")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create virtual environment
    if not create_virtual_environment():
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Setup directories
    if not setup_directories():
        return False
    
    # Verify installation
    if not verify_installation():
        print("⚠️  Some packages may not be properly installed")
        print("   Try activating the virtual environment and running setup again")
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Activate virtual environment:")
    if os.name == 'nt':
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("2. Run a quick test:")
    print("   python main.py train --config configs/quick_test.yaml")
    print("3. Check the documentation:")
    print("   cat README.md")
    print("\nHappy training! 🤖🎮")
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
