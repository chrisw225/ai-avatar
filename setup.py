#!/usr/bin/env python3
"""
Setup script for AI Chatbot with Voice and Lip Sync
This script handles the installation of external repositories and dependencies
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, cwd=None, check=True):
    """Run a command and handle errors"""
    print(f"Running: {command}")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=cwd, 
            check=check,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Error: {e.stderr}")
        if check:
            sys.exit(1)
        return e

def create_virtual_environment():
    """Create and activate virtual environment"""
    print("Creating virtual environment...")
    
    # Create .venv directory
    run_command("python -m venv .venv")
    
    # Get activation script path based on OS
    if platform.system() == "Windows":
        activate_script = ".venv\\Scripts\\activate"
        pip_path = ".venv\\Scripts\\pip"
        python_path = ".venv\\Scripts\\python"
    else:
        activate_script = ".venv/bin/activate"
        pip_path = ".venv/bin/pip"
        python_path = ".venv/bin/python"
    
    print(f"Virtual environment created. Activation script: {activate_script}")
    return pip_path, python_path

def install_pytorch():
    """Install PyTorch with CUDA support"""
    print("Installing PyTorch with CUDA support...")
    
    # For RTX 4090, install PyTorch with CUDA 11.8 or 12.1
    pytorch_command = (
        "pip install torch torchvision torchaudio "
        "--index-url https://download.pytorch.org/whl/cu121"
    )
    
    run_command(pytorch_command)

def install_requirements(pip_path):
    """Install requirements from requirements.txt"""
    print("Installing requirements...")
    run_command(f"{pip_path} install -r requirements.txt")

def clone_and_setup_gpt_sovits():
    """Clone and setup GPT-SoVITS"""
    print("Setting up GPT-SoVITS...")
    
    repo_dir = "external/GPT-SoVITS"
    if not os.path.exists(repo_dir):
        os.makedirs("external", exist_ok=True)
        run_command(
            "git clone https://github.com/RVC-Boss/GPT-SoVITS.git",
            cwd="external"
        )
    
    # Install GPT-SoVITS requirements
    gpt_sovits_req = os.path.join(repo_dir, "requirements.txt")
    if os.path.exists(gpt_sovits_req):
        run_command(f"pip install -r {gpt_sovits_req}")

def clone_and_setup_sadtalker():
    """Clone and setup SadTalker"""
    print("Setting up SadTalker...")
    
    repo_dir = "external/SadTalker"
    if not os.path.exists(repo_dir):
        os.makedirs("external", exist_ok=True)
        run_command(
            "git clone https://github.com/OpenTalker/SadTalker.git",
            cwd="external"
        )
    
    # Install SadTalker requirements
    sadtalker_req = os.path.join(repo_dir, "requirements.txt")
    if os.path.exists(sadtalker_req):
        run_command(f"pip install -r {sadtalker_req}")

def download_models():
    """Download required model files"""
    print("Downloading required models...")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Note: Model downloads will be handled by individual services
    # when they first run to avoid large initial download
    print("Model downloads will be handled automatically on first run")

def create_directories():
    """Create necessary directories"""
    directories = [
        "app",
        "app/services",
        "app/models",
        "app/utils",
        "frontend",
        "models",
        "temp",
        "logs",
        "external"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def main():
    """Main setup function"""
    print("Starting AI Chatbot setup...")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.system()}")
    
    # Create directories
    create_directories()
    
    # Create virtual environment
    pip_path, python_path = create_virtual_environment()
    
    print("\n" + "="*50)
    print("IMPORTANT: Activate your virtual environment before continuing!")
    if platform.system() == "Windows":
        print("Run: .venv\\Scripts\\activate")
    else:
        print("Run: source .venv/bin/activate")
    print("="*50 + "\n")
    
    # Install PyTorch first
    install_pytorch()
    
    # Install requirements
    install_requirements(pip_path)
    
    # Clone and setup external repositories
    clone_and_setup_gpt_sovits()
    clone_and_setup_sadtalker()
    
    # Download models
    download_models()
    
    print("\n" + "="*50)
    print("Setup completed successfully!")
    print("Next steps:")
    print("1. Activate virtual environment")
    print("2. Run: python app/main.py")
    print("="*50)

if __name__ == "__main__":
    main() 