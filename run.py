#!/usr/bin/env python3
"""
Startup script for AI Chatbot with Voice and Lip Sync
Provides easy way to start the application with proper configuration
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path

def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def check_dependencies():
    """Check if required dependencies are installed"""
    logger = logging.getLogger(__name__)
    
    try:
        import fastapi
        import uvicorn
        import torch
        import whisper
        import librosa
        logger.info("✓ Core dependencies found")
    except ImportError as e:
        logger.error(f"✗ Missing dependency: {e}")
        logger.error("Please run: pip install -r requirements.txt")
        return False
    
    # Check for FFmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      capture_output=True, check=True)
        logger.info("✓ FFmpeg found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("⚠ FFmpeg not found - some audio processing may fail")
    
    return True

def check_ollama():
    """Check if Ollama is running"""
    logger = logging.getLogger(__name__)
    
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info("✓ Ollama server is running")
            return True
    except Exception:
        pass
    
    logger.warning("⚠ Ollama server not accessible at http://localhost:11434")
    logger.warning("  Please start Ollama: ollama serve")
    return False

def create_directories():
    """Create necessary directories"""
    logger = logging.getLogger(__name__)
    
    directories = ['temp', 'models', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"✓ Directory created/verified: {directory}")

def check_avatar():
    """Check if avatar image exists"""
    logger = logging.getLogger(__name__)
    
    avatar_path = Path("avatar.png")
    if avatar_path.exists():
        logger.info("✓ Avatar image found")
        return True
    else:
        logger.warning("⚠ Avatar image not found at avatar.png")
        logger.warning("  Please add your avatar image for lip sync functionality")
        return False

def create_env_file():
    """Create .env file if it doesn't exist"""
    logger = logging.getLogger(__name__)
    
    env_path = Path(".env")
    if env_path.exists():
        logger.info("✓ .env file found")
        return
    
    logger.info("Creating default .env file...")
    
    env_content = """# Server Configuration
HOST=0.0.0.0
PORT=8000
RELOAD=true
LOG_LEVEL=info

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

# Whisper Configuration
WHISPER_MODEL=base
WHISPER_DEVICE=auto

# File Paths
TEMP_DIR=./temp
MODELS_DIR=./models
AVATAR_PATH=./avatar.png

# API Keys (if needed)
# OPENAI_API_KEY=your_openai_key_here
"""
    
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    logger.info("✓ Created default .env file")

def main():
    """Main startup function"""
    parser = argparse.ArgumentParser(description="AI Chatbot Startup Script")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--check-only", action="store_true", help="Only check dependencies")
    parser.add_argument("--skip-checks", action="store_true", help="Skip dependency checks")
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🤖 AI Chatbot with Voice and Lip Sync")
    logger.info("=" * 50)
    
    # Create necessary directories
    create_directories()
    
    # Create .env file if needed
    create_env_file()
    
    if not args.skip_checks:
        # Check dependencies
        if not check_dependencies():
            sys.exit(1)
        
        # Check Ollama
        check_ollama()
        
        # Check avatar
        check_avatar()
    
    if args.check_only:
        logger.info("✓ Dependency check complete")
        return
    
    # Start the server
    logger.info("🚀 Starting AI Chatbot server...")
    logger.info(f"   Host: {args.host}")
    logger.info(f"   Port: {args.port}")
    logger.info(f"   Reload: {args.reload}")
    logger.info("=" * 50)
    
    try:
        import uvicorn
        uvicorn.run(
            "app.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("\n👋 Shutting down AI Chatbot server...")
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 