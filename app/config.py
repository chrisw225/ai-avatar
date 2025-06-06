"""
Configuration management for AI Chatbot
Handles environment variables and default settings
"""

import os
from typing import Optional
from pydantic import BaseSettings, Field
from pathlib import Path

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    frontend_port: int = Field(default=3000, env="FRONTEND_PORT")
    
    # Ollama Configuration (Configurable as requested)
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.1:8b", env="OLLAMA_MODEL")
    
    # Audio Configuration
    audio_sample_rate: int = Field(default=16000, env="AUDIO_SAMPLE_RATE")
    audio_chunk_duration: float = Field(default=3.0, env="AUDIO_CHUNK_DURATION")
    audio_format: str = Field(default="wav", env="AUDIO_FORMAT")
    
    # Video Configuration
    video_fps: int = Field(default=30, env="VIDEO_FPS")
    video_resolution: int = Field(default=512, env="VIDEO_RESOLUTION")
    avatar_path: str = Field(default="avatar.png", env="AVATAR_PATH")
    
    # Model Paths
    whisper_model: str = Field(default="medium", env="WHISPER_MODEL")
    whisper_device: str = Field(default="cuda", env="WHISPER_DEVICE")
    gpt_sovits_model_path: str = Field(default="external/GPT-SoVITS", env="GPT_SOVITS_MODEL_PATH")
    sadtalker_model_path: str = Field(default="external/SadTalker", env="SADTALKER_MODEL_PATH")
    
    # Performance Settings (Optimized for RTX 4090)
    max_concurrent_sessions: int = Field(default=3, env="MAX_CONCURRENT_SESSIONS")
    gpu_memory_fraction: float = Field(default=0.8, env="GPU_MEMORY_FRACTION")
    enable_model_caching: bool = Field(default=True, env="ENABLE_MODEL_CACHING")
    batch_size: int = Field(default=1, env="BATCH_SIZE")
    
    # Temporary Files
    temp_dir: str = Field(default="temp", env="TEMP_DIR")
    cleanup_temp_files: bool = Field(default=True, env="CLEANUP_TEMP_FILES")
    max_temp_file_age: int = Field(default=3600, env="MAX_TEMP_FILE_AGE")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/chatbot.log", env="LOG_FILE")
    
    # Development
    debug: bool = Field(default=False, env="DEBUG")
    reload: bool = Field(default=True, env="RELOAD")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.temp_dir,
            os.path.dirname(self.log_file),
            "models",
            "external"
        ]
        
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
    
    @property
    def available_models(self) -> list:
        """List of available Ollama models"""
        return [
            "llama3.1:8b",
            "llama3.2:3b",
            "llama2:7b",
            "codellama:7b",
            "mistral:7b"
        ]
    
    def get_model_config(self) -> dict:
        """Get current model configuration"""
        return {
            "base_url": self.ollama_base_url,
            "model": self.ollama_model,
            "available_models": self.available_models
        }

# Global settings instance
settings = Settings()

def get_settings() -> Settings:
    """Get application settings"""
    return settings

def update_ollama_config(base_url: Optional[str] = None, model: Optional[str] = None):
    """Update Ollama configuration at runtime"""
    global settings
    
    if base_url:
        settings.ollama_base_url = base_url
    
    if model and model in settings.available_models:
        settings.ollama_model = model
    
    return settings.get_model_config() 