"""
Configuration settings for the AI Chatbot application.
Loads settings from environment variables with sensible defaults.
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Server configuration
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    RELOAD: bool = Field(default=False, description="Enable auto-reload")
    FRONTEND_PORT: int = Field(default=3000, description="Frontend port")
    
    # Logging configuration
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FILE: str = Field(default="logs/app.log", description="Log file path")
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key")
    ELEVENLABS_API_KEY: Optional[str] = Field(default=None, description="ElevenLabs API key")
    HUGGINGFACE_API_KEY: Optional[str] = Field(default=None, description="HuggingFace API key")
    
    # Model configuration
    STT_MODEL: str = Field(default="whisper", description="Speech-to-text model")
    LLM_MODEL: str = Field(default="ollama", description="Language model")
    TTS_MODEL: str = Field(default="coqui", description="Text-to-speech model")
    
    # Ollama configuration
    OLLAMA_HOST: str = Field(default="localhost:11434", description="Ollama server host")
    OLLAMA_MODEL: str = Field(default="llama3.1:8b", description="Ollama model name")
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", description="Ollama base URL")
    
    # File storage configuration
    UPLOAD_DIR: str = Field(default="uploads", description="Upload directory")
    OUTPUT_DIR: str = Field(default="outputs", description="Output directory")
    MODELS_DIR: str = Field(default="models", description="Models directory")
    TEMP_DIR: str = Field(default="temp", description="Temporary directory")
    
    # File size limits (in bytes)
    MAX_AUDIO_FILE_SIZE: int = Field(default=50 * 1024 * 1024, description="Max audio file size")
    MAX_VIDEO_FILE_SIZE: int = Field(default=100 * 1024 * 1024, description="Max video file size")
    MAX_IMAGE_FILE_SIZE: int = Field(default=10 * 1024 * 1024, description="Max image file size")
    
    # Audio configuration
    AUDIO_SAMPLE_RATE: int = Field(default=16000, description="Audio sample rate")
    AUDIO_CHANNELS: int = Field(default=1, description="Audio channels")
    AUDIO_CHUNK_DURATION: float = Field(default=3.0, description="Audio chunk duration")
    AUDIO_FORMAT: str = Field(default="wav", description="Audio format")
    
    # Video configuration
    VIDEO_FPS: int = Field(default=30, description="Video FPS")
    VIDEO_RESOLUTION: str = Field(default="512x512", description="Video resolution")
    
    # Session configuration
    SESSION_TIMEOUT: int = Field(default=3600, description="Session timeout in seconds")
    MAX_SESSIONS: int = Field(default=100, description="Maximum concurrent sessions")
    
    # Security configuration
    ALLOWED_AUDIO_EXTENSIONS: List[str] = Field(
        default=[".wav", ".mp3", ".m4a", ".flac"],
        description="Allowed audio file extensions"
    )
    ALLOWED_VIDEO_EXTENSIONS: List[str] = Field(
        default=[".mp4", ".avi", ".mov"],
        description="Allowed video file extensions"
    )
    ALLOWED_IMAGE_EXTENSIONS: List[str] = Field(
        default=[".png", ".jpg", ".jpeg", ".bmp", ".tiff"],
        description="Allowed image file extensions"
    )
    
    # Rate limiting
    RATE_LIMIT_REQUESTS: int = Field(default=100, description="Rate limit requests per window")
    RATE_LIMIT_WINDOW: int = Field(default=3600, description="Rate limit window in seconds")
    
    # Performance configuration
    MAX_CONCURRENT_REQUESTS: int = Field(default=10, description="Max concurrent requests")
    REQUEST_TIMEOUT: int = Field(default=300, description="Request timeout in seconds")
    MAX_CONCURRENT_SESSIONS: int = Field(default=1, description="Max concurrent sessions")
    
    # Development configuration
    DEBUG: bool = Field(default=False, description="Debug mode")
    TESTING: bool = Field(default=False, description="Testing mode")
    
    # Model paths and configuration
    AVATAR_PATH: str = Field(default="avatar.png", description="Avatar image path")
    WHISPER_MODEL: str = Field(default="medium", description="Whisper model size")
    WHISPER_DEVICE: str = Field(default="cpu", description="Whisper device")
    GPT_SOVITS_MODEL_PATH: str = Field(default="external/GPT-SoVITS", description="GPT-SoVITS model path")
    SADTALKER_MODEL_PATH: str = Field(default="external/SadTalker", description="SadTalker model path")
    
    # Performance settings
    GPU_MEMORY_FRACTION: float = Field(default=0.5, description="GPU memory fraction")
    ENABLE_MODEL_CACHING: bool = Field(default=True, description="Enable model caching")
    BATCH_SIZE: int = Field(default=1, description="Batch size")
    
    # Cleanup settings
    CLEANUP_TEMP_FILES: bool = Field(default=True, description="Cleanup temporary files")
    MAX_TEMP_FILE_AGE: int = Field(default=3600, description="Max temp file age in seconds")
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Create global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the application settings."""
    return settings


def validate_settings() -> bool:
    """Validate that all required settings are properly configured."""
    errors = []
    
    # Check required directories
    required_dirs = [
        settings.UPLOAD_DIR,
        settings.OUTPUT_DIR,
        settings.MODELS_DIR,
        os.path.dirname(settings.LOG_FILE)
    ]
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create directory {directory}: {e}")
    
    # Check API keys for external services
    if settings.TTS_MODEL == "elevenlabs" and not settings.ELEVENLABS_API_KEY:
        errors.append("ELEVENLABS_API_KEY is required when using ElevenLabs TTS")
    
    if settings.STT_MODEL.startswith("openai/") and not settings.OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is required when using OpenAI models")
    
    # Validate file size limits
    if settings.MAX_AUDIO_FILE_SIZE <= 0:
        errors.append("MAX_AUDIO_FILE_SIZE must be positive")
    
    if settings.MAX_VIDEO_FILE_SIZE <= 0:
        errors.append("MAX_VIDEO_FILE_SIZE must be positive")
    
    # Validate audio configuration
    if settings.AUDIO_SAMPLE_RATE <= 0:
        errors.append("AUDIO_SAMPLE_RATE must be positive")
    
    if settings.AUDIO_CHANNELS not in [1, 2]:
        errors.append("AUDIO_CHANNELS must be 1 (mono) or 2 (stereo)")
    
    # Validate video configuration
    if settings.VIDEO_FPS <= 0:
        errors.append("VIDEO_FPS must be positive")
    
    # Validate session configuration
    if settings.SESSION_TIMEOUT <= 0:
        errors.append("SESSION_TIMEOUT must be positive")
    
    if settings.MAX_SESSIONS <= 0:
        errors.append("MAX_SESSIONS must be positive")
    
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


def print_settings():
    """Print current settings (excluding sensitive information)."""
    print("Current Application Settings:")
    print(f"  Host: {settings.HOST}")
    print(f"  Port: {settings.PORT}")
    print(f"  Log Level: {settings.LOG_LEVEL}")
    print(f"  STT Model: {settings.STT_MODEL}")
    print(f"  LLM Model: {settings.LLM_MODEL}")
    print(f"  TTS Model: {settings.TTS_MODEL}")
    print(f"  Ollama Host: {settings.OLLAMA_HOST}")
    print(f"  Upload Directory: {settings.UPLOAD_DIR}")
    print(f"  Output Directory: {settings.OUTPUT_DIR}")
    print(f"  Models Directory: {settings.MODELS_DIR}")
    print(f"  Debug Mode: {settings.DEBUG}")
    print(f"  Testing Mode: {settings.TESTING}") 