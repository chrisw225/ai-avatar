"""
Text-to-Speech Service using GPT-SoVITS
Handles high-quality voice synthesis with voice cloning capabilities
"""

import asyncio
import logging
import time
import os
import sys
import tempfile
import subprocess
from typing import Optional, Dict, Any, List
from pathlib import Path

import torch
import numpy as np
import soundfile as sf

from ..config import get_settings

logger = logging.getLogger(__name__)

class TTSService:
    """Text-to-Speech service using GPT-SoVITS"""
    
    def __init__(self):
        self.settings = get_settings()
        self.is_initialized = False
        self.gpt_sovits_path = Path(self.settings.gpt_sovits_model_path)
        self.device = self._get_device()
        self.current_voice_model = None
        
        # Default voice settings (will be replaced with trained model)
        self.default_voice_config = {
            "reference_audio": None,
            "reference_text": "Hello, this is a test of the text to speech system.",
            "language": "en"
        }
        
    def _get_device(self) -> str:
        """Determine the best device for processing"""
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    async def initialize(self) -> None:
        """Initialize the TTS service"""
        if self.is_initialized:
            return
            
        try:
            logger.info("Initializing TTS service with GPT-SoVITS")
            logger.info(f"Using device: {self.device}")
            
            # Check if GPT-SoVITS is available
            if not self.gpt_sovits_path.exists():
                raise Exception(f"GPT-SoVITS not found at {self.gpt_sovits_path}")
            
            # Add GPT-SoVITS to Python path
            sys.path.insert(0, str(self.gpt_sovits_path))
            
            # Initialize GPT-SoVITS (this will be implemented based on actual API)
            await self._setup_gpt_sovits()
            
            # Load default voice model
            await self._load_default_voice()
            
            self.is_initialized = True
            logger.info("TTS Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS service: {e}")
            raise
    
    async def _setup_gpt_sovits(self) -> None:
        """Setup GPT-SoVITS environment"""
        try:
            # This is a placeholder for GPT-SoVITS initialization
            # The actual implementation will depend on GPT-SoVITS API
            logger.info("Setting up GPT-SoVITS environment")
            
            # Check for required models and download if needed
            models_dir = self.gpt_sovits_path / "pretrained_models"
            if not models_dir.exists():
                logger.info("Downloading GPT-SoVITS pretrained models...")
                await self._download_pretrained_models()
            
        except Exception as e:
            logger.error(f"GPT-SoVITS setup failed: {e}")
            raise
    
    async def _download_pretrained_models(self) -> None:
        """Download pretrained models for GPT-SoVITS"""
        # This would download the required pretrained models
        # Implementation depends on GPT-SoVITS requirements
        logger.info("Pretrained models download would be implemented here")
        pass
    
    async def _load_default_voice(self) -> None:
        """Load default voice model"""
        try:
            # For now, we'll use a placeholder default voice
            # In a real implementation, this would load a pre-trained voice model
            self.current_voice_model = "default"
            logger.info("Default voice model loaded")
            
        except Exception as e:
            logger.error(f"Failed to load default voice: {e}")
            raise
    
    async def synthesize_speech(
        self, 
        text: str, 
        voice_model: Optional[str] = None,
        language: str = "en",
        speed: float = 1.0,
        emotion: str = "neutral"
    ) -> Dict[str, Any]:
        """
        Synthesize speech from text
        
        Args:
            text: Text to synthesize
            voice_model: Voice model to use (None for default)
            language: Language code
            speed: Speech speed multiplier
            emotion: Emotion for synthesis
            
        Returns:
            Dictionary with audio file path and metadata
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            logger.info(f"Synthesizing speech: '{text[:50]}...'")
            
            # Create temporary output file
            output_file = tempfile.NamedTemporaryFile(
                suffix=".wav", 
                delete=False,
                dir=self.settings.temp_dir
            )
            output_path = output_file.name
            output_file.close()
            
            # Use specified voice model or default
            model_to_use = voice_model or self.current_voice_model
            
            # Synthesize speech using GPT-SoVITS
            success = await self._run_gpt_sovits_synthesis(
                text=text,
                output_path=output_path,
                voice_model=model_to_use,
                language=language,
                speed=speed,
                emotion=emotion
            )
            
            if not success:
                raise Exception("GPT-SoVITS synthesis failed")
            
            # Get audio duration
            duration = self._get_audio_duration(output_path)
            processing_time = time.time() - start_time
            
            result = {
                "audio_file": output_path,
                "text": text,
                "duration": duration,
                "voice_model": model_to_use,
                "language": language,
                "processing_time": processing_time,
                "sample_rate": self.settings.audio_sample_rate
            }
            
            logger.info(f"Speech synthesis completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            return {
                "audio_file": None,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def _run_gpt_sovits_synthesis(
        self,
        text: str,
        output_path: str,
        voice_model: str,
        language: str,
        speed: float,
        emotion: str
    ) -> bool:
        """Run GPT-SoVITS synthesis"""
        try:
            # This is a placeholder implementation
            # The actual implementation would call GPT-SoVITS API or command line tool
            
            # For now, create a simple sine wave as placeholder
            sample_rate = self.settings.audio_sample_rate
            duration = len(text) * 0.1  # Rough estimate: 0.1 seconds per character
            
            # Generate a simple tone (placeholder)
            t = np.linspace(0, duration, int(sample_rate * duration))
            frequency = 440  # A4 note
            audio = 0.3 * np.sin(2 * np.pi * frequency * t)
            
            # Add some variation to make it less monotonous
            audio *= np.exp(-t * 0.5)  # Fade out
            
            # Save as WAV file
            sf.write(output_path, audio, sample_rate)
            
            logger.info(f"Placeholder audio generated: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"GPT-SoVITS synthesis error: {e}")
            return False
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file"""
        try:
            import librosa
            duration = librosa.get_duration(filename=audio_path)
            return duration
        except Exception:
            # Fallback calculation
            try:
                data, sr = sf.read(audio_path)
                return len(data) / sr
            except Exception:
                return 0.0
    
    async def train_voice_model(
        self, 
        reference_audio_path: str,
        reference_text: str,
        model_name: str,
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Train a new voice model from reference audio
        
        Args:
            reference_audio_path: Path to reference audio file
            reference_text: Text corresponding to reference audio
            model_name: Name for the new voice model
            language: Language of the reference audio
            
        Returns:
            Dictionary with training results
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            logger.info(f"Training voice model: {model_name}")
            
            # Validate reference audio
            if not os.path.exists(reference_audio_path):
                raise Exception(f"Reference audio not found: {reference_audio_path}")
            
            # Check audio duration (should be around 1 minute as per requirements)
            duration = self._get_audio_duration(reference_audio_path)
            if duration < 30 or duration > 120:
                logger.warning(f"Reference audio duration ({duration:.1f}s) is not optimal (30-120s recommended)")
            
            # Run GPT-SoVITS training
            success = await self._run_gpt_sovits_training(
                reference_audio_path=reference_audio_path,
                reference_text=reference_text,
                model_name=model_name,
                language=language
            )
            
            if not success:
                raise Exception("Voice model training failed")
            
            processing_time = time.time() - start_time
            
            result = {
                "model_name": model_name,
                "reference_audio": reference_audio_path,
                "reference_text": reference_text,
                "language": language,
                "training_time": processing_time,
                "status": "completed"
            }
            
            logger.info(f"Voice model training completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Voice model training failed: {e}")
            return {
                "model_name": model_name,
                "error": str(e),
                "training_time": time.time() - start_time,
                "status": "failed"
            }
    
    async def _run_gpt_sovits_training(
        self,
        reference_audio_path: str,
        reference_text: str,
        model_name: str,
        language: str
    ) -> bool:
        """Run GPT-SoVITS voice model training"""
        try:
            # This is a placeholder for actual GPT-SoVITS training
            # The real implementation would:
            # 1. Preprocess the reference audio
            # 2. Extract voice features
            # 3. Fine-tune the model
            # 4. Save the trained model
            
            logger.info(f"Training voice model with GPT-SoVITS (placeholder)")
            
            # Simulate training time
            await asyncio.sleep(2)
            
            # Save model configuration
            model_config = {
                "name": model_name,
                "reference_audio": reference_audio_path,
                "reference_text": reference_text,
                "language": language,
                "created_at": time.time()
            }
            
            # In real implementation, save the trained model files
            logger.info(f"Voice model '{model_name}' training completed (placeholder)")
            return True
            
        except Exception as e:
            logger.error(f"GPT-SoVITS training error: {e}")
            return False
    
    async def list_voice_models(self) -> List[Dict[str, Any]]:
        """List available voice models"""
        try:
            # This would list all trained voice models
            # For now, return default model
            models = [
                {
                    "name": "default",
                    "language": "en",
                    "description": "Default voice model",
                    "created_at": time.time()
                }
            ]
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to list voice models: {e}")
            return []
    
    async def delete_voice_model(self, model_name: str) -> bool:
        """Delete a voice model"""
        try:
            if model_name == "default":
                logger.warning("Cannot delete default voice model")
                return False
            
            # This would delete the specified voice model
            logger.info(f"Voice model '{model_name}' deleted (placeholder)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete voice model: {e}")
            return False
    
    async def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return [
            "en", "zh", "ja", "ko", "es", "fr", "de", "it", "pt", "ru",
            "ar", "hi", "th", "vi", "id", "ms", "tr", "pl", "nl", "sv"
        ]
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Test synthesis with short text
            test_result = await self.synthesize_speech("Test", voice_model="default")
            
            return {
                "status": "healthy",
                "device": self.device,
                "current_voice_model": self.current_voice_model,
                "gpt_sovits_path": str(self.gpt_sovits_path),
                "test_synthesis_time": test_result.get("processing_time", 0),
                "initialized": self.is_initialized
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        # Clean up temporary files
        if self.settings.cleanup_temp_files:
            temp_dir = Path(self.settings.temp_dir)
            for file in temp_dir.glob("*.wav"):
                try:
                    if file.stat().st_mtime < time.time() - self.settings.max_temp_file_age:
                        file.unlink()
                except Exception:
                    pass
        
        self.is_initialized = False
        logger.info("TTS Service cleaned up")

# Global TTS service instance
tts_service = TTSService() 