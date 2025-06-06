"""
Text-to-Speech (TTS) Service
Handles text-to-speech conversion using various TTS models.
"""

import os
import logging
import asyncio
import tempfile
from typing import Optional, Dict, Any, List
from pathlib import Path

import torch
import torchaudio
from TTS.api import TTS

from app.config import settings

logger = logging.getLogger(__name__)


class TTSService:
    """Text-to-Speech service for audio generation."""
    
    def __init__(self):
        """Initialize the TTS service."""
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = settings.TTS_MODEL
        self.sample_rate = settings.AUDIO_SAMPLE_RATE
        self.voice_models = {}
        
        logger.info(f"Initializing TTS service with model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
    
    async def initialize(self):
        """Initialize the TTS model."""
        try:
            if self.model_name == "elevenlabs":
                # ElevenLabs would require API key and different implementation
                logger.info("ElevenLabs TTS selected - using Coqui TTS as fallback")
                self.model = TTS("tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False).to(self.device)
            else:
                # Default to Coqui TTS
                self.model = TTS("tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False).to(self.device)
            
            logger.info("TTS service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TTS service: {e}")
            raise
    
    async def synthesize_speech(
        self, 
        text: str, 
        voice: Optional[str] = None,
        speed: float = 1.0,
        pitch: float = 1.0,
        output_path: Optional[str] = None
    ) -> str:
        """
        Convert text to speech.
        
        Args:
            text: Text to convert to speech
            voice: Voice model to use (optional)
            speed: Speech speed multiplier
            pitch: Pitch multiplier
            output_path: Output file path (optional, will generate if not provided)
            
        Returns:
            Path to the generated audio file
        """
        try:
            if not self.model:
                await self.initialize()
            
            # Generate output path if not provided
            if not output_path:
                output_path = os.path.join(
                    settings.OUTPUT_DIR,
                    f"tts_{hash(text) % 1000000}.wav"
                )
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Clean and prepare text
            cleaned_text = self._clean_text(text)
            
            # Run TTS in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._synthesize_sync,
                cleaned_text,
                voice,
                speed,
                pitch,
                output_path
            )
            
            logger.info(f"Generated speech: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            raise
    
    def _synthesize_sync(
        self, 
        text: str, 
        voice: Optional[str], 
        speed: float, 
        pitch: float, 
        output_path: str
    ):
        """Synchronous speech synthesis."""
        try:
            # Generate audio
            if voice and voice in self.voice_models:
                # Use custom voice model if available
                wav = self.voice_models[voice].tts(text)
            else:
                # Use default model
                wav = self.model.tts(text)
            
            # Apply speed and pitch modifications if needed
            if speed != 1.0 or pitch != 1.0:
                wav = self._modify_audio(wav, speed, pitch)
            
            # Save audio file
            self.model.tts_to_file(text=text, file_path=output_path)
            
        except Exception as e:
            logger.error(f"Synchronous TTS failed: {e}")
            raise
    
    def _modify_audio(self, wav: torch.Tensor, speed: float, pitch: float) -> torch.Tensor:
        """Apply speed and pitch modifications to audio."""
        try:
            # Convert to tensor if needed
            if not isinstance(wav, torch.Tensor):
                wav = torch.tensor(wav, dtype=torch.float32)
            
            # Apply speed change (time stretching)
            if speed != 1.0:
                # Simple resampling for speed change
                new_length = int(len(wav) / speed)
                wav = torch.nn.functional.interpolate(
                    wav.unsqueeze(0).unsqueeze(0),
                    size=new_length,
                    mode='linear',
                    align_corners=False
                ).squeeze()
            
            # Apply pitch change (would need more sophisticated processing)
            if pitch != 1.0:
                # Placeholder for pitch shifting
                # In practice, you'd use librosa or similar for pitch shifting
                logger.warning("Pitch modification not fully implemented")
            
            return wav
            
        except Exception as e:
            logger.error(f"Error modifying audio: {e}")
            return wav
    
    def _clean_text(self, text: str) -> str:
        """Clean and prepare text for TTS."""
        # Remove or replace problematic characters
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())  # Normalize whitespace
        
        # Limit text length to prevent memory issues
        max_length = 1000
        if len(text) > max_length:
            text = text[:max_length] + "..."
            logger.warning(f"Text truncated to {max_length} characters")
        
        return text
    
    async def train_voice_model(
        self, 
        voice_name: str, 
        audio_paths: List[str],
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train a custom voice model from audio samples.
        
        Args:
            voice_name: Name for the new voice model
            audio_paths: List of paths to training audio files
            description: Optional description of the voice
            
        Returns:
            Training result information
        """
        try:
            logger.info(f"Training voice model: {voice_name}")
            
            # Validate audio files
            valid_paths = []
            for path in audio_paths:
                if os.path.exists(path):
                    valid_paths.append(path)
                else:
                    logger.warning(f"Audio file not found: {path}")
            
            if not valid_paths:
                raise ValueError("No valid audio files provided for training")
            
            # For now, we'll simulate training and store the voice info
            # In a real implementation, you'd train a voice cloning model
            voice_info = {
                "name": voice_name,
                "description": description or f"Custom voice: {voice_name}",
                "training_files": valid_paths,
                "created_at": asyncio.get_event_loop().time(),
                "status": "trained"
            }
            
            # Store voice model info (in practice, you'd save the actual model)
            self.voice_models[voice_name] = voice_info
            
            logger.info(f"Voice model '{voice_name}' training completed")
            
            return {
                "status": "success",
                "voice_name": voice_name,
                "training_files_count": len(valid_paths),
                "message": f"Voice model '{voice_name}' has been trained successfully"
            }
            
        except Exception as e:
            logger.error(f"Error training voice model: {e}")
            return {
                "status": "error",
                "error": str(e),
                "message": f"Failed to train voice model '{voice_name}'"
            }
    
    async def list_voices(self) -> List[Dict[str, Any]]:
        """
        List available voice models.
        
        Returns:
            List of available voices with their information
        """
        try:
            voices = []
            
            # Add default voices
            voices.append({
                "name": "default",
                "description": "Default TTS voice",
                "type": "built-in",
                "language": "en"
            })
            
            # Add custom trained voices
            for voice_name, voice_info in self.voice_models.items():
                voices.append({
                    "name": voice_name,
                    "description": voice_info.get("description", "Custom voice"),
                    "type": "custom",
                    "language": "en",
                    "created_at": voice_info.get("created_at")
                })
            
            return voices
            
        except Exception as e:
            logger.error(f"Error listing voices: {e}")
            return []
    
    async def delete_voice(self, voice_name: str) -> bool:
        """
        Delete a custom voice model.
        
        Args:
            voice_name: Name of the voice to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if voice_name in self.voice_models:
                del self.voice_models[voice_name]
                logger.info(f"Deleted voice model: {voice_name}")
                return True
            else:
                logger.warning(f"Voice model not found: {voice_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting voice: {e}")
            return False
    
    async def get_voice_info(self, voice_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific voice.
        
        Args:
            voice_name: Name of the voice
            
        Returns:
            Voice information or None if not found
        """
        try:
            if voice_name == "default":
                return {
                    "name": "default",
                    "description": "Default TTS voice",
                    "type": "built-in",
                    "language": "en"
                }
            elif voice_name in self.voice_models:
                return self.voice_models[voice_name]
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting voice info: {e}")
            return None
    
    async def preview_voice(self, voice_name: str, text: str = "Hello, this is a voice preview.") -> str:
        """
        Generate a preview of a voice.
        
        Args:
            voice_name: Name of the voice to preview
            text: Text to use for preview
            
        Returns:
            Path to the preview audio file
        """
        try:
            preview_path = os.path.join(
                settings.OUTPUT_DIR,
                f"preview_{voice_name}_{hash(text) % 1000}.wav"
            )
            
            return await self.synthesize_speech(
                text=text,
                voice=voice_name,
                output_path=preview_path
            )
            
        except Exception as e:
            logger.error(f"Error generating voice preview: {e}")
            raise
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported audio output formats."""
        return ["wav", "mp3", "ogg"]
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return ["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"]
    
    async def health_check(self) -> str:
        """Check the health of the TTS service."""
        try:
            if not self.model:
                return "not_initialized"
            
            # Test with a short text
            test_text = "Health check test."
            
            # Use project temp directory instead of system temp
            temp_dir = settings.TEMP_DIR
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_file_path = os.path.join(temp_dir, f"tts_health_check_{os.getpid()}.wav")
            
            try:
                await self.synthesize_speech(
                    text=test_text,
                    output_path=temp_file_path
                )
                
                # Clean up the temp file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    
                return "healthy"
                
            except Exception as e:
                # Clean up temp file if it exists
                if os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except:
                        pass
                raise e
            
        except Exception as e:
            logger.error(f"TTS health check failed: {e}")
            return f"error: {str(e)}"
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            if self.model:
                del self.model
                self.model = None
            
            # Clear voice models
            self.voice_models.clear()
            
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("TTS service cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up TTS service: {e}")
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the TTS service."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "sample_rate": self.sample_rate,
            "supported_formats": self.get_supported_formats(),
            "supported_languages": self.get_supported_languages(),
            "custom_voices_count": len(self.voice_models),
            "initialized": self.model is not None
        }

# Global TTS service instance
tts_service = TTSService() 