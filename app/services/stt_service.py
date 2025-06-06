"""
Speech-to-Text (STT) Service
Handles audio transcription using various STT models.
"""

import os
import logging
import asyncio
from typing import Optional, Dict, Any
from pathlib import Path

import torch
import whisper
import librosa
import soundfile as sf

from app.config import settings

logger = logging.getLogger(__name__)


class STTService:
    """Speech-to-Text service for audio transcription."""
    
    def __init__(self):
        """Initialize the STT service."""
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = settings.STT_MODEL
        self.sample_rate = settings.AUDIO_SAMPLE_RATE
        
        logger.info(f"Initializing STT service with model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
    
    async def initialize(self):
        """Initialize the STT model."""
        try:
            if self.model_name.startswith("openai/whisper"):
                model_size = self.model_name.split("-")[-1] if "-" in self.model_name else "base"
                self.model = whisper.load_model(model_size, device=self.device)
                logger.info(f"Loaded Whisper model: {model_size}")
            else:
                # Default to Whisper base model
                self.model = whisper.load_model("base", device=self.device)
                logger.info("Loaded default Whisper base model")
                
        except Exception as e:
            logger.error(f"Failed to initialize STT model: {e}")
            raise
    
    async def transcribe_audio(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to the audio file
            language: Optional language code for transcription
            
        Returns:
            Dictionary containing transcription results
        """
        try:
            if not self.model:
                await self.initialize()
            
            # Validate audio file
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Preprocess audio
            processed_audio_path = await self._preprocess_audio(audio_path)
            
            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._transcribe_sync, 
                processed_audio_path, 
                language
            )
            
            # Clean up processed audio if it's different from original
            if processed_audio_path != audio_path:
                try:
                    os.remove(processed_audio_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up processed audio: {e}")
            
            return {
                "text": result["text"].strip(),
                "language": result.get("language", "unknown"),
                "confidence": self._calculate_confidence(result),
                "segments": result.get("segments", []),
                "duration": result.get("duration", 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            raise
    
    def _transcribe_sync(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Synchronous transcription method."""
        options = {
            "fp16": self.device == "cuda",
            "language": language,
            "task": "transcribe"
        }
        
        # Remove None values
        options = {k: v for k, v in options.items() if v is not None}
        
        result = self.model.transcribe(audio_path, **options)
        return result
    
    async def _preprocess_audio(self, audio_path: str) -> str:
        """
        Preprocess audio file for optimal transcription.
        
        Args:
            audio_path: Path to the original audio file
            
        Returns:
            Path to the processed audio file
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Check if preprocessing is needed
            if sr == self.sample_rate and len(audio.shape) == 1:
                return audio_path  # No preprocessing needed
            
            # Create processed file path
            processed_path = audio_path.replace(
                Path(audio_path).suffix, 
                f"_processed{Path(audio_path).suffix}"
            )
            
            # Save processed audio
            sf.write(processed_path, audio, self.sample_rate)
            
            logger.debug(f"Preprocessed audio: {audio_path} -> {processed_path}")
            return processed_path
            
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            return audio_path  # Return original if preprocessing fails
    
    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """
        Calculate confidence score from transcription result.
        
        Args:
            result: Whisper transcription result
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            if "segments" in result and result["segments"]:
                # Calculate average confidence from segments
                confidences = []
                for segment in result["segments"]:
                    if "avg_logprob" in segment:
                        # Convert log probability to confidence
                        confidence = min(1.0, max(0.0, (segment["avg_logprob"] + 1.0)))
                        confidences.append(confidence)
                
                if confidences:
                    return sum(confidences) / len(confidences)
            
            # Default confidence based on text length and quality
            text = result.get("text", "").strip()
            if len(text) > 10:
                return 0.8
            elif len(text) > 0:
                return 0.6
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Error calculating confidence: {e}")
            return 0.5  # Default confidence
    
    async def detect_language(self, audio_path: str) -> str:
        """
        Detect the language of the audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Detected language code
        """
        try:
            if not self.model:
                await self.initialize()
            
            # Load audio for language detection
            audio = whisper.load_audio(audio_path)
            audio = whisper.pad_or_trim(audio)
            
            # Make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            
            # Detect the spoken language
            _, probs = self.model.detect_language(mel)
            detected_language = max(probs, key=probs.get)
            
            logger.info(f"Detected language: {detected_language} (confidence: {probs[detected_language]:.2f})")
            return detected_language
            
        except Exception as e:
            logger.error(f"Error detecting language: {e}")
            return "en"  # Default to English
    
    async def health_check(self) -> str:
        """Check the health of the STT service."""
        try:
            if not self.model:
                return "not_initialized"
            
            # Test with a small dummy audio
            test_audio = torch.zeros(16000).to(self.device)  # 1 second of silence
            
            # Quick test transcription
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.model.transcribe(test_audio.cpu().numpy())
            )
            
            return "healthy"
            
        except Exception as e:
            logger.error(f"STT health check failed: {e}")
            return f"error: {str(e)}"
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            if self.model:
                del self.model
                self.model = None
            
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("STT service cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up STT service: {e}")
    
    def get_supported_formats(self) -> list:
        """Get list of supported audio formats."""
        return ["wav", "mp3", "m4a", "flac", "ogg", "aac"]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "sample_rate": self.sample_rate,
            "supported_formats": self.get_supported_formats(),
            "initialized": self.model is not None
        }

# Global STT service instance
stt_service = STTService() 