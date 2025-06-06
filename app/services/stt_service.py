"""
Speech-to-Text Service using Faster-Whisper
Handles real-time audio transcription with chunked processing
"""

import asyncio
import logging
import time
import tempfile
import os
from typing import Optional, AsyncGenerator, List
from pathlib import Path

import torch
import numpy as np
from faster_whisper import WhisperModel
import librosa
import soundfile as sf

from ..config import get_settings

logger = logging.getLogger(__name__)

class STTService:
    """Speech-to-Text service using Faster-Whisper"""
    
    def __init__(self):
        self.settings = get_settings()
        self.model: Optional[WhisperModel] = None
        self.is_initialized = False
        self.device = self._get_device()
        
    def _get_device(self) -> str:
        """Determine the best device for processing"""
        if torch.cuda.is_available() and self.settings.whisper_device == "cuda":
            return "cuda"
        return "cpu"
    
    async def initialize(self) -> None:
        """Initialize the Whisper model"""
        if self.is_initialized:
            return
            
        try:
            logger.info(f"Initializing Whisper model: {self.settings.whisper_model}")
            logger.info(f"Using device: {self.device}")
            
            # Initialize Faster-Whisper model
            self.model = WhisperModel(
                self.settings.whisper_model,
                device=self.device,
                compute_type="float16" if self.device == "cuda" else "int8"
            )
            
            self.is_initialized = True
            logger.info("STT Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize STT service: {e}")
            raise
    
    async def transcribe_audio(self, audio_file_path: str, language: Optional[str] = None) -> dict:
        """
        Transcribe audio file to text
        
        Args:
            audio_file_path: Path to audio file
            language: Language code (optional, auto-detect if None)
            
        Returns:
            Dictionary with transcription results
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            logger.info(f"Transcribing audio file: {audio_file_path}")
            
            # Transcribe with Faster-Whisper
            segments, info = self.model.transcribe(
                audio_file_path,
                language=language,
                beam_size=5,
                best_of=5,
                temperature=0.0,
                condition_on_previous_text=False
            )
            
            # Collect all segments
            transcription_segments = []
            full_text = ""
            
            for segment in segments:
                segment_data = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "confidence": getattr(segment, 'avg_logprob', 0.0)
                }
                transcription_segments.append(segment_data)
                full_text += segment.text.strip() + " "
            
            processing_time = time.time() - start_time
            
            result = {
                "text": full_text.strip(),
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "segments": transcription_segments,
                "processing_time": processing_time,
                "model_used": self.settings.whisper_model
            }
            
            logger.info(f"Transcription completed in {processing_time:.2f}s")
            logger.info(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                "text": "",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def transcribe_audio_chunk(self, audio_data: bytes, sample_rate: int = 16000) -> dict:
        """
        Transcribe audio chunk for real-time processing
        
        Args:
            audio_data: Raw audio bytes
            sample_rate: Audio sample rate
            
        Returns:
            Dictionary with transcription results
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Create temporary file for audio chunk
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                
                # Convert bytes to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                audio_float = audio_array.astype(np.float32) / 32768.0
                
                # Save as WAV file
                sf.write(temp_path, audio_float, sample_rate)
                
                # Transcribe the chunk
                result = await self.transcribe_audio(temp_path)
                
                # Clean up temporary file
                os.unlink(temp_path)
                
                return result
                
        except Exception as e:
            logger.error(f"Chunk transcription failed: {e}")
            return {
                "text": "",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def transcribe_stream(self, audio_stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[dict, None]:
        """
        Transcribe streaming audio in real-time
        
        Args:
            audio_stream: Async generator yielding audio chunks
            
        Yields:
            Transcription results for each chunk
        """
        if not self.is_initialized:
            await self.initialize()
        
        logger.info("Starting streaming transcription")
        
        chunk_buffer = []
        buffer_duration = 0.0
        target_duration = self.settings.audio_chunk_duration
        
        async for audio_chunk in audio_stream:
            try:
                chunk_buffer.append(audio_chunk)
                
                # Estimate buffer duration (assuming 16-bit audio at sample rate)
                chunk_samples = len(audio_chunk) // 2
                chunk_duration = chunk_samples / self.settings.audio_sample_rate
                buffer_duration += chunk_duration
                
                # Process when we have enough audio
                if buffer_duration >= target_duration:
                    # Combine chunks
                    combined_audio = b''.join(chunk_buffer)
                    
                    # Transcribe the combined chunk
                    result = await self.transcribe_audio_chunk(
                        combined_audio, 
                        self.settings.audio_sample_rate
                    )
                    
                    # Yield result if we got text
                    if result.get("text"):
                        yield result
                    
                    # Reset buffer
                    chunk_buffer = []
                    buffer_duration = 0.0
                    
            except Exception as e:
                logger.error(f"Stream transcription error: {e}")
                yield {
                    "text": "",
                    "error": str(e),
                    "processing_time": 0.0
                }
    
    def preprocess_audio(self, audio_file_path: str, target_sample_rate: int = 16000) -> str:
        """
        Preprocess audio file for optimal transcription
        
        Args:
            audio_file_path: Input audio file path
            target_sample_rate: Target sample rate for processing
            
        Returns:
            Path to preprocessed audio file
        """
        try:
            # Load audio file
            audio, sr = librosa.load(audio_file_path, sr=target_sample_rate)
            
            # Apply noise reduction (simple high-pass filter)
            audio = librosa.effects.preemphasis(audio)
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            # Create output path
            output_path = audio_file_path.replace('.wav', '_processed.wav')
            
            # Save preprocessed audio
            sf.write(output_path, audio, target_sample_rate)
            
            logger.info(f"Audio preprocessed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            return audio_file_path  # Return original if preprocessing fails
    
    async def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        # Whisper supported languages
        return [
            "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl",
            "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro",
            "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy",
            "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk", "br", "eu",
            "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km",
            "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo",
            "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg",
            "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"
        ]
    
    async def health_check(self) -> dict:
        """Check service health"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            return {
                "status": "healthy",
                "model": self.settings.whisper_model,
                "device": self.device,
                "initialized": self.is_initialized
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        if self.model:
            # Faster-Whisper doesn't need explicit cleanup
            self.model = None
        self.is_initialized = False
        logger.info("STT Service cleaned up")

# Global STT service instance
stt_service = STTService() 