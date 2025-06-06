"""
Audio processing utilities for the AI Chatbot application
Handles audio format conversion, validation, and processing
"""

import os
import logging
import tempfile
import subprocess
from typing import Optional, Tuple, Dict, Any
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from pydub.utils import which

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Audio processing utilities"""
    
    def __init__(self):
        self.supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac']
        self.target_sample_rate = 16000  # Standard for speech processing
        self.target_channels = 1  # Mono
        
        # Check for ffmpeg availability
        self.ffmpeg_available = which("ffmpeg") is not None
        if not self.ffmpeg_available:
            logger.warning("FFmpeg not found. Some audio processing features may be limited.")
    
    def validate_audio_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate audio file and return metadata
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with validation results and metadata
        """
        try:
            if not os.path.exists(file_path):
                return {"valid": False, "error": "File does not exist"}
            
            # Check file extension
            _, ext = os.path.splitext(file_path.lower())
            if ext not in self.supported_formats:
                return {
                    "valid": False, 
                    "error": f"Unsupported format: {ext}. Supported: {self.supported_formats}"
                }
            
            # Try to load audio metadata
            try:
                duration = librosa.get_duration(filename=file_path)
                sample_rate = librosa.get_samplerate(file_path)
                
                # Additional checks with pydub for more formats
                audio_segment = AudioSegment.from_file(file_path)
                channels = audio_segment.channels
                frame_rate = audio_segment.frame_rate
                
                return {
                    "valid": True,
                    "duration": duration,
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "frame_rate": frame_rate,
                    "format": ext,
                    "file_size": os.path.getsize(file_path)
                }
                
            except Exception as e:
                return {"valid": False, "error": f"Cannot read audio file: {str(e)}"}
                
        except Exception as e:
            logger.error(f"Audio validation error: {e}")
            return {"valid": False, "error": str(e)}
    
    def convert_to_wav(self, input_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert audio file to WAV format with standard parameters
        
        Args:
            input_path: Path to input audio file
            output_path: Path for output WAV file (optional)
            
        Returns:
            Dictionary with conversion results
        """
        try:
            if output_path is None:
                output_path = os.path.splitext(input_path)[0] + "_converted.wav"
            
            # Load audio with librosa
            audio_data, original_sr = librosa.load(input_path, sr=None)
            
            # Resample if necessary
            if original_sr != self.target_sample_rate:
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=original_sr, 
                    target_sr=self.target_sample_rate
                )
            
            # Ensure mono
            if len(audio_data.shape) > 1:
                audio_data = librosa.to_mono(audio_data)
            
            # Normalize audio
            audio_data = self.normalize_audio(audio_data)
            
            # Save as WAV
            sf.write(
                output_path, 
                audio_data, 
                self.target_sample_rate, 
                subtype='PCM_16'
            )
            
            return {
                "success": True,
                "output_path": output_path,
                "original_sample_rate": original_sr,
                "target_sample_rate": self.target_sample_rate,
                "duration": len(audio_data) / self.target_sample_rate
            }
            
        except Exception as e:
            logger.error(f"Audio conversion error: {e}")
            return {"success": False, "error": str(e)}
    
    def normalize_audio(self, audio_data: np.ndarray, target_db: float = -20.0) -> np.ndarray:
        """
        Normalize audio to target dB level
        
        Args:
            audio_data: Audio data as numpy array
            target_db: Target dB level
            
        Returns:
            Normalized audio data
        """
        try:
            # Calculate RMS
            rms = np.sqrt(np.mean(audio_data ** 2))
            
            if rms > 0:
                # Convert target dB to linear scale
                target_rms = 10 ** (target_db / 20.0)
                
                # Apply normalization
                scaling_factor = target_rms / rms
                normalized_audio = audio_data * scaling_factor
                
                # Prevent clipping
                max_val = np.max(np.abs(normalized_audio))
                if max_val > 1.0:
                    normalized_audio = normalized_audio / max_val * 0.95
                
                return normalized_audio
            else:
                return audio_data
                
        except Exception as e:
            logger.error(f"Audio normalization error: {e}")
            return audio_data
    
    def split_audio_by_silence(
        self, 
        audio_path: str, 
        min_silence_len: int = 1000,
        silence_thresh: int = -40,
        keep_silence: int = 500
    ) -> Dict[str, Any]:
        """
        Split audio file by silence detection
        
        Args:
            audio_path: Path to audio file
            min_silence_len: Minimum silence length in ms
            silence_thresh: Silence threshold in dB
            keep_silence: Amount of silence to keep in ms
            
        Returns:
            Dictionary with split results
        """
        try:
            # Load audio with pydub
            audio = AudioSegment.from_file(audio_path)
            
            # Split on silence
            chunks = audio.split_on_silence(
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh,
                keep_silence=keep_silence
            )
            
            # Save chunks
            chunk_paths = []
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            output_dir = os.path.dirname(audio_path)
            
            for i, chunk in enumerate(chunks):
                if len(chunk) > 500:  # Only save chunks longer than 500ms
                    chunk_path = os.path.join(output_dir, f"{base_name}_chunk_{i:03d}.wav")
                    chunk.export(chunk_path, format="wav")
                    chunk_paths.append(chunk_path)
            
            return {
                "success": True,
                "chunk_count": len(chunk_paths),
                "chunk_paths": chunk_paths,
                "total_duration": len(audio) / 1000.0
            }
            
        except Exception as e:
            logger.error(f"Audio splitting error: {e}")
            return {"success": False, "error": str(e)}
    
    def merge_audio_files(self, audio_paths: list, output_path: str) -> Dict[str, Any]:
        """
        Merge multiple audio files into one
        
        Args:
            audio_paths: List of audio file paths
            output_path: Output file path
            
        Returns:
            Dictionary with merge results
        """
        try:
            if not audio_paths:
                return {"success": False, "error": "No audio files provided"}
            
            # Load first audio file
            merged_audio = AudioSegment.from_file(audio_paths[0])
            
            # Append other files
            for audio_path in audio_paths[1:]:
                if os.path.exists(audio_path):
                    audio_segment = AudioSegment.from_file(audio_path)
                    merged_audio += audio_segment
                else:
                    logger.warning(f"Audio file not found: {audio_path}")
            
            # Export merged audio
            merged_audio.export(output_path, format="wav")
            
            return {
                "success": True,
                "output_path": output_path,
                "duration": len(merged_audio) / 1000.0,
                "files_merged": len(audio_paths)
            }
            
        except Exception as e:
            logger.error(f"Audio merging error: {e}")
            return {"success": False, "error": str(e)}
    
    def extract_audio_features(self, audio_path: str) -> Dict[str, Any]:
        """
        Extract audio features for analysis
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with extracted features
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.target_sample_rate)
            
            # Extract features
            features = {}
            
            # Basic features
            features['duration'] = len(y) / sr
            features['sample_rate'] = sr
            features['rms_energy'] = float(np.sqrt(np.mean(y ** 2)))
            features['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = [float(x) for x in np.mean(mfccs, axis=1)]
            features['mfcc_std'] = [float(x) for x in np.std(mfccs, axis=1)]
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo)
            
            return {
                "success": True,
                "features": features
            }
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return {"success": False, "error": str(e)}
    
    def apply_noise_reduction(self, audio_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Apply basic noise reduction to audio
        
        Args:
            audio_path: Path to input audio file
            output_path: Path for output file (optional)
            
        Returns:
            Dictionary with processing results
        """
        try:
            if output_path is None:
                output_path = os.path.splitext(audio_path)[0] + "_denoised.wav"
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.target_sample_rate)
            
            # Simple spectral gating for noise reduction
            # This is a basic implementation - for better results, use specialized libraries
            
            # Compute short-time Fourier transform
            stft = librosa.stft(y)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise floor from first 0.5 seconds
            noise_duration = min(int(0.5 * sr / 512), magnitude.shape[1])
            noise_profile = np.mean(magnitude[:, :noise_duration], axis=1, keepdims=True)
            
            # Apply spectral gating
            gate_threshold = 2.0  # Adjust as needed
            mask = magnitude > (noise_profile * gate_threshold)
            
            # Apply mask
            cleaned_magnitude = magnitude * mask
            
            # Reconstruct audio
            cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
            cleaned_audio = librosa.istft(cleaned_stft)
            
            # Normalize
            cleaned_audio = self.normalize_audio(cleaned_audio)
            
            # Save cleaned audio
            sf.write(output_path, cleaned_audio, sr, subtype='PCM_16')
            
            return {
                "success": True,
                "output_path": output_path,
                "noise_reduction_applied": True
            }
            
        except Exception as e:
            logger.error(f"Noise reduction error: {e}")
            return {"success": False, "error": str(e)}
    
    def create_silence(self, duration_seconds: float, output_path: str) -> Dict[str, Any]:
        """
        Create a silent audio file
        
        Args:
            duration_seconds: Duration of silence in seconds
            output_path: Output file path
            
        Returns:
            Dictionary with creation results
        """
        try:
            # Create silent audio
            samples = int(duration_seconds * self.target_sample_rate)
            silence = np.zeros(samples, dtype=np.float32)
            
            # Save as WAV
            sf.write(output_path, silence, self.target_sample_rate, subtype='PCM_16')
            
            return {
                "success": True,
                "output_path": output_path,
                "duration": duration_seconds
            }
            
        except Exception as e:
            logger.error(f"Silence creation error: {e}")
            return {"success": False, "error": str(e)}
    
    def get_audio_duration(self, audio_path: str) -> float:
        """
        Get audio file duration in seconds
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        try:
            return librosa.get_duration(filename=audio_path)
        except Exception as e:
            logger.error(f"Duration calculation error: {e}")
            return 0.0
    
    def convert_sample_rate(
        self, 
        input_path: str, 
        output_path: str, 
        target_sr: int
    ) -> Dict[str, Any]:
        """
        Convert audio file to different sample rate
        
        Args:
            input_path: Input audio file path
            output_path: Output audio file path
            target_sr: Target sample rate
            
        Returns:
            Dictionary with conversion results
        """
        try:
            # Load audio
            y, original_sr = librosa.load(input_path, sr=None)
            
            # Resample
            if original_sr != target_sr:
                y_resampled = librosa.resample(y, orig_sr=original_sr, target_sr=target_sr)
            else:
                y_resampled = y
            
            # Save resampled audio
            sf.write(output_path, y_resampled, target_sr, subtype='PCM_16')
            
            return {
                "success": True,
                "output_path": output_path,
                "original_sample_rate": original_sr,
                "target_sample_rate": target_sr
            }
            
        except Exception as e:
            logger.error(f"Sample rate conversion error: {e}")
            return {"success": False, "error": str(e)} 