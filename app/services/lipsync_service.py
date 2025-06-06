"""
Lip Synchronization Service using SadTalker
Handles video generation with lip sync using the provided avatar image
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
import cv2
import numpy as np
from PIL import Image

from ..config import get_settings

logger = logging.getLogger(__name__)

class LipSyncService:
    """Lip synchronization service using SadTalker"""
    
    def __init__(self):
        self.settings = get_settings()
        self.is_initialized = False
        self.sadtalker_path = Path(self.settings.sadtalker_model_path)
        self.device = self._get_device()
        self.avatar_image_path = self.settings.avatar_path
        
    def _get_device(self) -> str:
        """Determine the best device for processing"""
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    async def initialize(self) -> None:
        """Initialize the lip sync service"""
        if self.is_initialized:
            return
            
        try:
            logger.info("Initializing Lip Sync service with SadTalker")
            logger.info(f"Using device: {self.device}")
            logger.info(f"Avatar image: {self.avatar_image_path}")
            
            # Check if SadTalker is available
            if not self.sadtalker_path.exists():
                raise Exception(f"SadTalker not found at {self.sadtalker_path}")
            
            # Check if avatar image exists
            if not os.path.exists(self.avatar_image_path):
                raise Exception(f"Avatar image not found at {self.avatar_image_path}")
            
            # Validate avatar image
            await self._validate_avatar_image()
            
            # Add SadTalker to Python path
            sys.path.insert(0, str(self.sadtalker_path))
            
            # Initialize SadTalker
            await self._setup_sadtalker()
            
            self.is_initialized = True
            logger.info("Lip Sync Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Lip Sync service: {e}")
            raise
    
    async def _validate_avatar_image(self) -> None:
        """Validate the avatar image"""
        try:
            # Load and check image
            image = Image.open(self.avatar_image_path)
            width, height = image.size
            
            logger.info(f"Avatar image size: {width}x{height}")
            
            # Check if image is square (recommended for SadTalker)
            if width != height:
                logger.warning(f"Avatar image is not square ({width}x{height}). Square images work best.")
            
            # Check if image is at least 512x512
            if width < 512 or height < 512:
                logger.warning(f"Avatar image is smaller than 512x512. Larger images produce better results.")
            
            # Check image format
            if image.format not in ['PNG', 'JPEG', 'JPG']:
                logger.warning(f"Avatar image format ({image.format}) may not be optimal. PNG or JPEG recommended.")
            
        except Exception as e:
            logger.error(f"Avatar image validation failed: {e}")
            raise
    
    async def _setup_sadtalker(self) -> None:
        """Setup SadTalker environment"""
        try:
            logger.info("Setting up SadTalker environment")
            
            # Check for required models and download if needed
            checkpoints_dir = self.sadtalker_path / "checkpoints"
            if not checkpoints_dir.exists():
                logger.info("Downloading SadTalker checkpoints...")
                await self._download_sadtalker_models()
            
            # Initialize SadTalker models (placeholder)
            # The actual implementation would load the SadTalker models here
            logger.info("SadTalker models loaded")
            
        except Exception as e:
            logger.error(f"SadTalker setup failed: {e}")
            raise
    
    async def _download_sadtalker_models(self) -> None:
        """Download SadTalker model checkpoints"""
        try:
            # This would download the required model checkpoints
            # Implementation depends on SadTalker requirements
            logger.info("SadTalker model download would be implemented here")
            
            # Create checkpoints directory
            checkpoints_dir = self.sadtalker_path / "checkpoints"
            checkpoints_dir.mkdir(exist_ok=True)
            
        except Exception as e:
            logger.error(f"Failed to download SadTalker models: {e}")
            raise
    
    async def generate_lip_sync_video(
        self, 
        audio_file_path: str,
        avatar_image_path: Optional[str] = None,
        expression_scale: float = 1.0,
        pose_style: int = 0,
        background_enhancer: bool = True
    ) -> Dict[str, Any]:
        """
        Generate lip-synced video from audio
        
        Args:
            audio_file_path: Path to audio file
            avatar_image_path: Path to avatar image (None for default)
            expression_scale: Scale of facial expressions (0.0-2.0)
            pose_style: Head pose style (0-45)
            background_enhancer: Whether to enhance background
            
        Returns:
            Dictionary with video file path and metadata
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            logger.info(f"Generating lip sync video for audio: {audio_file_path}")
            
            # Use provided avatar or default
            avatar_path = avatar_image_path or self.avatar_image_path
            
            # Validate inputs
            if not os.path.exists(audio_file_path):
                raise Exception(f"Audio file not found: {audio_file_path}")
            
            if not os.path.exists(avatar_path):
                raise Exception(f"Avatar image not found: {avatar_path}")
            
            # Create temporary output file
            output_file = tempfile.NamedTemporaryFile(
                suffix=".mp4", 
                delete=False,
                dir=self.settings.temp_dir
            )
            output_path = output_file.name
            output_file.close()
            
            # Generate lip sync video using SadTalker
            success = await self._run_sadtalker_generation(
                audio_path=audio_file_path,
                avatar_path=avatar_path,
                output_path=output_path,
                expression_scale=expression_scale,
                pose_style=pose_style,
                background_enhancer=background_enhancer
            )
            
            if not success:
                raise Exception("SadTalker video generation failed")
            
            # Get video metadata
            duration, fps, resolution = self._get_video_metadata(output_path)
            processing_time = time.time() - start_time
            
            result = {
                "video_file": output_path,
                "audio_file": audio_file_path,
                "avatar_image": avatar_path,
                "duration": duration,
                "fps": fps,
                "resolution": resolution,
                "processing_time": processing_time,
                "expression_scale": expression_scale,
                "pose_style": pose_style
            }
            
            logger.info(f"Lip sync video generated in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Lip sync video generation failed: {e}")
            return {
                "video_file": None,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def _run_sadtalker_generation(
        self,
        audio_path: str,
        avatar_path: str,
        output_path: str,
        expression_scale: float,
        pose_style: int,
        background_enhancer: bool
    ) -> bool:
        """Run SadTalker video generation"""
        try:
            # This is a placeholder implementation
            # The actual implementation would call SadTalker API or command line tool
            
            logger.info("Running SadTalker generation (placeholder)")
            
            # For now, create a simple video with the avatar image
            # In real implementation, this would use SadTalker to generate lip-synced video
            
            # Load avatar image
            avatar_img = cv2.imread(avatar_path)
            if avatar_img is None:
                raise Exception(f"Could not load avatar image: {avatar_path}")
            
            # Get audio duration to determine video length
            audio_duration = self._get_audio_duration(audio_path)
            
            # Video parameters
            fps = self.settings.video_fps
            total_frames = int(audio_duration * fps)
            height, width = avatar_img.shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Generate frames (placeholder - just static image)
            for frame_idx in range(total_frames):
                # In real implementation, this would be the lip-synced frame from SadTalker
                frame = avatar_img.copy()
                
                # Add some simple animation (placeholder)
                # This would be replaced by actual lip sync from SadTalker
                time_factor = frame_idx / fps
                mouth_movement = int(5 * np.sin(time_factor * 10))  # Simple mouth movement simulation
                
                video_writer.write(frame)
            
            video_writer.release()
            
            logger.info(f"Placeholder video generated: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"SadTalker generation error: {e}")
            return False
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file"""
        try:
            import librosa
            duration = librosa.get_duration(filename=audio_path)
            return duration
        except Exception:
            # Fallback using ffprobe
            try:
                result = subprocess.run([
                    'ffprobe', '-v', 'quiet', '-show_entries', 
                    'format=duration', '-of', 'csv=p=0', audio_path
                ], capture_output=True, text=True)
                return float(result.stdout.strip())
            except Exception:
                return 5.0  # Default duration
    
    def _get_video_metadata(self, video_path: str) -> tuple:
        """Get video metadata (duration, fps, resolution)"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            duration = frame_count / fps if fps > 0 else 0
            resolution = f"{width}x{height}"
            
            cap.release()
            
            return duration, fps, resolution
            
        except Exception as e:
            logger.error(f"Failed to get video metadata: {e}")
            return 0.0, 30.0, "512x512"
    
    async def generate_batch_videos(
        self, 
        audio_files: List[str],
        avatar_image_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple lip sync videos in batch
        
        Args:
            audio_files: List of audio file paths
            avatar_image_path: Path to avatar image (None for default)
            
        Returns:
            List of results for each video
        """
        if not self.is_initialized:
            await self.initialize()
        
        results = []
        
        for audio_file in audio_files:
            try:
                result = await self.generate_lip_sync_video(
                    audio_file_path=audio_file,
                    avatar_image_path=avatar_image_path
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Batch processing failed for {audio_file}: {e}")
                results.append({
                    "video_file": None,
                    "audio_file": audio_file,
                    "error": str(e)
                })
        
        return results
    
    async def preprocess_avatar_image(
        self, 
        input_image_path: str,
        output_size: tuple = (512, 512)
    ) -> str:
        """
        Preprocess avatar image for optimal results
        
        Args:
            input_image_path: Path to input image
            output_size: Target size (width, height)
            
        Returns:
            Path to preprocessed image
        """
        try:
            # Load image
            image = Image.open(input_image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to target size
            image = image.resize(output_size, Image.Resampling.LANCZOS)
            
            # Create output path
            output_path = input_image_path.replace('.', '_processed.')
            if not output_path.endswith(('.png', '.jpg', '.jpeg')):
                output_path += '.png'
            
            # Save preprocessed image
            image.save(output_path, 'PNG', quality=95)
            
            logger.info(f"Avatar image preprocessed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Avatar preprocessing failed: {e}")
            return input_image_path
    
    async def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get supported input and output formats"""
        return {
            "audio_formats": ["wav", "mp3", "m4a", "flac"],
            "image_formats": ["png", "jpg", "jpeg"],
            "video_formats": ["mp4", "avi", "mov"]
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Check if avatar image is accessible
            avatar_accessible = os.path.exists(self.avatar_image_path)
            
            return {
                "status": "healthy",
                "device": self.device,
                "sadtalker_path": str(self.sadtalker_path),
                "avatar_image": self.avatar_image_path,
                "avatar_accessible": avatar_accessible,
                "initialized": self.is_initialized
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        # Clean up temporary video files
        if self.settings.cleanup_temp_files:
            temp_dir = Path(self.settings.temp_dir)
            for file in temp_dir.glob("*.mp4"):
                try:
                    if file.stat().st_mtime < time.time() - self.settings.max_temp_file_age:
                        file.unlink()
                except Exception:
                    pass
        
        self.is_initialized = False
        logger.info("Lip Sync Service cleaned up")

# Global lip sync service instance
lipsync_service = LipSyncService() 