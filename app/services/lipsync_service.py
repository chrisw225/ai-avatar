"""
Lip Sync Service
Handles lip synchronization and avatar video generation.
"""

import os
import logging
import asyncio
import tempfile
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)


class LipSyncService:
    """Lip synchronization service for avatar video generation."""
    
    def __init__(self):
        """Initialize the LipSync service."""
        self.model = None
        self.avatar_image_path = None
        self.video_fps = settings.VIDEO_FPS
        self.video_resolution = self._parse_resolution(settings.VIDEO_RESOLUTION)
        
        logger.info("Initializing LipSync service")
        logger.info(f"Video FPS: {self.video_fps}")
        logger.info(f"Video resolution: {self.video_resolution}")
    
    def _parse_resolution(self, resolution_str: str) -> Tuple[int, int]:
        """Parse resolution string like '512x512' to tuple (width, height)."""
        try:
            width, height = map(int, resolution_str.split('x'))
            return (width, height)
        except:
            logger.warning(f"Invalid resolution format: {resolution_str}, using default 512x512")
            return (512, 512)
    
    async def initialize(self):
        """Initialize the LipSync service."""
        try:
            # Check for default avatar image
            avatar_path = "avatar.png"
            if os.path.exists(avatar_path):
                self.avatar_image_path = avatar_path
                logger.info(f"Found avatar image: {avatar_path}")
            else:
                # Create a default avatar if none exists
                await self._create_default_avatar()
            
            logger.info("LipSync service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LipSync service: {e}")
            raise
    
    async def _create_default_avatar(self):
        """Create a default avatar image."""
        try:
            # Create a simple default avatar
            width, height = self.video_resolution
            
            # Create a simple face-like image
            avatar = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray background
            
            # Draw a simple face
            center_x, center_y = width // 2, height // 2
            
            # Face outline (circle)
            cv2.circle(avatar, (center_x, center_y), min(width, height) // 3, (200, 180, 160), -1)
            
            # Eyes
            eye_y = center_y - height // 8
            eye_offset = width // 8
            cv2.circle(avatar, (center_x - eye_offset, eye_y), width // 20, (50, 50, 50), -1)
            cv2.circle(avatar, (center_x + eye_offset, eye_y), width // 20, (50, 50, 50), -1)
            
            # Mouth area (will be animated)
            mouth_y = center_y + height // 8
            cv2.ellipse(avatar, (center_x, mouth_y), (width // 12, height // 24), 0, 0, 180, (100, 50, 50), 2)
            
            # Save default avatar
            self.avatar_image_path = "avatar.png"
            cv2.imwrite(self.avatar_image_path, avatar)
            
            logger.info(f"Created default avatar: {self.avatar_image_path}")
            
        except Exception as e:
            logger.error(f"Error creating default avatar: {e}")
            raise
    
    async def generate_lipsync_video(
        self, 
        audio_path: str, 
        avatar_path: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a lip-synced video from audio and avatar image.
        
        Args:
            audio_path: Path to the audio file
            avatar_path: Path to the avatar image (optional, uses default if not provided)
            output_path: Output video path (optional, will generate if not provided)
            
        Returns:
            Path to the generated video file
        """
        try:
            if not self.avatar_image_path and not avatar_path:
                await self.initialize()
            
            # Use provided avatar or default
            avatar_to_use = avatar_path or self.avatar_image_path
            
            if not os.path.exists(avatar_to_use):
                raise FileNotFoundError(f"Avatar image not found: {avatar_to_use}")
            
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Generate output path if not provided
            if not output_path:
                output_path = os.path.join(
                    settings.OUTPUT_DIR,
                    f"lipsync_{hash(audio_path) % 1000000}.mp4"
                )
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Get audio duration
            audio_duration = await self._get_audio_duration(audio_path)
            
            # Run lip sync generation in thread pool
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._generate_video_sync,
                avatar_to_use,
                audio_path,
                output_path,
                audio_duration
            )
            
            logger.info(f"Generated lip-sync video: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating lip-sync video: {e}")
            raise
    
    def _generate_video_sync(
        self, 
        avatar_path: str, 
        audio_path: str, 
        output_path: str, 
        duration: float
    ):
        """Generate video synchronously."""
        try:
            # Load avatar image
            avatar_img = cv2.imread(avatar_path)
            if avatar_img is None:
                raise ValueError(f"Could not load avatar image: {avatar_path}")
            
            # Resize avatar to target resolution
            avatar_img = cv2.resize(avatar_img, self.video_resolution)
            
            # Calculate total frames
            total_frames = int(duration * self.video_fps)
            
            # Set up video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                output_path, 
                fourcc, 
                self.video_fps, 
                self.video_resolution
            )
            
            try:
                # Generate frames with simple mouth animation
                for frame_idx in range(total_frames):
                    frame = self._generate_frame(avatar_img.copy(), frame_idx, total_frames)
                    video_writer.write(frame)
                
            finally:
                video_writer.release()
            
            # Add audio to video using ffmpeg (if available)
            self._add_audio_to_video_sync(output_path, audio_path)
            
        except Exception as e:
            logger.error(f"Synchronous video generation failed: {e}")
            raise
    
    def _generate_frame(self, avatar_img: np.ndarray, frame_idx: int, total_frames: int) -> np.ndarray:
        """Generate a single frame with mouth animation."""
        try:
            height, width = avatar_img.shape[:2]
            center_x, center_y = width // 2, height // 2
            
            # Simple mouth animation based on frame index
            # This creates a basic opening/closing mouth effect
            mouth_y = center_y + height // 8
            
            # Create a simple oscillating mouth movement
            time_factor = (frame_idx / total_frames) * 10  # Adjust speed
            mouth_openness = abs(np.sin(time_factor)) * 0.5 + 0.2  # 0.2 to 0.7 range
            
            # Draw mouth
            mouth_width = int(width // 12 * (1 + mouth_openness))
            mouth_height = int(height // 24 * mouth_openness)
            
            # Clear previous mouth area
            cv2.ellipse(
                avatar_img, 
                (center_x, mouth_y), 
                (width // 10, height // 20), 
                0, 0, 360, 
                (200, 180, 160),  # Face color
                -1
            )
            
            # Draw new mouth
            if mouth_openness > 0.4:
                # Open mouth
                cv2.ellipse(
                    avatar_img, 
                    (center_x, mouth_y), 
                    (mouth_width, mouth_height), 
                    0, 0, 360, 
                    (50, 20, 20), 
                    -1
                )
            else:
                # Closed mouth
                cv2.ellipse(
                    avatar_img, 
                    (center_x, mouth_y), 
                    (mouth_width, mouth_height // 2), 
                    0, 0, 180, 
                    (100, 50, 50), 
                    2
                )
            
            return avatar_img
            
        except Exception as e:
            logger.error(f"Error generating frame: {e}")
            return avatar_img
    
    async def _get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file."""
        try:
            # Try using librosa if available
            try:
                import librosa
                duration = librosa.get_duration(filename=audio_path)
                return duration
            except ImportError:
                pass
            
            # Fallback: use ffprobe if available
            import subprocess
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 
                'format=duration', '-of', 'csv=p=0', audio_path
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                return float(result.stdout.strip())
            
            # Final fallback: estimate based on file size (very rough)
            file_size = os.path.getsize(audio_path)
            estimated_duration = file_size / (44100 * 2 * 2)  # Rough estimate for 16-bit stereo
            logger.warning(f"Using estimated duration: {estimated_duration:.2f}s")
            return estimated_duration
            
        except Exception as e:
            logger.error(f"Error getting audio duration: {e}")
            return 5.0  # Default fallback duration
    
    async def _add_audio_to_video(self, video_path: str, audio_path: str):
        """Add audio track to video using ffmpeg."""
        try:
            import subprocess
            
            # Create temporary output path
            temp_output = video_path.replace('.mp4', '_with_audio.mp4')
            
            # Run ffmpeg to combine video and audio
            cmd = [
                'ffmpeg', '-y',  # -y to overwrite output file
                '-i', video_path,  # Input video
                '-i', audio_path,  # Input audio
                '-c:v', 'copy',    # Copy video stream
                '-c:a', 'aac',     # Encode audio as AAC
                '-shortest',       # Stop at shortest stream
                temp_output
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Replace original video with the one that has audio
                os.replace(temp_output, video_path)
                logger.info("Successfully added audio to video")
            else:
                logger.warning(f"Failed to add audio to video: {result.stderr}")
                # Clean up temp file if it exists
                if os.path.exists(temp_output):
                    os.remove(temp_output)
                    
        except Exception as e:
            logger.warning(f"Could not add audio to video: {e}")
    
    def _add_audio_to_video_sync(self, video_path: str, audio_path: str):
        """Add audio track to video using ffmpeg (synchronous version)."""
        try:
            import subprocess
            
            # Create temporary output path
            temp_output = video_path.replace('.mp4', '_with_audio.mp4')
            
            # Run ffmpeg to combine video and audio
            cmd = [
                'ffmpeg', '-y',  # -y to overwrite output file
                '-i', video_path,  # Input video
                '-i', audio_path,  # Input audio
                '-c:v', 'copy',    # Copy video stream
                '-c:a', 'aac',     # Encode audio as AAC
                '-shortest',       # Stop at shortest stream
                temp_output
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Replace original video with the one that has audio
                os.replace(temp_output, video_path)
                logger.info("Successfully added audio to video")
            else:
                logger.warning(f"Failed to add audio to video: {result.stderr}")
                # Clean up temp file if it exists
                if os.path.exists(temp_output):
                    os.remove(temp_output)
                    
        except Exception as e:
            logger.warning(f"Could not add audio to video: {e}")
    
    async def set_avatar_image(self, image_path: str) -> bool:
        """
        Set a new avatar image.
        
        Args:
            image_path: Path to the new avatar image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Avatar image not found: {image_path}")
            
            # Validate image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Invalid image file: {image_path}")
            
            self.avatar_image_path = image_path
            logger.info(f"Avatar image updated: {image_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting avatar image: {e}")
            return False
    
    async def get_avatar_info(self) -> Dict[str, Any]:
        """Get information about the current avatar."""
        try:
            if not self.avatar_image_path or not os.path.exists(self.avatar_image_path):
                return {"error": "No avatar image available"}
            
            # Get image dimensions
            img = cv2.imread(self.avatar_image_path)
            height, width = img.shape[:2]
            
            # Get file size
            file_size = os.path.getsize(self.avatar_image_path)
            
            return {
                "path": self.avatar_image_path,
                "width": width,
                "height": height,
                "file_size": file_size,
                "target_resolution": self.video_resolution
            }
            
        except Exception as e:
            logger.error(f"Error getting avatar info: {e}")
            return {"error": str(e)}
    
    def get_supported_formats(self) -> Dict[str, list]:
        """Get supported file formats."""
        return {
            "image_formats": ["png", "jpg", "jpeg", "bmp", "tiff"],
            "video_formats": ["mp4", "avi", "mov"],
            "audio_formats": ["wav", "mp3", "m4a", "flac"]
        }
    
    async def health_check(self) -> str:
        """Check the health of the LipSync service."""
        try:
            if not self.avatar_image_path:
                return "no_avatar"
            
            if not os.path.exists(self.avatar_image_path):
                return "avatar_missing"
            
            # Test image loading
            img = cv2.imread(self.avatar_image_path)
            if img is None:
                return "avatar_invalid"
            
            return "healthy"
            
        except Exception as e:
            logger.error(f"LipSync health check failed: {e}")
            return f"error: {str(e)}"
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            # Clean up temporary files if any
            logger.info("LipSync service cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up LipSync service: {e}")
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the LipSync service."""
        return {
            "avatar_path": self.avatar_image_path,
            "video_fps": self.video_fps,
            "video_resolution": self.video_resolution,
            "supported_formats": self.get_supported_formats(),
            "initialized": self.avatar_image_path is not None
        }

# Global lip sync service instance
lipsync_service = LipSyncService() 