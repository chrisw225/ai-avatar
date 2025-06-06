"""
FFmpeg utilities for AI Chatbot application
Handles FFmpeg installation and usage without requiring system installation
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

class FFmpegManager:
    """Manages FFmpeg installation and execution without system dependencies"""
    
    def __init__(self):
        self.ffmpeg_path: Optional[str] = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize FFmpeg, downloading if necessary"""
        if self._initialized:
            return True
        
        try:
            # Try to get FFmpeg from imageio-ffmpeg first
            self.ffmpeg_path = self._get_imageio_ffmpeg()
            if self.ffmpeg_path:
                logger.info(f"✓ FFmpeg found via imageio-ffmpeg: {self.ffmpeg_path}")
                self._initialized = True
                return True
            
            # Try ffmpeg-python
            self.ffmpeg_path = self._get_ffmpeg_python()
            if self.ffmpeg_path:
                logger.info(f"✓ FFmpeg found via ffmpeg-python: {self.ffmpeg_path}")
                self._initialized = True
                return True
            
            # Fallback to system FFmpeg
            self.ffmpeg_path = self._get_system_ffmpeg()
            if self.ffmpeg_path:
                logger.info(f"✓ System FFmpeg found: {self.ffmpeg_path}")
                self._initialized = True
                return True
            
            logger.error("❌ FFmpeg not available")
            return False
            
        except Exception as e:
            logger.error(f"Failed to initialize FFmpeg: {e}")
            return False
    
    def _get_imageio_ffmpeg(self) -> Optional[str]:
        """Get FFmpeg from imageio-ffmpeg (auto-downloads if needed)"""
        try:
            import imageio_ffmpeg as ffmpeg
            ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
            
            # Test if it works
            result = subprocess.run(
                [ffmpeg_exe, '-version'], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0:
                return ffmpeg_exe
                
        except ImportError:
            logger.warning("imageio-ffmpeg not installed")
        except Exception as e:
            logger.warning(f"Failed to get imageio-ffmpeg: {e}")
        
        return None
    
    def _get_ffmpeg_python(self) -> Optional[str]:
        """Get FFmpeg from ffmpeg-python"""
        try:
            import ffmpeg
            # ffmpeg-python uses system ffmpeg, so we need to check if it exists
            return self._get_system_ffmpeg()
        except ImportError:
            logger.warning("ffmpeg-python not installed")
        except Exception as e:
            logger.warning(f"Failed to get ffmpeg-python: {e}")
        
        return None
    
    def _get_system_ffmpeg(self) -> Optional[str]:
        """Try to find system FFmpeg"""
        try:
            # Try common FFmpeg command names
            for cmd in ['ffmpeg', 'ffmpeg.exe']:
                try:
                    result = subprocess.run(
                        [cmd, '-version'], 
                        capture_output=True, 
                        text=True, 
                        timeout=10
                    )
                    if result.returncode == 0:
                        return cmd
                except FileNotFoundError:
                    continue
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"System FFmpeg check failed: {e}")
        
        return None
    
    def is_available(self) -> bool:
        """Check if FFmpeg is available"""
        if not self._initialized:
            self.initialize()
        return self.ffmpeg_path is not None
    
    def get_version(self) -> Optional[str]:
        """Get FFmpeg version"""
        if not self.is_available():
            return None
        
        try:
            result = subprocess.run(
                [self.ffmpeg_path, '-version'], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0:
                # Extract version from first line
                first_line = result.stdout.split('\n')[0]
                if 'version' in first_line:
                    return first_line.split('version')[1].split()[0]
            
        except Exception as e:
            logger.error(f"Failed to get FFmpeg version: {e}")
        
        return None
    
    def convert_audio(
        self, 
        input_path: str, 
        output_path: str, 
        sample_rate: int = 16000,
        channels: int = 1,
        format: str = 'wav'
    ) -> bool:
        """Convert audio file using FFmpeg"""
        if not self.is_available():
            logger.error("FFmpeg not available for audio conversion")
            return False
        
        try:
            cmd = [
                self.ffmpeg_path,
                '-i', input_path,
                '-ar', str(sample_rate),
                '-ac', str(channels),
                '-f', format,
                '-y',  # Overwrite output file
                output_path
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=60
            )
            
            if result.returncode == 0:
                logger.info(f"✓ Audio converted: {input_path} -> {output_path}")
                return True
            else:
                logger.error(f"FFmpeg conversion failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg conversion timed out")
            return False
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return False
    
    def convert_audio_with_ffmpeg_python(
        self,
        input_path: str,
        output_path: str,
        sample_rate: int = 16000,
        channels: int = 1
    ) -> bool:
        """Convert audio using ffmpeg-python (easier syntax)"""
        try:
            import ffmpeg
            
            stream = ffmpeg.input(input_path)
            stream = ffmpeg.output(
                stream, 
                output_path,
                ar=sample_rate,
                ac=channels,
                f='wav'
            )
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            
            logger.info(f"✓ Audio converted with ffmpeg-python: {input_path} -> {output_path}")
            return True
            
        except ImportError:
            logger.warning("ffmpeg-python not available, falling back to direct FFmpeg")
            return self.convert_audio(input_path, output_path, sample_rate, channels)
        except Exception as e:
            logger.error(f"ffmpeg-python conversion failed: {e}")
            return False
    
    def extract_audio_from_video(
        self, 
        video_path: str, 
        audio_path: str,
        sample_rate: int = 16000
    ) -> bool:
        """Extract audio from video file"""
        try:
            import ffmpeg
            
            stream = ffmpeg.input(video_path)
            stream = ffmpeg.output(
                stream,
                audio_path,
                vn=None,  # No video
                ar=sample_rate,
                ac=1,  # Mono
                f='wav'
            )
            ffmpeg.run(stream, overwrite_output=True, quiet=True)
            
            logger.info(f"✓ Audio extracted with ffmpeg-python: {video_path} -> {audio_path}")
            return True
            
        except ImportError:
            logger.warning("ffmpeg-python not available, falling back to direct FFmpeg")
            return self._extract_audio_direct(video_path, audio_path, sample_rate)
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            return False
    
    def _extract_audio_direct(
        self, 
        video_path: str, 
        audio_path: str,
        sample_rate: int = 16000
    ) -> bool:
        """Extract audio using direct FFmpeg command"""
        if not self.is_available():
            logger.error("FFmpeg not available for audio extraction")
            return False
        
        try:
            cmd = [
                self.ffmpeg_path,
                '-i', video_path,
                '-vn',  # No video
                '-ar', str(sample_rate),
                '-ac', '1',  # Mono
                '-f', 'wav',
                '-y',
                audio_path
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=120
            )
            
            if result.returncode == 0:
                logger.info(f"✓ Audio extracted: {video_path} -> {audio_path}")
                return True
            else:
                logger.error(f"Audio extraction failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            return False
    
    def get_media_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get media file information using ffmpeg-python"""
        try:
            import ffmpeg
            
            probe = ffmpeg.probe(file_path)
            
            info = {
                'format': probe.get('format', {}),
                'streams': probe.get('streams', []),
                'duration': float(probe['format'].get('duration', 0)),
                'size': int(probe['format'].get('size', 0))
            }
            
            return info
            
        except ImportError:
            logger.warning("ffmpeg-python not available for media info")
            return self._get_media_info_direct(file_path)
        except Exception as e:
            logger.error(f"Failed to get media info: {e}")
            return None
    
    def _get_media_info_direct(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get media file information using direct FFmpeg"""
        if not self.is_available():
            return None
        
        try:
            cmd = [
                self.ffmpeg_path,
                '-i', file_path,
                '-f', 'null',
                '-'
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            
            # Parse stderr for media info
            stderr = result.stderr
            info = {}
            
            # Extract duration
            if 'Duration:' in stderr:
                duration_line = [line for line in stderr.split('\n') if 'Duration:' in line][0]
                duration_str = duration_line.split('Duration:')[1].split(',')[0].strip()
                info['duration'] = duration_str
            
            # Extract stream info
            streams = []
            for line in stderr.split('\n'):
                if 'Stream #' in line:
                    streams.append(line.strip())
            info['streams'] = streams
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get media info: {e}")
            return None
    
    def create_video_from_images_and_audio(
        self,
        image_path: str,
        audio_path: str,
        output_path: str,
        fps: int = 25
    ) -> bool:
        """Create video from static image and audio using ffmpeg-python"""
        try:
            import ffmpeg
            
            # Input streams
            image_input = ffmpeg.input(image_path, loop=1)
            audio_input = ffmpeg.input(audio_path)
            
            # Create output
            output = ffmpeg.output(
                image_input,
                audio_input,
                output_path,
                vcodec='libx264',
                tune='stillimage',
                acodec='aac',
                audio_bitrate='192k',
                pix_fmt='yuv420p',
                shortest=None,
                r=fps
            )
            
            ffmpeg.run(output, overwrite_output=True, quiet=True)
            
            logger.info(f"✓ Video created with ffmpeg-python: {output_path}")
            return True
            
        except ImportError:
            logger.warning("ffmpeg-python not available, falling back to direct FFmpeg")
            return self._create_video_direct(image_path, audio_path, output_path, fps)
        except Exception as e:
            logger.error(f"Video creation failed: {e}")
            return False
    
    def _create_video_direct(
        self,
        image_path: str,
        audio_path: str,
        output_path: str,
        fps: int = 25
    ) -> bool:
        """Create video using direct FFmpeg command"""
        if not self.is_available():
            logger.error("FFmpeg not available for video creation")
            return False
        
        try:
            cmd = [
                self.ffmpeg_path,
                '-loop', '1',
                '-i', image_path,
                '-i', audio_path,
                '-c:v', 'libx264',
                '-tune', 'stillimage',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-pix_fmt', 'yuv420p',
                '-shortest',
                '-r', str(fps),
                '-y',
                output_path
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300
            )
            
            if result.returncode == 0:
                logger.info(f"✓ Video created: {output_path}")
                return True
            else:
                logger.error(f"Video creation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Video creation failed: {e}")
            return False

# Global FFmpeg manager instance
ffmpeg_manager = FFmpegManager()

def get_ffmpeg_manager() -> FFmpegManager:
    """Get the global FFmpeg manager instance"""
    return ffmpeg_manager

def ensure_ffmpeg() -> bool:
    """Ensure FFmpeg is available"""
    return ffmpeg_manager.initialize()

def convert_audio(input_path: str, output_path: str, **kwargs) -> bool:
    """Convenience function for audio conversion"""
    return ffmpeg_manager.convert_audio_with_ffmpeg_python(input_path, output_path, **kwargs)

def extract_audio(video_path: str, audio_path: str, **kwargs) -> bool:
    """Convenience function for audio extraction"""
    return ffmpeg_manager.extract_audio_from_video(video_path, audio_path, **kwargs)

def get_media_info(file_path: str) -> Optional[Dict[str, Any]]:
    """Convenience function for getting media info"""
    return ffmpeg_manager.get_media_info(file_path)