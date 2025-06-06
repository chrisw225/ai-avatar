"""
Validation Utilities
Contains validators for different file types and input validation.
"""

import os
import logging
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


class BaseValidator:
    """Base class for file validators."""
    
    def __init__(self, max_file_size: int, allowed_extensions: List[str]):
        """
        Initialize base validator.
        
        Args:
            max_file_size: Maximum file size in bytes
            allowed_extensions: List of allowed file extensions
        """
        self.max_file_size = max_file_size
        self.allowed_extensions = [ext.lower() for ext in allowed_extensions]
    
    def validate_file_size(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate file size.
        
        Args:
            file_path: Path to file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not os.path.exists(file_path):
                return False, "File not found"
            
            file_size = os.path.getsize(file_path)
            
            if file_size > self.max_file_size:
                max_mb = self.max_file_size / (1024 * 1024)
                actual_mb = file_size / (1024 * 1024)
                return False, f"File size ({actual_mb:.1f}MB) exceeds maximum allowed size ({max_mb:.1f}MB)"
            
            return True, ""
            
        except Exception as e:
            logger.error(f"Error validating file size for {file_path}: {e}")
            return False, f"Error checking file size: {str(e)}"
    
    def validate_extension(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate file extension.
        
        Args:
            file_path: Path to file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if not file_ext:
                return False, "File has no extension"
            
            if file_ext not in self.allowed_extensions:
                return False, f"File extension '{file_ext}' not allowed. Allowed: {', '.join(self.allowed_extensions)}"
            
            return True, ""
            
        except Exception as e:
            logger.error(f"Error validating file extension for {file_path}: {e}")
            return False, f"Error checking file extension: {str(e)}"
    
    def validate_basic(self, file_path: str) -> Tuple[bool, str]:
        """
        Perform basic validation (size and extension).
        
        Args:
            file_path: Path to file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check file size
        is_valid, error = self.validate_file_size(file_path)
        if not is_valid:
            return False, error
        
        # Check file extension
        is_valid, error = self.validate_extension(file_path)
        if not is_valid:
            return False, error
        
        return True, ""


class AudioValidator(BaseValidator):
    """Validator for audio files."""
    
    def __init__(self):
        """Initialize audio validator."""
        super().__init__(
            max_file_size=settings.MAX_AUDIO_FILE_SIZE,
            allowed_extensions=settings.ALLOWED_AUDIO_EXTENSIONS
        )
    
    def validate(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Basic validation
            is_valid, error = self.validate_basic(file_path)
            if not is_valid:
                return False, error
            
            # Try to validate audio format using librosa if available
            try:
                import librosa
                
                # Try to load audio metadata
                duration = librosa.get_duration(filename=file_path)
                
                # Check duration limits (optional)
                max_duration = 600  # 10 minutes
                if duration > max_duration:
                    return False, f"Audio duration ({duration:.1f}s) exceeds maximum allowed ({max_duration}s)"
                
                logger.debug(f"Audio file validated: {file_path} (duration: {duration:.1f}s)")
                
            except ImportError:
                logger.warning("librosa not available, skipping detailed audio validation")
            except Exception as e:
                logger.warning(f"Could not validate audio format for {file_path}: {e}")
                # Don't fail validation if we can't check format
            
            return True, ""
            
        except Exception as e:
            logger.error(f"Error validating audio file {file_path}: {e}")
            return False, f"Audio validation error: {str(e)}"
    
    def get_audio_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get audio file information.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with audio information
        """
        try:
            info = {
                "path": file_path,
                "size": os.path.getsize(file_path),
                "extension": Path(file_path).suffix.lower()
            }
            
            # Try to get detailed audio info
            try:
                import librosa
                
                duration = librosa.get_duration(filename=file_path)
                sr = librosa.get_samplerate(filename=file_path)
                
                info.update({
                    "duration": duration,
                    "sample_rate": sr,
                    "channels": "unknown"  # librosa doesn't easily provide this
                })
                
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"Could not get detailed audio info for {file_path}: {e}")
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting audio info for {file_path}: {e}")
            return {"error": str(e)}


class VideoValidator(BaseValidator):
    """Validator for video files."""
    
    def __init__(self):
        """Initialize video validator."""
        super().__init__(
            max_file_size=settings.MAX_VIDEO_FILE_SIZE,
            allowed_extensions=settings.ALLOWED_VIDEO_EXTENSIONS
        )
    
    def validate(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate video file.
        
        Args:
            file_path: Path to video file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Basic validation
            is_valid, error = self.validate_basic(file_path)
            if not is_valid:
                return False, error
            
            # Try to validate video format using cv2 if available
            try:
                import cv2
                
                cap = cv2.VideoCapture(file_path)
                
                if not cap.isOpened():
                    return False, "Could not open video file"
                
                # Get basic video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                cap.release()
                
                # Check video properties
                if fps <= 0 or frame_count <= 0:
                    return False, "Invalid video properties"
                
                duration = frame_count / fps
                max_duration = 1800  # 30 minutes
                if duration > max_duration:
                    return False, f"Video duration ({duration:.1f}s) exceeds maximum allowed ({max_duration}s)"
                
                logger.debug(f"Video file validated: {file_path} ({width}x{height}, {duration:.1f}s)")
                
            except ImportError:
                logger.warning("cv2 not available, skipping detailed video validation")
            except Exception as e:
                logger.warning(f"Could not validate video format for {file_path}: {e}")
                # Don't fail validation if we can't check format
            
            return True, ""
            
        except Exception as e:
            logger.error(f"Error validating video file {file_path}: {e}")
            return False, f"Video validation error: {str(e)}"
    
    def get_video_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get video file information.
        
        Args:
            file_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        try:
            info = {
                "path": file_path,
                "size": os.path.getsize(file_path),
                "extension": Path(file_path).suffix.lower()
            }
            
            # Try to get detailed video info
            try:
                import cv2
                
                cap = cv2.VideoCapture(file_path)
                
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    duration = frame_count / fps if fps > 0 else 0
                    
                    info.update({
                        "width": width,
                        "height": height,
                        "fps": fps,
                        "frame_count": frame_count,
                        "duration": duration
                    })
                
                cap.release()
                
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"Could not get detailed video info for {file_path}: {e}")
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting video info for {file_path}: {e}")
            return {"error": str(e)}


class ImageValidator(BaseValidator):
    """Validator for image files."""
    
    def __init__(self):
        """Initialize image validator."""
        super().__init__(
            max_file_size=settings.MAX_IMAGE_FILE_SIZE,
            allowed_extensions=settings.ALLOWED_IMAGE_EXTENSIONS
        )
    
    def validate(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate image file.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Basic validation
            is_valid, error = self.validate_basic(file_path)
            if not is_valid:
                return False, error
            
            # Try to validate image format using PIL if available
            try:
                from PIL import Image
                
                with Image.open(file_path) as img:
                    # Check if image can be opened
                    img.verify()
                
                # Reopen for getting properties (verify() closes the image)
                with Image.open(file_path) as img:
                    width, height = img.size
                    
                    # Check image dimensions
                    max_dimension = 4096
                    if width > max_dimension or height > max_dimension:
                        return False, f"Image dimensions ({width}x{height}) exceed maximum allowed ({max_dimension}x{max_dimension})"
                    
                    min_dimension = 32
                    if width < min_dimension or height < min_dimension:
                        return False, f"Image dimensions ({width}x{height}) below minimum required ({min_dimension}x{min_dimension})"
                
                logger.debug(f"Image file validated: {file_path} ({width}x{height})")
                
            except ImportError:
                logger.warning("PIL not available, skipping detailed image validation")
            except Exception as e:
                return False, f"Invalid image file: {str(e)}"
            
            return True, ""
            
        except Exception as e:
            logger.error(f"Error validating image file {file_path}: {e}")
            return False, f"Image validation error: {str(e)}"
    
    def get_image_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get image file information.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Dictionary with image information
        """
        try:
            info = {
                "path": file_path,
                "size": os.path.getsize(file_path),
                "extension": Path(file_path).suffix.lower()
            }
            
            # Try to get detailed image info
            try:
                from PIL import Image
                
                with Image.open(file_path) as img:
                    width, height = img.size
                    mode = img.mode
                    format_name = img.format
                    
                    info.update({
                        "width": width,
                        "height": height,
                        "mode": mode,
                        "format": format_name
                    })
                
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"Could not get detailed image info for {file_path}: {e}")
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting image info for {file_path}: {e}")
            return {"error": str(e)}


def validate_file_by_type(file_path: str, file_type: str) -> Tuple[bool, str]:
    """
    Validate file based on its type.
    
    Args:
        file_path: Path to file
        file_type: Type of file ('audio', 'video', 'image')
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if file_type == "audio":
            validator = AudioValidator()
        elif file_type == "video":
            validator = VideoValidator()
        elif file_type == "image":
            validator = ImageValidator()
        else:
            return False, f"Unknown file type: {file_type}"
        
        return validator.validate(file_path)
        
    except Exception as e:
        logger.error(f"Error validating file {file_path} as {file_type}: {e}")
        return False, f"Validation error: {str(e)}"


def get_file_info_by_type(file_path: str, file_type: str) -> Dict[str, Any]:
    """
    Get file information based on its type.
    
    Args:
        file_path: Path to file
        file_type: Type of file ('audio', 'video', 'image')
        
    Returns:
        Dictionary with file information
    """
    try:
        if file_type == "audio":
            validator = AudioValidator()
            return validator.get_audio_info(file_path)
        elif file_type == "video":
            validator = VideoValidator()
            return validator.get_video_info(file_path)
        elif file_type == "image":
            validator = ImageValidator()
            return validator.get_image_info(file_path)
        else:
            return {"error": f"Unknown file type: {file_type}"}
        
    except Exception as e:
        logger.error(f"Error getting file info for {file_path} as {file_type}: {e}")
        return {"error": str(e)} 