"""
File Manager Utility
Handles file operations, cleanup, and management for the AI chatbot.
"""

import os
import logging
import shutil
import tempfile
import hashlib
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta

from app.config import settings

logger = logging.getLogger(__name__)


class FileManager:
    """Manages file operations and cleanup for the application."""
    
    def __init__(self):
        """Initialize the FileManager."""
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.output_dir = Path(settings.OUTPUT_DIR)
        self.temp_dir = Path(tempfile.gettempdir()) / "ai_chatbot"
        
        # Create directories
        self._ensure_directories()
        
        logger.info("FileManager initialized")
        logger.info(f"Upload dir: {self.upload_dir}")
        logger.info(f"Output dir: {self.output_dir}")
        logger.info(f"Temp dir: {self.temp_dir}")
    
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        try:
            self.upload_dir.mkdir(parents=True, exist_ok=True)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (self.upload_dir / "audio").mkdir(exist_ok=True)
            (self.upload_dir / "images").mkdir(exist_ok=True)
            (self.upload_dir / "videos").mkdir(exist_ok=True)
            
            (self.output_dir / "audio").mkdir(exist_ok=True)
            (self.output_dir / "videos").mkdir(exist_ok=True)
            (self.output_dir / "conversations").mkdir(exist_ok=True)
            
        except Exception as e:
            logger.error(f"Error creating directories: {e}")
            raise
    
    async def save_uploaded_file(
        self, 
        file_content: bytes, 
        filename: str, 
        file_type: str = "audio"
    ) -> str:
        """
        Save uploaded file to appropriate directory.
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            file_type: Type of file ('audio', 'image', 'video')
            
        Returns:
            Path to saved file
        """
        try:
            # Sanitize filename
            safe_filename = self._sanitize_filename(filename)
            
            # Generate unique filename to avoid conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name, ext = os.path.splitext(safe_filename)
            unique_filename = f"{timestamp}_{name}{ext}"
            
            # Determine target directory
            target_dir = self.upload_dir / file_type
            target_path = target_dir / unique_filename
            
            # Save file
            with open(target_path, 'wb') as f:
                f.write(file_content)
            
            logger.info(f"Saved uploaded file: {target_path}")
            return str(target_path)
            
        except Exception as e:
            logger.error(f"Error saving uploaded file: {e}")
            raise
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to remove dangerous characters."""
        # Remove path separators and other dangerous characters
        safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_"
        sanitized = "".join(c for c in filename if c in safe_chars)
        
        # Ensure filename is not empty and has reasonable length
        if not sanitized:
            sanitized = "file"
        
        if len(sanitized) > 100:
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:95] + ext
        
        return sanitized
    
    async def create_temp_file(self, suffix: str = "", prefix: str = "temp_") -> str:
        """
        Create a temporary file.
        
        Args:
            suffix: File suffix/extension
            prefix: File prefix
            
        Returns:
            Path to temporary file
        """
        try:
            temp_file = tempfile.NamedTemporaryFile(
                suffix=suffix,
                prefix=prefix,
                dir=self.temp_dir,
                delete=False
            )
            temp_path = temp_file.name
            temp_file.close()
            
            logger.debug(f"Created temp file: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Error creating temp file: {e}")
            raise
    
    async def move_to_output(self, source_path: str, output_filename: str) -> str:
        """
        Move file to output directory.
        
        Args:
            source_path: Source file path
            output_filename: Output filename
            
        Returns:
            Path to moved file
        """
        try:
            source = Path(source_path)
            if not source.exists():
                raise FileNotFoundError(f"Source file not found: {source_path}")
            
            # Determine output subdirectory based on file extension
            ext = source.suffix.lower()
            if ext in ['.mp3', '.wav', '.m4a', '.flac']:
                output_subdir = self.output_dir / "audio"
            elif ext in ['.mp4', '.avi', '.mov']:
                output_subdir = self.output_dir / "videos"
            else:
                output_subdir = self.output_dir
            
            output_path = output_subdir / output_filename
            
            # Move file
            shutil.move(str(source), str(output_path))
            
            logger.info(f"Moved file to output: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error moving file to output: {e}")
            raise
    
    async def copy_file(self, source_path: str, dest_path: str) -> str:
        """
        Copy file to destination.
        
        Args:
            source_path: Source file path
            dest_path: Destination file path
            
        Returns:
            Destination file path
        """
        try:
            source = Path(source_path)
            dest = Path(dest_path)
            
            if not source.exists():
                raise FileNotFoundError(f"Source file not found: {source_path}")
            
            # Ensure destination directory exists
            dest.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(str(source), str(dest))
            
            logger.debug(f"Copied file: {source} -> {dest}")
            return str(dest)
            
        except Exception as e:
            logger.error(f"Error copying file: {e}")
            raise
    
    async def delete_file(self, file_path: str) -> bool:
        """
        Delete a file.
        
        Args:
            file_path: Path to file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_obj = Path(file_path)
            if file_obj.exists():
                file_obj.unlink()
                logger.debug(f"Deleted file: {file_path}")
                return True
            else:
                logger.warning(f"File not found for deletion: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            return False
    
    async def cleanup_temp_files(self, max_age_hours: int = 24):
        """
        Clean up temporary files older than specified age.
        
        Args:
            max_age_hours: Maximum age of temp files in hours
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            deleted_count = 0
            
            for file_path in self.temp_dir.rglob("*"):
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_time:
                        try:
                            file_path.unlink()
                            deleted_count += 1
                        except Exception as e:
                            logger.warning(f"Could not delete temp file {file_path}: {e}")
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} temporary files")
                
        except Exception as e:
            logger.error(f"Error cleaning up temp files: {e}")
    
    async def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file information
        """
        try:
            file_obj = Path(file_path)
            
            if not file_obj.exists():
                return {"error": "File not found"}
            
            stat = file_obj.stat()
            
            return {
                "path": str(file_obj),
                "name": file_obj.name,
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "extension": file_obj.suffix,
                "is_file": file_obj.is_file(),
                "is_dir": file_obj.is_dir()
            }
            
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            return {"error": str(e)}
    
    async def list_files(
        self, 
        directory: str, 
        pattern: str = "*",
        recursive: bool = False
    ) -> List[Dict[str, Any]]:
        """
        List files in a directory.
        
        Args:
            directory: Directory path
            pattern: File pattern to match
            recursive: Whether to search recursively
            
        Returns:
            List of file information dictionaries
        """
        try:
            dir_path = Path(directory)
            
            if not dir_path.exists():
                return []
            
            files = []
            
            if recursive:
                file_paths = dir_path.rglob(pattern)
            else:
                file_paths = dir_path.glob(pattern)
            
            for file_path in file_paths:
                if file_path.is_file():
                    file_info = await self.get_file_info(str(file_path))
                    files.append(file_info)
            
            return files
            
        except Exception as e:
            logger.error(f"Error listing files in {directory}: {e}")
            return []
    
    def calculate_file_hash(self, file_path: str, algorithm: str = "md5") -> str:
        """
        Calculate hash of a file.
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
            
        Returns:
            File hash as hex string
        """
        try:
            hash_obj = hashlib.new(algorithm)
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            
            return hash_obj.hexdigest()
            
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    async def get_directory_size(self, directory: str) -> int:
        """
        Get total size of a directory.
        
        Args:
            directory: Directory path
            
        Returns:
            Total size in bytes
        """
        try:
            dir_path = Path(directory)
            
            if not dir_path.exists():
                return 0
            
            total_size = 0
            for file_path in dir_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            
            return total_size
            
        except Exception as e:
            logger.error(f"Error calculating directory size for {directory}: {e}")
            return 0
    
    async def cleanup_old_files(
        self, 
        directory: str, 
        max_age_days: int = 7,
        pattern: str = "*"
    ):
        """
        Clean up old files in a directory.
        
        Args:
            directory: Directory to clean
            max_age_days: Maximum age of files in days
            pattern: File pattern to match
        """
        try:
            dir_path = Path(directory)
            
            if not dir_path.exists():
                return
            
            cutoff_time = datetime.now() - timedelta(days=max_age_days)
            deleted_count = 0
            
            for file_path in dir_path.glob(pattern):
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_time:
                        try:
                            file_path.unlink()
                            deleted_count += 1
                        except Exception as e:
                            logger.warning(f"Could not delete old file {file_path}: {e}")
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old files from {directory}")
                
        except Exception as e:
            logger.error(f"Error cleaning up old files in {directory}: {e}")
    
    def get_available_space(self, path: str) -> int:
        """
        Get available disk space for a path.
        
        Args:
            path: Path to check
            
        Returns:
            Available space in bytes
        """
        try:
            stat = shutil.disk_usage(path)
            return stat.free
            
        except Exception as e:
            logger.error(f"Error getting available space for {path}: {e}")
            return 0
    
    def get_manager_info(self) -> Dict[str, Any]:
        """Get information about the FileManager."""
        return {
            "upload_dir": str(self.upload_dir),
            "output_dir": str(self.output_dir),
            "temp_dir": str(self.temp_dir),
            "upload_space": self.get_available_space(str(self.upload_dir)),
            "output_space": self.get_available_space(str(self.output_dir)),
            "temp_space": self.get_available_space(str(self.temp_dir))
        } 