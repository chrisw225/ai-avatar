"""
File management utilities for the AI Chatbot application
Handles file operations, cleanup, and temporary file management
"""

import os
import shutil
import tempfile
import logging
import hashlib
import mimetypes
from typing import Optional, List, Dict, Any
from pathlib import Path
import time
import json

logger = logging.getLogger(__name__)

class FileManager:
    """File management utilities"""
    
    def __init__(self, temp_dir: Optional[str] = None, max_file_age_hours: int = 24):
        """
        Initialize FileManager
        
        Args:
            temp_dir: Custom temporary directory path
            max_file_age_hours: Maximum age for temporary files before cleanup
        """
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.max_file_age_hours = max_file_age_hours
        self.app_temp_dir = os.path.join(self.temp_dir, "ai_chatbot")
        
        # Create application temp directory
        self.ensure_directory_exists(self.app_temp_dir)
        
        # Supported file types
        self.audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}
        self.video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        self.text_extensions = {'.txt', '.json', '.csv', '.log'}
    
    def ensure_directory_exists(self, directory_path: str) -> bool:
        """
        Ensure directory exists, create if it doesn't
        
        Args:
            directory_path: Path to directory
            
        Returns:
            True if directory exists or was created successfully
        """
        try:
            Path(directory_path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {directory_path}: {e}")
            return False
    
    def get_temp_file_path(self, extension: str = "", prefix: str = "temp_") -> str:
        """
        Generate a temporary file path
        
        Args:
            extension: File extension (with or without dot)
            prefix: File prefix
            
        Returns:
            Temporary file path
        """
        if extension and not extension.startswith('.'):
            extension = '.' + extension
        
        timestamp = str(int(time.time() * 1000))
        filename = f"{prefix}{timestamp}{extension}"
        return os.path.join(self.app_temp_dir, filename)
    
    def save_uploaded_file(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Save uploaded file content to temporary directory
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            
        Returns:
            Dictionary with save results
        """
        try:
            # Generate safe filename
            safe_filename = self.sanitize_filename(filename)
            file_path = os.path.join(self.app_temp_dir, safe_filename)
            
            # Ensure unique filename
            counter = 1
            base_path = file_path
            while os.path.exists(file_path):
                name, ext = os.path.splitext(base_path)
                file_path = f"{name}_{counter}{ext}"
                counter += 1
            
            # Save file
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            # Get file info
            file_info = self.get_file_info(file_path)
            
            return {
                "success": True,
                "file_path": file_path,
                "original_filename": filename,
                "saved_filename": os.path.basename(file_path),
                "file_info": file_info
            }
            
        except Exception as e:
            logger.error(f"Failed to save uploaded file: {e}")
            return {"success": False, "error": str(e)}
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to remove dangerous characters
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove path separators and dangerous characters
        dangerous_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\0']
        sanitized = filename
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '_')
        
        # Remove leading/trailing spaces and dots
        sanitized = sanitized.strip(' .')
        
        # Ensure filename is not empty
        if not sanitized:
            sanitized = "unnamed_file"
        
        # Limit filename length
        if len(sanitized) > 255:
            name, ext = os.path.splitext(sanitized)
            max_name_length = 255 - len(ext)
            sanitized = name[:max_name_length] + ext
        
        return sanitized
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get comprehensive file information
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file information
        """
        try:
            if not os.path.exists(file_path):
                return {"exists": False}
            
            stat = os.stat(file_path)
            
            # Basic info
            info = {
                "exists": True,
                "path": file_path,
                "filename": os.path.basename(file_path),
                "size": stat.st_size,
                "size_human": self.format_file_size(stat.st_size),
                "created": stat.st_ctime,
                "modified": stat.st_mtime,
                "accessed": stat.st_atime,
                "is_file": os.path.isfile(file_path),
                "is_directory": os.path.isdir(file_path)
            }
            
            if os.path.isfile(file_path):
                # File-specific info
                _, ext = os.path.splitext(file_path.lower())
                info["extension"] = ext
                info["mime_type"] = mimetypes.guess_type(file_path)[0]
                info["file_type"] = self.get_file_type(ext)
                
                # Calculate file hash
                info["md5_hash"] = self.calculate_file_hash(file_path)
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get file info for {file_path}: {e}")
            return {"exists": False, "error": str(e)}
    
    def get_file_type(self, extension: str) -> str:
        """
        Determine file type category from extension
        
        Args:
            extension: File extension
            
        Returns:
            File type category
        """
        extension = extension.lower()
        
        if extension in self.audio_extensions:
            return "audio"
        elif extension in self.video_extensions:
            return "video"
        elif extension in self.image_extensions:
            return "image"
        elif extension in self.text_extensions:
            return "text"
        else:
            return "unknown"
    
    def format_file_size(self, size_bytes: int) -> str:
        """
        Format file size in human-readable format
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            Formatted size string
        """
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        size = float(size_bytes)
        
        while size >= 1024.0 and i < len(size_names) - 1:
            size /= 1024.0
            i += 1
        
        return f"{size:.1f} {size_names[i]}"
    
    def calculate_file_hash(self, file_path: str, algorithm: str = "md5") -> str:
        """
        Calculate file hash
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm (md5, sha1, sha256)
            
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
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return ""
    
    def copy_file(self, source_path: str, destination_path: str) -> Dict[str, Any]:
        """
        Copy file from source to destination
        
        Args:
            source_path: Source file path
            destination_path: Destination file path
            
        Returns:
            Dictionary with copy results
        """
        try:
            if not os.path.exists(source_path):
                return {"success": False, "error": "Source file does not exist"}
            
            # Ensure destination directory exists
            dest_dir = os.path.dirname(destination_path)
            self.ensure_directory_exists(dest_dir)
            
            # Copy file
            shutil.copy2(source_path, destination_path)
            
            return {
                "success": True,
                "source_path": source_path,
                "destination_path": destination_path,
                "file_info": self.get_file_info(destination_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to copy file from {source_path} to {destination_path}: {e}")
            return {"success": False, "error": str(e)}
    
    def move_file(self, source_path: str, destination_path: str) -> Dict[str, Any]:
        """
        Move file from source to destination
        
        Args:
            source_path: Source file path
            destination_path: Destination file path
            
        Returns:
            Dictionary with move results
        """
        try:
            if not os.path.exists(source_path):
                return {"success": False, "error": "Source file does not exist"}
            
            # Ensure destination directory exists
            dest_dir = os.path.dirname(destination_path)
            self.ensure_directory_exists(dest_dir)
            
            # Move file
            shutil.move(source_path, destination_path)
            
            return {
                "success": True,
                "source_path": source_path,
                "destination_path": destination_path,
                "file_info": self.get_file_info(destination_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to move file from {source_path} to {destination_path}: {e}")
            return {"success": False, "error": str(e)}
    
    def delete_file(self, file_path: str) -> Dict[str, Any]:
        """
        Delete file
        
        Args:
            file_path: Path to file to delete
            
        Returns:
            Dictionary with deletion results
        """
        try:
            if not os.path.exists(file_path):
                return {"success": True, "message": "File does not exist"}
            
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
            
            return {
                "success": True,
                "deleted_path": file_path,
                "message": "File deleted successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            return {"success": False, "error": str(e)}
    
    def cleanup_old_files(self, directory: Optional[str] = None) -> Dict[str, Any]:
        """
        Clean up old temporary files
        
        Args:
            directory: Directory to clean (defaults to app temp directory)
            
        Returns:
            Dictionary with cleanup results
        """
        try:
            target_dir = directory or self.app_temp_dir
            
            if not os.path.exists(target_dir):
                return {"success": True, "files_deleted": 0, "message": "Directory does not exist"}
            
            current_time = time.time()
            max_age_seconds = self.max_file_age_hours * 3600
            
            deleted_files = []
            total_size_freed = 0
            
            for root, dirs, files in os.walk(target_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    try:
                        file_age = current_time - os.path.getmtime(file_path)
                        
                        if file_age > max_age_seconds:
                            file_size = os.path.getsize(file_path)
                            os.unlink(file_path)
                            deleted_files.append(file_path)
                            total_size_freed += file_size
                            
                    except Exception as e:
                        logger.warning(f"Failed to delete old file {file_path}: {e}")
            
            return {
                "success": True,
                "files_deleted": len(deleted_files),
                "size_freed": total_size_freed,
                "size_freed_human": self.format_file_size(total_size_freed),
                "deleted_files": deleted_files
            }
            
        except Exception as e:
            logger.error(f"Failed to cleanup old files: {e}")
            return {"success": False, "error": str(e)}
    
    def list_directory_contents(self, directory_path: str) -> Dict[str, Any]:
        """
        List directory contents with detailed information
        
        Args:
            directory_path: Path to directory
            
        Returns:
            Dictionary with directory contents
        """
        try:
            if not os.path.exists(directory_path):
                return {"success": False, "error": "Directory does not exist"}
            
            if not os.path.isdir(directory_path):
                return {"success": False, "error": "Path is not a directory"}
            
            contents = []
            total_size = 0
            
            for item in os.listdir(directory_path):
                item_path = os.path.join(directory_path, item)
                item_info = self.get_file_info(item_path)
                contents.append(item_info)
                
                if item_info.get("is_file", False):
                    total_size += item_info.get("size", 0)
            
            # Sort by name
            contents.sort(key=lambda x: x.get("filename", ""))
            
            return {
                "success": True,
                "directory_path": directory_path,
                "total_items": len(contents),
                "total_size": total_size,
                "total_size_human": self.format_file_size(total_size),
                "contents": contents
            }
            
        except Exception as e:
            logger.error(f"Failed to list directory contents for {directory_path}: {e}")
            return {"success": False, "error": str(e)}
    
    def create_backup(self, source_path: str, backup_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Create backup of file or directory
        
        Args:
            source_path: Path to source file/directory
            backup_dir: Backup directory (optional)
            
        Returns:
            Dictionary with backup results
        """
        try:
            if not os.path.exists(source_path):
                return {"success": False, "error": "Source path does not exist"}
            
            # Default backup directory
            if backup_dir is None:
                backup_dir = os.path.join(self.app_temp_dir, "backups")
            
            self.ensure_directory_exists(backup_dir)
            
            # Generate backup filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            source_name = os.path.basename(source_path)
            backup_name = f"{source_name}_{timestamp}"
            backup_path = os.path.join(backup_dir, backup_name)
            
            # Create backup
            if os.path.isfile(source_path):
                shutil.copy2(source_path, backup_path)
            elif os.path.isdir(source_path):
                shutil.copytree(source_path, backup_path)
            
            return {
                "success": True,
                "source_path": source_path,
                "backup_path": backup_path,
                "backup_info": self.get_file_info(backup_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to create backup of {source_path}: {e}")
            return {"success": False, "error": str(e)}
    
    def save_json_data(self, data: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """
        Save data as JSON file
        
        Args:
            data: Data to save
            file_path: Output file path
            
        Returns:
            Dictionary with save results
        """
        try:
            # Ensure directory exists
            directory = os.path.dirname(file_path)
            if directory:
                self.ensure_directory_exists(directory)
            
            # Save JSON data
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            return {
                "success": True,
                "file_path": file_path,
                "file_info": self.get_file_info(file_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to save JSON data to {file_path}: {e}")
            return {"success": False, "error": str(e)}
    
    def load_json_data(self, file_path: str) -> Dict[str, Any]:
        """
        Load data from JSON file
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Dictionary with loaded data
        """
        try:
            if not os.path.exists(file_path):
                return {"success": False, "error": "File does not exist"}
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return {
                "success": True,
                "data": data,
                "file_info": self.get_file_info(file_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to load JSON data from {file_path}: {e}")
            return {"success": False, "error": str(e)}
    
    def get_disk_usage(self, path: str = None) -> Dict[str, Any]:
        """
        Get disk usage information
        
        Args:
            path: Path to check (defaults to temp directory)
            
        Returns:
            Dictionary with disk usage information
        """
        try:
            target_path = path or self.app_temp_dir
            
            if not os.path.exists(target_path):
                return {"success": False, "error": "Path does not exist"}
            
            # Get disk usage
            total, used, free = shutil.disk_usage(target_path)
            
            return {
                "success": True,
                "path": target_path,
                "total": total,
                "used": used,
                "free": free,
                "total_human": self.format_file_size(total),
                "used_human": self.format_file_size(used),
                "free_human": self.format_file_size(free),
                "usage_percent": (used / total) * 100 if total > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get disk usage for {path}: {e}")
            return {"success": False, "error": str(e)} 