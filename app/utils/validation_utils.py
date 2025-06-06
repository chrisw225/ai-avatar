"""
Input validation utilities for the AI Chatbot application
Handles validation of user inputs, file uploads, and API parameters
"""

import re
import os
import logging
from typing import Optional, Dict, Any, List, Union
import mimetypes
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class InputValidator:
    """Input validation utilities"""
    
    def __init__(self):
        # Common validation patterns
        self.email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        self.url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        # File validation settings
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.allowed_audio_types = {
            'audio/wav', 'audio/mpeg', 'audio/mp4', 'audio/flac', 
            'audio/ogg', 'audio/aac', 'audio/x-wav'
        }
        self.allowed_video_types = {
            'video/mp4', 'video/avi', 'video/quicktime', 'video/x-msvideo',
            'video/webm', 'video/x-matroska'
        }
        self.allowed_image_types = {
            'image/jpeg', 'image/png', 'image/bmp', 'image/gif', 
            'image/tiff', 'image/webp'
        }
        
        # Text validation settings
        self.max_text_length = 10000
        self.min_text_length = 1
        
        # Dangerous patterns to check for
        self.dangerous_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',  # JavaScript URLs
            r'on\w+\s*=',  # Event handlers
            r'<iframe[^>]*>.*?</iframe>',  # Iframes
            r'<object[^>]*>.*?</object>',  # Objects
            r'<embed[^>]*>.*?</embed>',  # Embeds
        ]
    
    def validate_text_input(
        self, 
        text: str, 
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        allow_html: bool = False,
        required: bool = True
    ) -> Dict[str, Any]:
        """
        Validate text input
        
        Args:
            text: Text to validate
            min_length: Minimum length (optional)
            max_length: Maximum length (optional)
            allow_html: Whether to allow HTML content
            required: Whether the field is required
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Check if required
            if required and (not text or not text.strip()):
                return {"valid": False, "error": "Text input is required"}
            
            # If not required and empty, it's valid
            if not required and (not text or not text.strip()):
                return {"valid": True, "cleaned_text": ""}
            
            # Length validation
            min_len = min_length or self.min_text_length
            max_len = max_length or self.max_text_length
            
            if len(text) < min_len:
                return {"valid": False, "error": f"Text must be at least {min_len} characters"}
            
            if len(text) > max_len:
                return {"valid": False, "error": f"Text must not exceed {max_len} characters"}
            
            # Security validation
            if not allow_html:
                security_check = self.check_for_dangerous_content(text)
                if not security_check["safe"]:
                    return {"valid": False, "error": f"Potentially dangerous content detected: {security_check['reason']}"}
            
            # Clean and normalize text
            cleaned_text = self.clean_text(text, allow_html)
            
            return {
                "valid": True,
                "cleaned_text": cleaned_text,
                "original_length": len(text),
                "cleaned_length": len(cleaned_text)
            }
            
        except Exception as e:
            logger.error(f"Text validation error: {e}")
            return {"valid": False, "error": str(e)}
    
    def validate_email(self, email: str) -> Dict[str, Any]:
        """
        Validate email address
        
        Args:
            email: Email address to validate
            
        Returns:
            Dictionary with validation results
        """
        try:
            if not email or not email.strip():
                return {"valid": False, "error": "Email is required"}
            
            email = email.strip().lower()
            
            if not self.email_pattern.match(email):
                return {"valid": False, "error": "Invalid email format"}
            
            # Additional checks
            if len(email) > 254:  # RFC 5321 limit
                return {"valid": False, "error": "Email address too long"}
            
            local, domain = email.split('@')
            if len(local) > 64:  # RFC 5321 limit
                return {"valid": False, "error": "Email local part too long"}
            
            return {"valid": True, "email": email}
            
        except Exception as e:
            logger.error(f"Email validation error: {e}")
            return {"valid": False, "error": str(e)}
    
    def validate_url(self, url: str, require_https: bool = False) -> Dict[str, Any]:
        """
        Validate URL
        
        Args:
            url: URL to validate
            require_https: Whether to require HTTPS
            
        Returns:
            Dictionary with validation results
        """
        try:
            if not url or not url.strip():
                return {"valid": False, "error": "URL is required"}
            
            url = url.strip()
            
            # Basic pattern check
            if not self.url_pattern.match(url):
                return {"valid": False, "error": "Invalid URL format"}
            
            # Parse URL for additional validation
            parsed = urlparse(url)
            
            if require_https and parsed.scheme != 'https':
                return {"valid": False, "error": "HTTPS is required"}
            
            if not parsed.netloc:
                return {"valid": False, "error": "Invalid domain"}
            
            return {
                "valid": True,
                "url": url,
                "scheme": parsed.scheme,
                "domain": parsed.netloc,
                "path": parsed.path
            }
            
        except Exception as e:
            logger.error(f"URL validation error: {e}")
            return {"valid": False, "error": str(e)}
    
    def validate_file_upload(
        self, 
        file_path: str, 
        allowed_types: Optional[List[str]] = None,
        max_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Validate uploaded file
        
        Args:
            file_path: Path to uploaded file
            allowed_types: List of allowed MIME types
            max_size: Maximum file size in bytes
            
        Returns:
            Dictionary with validation results
        """
        try:
            if not os.path.exists(file_path):
                return {"valid": False, "error": "File does not exist"}
            
            # Get file info
            file_size = os.path.getsize(file_path)
            mime_type, _ = mimetypes.guess_type(file_path)
            
            # Size validation
            max_allowed_size = max_size or self.max_file_size
            if file_size > max_allowed_size:
                return {
                    "valid": False, 
                    "error": f"File size ({self.format_file_size(file_size)}) exceeds maximum allowed size ({self.format_file_size(max_allowed_size)})"
                }
            
            # MIME type validation
            if allowed_types and mime_type not in allowed_types:
                return {
                    "valid": False,
                    "error": f"File type '{mime_type}' not allowed. Allowed types: {', '.join(allowed_types)}"
                }
            
            # Additional security checks
            security_check = self.check_file_security(file_path)
            if not security_check["safe"]:
                return {"valid": False, "error": f"File security check failed: {security_check['reason']}"}
            
            return {
                "valid": True,
                "file_path": file_path,
                "file_size": file_size,
                "file_size_human": self.format_file_size(file_size),
                "mime_type": mime_type,
                "file_type": self.get_file_category(mime_type)
            }
            
        except Exception as e:
            logger.error(f"File validation error: {e}")
            return {"valid": False, "error": str(e)}
    
    def validate_audio_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate audio file specifically
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with validation results
        """
        return self.validate_file_upload(
            file_path, 
            allowed_types=list(self.allowed_audio_types)
        )
    
    def validate_video_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate video file specifically
        
        Args:
            file_path: Path to video file
            
        Returns:
            Dictionary with validation results
        """
        return self.validate_file_upload(
            file_path, 
            allowed_types=list(self.allowed_video_types)
        )
    
    def validate_image_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate image file specifically
        
        Args:
            file_path: Path to image file
            
        Returns:
            Dictionary with validation results
        """
        return self.validate_file_upload(
            file_path, 
            allowed_types=list(self.allowed_image_types)
        )
    
    def validate_numeric_input(
        self, 
        value: Union[str, int, float], 
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_float: bool = True
    ) -> Dict[str, Any]:
        """
        Validate numeric input
        
        Args:
            value: Value to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            allow_float: Whether to allow floating point numbers
            
        Returns:
            Dictionary with validation results
        """
        try:
            # Convert to number
            if isinstance(value, str):
                if not value.strip():
                    return {"valid": False, "error": "Numeric value is required"}
                
                try:
                    if allow_float:
                        numeric_value = float(value)
                    else:
                        numeric_value = int(value)
                except ValueError:
                    return {"valid": False, "error": "Invalid numeric format"}
            else:
                numeric_value = value
            
            # Range validation
            if min_value is not None and numeric_value < min_value:
                return {"valid": False, "error": f"Value must be at least {min_value}"}
            
            if max_value is not None and numeric_value > max_value:
                return {"valid": False, "error": f"Value must not exceed {max_value}"}
            
            return {
                "valid": True,
                "value": numeric_value,
                "is_integer": isinstance(numeric_value, int) or numeric_value.is_integer()
            }
            
        except Exception as e:
            logger.error(f"Numeric validation error: {e}")
            return {"valid": False, "error": str(e)}
    
    def validate_session_id(self, session_id: str) -> Dict[str, Any]:
        """
        Validate session ID format
        
        Args:
            session_id: Session ID to validate
            
        Returns:
            Dictionary with validation results
        """
        try:
            if not session_id or not session_id.strip():
                return {"valid": False, "error": "Session ID is required"}
            
            session_id = session_id.strip()
            
            # Check format (UUID-like or alphanumeric)
            if not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
                return {"valid": False, "error": "Invalid session ID format"}
            
            # Length check
            if len(session_id) < 8 or len(session_id) > 128:
                return {"valid": False, "error": "Session ID length must be between 8 and 128 characters"}
            
            return {"valid": True, "session_id": session_id}
            
        except Exception as e:
            logger.error(f"Session ID validation error: {e}")
            return {"valid": False, "error": str(e)}
    
    def check_for_dangerous_content(self, text: str) -> Dict[str, Any]:
        """
        Check text for potentially dangerous content
        
        Args:
            text: Text to check
            
        Returns:
            Dictionary with safety check results
        """
        try:
            text_lower = text.lower()
            
            # Check for dangerous patterns
            for pattern in self.dangerous_patterns:
                if re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL):
                    return {
                        "safe": False,
                        "reason": f"Potentially dangerous pattern detected: {pattern}"
                    }
            
            # Check for suspicious keywords
            suspicious_keywords = [
                'eval(', 'exec(', 'system(', 'shell_exec(', 'passthru(',
                'file_get_contents(', 'file_put_contents(', 'fopen(',
                'include(', 'require(', 'import(', '__import__('
            ]
            
            for keyword in suspicious_keywords:
                if keyword in text_lower:
                    return {
                        "safe": False,
                        "reason": f"Suspicious keyword detected: {keyword}"
                    }
            
            return {"safe": True}
            
        except Exception as e:
            logger.error(f"Dangerous content check error: {e}")
            return {"safe": False, "reason": str(e)}
    
    def check_file_security(self, file_path: str) -> Dict[str, Any]:
        """
        Perform basic security checks on uploaded file
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with security check results
        """
        try:
            # Check file extension
            _, ext = os.path.splitext(file_path.lower())
            
            # Dangerous extensions
            dangerous_extensions = {
                '.exe', '.bat', '.cmd', '.com', '.pif', '.scr', '.vbs', '.js',
                '.jar', '.php', '.asp', '.aspx', '.jsp', '.py', '.rb', '.pl'
            }
            
            if ext in dangerous_extensions:
                return {
                    "safe": False,
                    "reason": f"Potentially dangerous file extension: {ext}"
                }
            
            # Check file size (prevent zip bombs, etc.)
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                return {
                    "safe": False,
                    "reason": f"File size too large: {self.format_file_size(file_size)}"
                }
            
            # Basic content check for text files
            if ext in {'.txt', '.json', '.csv', '.log'}:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(1024)  # Read first 1KB
                        content_check = self.check_for_dangerous_content(content)
                        if not content_check["safe"]:
                            return content_check
                except Exception:
                    pass  # If we can't read it, assume it's binary and safe
            
            return {"safe": True}
            
        except Exception as e:
            logger.error(f"File security check error: {e}")
            return {"safe": False, "reason": str(e)}
    
    def clean_text(self, text: str, allow_html: bool = False) -> str:
        """
        Clean and sanitize text input
        
        Args:
            text: Text to clean
            allow_html: Whether to preserve HTML
            
        Returns:
            Cleaned text
        """
        try:
            if not text:
                return ""
            
            # Basic cleaning
            cleaned = text.strip()
            
            # Remove null bytes
            cleaned = cleaned.replace('\x00', '')
            
            # Normalize whitespace
            cleaned = re.sub(r'\s+', ' ', cleaned)
            
            # If HTML not allowed, remove HTML tags
            if not allow_html:
                cleaned = re.sub(r'<[^>]+>', '', cleaned)
                
                # Decode HTML entities
                import html
                cleaned = html.unescape(cleaned)
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Text cleaning error: {e}")
            return text  # Return original if cleaning fails
    
    def get_file_category(self, mime_type: Optional[str]) -> str:
        """
        Get file category from MIME type
        
        Args:
            mime_type: MIME type
            
        Returns:
            File category
        """
        if not mime_type:
            return "unknown"
        
        if mime_type in self.allowed_audio_types:
            return "audio"
        elif mime_type in self.allowed_video_types:
            return "video"
        elif mime_type in self.allowed_image_types:
            return "image"
        elif mime_type.startswith('text/'):
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
    
    def validate_api_parameters(self, params: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate API parameters against schema
        
        Args:
            params: Parameters to validate
            schema: Validation schema
            
        Returns:
            Dictionary with validation results
        """
        try:
            errors = []
            validated_params = {}
            
            for field_name, field_schema in schema.items():
                value = params.get(field_name)
                field_type = field_schema.get('type', 'string')
                required = field_schema.get('required', False)
                default = field_schema.get('default')
                
                # Check if required
                if required and value is None:
                    errors.append(f"Field '{field_name}' is required")
                    continue
                
                # Use default if not provided
                if value is None and default is not None:
                    value = default
                
                # Skip validation if value is None and not required
                if value is None:
                    continue
                
                # Type-specific validation
                if field_type == 'string':
                    validation = self.validate_text_input(
                        str(value),
                        min_length=field_schema.get('min_length'),
                        max_length=field_schema.get('max_length'),
                        required=required
                    )
                elif field_type == 'number':
                    validation = self.validate_numeric_input(
                        value,
                        min_value=field_schema.get('min_value'),
                        max_value=field_schema.get('max_value'),
                        allow_float=field_schema.get('allow_float', True)
                    )
                elif field_type == 'email':
                    validation = self.validate_email(str(value))
                elif field_type == 'url':
                    validation = self.validate_url(str(value))
                else:
                    validation = {"valid": True, "value": value}
                
                if not validation["valid"]:
                    errors.append(f"Field '{field_name}': {validation['error']}")
                else:
                    validated_params[field_name] = validation.get('value', value)
            
            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "validated_params": validated_params
            }
            
        except Exception as e:
            logger.error(f"API parameter validation error: {e}")
            return {"valid": False, "errors": [str(e)], "validated_params": {}} 