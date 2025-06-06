"""
Conversation Service for AI Chatbot

This service manages conversation sessions, message history, and session state.
It provides functionality for creating sessions, managing conversation flow,
and maintaining context across multiple interactions.
"""

import logging
import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a single message in a conversation."""
    id: str
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationSession:
    """Represents a conversation session with message history."""
    session_id: str
    created_at: datetime
    last_activity: datetime
    messages: List[Message] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True


class ConversationService:
    """Service for managing conversation sessions and message history."""
    
    def __init__(self, max_sessions: int = 100, session_timeout: int = 3600):
        """
        Initialize the conversation service.
        
        Args:
            max_sessions: Maximum number of concurrent sessions
            session_timeout: Session timeout in seconds
        """
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout
        self.sessions: Dict[str, ConversationSession] = {}
        self.logger = logging.getLogger(__name__)
        
        # Start cleanup task
        self._cleanup_task = None
        
    async def initialize(self):
        """Initialize the conversation service."""
        self.logger.info("Initializing Conversation service")
        
        # Start periodic cleanup
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        
        self.logger.info("Conversation service initialized successfully")
        
    async def create_session(self, session_id: Optional[str] = None) -> str:
        """
        Create a new conversation session.
        
        Args:
            session_id: Optional session ID, generates one if not provided
            
        Returns:
            Session ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
            
        # Check if session already exists
        if session_id in self.sessions:
            self.logger.warning(f"Session {session_id} already exists, returning existing session")
            return session_id
            
        # Check session limit
        if len(self.sessions) >= self.max_sessions:
            await self._cleanup_old_sessions()
            
        # Create new session
        now = datetime.now()
        session = ConversationSession(
            session_id=session_id,
            created_at=now,
            last_activity=now
        )
        
        self.sessions[session_id] = session
        self.logger.info(f"Created new conversation session: {session_id}")
        
        return session_id
        
    async def add_message(self, session_id: str, role: str, content: str, 
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a message to a conversation session.
        
        Args:
            session_id: Session ID
            role: Message role ('user', 'assistant', 'system')
            content: Message content
            metadata: Optional message metadata
            
        Returns:
            Message ID
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
            
        session = self.sessions[session_id]
        
        # Update session activity
        session.last_activity = datetime.now()
        
        # Create message
        message_id = str(uuid.uuid4())
        message = Message(
            id=message_id,
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        session.messages.append(message)
        
        self.logger.debug(f"Added message to session {session_id}: {role} - {content[:50]}...")
        
        return message_id
        
    async def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """
        Get a conversation session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Conversation session or None if not found
        """
        return self.sessions.get(session_id)
        
    async def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[Message]:
        """
        Get messages from a conversation session.
        
        Args:
            session_id: Session ID
            limit: Optional limit on number of messages to return
            
        Returns:
            List of messages
        """
        if session_id not in self.sessions:
            return []
            
        messages = self.sessions[session_id].messages
        
        if limit:
            return messages[-limit:]
        return messages
        
    async def get_conversation_history(self, session_id: str, 
                                     format_for_llm: bool = False) -> List[Dict[str, str]]:
        """
        Get conversation history formatted for LLM or display.
        
        Args:
            session_id: Session ID
            format_for_llm: Whether to format for LLM consumption
            
        Returns:
            List of message dictionaries
        """
        messages = await self.get_messages(session_id)
        
        if format_for_llm:
            # Format for LLM (role and content only)
            return [{"role": msg.role, "content": msg.content} for msg in messages]
        else:
            # Format for display (include all fields)
            return [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": msg.metadata
                }
                for msg in messages
            ]
            
    async def update_session_context(self, session_id: str, context: Dict[str, Any]):
        """
        Update session context.
        
        Args:
            session_id: Session ID
            context: Context data to update
        """
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
            
        self.sessions[session_id].context.update(context)
        self.sessions[session_id].last_activity = datetime.now()
        
    async def clear_session(self, session_id: str):
        """
        Clear a conversation session.
        
        Args:
            session_id: Session ID
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.logger.info(f"Cleared session: {session_id}")
            
    async def clear_session_messages(self, session_id: str):
        """
        Clear messages from a session but keep the session.
        
        Args:
            session_id: Session ID
        """
        if session_id in self.sessions:
            self.sessions[session_id].messages.clear()
            self.sessions[session_id].last_activity = datetime.now()
            self.logger.info(f"Cleared messages for session: {session_id}")
            
    async def get_active_sessions(self) -> List[str]:
        """
        Get list of active session IDs.
        
        Returns:
            List of active session IDs
        """
        return [sid for sid, session in self.sessions.items() if session.is_active]
        
    async def get_session_stats(self) -> Dict[str, Any]:
        """
        Get session statistics.
        
        Returns:
            Dictionary with session statistics
        """
        now = datetime.now()
        active_sessions = 0
        total_messages = 0
        
        for session in self.sessions.values():
            if session.is_active:
                active_sessions += 1
            total_messages += len(session.messages)
            
        return {
            "total_sessions": len(self.sessions),
            "active_sessions": active_sessions,
            "total_messages": total_messages,
            "max_sessions": self.max_sessions,
            "session_timeout": self.session_timeout
        }
        
    async def _cleanup_old_sessions(self):
        """Clean up old and inactive sessions."""
        now = datetime.now()
        timeout_threshold = now - timedelta(seconds=self.session_timeout)
        
        sessions_to_remove = []
        for session_id, session in self.sessions.items():
            if session.last_activity < timeout_threshold:
                sessions_to_remove.append(session_id)
                
        for session_id in sessions_to_remove:
            del self.sessions[session_id]
            self.logger.info(f"Cleaned up expired session: {session_id}")
            
        if sessions_to_remove:
            self.logger.info(f"Cleaned up {len(sessions_to_remove)} expired sessions")
            
    async def _periodic_cleanup(self):
        """Periodic cleanup task."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._cleanup_old_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in periodic cleanup: {e}")
                
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the conversation service.
        
        Returns:
            Health check results
        """
        try:
            stats = await self.get_session_stats()
            
            return {
                "status": "healthy",
                "service": "conversation",
                "stats": stats,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Conversation service health check failed: {e}")
            return {
                "status": "unhealthy",
                "service": "conversation",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    async def cleanup(self):
        """Clean up the conversation service."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
                
        self.sessions.clear()
        self.logger.info("Conversation service cleaned up")
        
    async def process_message(self, message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a message and generate a response.
        
        Args:
            message: User message
            session_id: Optional session ID
            
        Returns:
            Dictionary with response data
        """
        try:
            # Create session if not provided
            if session_id is None:
                session_id = await self.create_session()
            elif session_id not in self.sessions:
                session_id = await self.create_session(session_id)
            
            # Add user message to conversation
            await self.add_message(session_id, "user", message)
            
            # For now, return a simple response
            # In a full implementation, this would call the LLM service
            response_text = f"I received your message: {message}"
            
            # Add assistant response to conversation
            await self.add_message(session_id, "assistant", response_text)
            
            return {
                "response": response_text,
                "session_id": session_id,
                "audio_url": None,
                "video_url": None
            }
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            raise
            
    async def process_voice_message(self, audio_path: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a voice message and generate a response.
        
        Args:
            audio_path: Path to the audio file
            session_id: Optional session ID
            
        Returns:
            Dictionary with response data
        """
        try:
            # Create session if not provided
            if session_id is None:
                session_id = await self.create_session()
            elif session_id not in self.sessions:
                session_id = await self.create_session(session_id)
            
            # For now, return a simple response
            # In a full implementation, this would:
            # 1. Use STT service to transcribe audio
            # 2. Process the transcribed text
            # 3. Generate TTS audio response
            # 4. Generate lip-sync video
            
            response_text = f"I received your voice message from {audio_path}"
            
            # Add messages to conversation
            await self.add_message(session_id, "user", "[Voice message]")
            await self.add_message(session_id, "assistant", response_text)
            
            return {
                "response": response_text,
                "session_id": session_id,
                "audio_url": None,
                "video_url": None
            }
            
        except Exception as e:
            self.logger.error(f"Error processing voice message: {e}")
            raise
            
    async def register_websocket(self, session_id: str, websocket):
        """
        Register a WebSocket connection for a session.
        
        Args:
            session_id: Session ID
            websocket: WebSocket connection
        """
        if session_id not in self.sessions:
            await self.create_session(session_id)
            
        # Store websocket in session context
        await self.update_session_context(session_id, {"websocket": websocket})
        self.logger.info(f"WebSocket registered for session: {session_id}")
        
    async def handle_websocket_message(self, session_id: str, data: Dict[str, Any]):
        """
        Handle a WebSocket message.
        
        Args:
            session_id: Session ID
            data: Message data
        """
        try:
            message = data.get("message", "")
            if not message:
                return
                
            # Process the message
            response_data = await self.process_message(message, session_id)
            
            # Get websocket from session context
            session = await self.get_session(session_id)
            if session and "websocket" in session.context:
                websocket = session.context["websocket"]
                await websocket.send_json(response_data)
                
        except Exception as e:
            self.logger.error(f"Error handling WebSocket message: {e}")
            raise 