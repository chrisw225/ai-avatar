"""
Large Language Model (LLM) Service
Handles text generation and conversation using various LLM models.
"""

import logging
import asyncio
import json
from typing import Optional, Dict, Any, List
import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Large Language Model service for text generation and conversation."""
    
    def __init__(self):
        """Initialize the LLM service."""
        self.model_name = settings.LLM_MODEL
        self.ollama_host = settings.OLLAMA_HOST
        self.client = None
        self.conversation_history = {}
        
        logger.info(f"Initializing LLM service with model: {self.model_name}")
        logger.info(f"Ollama host: {self.ollama_host}")
    
    async def initialize(self):
        """Initialize the LLM service."""
        try:
            # Create HTTP client for Ollama API
            self.client = httpx.AsyncClient(timeout=30.0)
            
            # Test connection to Ollama
            await self._test_ollama_connection()
            
            logger.info("LLM service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {e}")
            raise
    
    async def _test_ollama_connection(self):
        """Test connection to Ollama server."""
        try:
            response = await self.client.get(f"{self.ollama_host}/api/tags")
            if response.status_code == 200:
                models = response.json()
                logger.info(f"Connected to Ollama. Available models: {[m['name'] for m in models.get('models', [])]}")
            else:
                raise Exception(f"Ollama server returned status {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Could not connect to Ollama: {e}")
            logger.info("LLM service will continue without Ollama (limited functionality)")
    
    async def generate_response(
        self, 
        message: str, 
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Generate a response to a user message.
        
        Args:
            message: User's input message
            session_id: Optional session ID for conversation context
            system_prompt: Optional system prompt to guide the model
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            
        Returns:
            Dictionary containing the generated response and metadata
        """
        try:
            if not self.client:
                await self.initialize()
            
            # Get conversation history
            history = self._get_conversation_history(session_id)
            
            # Build messages for the model
            messages = []
            
            # Add system prompt if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            elif not history:
                # Default system prompt for new conversations
                messages.append({
                    "role": "system", 
                    "content": "You are a helpful AI assistant. Be concise, friendly, and informative in your responses."
                })
            
            # Add conversation history
            messages.extend(history)
            
            # Add current user message
            messages.append({"role": "user", "content": message})
            
            # Generate response using Ollama
            response_data = await self._call_ollama(messages, max_tokens, temperature)
            
            # Update conversation history
            if session_id:
                self._update_conversation_history(session_id, message, response_data["content"])
            
            return {
                "content": response_data["content"],
                "model": self.model_name,
                "session_id": session_id,
                "tokens_used": response_data.get("tokens_used", 0),
                "finish_reason": response_data.get("finish_reason", "stop")
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Fallback response
            return {
                "content": "I apologize, but I'm having trouble processing your request right now. Please try again.",
                "model": self.model_name,
                "session_id": session_id,
                "error": str(e)
            }
    
    async def _call_ollama(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int, 
        temperature: float
    ) -> Dict[str, Any]:
        """Call Ollama API to generate response."""
        try:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "top_k": 40
                }
            }
            
            response = await self.client.post(
                f"{self.ollama_host}/api/chat",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "content": result["message"]["content"],
                    "tokens_used": result.get("eval_count", 0),
                    "finish_reason": "stop" if result.get("done", False) else "length"
                }
            else:
                error_msg = f"Ollama API error: {response.status_code}"
                if response.text:
                    error_msg += f" - {response.text}"
                raise Exception(error_msg)
                
        except Exception as e:
            logger.error(f"Ollama API call failed: {e}")
            raise
    
    async def generate_streaming_response(
        self, 
        message: str, 
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ):
        """
        Generate a streaming response to a user message.
        
        Args:
            message: User's input message
            session_id: Optional session ID for conversation context
            system_prompt: Optional system prompt to guide the model
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            
        Yields:
            Chunks of the generated response
        """
        try:
            if not self.client:
                await self.initialize()
            
            # Get conversation history
            history = self._get_conversation_history(session_id)
            
            # Build messages for the model
            messages = []
            
            # Add system prompt if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            elif not history:
                messages.append({
                    "role": "system", 
                    "content": "You are a helpful AI assistant. Be concise, friendly, and informative in your responses."
                })
            
            # Add conversation history
            messages.extend(history)
            
            # Add current user message
            messages.append({"role": "user", "content": message})
            
            # Generate streaming response
            full_response = ""
            async for chunk in self._call_ollama_streaming(messages, max_tokens, temperature):
                full_response += chunk
                yield chunk
            
            # Update conversation history
            if session_id:
                self._update_conversation_history(session_id, message, full_response)
                
        except Exception as e:
            logger.error(f"Error generating streaming response: {e}")
            yield "I apologize, but I'm having trouble processing your request right now."
    
    async def _call_ollama_streaming(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int, 
        temperature: float
    ):
        """Call Ollama API with streaming response."""
        try:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": True,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "top_k": 40
                }
            }
            
            async with self.client.stream(
                "POST",
                f"{self.ollama_host}/api/chat",
                json=payload
            ) as response:
                if response.status_code == 200:
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if "message" in data and "content" in data["message"]:
                                    yield data["message"]["content"]
                            except json.JSONDecodeError:
                                continue
                else:
                    raise Exception(f"Ollama streaming API error: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Ollama streaming API call failed: {e}")
            raise
    
    def _get_conversation_history(self, session_id: Optional[str]) -> List[Dict[str, str]]:
        """Get conversation history for a session."""
        if not session_id:
            return []
        
        return self.conversation_history.get(session_id, [])
    
    def _update_conversation_history(self, session_id: str, user_message: str, assistant_response: str):
        """Update conversation history for a session."""
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        
        # Add user message and assistant response
        self.conversation_history[session_id].extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_response}
        ])
        
        # Keep only last 20 messages to prevent memory issues
        if len(self.conversation_history[session_id]) > 20:
            self.conversation_history[session_id] = self.conversation_history[session_id][-20:]
    
    async def summarize_text(self, text: str, max_length: int = 200) -> str:
        """
        Summarize a given text.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Summarized text
        """
        try:
            prompt = f"Please provide a concise summary of the following text in no more than {max_length} characters:\n\n{text}"
            
            response = await self.generate_response(
                message=prompt,
                system_prompt="You are a text summarization assistant. Provide clear, concise summaries.",
                max_tokens=max_length // 4,  # Rough estimate of tokens
                temperature=0.3
            )
            
            return response["content"]
            
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            return text[:max_length] + "..." if len(text) > max_length else text
    
    async def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama."""
        try:
            if not self.client:
                await self.initialize()
            
            response = await self.client.get(f"{self.ollama_host}/api/tags")
            if response.status_code == 200:
                models_data = response.json()
                return [model["name"] for model in models_data.get("models", [])]
            else:
                logger.warning(f"Could not fetch models: {response.status_code}")
                return [self.model_name]
                
        except Exception as e:
            logger.error(f"Error fetching available models: {e}")
            return [self.model_name]
    
    async def switch_model(self, model_name: str) -> bool:
        """
        Switch to a different model.
        
        Args:
            model_name: Name of the model to switch to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            available_models = await self.get_available_models()
            if model_name in available_models:
                self.model_name = model_name
                logger.info(f"Switched to model: {model_name}")
                return True
            else:
                logger.warning(f"Model {model_name} not available. Available models: {available_models}")
                return False
                
        except Exception as e:
            logger.error(f"Error switching model: {e}")
            return False
    
    def clear_conversation_history(self, session_id: str):
        """Clear conversation history for a session."""
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
            logger.info(f"Cleared conversation history for session: {session_id}")
    
    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of conversation for a session."""
        history = self.conversation_history.get(session_id, [])
        
        return {
            "session_id": session_id,
            "message_count": len(history),
            "last_message": history[-1]["content"] if history else None,
            "conversation_length": sum(len(msg["content"]) for msg in history)
        }
    
    async def health_check(self) -> str:
        """Check the health of the LLM service."""
        try:
            if not self.client:
                return "not_initialized"
            
            # Test with a simple message
            response = await self.generate_response(
                message="Hello, this is a health check.",
                max_tokens=50,
                temperature=0.1
            )
            
            if "error" in response:
                return f"error: {response['error']}"
            
            return "healthy"
            
        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
            return f"error: {str(e)}"
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            if self.client:
                await self.client.aclose()
                self.client = None
            
            # Clear conversation history
            self.conversation_history.clear()
            
            logger.info("LLM service cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up LLM service: {e}")
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get information about the LLM service."""
        return {
            "model_name": self.model_name,
            "ollama_host": self.ollama_host,
            "active_sessions": len(self.conversation_history),
            "total_conversations": sum(len(history) for history in self.conversation_history.values()),
            "initialized": self.client is not None
        } 