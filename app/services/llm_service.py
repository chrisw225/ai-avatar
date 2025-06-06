"""
Large Language Model Service using Ollama
Handles text generation with configurable model selection
"""

import asyncio
import logging
import time
import json
from typing import Optional, Dict, Any, List, AsyncGenerator

import httpx
from ..config import get_settings, update_ollama_config

logger = logging.getLogger(__name__)

class LLMService:
    """LLM service using Ollama with configurable models"""
    
    def __init__(self):
        self.settings = get_settings()
        self.client: Optional[httpx.AsyncClient] = None
        self.is_initialized = False
        self.available_models: List[str] = []
        
    async def initialize(self) -> None:
        """Initialize the LLM service"""
        if self.is_initialized:
            return
            
        try:
            logger.info(f"Initializing LLM service with Ollama at {self.settings.ollama_base_url}")
            
            # Create HTTP client
            self.client = httpx.AsyncClient(
                base_url=self.settings.ollama_base_url,
                timeout=httpx.Timeout(60.0)
            )
            
            # Check if Ollama is running
            await self._check_ollama_health()
            
            # Get available models
            await self._update_available_models()
            
            # Ensure the configured model is available
            await self._ensure_model_available()
            
            self.is_initialized = True
            logger.info("LLM Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {e}")
            raise
    
    async def _check_ollama_health(self) -> None:
        """Check if Ollama server is running"""
        try:
            response = await self.client.get("/api/tags")
            if response.status_code != 200:
                raise Exception(f"Ollama server returned status {response.status_code}")
        except httpx.ConnectError:
            raise Exception(f"Cannot connect to Ollama server at {self.settings.ollama_base_url}")
    
    async def _update_available_models(self) -> None:
        """Update list of available models"""
        try:
            response = await self.client.get("/api/tags")
            if response.status_code == 200:
                data = response.json()
                self.available_models = [model["name"] for model in data.get("models", [])]
                logger.info(f"Available models: {self.available_models}")
            else:
                logger.warning("Could not fetch available models")
        except Exception as e:
            logger.error(f"Error fetching available models: {e}")
    
    async def _ensure_model_available(self) -> None:
        """Ensure the configured model is available"""
        if self.settings.ollama_model not in self.available_models:
            logger.warning(f"Model {self.settings.ollama_model} not found. Attempting to pull...")
            await self.pull_model(self.settings.ollama_model)
    
    async def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from Ollama registry
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Pulling model: {model_name}")
            
            async with self.client.stream(
                "POST", 
                "/api/pull",
                json={"name": model_name}
            ) as response:
                if response.status_code != 200:
                    logger.error(f"Failed to pull model: {response.status_code}")
                    return False
                
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "status" in data:
                                logger.info(f"Pull status: {data['status']}")
                            if data.get("status") == "success":
                                logger.info(f"Model {model_name} pulled successfully")
                                await self._update_available_models()
                                return True
                        except json.JSONDecodeError:
                            continue
            
            return False
            
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    async def generate_response(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate response from LLM
        
        Args:
            prompt: User prompt
            context: Conversation context
            system_prompt: System prompt for behavior
            temperature: Response randomness (0.0-1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with response and metadata
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Build the full prompt
            full_prompt = self._build_prompt(prompt, context, system_prompt)
            
            logger.info(f"Generating response with model: {self.settings.ollama_model}")
            
            # Prepare request data
            request_data = {
                "model": self.settings.ollama_model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens or 2048,
                    "top_p": 0.9,
                    "top_k": 40
                }
            }
            
            # Make request to Ollama
            response = await self.client.post("/api/generate", json=request_data)
            
            if response.status_code != 200:
                raise Exception(f"Ollama API returned status {response.status_code}: {response.text}")
            
            data = response.json()
            processing_time = time.time() - start_time
            
            result = {
                "response": data.get("response", "").strip(),
                "model": self.settings.ollama_model,
                "processing_time": processing_time,
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                "done": data.get("done", False)
            }
            
            logger.info(f"Response generated in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return {
                "response": "",
                "error": str(e),
                "processing_time": time.time() - start_time,
                "model": self.settings.ollama_model
            }
    
    async def generate_streaming_response(
        self, 
        prompt: str, 
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate streaming response from LLM
        
        Args:
            prompt: User prompt
            context: Conversation context
            system_prompt: System prompt for behavior
            temperature: Response randomness (0.0-1.0)
            
        Yields:
            Dictionary with partial response and metadata
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Build the full prompt
            full_prompt = self._build_prompt(prompt, context, system_prompt)
            
            logger.info(f"Starting streaming response with model: {self.settings.ollama_model}")
            
            # Prepare request data
            request_data = {
                "model": self.settings.ollama_model,
                "prompt": full_prompt,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": 2048,
                    "top_p": 0.9,
                    "top_k": 40
                }
            }
            
            # Make streaming request to Ollama
            async with self.client.stream("POST", "/api/generate", json=request_data) as response:
                if response.status_code != 200:
                    yield {
                        "response": "",
                        "error": f"Ollama API returned status {response.status_code}",
                        "done": True
                    }
                    return
                
                full_response = ""
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            
                            if "response" in data:
                                chunk = data["response"]
                                full_response += chunk
                                
                                yield {
                                    "response": chunk,
                                    "full_response": full_response,
                                    "model": self.settings.ollama_model,
                                    "done": data.get("done", False),
                                    "processing_time": time.time() - start_time
                                }
                            
                            if data.get("done", False):
                                logger.info(f"Streaming response completed in {time.time() - start_time:.2f}s")
                                break
                                
                        except json.JSONDecodeError:
                            continue
            
        except Exception as e:
            logger.error(f"Streaming response failed: {e}")
            yield {
                "response": "",
                "error": str(e),
                "done": True,
                "processing_time": time.time() - start_time
            }
    
    def _build_prompt(self, prompt: str, context: Optional[str] = None, system_prompt: Optional[str] = None) -> str:
        """Build the full prompt with context and system instructions"""
        parts = []
        
        # Add system prompt
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        else:
            parts.append("System: You are a helpful AI assistant. Provide clear, concise, and accurate responses.")
        
        # Add conversation context
        if context:
            parts.append(f"Context:\n{context}")
        
        # Add current prompt
        parts.append(f"Human: {prompt}")
        parts.append("Assistant:")
        
        return "\n\n".join(parts)
    
    async def get_available_models(self) -> List[str]:
        """Get list of available models"""
        if not self.is_initialized:
            await self.initialize()
        
        await self._update_available_models()
        return self.available_models
    
    async def switch_model(self, model_name: str) -> bool:
        """
        Switch to a different model
        
        Args:
            model_name: Name of the model to switch to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if model_name not in self.available_models:
                logger.info(f"Model {model_name} not available, attempting to pull...")
                if not await self.pull_model(model_name):
                    return False
            
            # Update configuration
            update_ollama_config(model=model_name)
            logger.info(f"Switched to model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch model: {e}")
            return False
    
    async def update_base_url(self, base_url: str) -> bool:
        """
        Update Ollama base URL
        
        Args:
            base_url: New base URL for Ollama
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Update configuration
            update_ollama_config(base_url=base_url)
            
            # Reinitialize with new URL
            if self.client:
                await self.client.aclose()
            
            self.is_initialized = False
            await self.initialize()
            
            logger.info(f"Updated Ollama base URL to: {base_url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update base URL: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Test with a simple prompt
            test_response = await self.generate_response("Hello", max_tokens=10)
            
            return {
                "status": "healthy",
                "base_url": self.settings.ollama_base_url,
                "current_model": self.settings.ollama_model,
                "available_models": self.available_models,
                "test_response_time": test_response.get("processing_time", 0),
                "initialized": self.is_initialized
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "base_url": self.settings.ollama_base_url,
                "current_model": self.settings.ollama_model
            }
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        if self.client:
            await self.client.aclose()
            self.client = None
        self.is_initialized = False
        logger.info("LLM Service cleaned up")

# Global LLM service instance
llm_service = LLMService() 