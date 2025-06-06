"""
AI Chatbot FastAPI Application
Main application entry point with API endpoints for voice and text chat.
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from app.services.stt_service import STTService
from app.services.llm_service import LLMService
from app.services.tts_service import TTSService
from app.services.lipsync_service import LipSyncService
from app.services.conversation_service import ConversationService
from app.utils.file_utils import FileManager
from app.utils.validation_utils import InputValidator
from app.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global services dictionary
services = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("Starting AI Chatbot services...")
    
    try:
        # Initialize services
        services["file_manager"] = FileManager()
        services["validator"] = InputValidator()
        services["stt"] = STTService()
        services["llm"] = LLMService()
        services["tts"] = TTSService()
        services["lipsync"] = LipSyncService()
        services["conversation"] = ConversationService()
        
        await services["stt"].initialize()
        await services["llm"].initialize()
        await services["tts"].initialize()
        await services["lipsync"].initialize()
        await services["conversation"].initialize()
        
        logger.info("All services initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Shutting down services...")
        
        service_list = [
            ("stt", services.get("stt")),
            ("llm", services.get("llm")),
            ("tts", services.get("tts")),
            ("lipsync", services.get("lipsync")),
            ("conversation", services.get("conversation"))
        ]
        
        for service_name, service in service_list:
            try:
                if service and hasattr(service, 'cleanup'):
                    await service.cleanup()
                logger.info(f"{service_name} cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up {service_name}: {e}")

# Create FastAPI app
app = FastAPI(
    title="AI Chatbot with Voice and Lip Sync",
    description="An AI chatbot with speech-to-text, text-to-speech, and lip synchronization capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests/responses
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    audio_url: Optional[str] = None
    video_url: Optional[str] = None

class VoiceTrainingRequest(BaseModel):
    voice_name: str
    description: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    services: Dict[str, str]
    version: str

# API Routes

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main application page."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Chatbot</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .endpoint { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
            .method { font-weight: bold; color: #007bff; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>AI Chatbot with Voice and Lip Sync</h1>
            <p>Welcome to the AI Chatbot API. Available endpoints:</p>
            
            <div class="endpoint">
                <div class="method">GET /health</div>
                <p>Check the health status of all services</p>
            </div>
            
            <div class="endpoint">
                <div class="method">POST /chat</div>
                <p>Send a text message to the chatbot</p>
            </div>
            
            <div class="endpoint">
                <div class="method">POST /voice-chat</div>
                <p>Send an audio file for voice conversation</p>
            </div>
            
            <div class="endpoint">
                <div class="method">POST /train-voice</div>
                <p>Train a new voice model with uploaded audio samples</p>
            </div>
            
            <div class="endpoint">
                <div class="method">GET /avatar-video/{session_id}</div>
                <p>Get the generated avatar video for a session</p>
            </div>
            
            <div class="endpoint">
                <div class="method">WebSocket /ws/{session_id}</div>
                <p>Real-time WebSocket connection for live conversation</p>
            </div>
            
            <p><a href="/docs">View API Documentation</a></p>
        </div>
    </body>
    </html>
    """

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health status of all services."""
    service_status = {}
    
    for service_name, service in services.items():
        try:
            if hasattr(service, 'health_check'):
                status = await service.health_check()
                # Handle different return types
                if isinstance(status, dict):
                    service_status[service_name] = status.get("status", "unknown")
                else:
                    service_status[service_name] = str(status)
            else:
                service_status[service_name] = "healthy"
        except Exception as e:
            service_status[service_name] = f"error: {str(e)}"
    
    overall_status = "healthy" if all(
        "healthy" in str(status).lower() for status in service_status.values()
    ) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        services=service_status,
        version="1.0.0"
    )

@app.post("/chat", response_model=ChatResponse)
async def text_chat(request: ChatRequest):
    """Handle text-based chat requests."""
    try:
        # Validate input
        validator = services["validator"]
        if not validator.validate_text_input(request.message):
            raise HTTPException(status_code=400, detail="Invalid message content")
        
        if request.session_id and not validator.validate_session_id(request.session_id):
            raise HTTPException(status_code=400, detail="Invalid session ID")
        
        # Get conversation service
        conversation_service = services["conversation"]
        
        # Process the message
        response_data = await conversation_service.process_message(
            message=request.message,
            session_id=request.session_id
        )
        
        return ChatResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error in text chat: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/voice-chat")
async def voice_chat(
    audio_file: UploadFile = File(...),
    session_id: Optional[str] = None
):
    """Handle voice-based chat requests."""
    try:
        # Validate file
        validator = services["validator"]
        if not validator.validate_audio_file(audio_file):
            raise HTTPException(status_code=400, detail="Invalid audio file")
        
        if session_id and not validator.validate_session_id(session_id):
            raise HTTPException(status_code=400, detail="Invalid session ID")
        
        # Save uploaded file
        file_manager = services["file_manager"]
        audio_path = await file_manager.save_uploaded_file(audio_file, "audio")
        
        # Get conversation service
        conversation_service = services["conversation"]
        
        # Process the voice message
        response_data = await conversation_service.process_voice_message(
            audio_path=audio_path,
            session_id=session_id
        )
        
        return ChatResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Error in voice chat: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/train-voice")
async def train_voice_model(
    request: VoiceTrainingRequest,
    audio_files: list[UploadFile] = File(...)
):
    """Train a new voice model with uploaded audio samples."""
    try:
        # Validate inputs
        validator = services["validator"]
        if not validator.validate_text_input(request.voice_name):
            raise HTTPException(status_code=400, detail="Invalid voice name")
        
        for audio_file in audio_files:
            if not validator.validate_audio_file(audio_file):
                raise HTTPException(status_code=400, detail=f"Invalid audio file: {audio_file.filename}")
        
        # Save uploaded files
        file_manager = services["file_manager"]
        audio_paths = []
        for audio_file in audio_files:
            path = await file_manager.save_uploaded_file(audio_file, "training")
            audio_paths.append(path)
        
        # Get TTS service and train voice
        tts_service = services["tts"]
        training_result = await tts_service.train_voice_model(
            voice_name=request.voice_name,
            audio_paths=audio_paths,
            description=request.description
        )
        
        return {"status": "success", "result": training_result}
        
    except Exception as e:
        logger.error(f"Error in voice training: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/avatar-video/{session_id}")
async def get_avatar_video(session_id: str):
    """Get the generated avatar video for a session."""
    try:
        # Validate session ID
        validator = services["validator"]
        if not validator.validate_session_id(session_id):
            raise HTTPException(status_code=400, detail="Invalid session ID")
        
        # Get file manager
        file_manager = services["file_manager"]
        
        # Find the video file for this session
        video_path = file_manager.get_session_video_path(session_id)
        
        if not video_path or not os.path.exists(video_path):
            raise HTTPException(status_code=404, detail="Video not found")
        
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename=f"avatar_{session_id}.mp4"
        )
        
    except Exception as e:
        logger.error(f"Error getting avatar video: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time conversation."""
    await websocket.accept()
    
    try:
        # Validate session ID
        validator = services["validator"]
        if not validator.validate_session_id(session_id):
            await websocket.close(code=1008, reason="Invalid session ID")
            return
        
        # Get conversation service
        conversation_service = services["conversation"]
        
        # Register WebSocket connection
        await conversation_service.register_websocket(session_id, websocket)
        
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            # Process the message
            await conversation_service.handle_websocket_message(session_id, data)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        await websocket.close(code=1011, reason="Internal server error")
    finally:
        # Cleanup
        if "conversation" in services:
            await services["conversation"].unregister_websocket(session_id)

# Mount static files (if needed)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    ) 