# AI Chatbot Development Request - Complete Open Source Solution

## Project Overview
We need to develop a real-time AI chatbot system that integrates Speech-to-Text, Large Language Model, Text-to-Speech, and lip-sync technologies using only open-source resources.

## What We Want You to Build

### Core Requirements
- **Real-time Voice Conversation**: Users can speak to the AI and get voice responses with synchronized video
- **Local Deployment**: Everything runs locally without external APIs
- **High Quality Output**: Professional-grade voice synthesis and lip synchronization
- **Cross-platform Compatibility**: Works on Windows, Linux, and macOS

### Technical Stack We've Chosen

1. **Speech Recognition (STT)**
   - **Use**: Faster-Whisper
   - **Reason**: Best open-source accuracy, offline capable, multi-language support
   - **Implementation**: Real-time audio streaming with chunked processing
   - **Model**: whisper-medium for production (balance of speed/accuracy)

2. **Large Language Model**
   - **Use**: Ollama (already set up locally)
   - **Models**: llama3.1:8b or llama3.2:3b depending on hardware
   - **Integration**: Local API calls, no external dependencies

3. **Text-to-Speech**
   - **Use**: GPT-SoVITS (https://github.com/RVC-Boss/GPT-SoVITS)
   - **Why**: Only 1 minute of voice data needed for high-quality voice cloning
   - **Features**: Cross-language support, few-shot voice conversion, excellent quality
   - **Setup**: Train custom voice model from provided audio samples

4. **Lip Synchronization**
   - **Use**: SadTalker
   - **Why**: Best open-source quality with natural head movements
   - **Processing**: Batch processing of short audio segments for near real-time
   - **Avatar**: Use provided `avatar.png` (512x512) as the test image for lip synchronization

5. **Backend Framework**
   - **Use**: Python + FastAPI
   - **Features**:
     - WebSocket for real-time communication
     - Async processing for multiple model pipelines
     - RESTful APIs for configuration and management

6. **Frontend**
   - **Use**: React + WebRTC
   - **Features**:
     - Real-time audio capture
     - Video playback with synchronized lip movements
     - Responsive design for multiple devices

## What We Need You to Code

### 1. Backend Architecture
```
/app
├── main.py                 # FastAPI main application
├── services/
│   ├── stt_service.py     # Faster-Whisper integration
│   ├── llm_service.py     # Ollama client
│   ├── tts_service.py     # GPT-SoVITS integration
│   └── lipsync_service.py # SadTalker integration
├── models/
│   └── conversation.py    # Conversation state management
└── utils/
    ├── audio_processing.py
    └── video_processing.py
```

### 2. Key APIs to Implement
- **POST /chat** - Text-based conversation
- **WS /voice-chat** - Real-time voice conversation
- **POST /train-voice** - Train custom TTS model
- **POST /generate-avatar** - Create lip-sync video using provided `avatar.png`
- **GET /health** - System health check

### 3. Frontend Components
- Audio recorder with real-time visualization
- Video player for avatar responses
- Chat history interface
- Voice model training interface
- System status dashboard

### 4. Processing Pipeline
```
User Audio → STT → LLM → TTS → Lip Sync → Video Response
     ↓
Real-time chunked processing with minimal latency
```

## Specific Implementation Requirements

### Audio Processing
- **Input**: 16kHz WAV format
- **Chunking**: 1-3 second segments for low latency
- **Quality**: Noise reduction and audio enhancement
- **Format**: Support for multiple audio codecs

### Video Generation
- **Output**: MP4 format, 30fps
- **Resolution**: 512x512 minimum, scalable to 1024x1024
- **Avatar**: Static base image (`avatar.png`, 512x512) with dynamic lip movements
- **Sync**: Perfect audio-visual synchronization

### Performance Targets
- **STT Latency**: < 500ms per chunk
- **LLM Response**: < 2 seconds
- **TTS Generation**: < 1 second for 5-second audio
- **Lip Sync**: < 3 seconds for 5-second video
- **Total Latency**: < 7 seconds end-to-end

### Memory Management
- **GPU Memory**: Efficient model loading/unloading
- **RAM Usage**: < 8GB total system usage
- **Storage**: Temporary file cleanup
- **Concurrent Users**: Support for 1-3 simultaneous sessions

### Configuration Management
- Model selection (different sizes based on hardware)
- Audio/video quality settings
- Language preferences
- Voice model management
- Performance monitoring

### Error Handling & Fallbacks
- Graceful degradation when models fail
- Audio/video format compatibility checks
- Network interruption handling
- Model loading failure recovery

## Development Priorities

### Phase 1: Core Pipeline
- Set up FastAPI backend with WebSocket support
- Integrate Faster-Whisper for STT
- Connect to Ollama for LLM responses
- Basic GPT-SoVITS integration

### Phase 2: Advanced Features
- SadTalker lip synchronization using `avatar.png`
- Real-time audio streaming
- Frontend React application
- Voice model training interface

### Phase 3: Optimization
- Performance tuning and caching
- Memory optimization
- Error handling and logging
- Documentation and deployment guides

## Hardware Assumptions
- **Minimum**: RTX 3060 (12GB VRAM), 16GB RAM
- **Recommended**: RTX 4080+ (16GB+ VRAM), 32GB RAM
- **OS**: Ubuntu 20.04+ or Windows 10+

## Expected Deliverables
- Complete source code with documentation
- Installation and setup scripts
- Docker configuration for easy deployment
- Training guides for custom voice models
- Performance benchmarking tools
- User manual and API documentation

## Success Criteria
- End-to-end voice conversation working smoothly
- High-quality voice synthesis indistinguishable from reference
- Natural lip synchronization without artifacts using `avatar.png`
- Stable performance for extended conversations
- Easy setup and deployment process

We want this to be a production-ready system that can serve as a foundation for various voice AI applications. Focus on code quality, performance, and user experience.