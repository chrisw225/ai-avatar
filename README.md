# AI Chatbot with Voice and Lip Sync

A real-time AI chatbot system that integrates Speech-to-Text (STT), Large Language Model (LLM), Text-to-Speech (TTS), and lip synchronization capabilities to create an immersive conversational AI experience.

## 🚀 Features

- **Real-time Voice Conversation**: Speak to the AI and get voice responses
- **Text-based Chat**: Traditional text-based conversation interface
- **Lip Sync Avatar**: AI-generated lip-synced video responses using your avatar
- **Multiple AI Models**: Support for various LLM models via Ollama
- **Voice Cloning**: Custom TTS voice model training
- **WebSocket Support**: Real-time bidirectional communication
- **REST API**: Comprehensive API for all functionalities
- **File Management**: Audio/video file processing and management
- **Session Management**: Conversation history and context management

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │   AI Services   │
│   (Web/Mobile)  │◄──►│   Backend       │◄──►│                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                         │
                              ▼                         ▼
                    ┌─────────────────┐    ┌─────────────────┐
                    │   WebSocket     │    │   File Storage  │
                    │   Manager       │    │   & Processing  │
                    └─────────────────┘    └─────────────────┘
```

### Core Components

1. **STT Service**: Speech recognition using OpenAI Whisper
2. **LLM Service**: Text generation using Ollama-compatible models
3. **TTS Service**: Voice synthesis using GPT-SoVITS
4. **Lip Sync Service**: Video generation using SadTalker
5. **Conversation Manager**: Session and context management
6. **File Manager**: Audio/video processing and storage

## 📋 Prerequisites

- Python 3.8+
- FFmpeg (for audio/video processing)
- CUDA-compatible GPU (optional, for faster processing)
- Ollama server running locally or remotely
- At least 8GB RAM (16GB+ recommended)

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ai-chatbot
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install System Dependencies

#### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install ffmpeg portaudio19-dev python3-dev
```

#### macOS:
```bash
brew install ffmpeg portaudio
```

#### Windows:
- Download and install FFmpeg from https://ffmpeg.org/
- Add FFmpeg to your system PATH

### 5. Setup Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (e.g., Llama 2)
ollama pull llama2
```

### 6. Download AI Models

The application will automatically download required models on first run:
- Whisper models for STT
- GPT-SoVITS models for TTS
- SadTalker models for lip sync

### 7. Configuration

Create a `.env` file in the project root:

```env
# Server Configuration
HOST=0.0.0.0
PORT=8000
RELOAD=true
LOG_LEVEL=info

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2

# Whisper Configuration
WHISPER_MODEL=base
WHISPER_DEVICE=auto

# File Paths
TEMP_DIR=./temp
MODELS_DIR=./models
AVATAR_PATH=./avatar.png

# API Keys (if needed)
OPENAI_API_KEY=your_openai_key_here
```

### 8. Add Your Avatar

Place your avatar image as `avatar.png` in the project root. Requirements:
- Square aspect ratio (1:1)
- Minimum 512x512 pixels
- Clear face visibility
- PNG or JPG format

## 🚀 Usage

### Starting the Server

```bash
# Development mode
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Text Chat
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?"}'
```

#### Voice Chat (WebSocket)
```javascript
const ws = new WebSocket('ws://localhost:8000/voice-chat');
ws.onopen = () => {
    // Send audio data or text messages
    ws.send(JSON.stringify({
        type: 'text',
        message: 'Hello!'
    }));
};
```

#### Train Voice Model
```bash
curl -X POST http://localhost:8000/train-voice \
  -F "model_name=my_voice" \
  -F "reference_text=Hello, this is my voice" \
  -F "reference_audio=@reference.wav"
```

#### Generate Avatar Video
```bash
curl -X POST http://localhost:8000/generate-avatar \
  -H "Content-Type: application/json" \
  -d '{
    "audio_file_path": "/path/to/audio.wav",
    "expression_scale": 1.0,
    "pose_style": 0
  }'
```

### WebSocket Events

#### Client to Server:
- `text`: Send text message
- `audio`: Send audio data (base64 encoded)
- `ping`: Keep connection alive

#### Server to Client:
- `connection`: Connection established
- `transcription`: Speech-to-text result
- `text_response`: Text-only response
- `response`: Complete response with audio/video
- `status`: Processing status updates
- `error`: Error messages
- `pong`: Ping response

## 🔧 Configuration

### Model Configuration

Update models via API:

```bash
# Update Ollama model
curl -X POST http://localhost:8000/config/ollama \
  -H "Content-Type: application/json" \
  -d '{"model": "llama2:13b"}'

# Get available models
curl http://localhost:8000/config/models
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Default LLM model | `llama2` |
| `WHISPER_MODEL` | Whisper model size | `base` |
| `WHISPER_DEVICE` | Processing device | `auto` |
| `TEMP_DIR` | Temporary files directory | `./temp` |
| `AVATAR_PATH` | Avatar image path | `./avatar.png` |

## 📁 Project Structure

```
ai-chatbot/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── config.py              # Configuration management
│   ├── models/
│   │   ├── __init__.py
│   │   └── conversation.py    # Conversation models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── stt_service.py     # Speech-to-Text service
│   │   ├── llm_service.py     # LLM service
│   │   ├── tts_service.py     # Text-to-Speech service
│   │   └── lipsync_service.py # Lip sync service
│   └── utils/
│       ├── __init__.py
│       ├── audio_utils.py     # Audio processing utilities
│       ├── file_utils.py      # File management utilities
│       └── validation_utils.py # Input validation utilities
├── models/                    # AI model storage
├── temp/                      # Temporary files
├── avatar.png                 # Default avatar image
├── requirements.txt           # Python dependencies
├── .env                       # Environment configuration
└── README.md                  # This file
```

## 🔍 API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_services.py

# Run with coverage
pytest --cov=app tests/
```

## 🐳 Docker Deployment

```dockerfile
# Dockerfile example
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t ai-chatbot .
docker run -p 8000:8000 ai-chatbot
```

## 🔧 Troubleshooting

### Common Issues

1. **Audio device not found**
   ```bash
   # Linux: Install ALSA development files
   sudo apt-get install libasound2-dev
   
   # macOS: Install PortAudio
   brew install portaudio
   ```

2. **CUDA out of memory**
   - Reduce batch sizes in configuration
   - Use smaller models
   - Enable CPU fallback

3. **Ollama connection failed**
   - Ensure Ollama is running: `ollama serve`
   - Check firewall settings
   - Verify OLLAMA_BASE_URL in configuration

4. **Model download fails**
   - Check internet connection
   - Verify disk space
   - Check model URLs in configuration

### Performance Optimization

1. **GPU Acceleration**
   ```bash
   # Install CUDA-enabled PyTorch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Memory Management**
   - Monitor memory usage with `/health` endpoint
   - Adjust model sizes based on available RAM
   - Enable automatic cleanup of old files

3. **Network Optimization**
   - Use local Ollama instance
   - Cache frequently used models
   - Optimize audio/video compression

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `pytest`
5. Commit your changes: `git commit -am 'Add feature'`
6. Push to the branch: `git push origin feature-name`
7. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [Ollama](https://ollama.ai/) for LLM integration
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) for voice synthesis
- [SadTalker](https://github.com/OpenTalker/SadTalker) for lip synchronization
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation at `/docs`
- Review the troubleshooting section above

---

**Note**: This is a development version. For production deployment, ensure proper security measures, rate limiting, and monitoring are in place.