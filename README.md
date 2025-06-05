# AI-Avatar

A local Windows application that performs near real-time lip-sync animation using a 2D image and locally deployed Text-to-Speech.

## Features

- Converts text to speech using pyttsx3 and gTTS with multi-language support
- Applies lip-sync to a static avatar image using Wav2Lip
- Supports both standard and real-time preview modes
- Fully local deployment - no cloud services required
- Optimized for minimal latency using GPU acceleration
- High-quality video output with MoviePy integration

## System Requirements

- Windows 10 or 11 (64-bit)
- NVIDIA GPU (GTX 1660 or higher recommended)
- Python 3.8+ (64-bit)
- FFmpeg installed and available in PATH

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/chrisw225/ai-avatar.git
   cd ai-avatar
   ```

2. Run the setup script which will:
   - Create a virtual environment
   - Install all required dependencies
   - Clone Wav2Lip and download required models
   ```
   setup.bat
   ```

3. Install FFmpeg (if not already installed):
   - Download from https://ffmpeg.org/download.html
   - Add to your system PATH

4. Place your avatar image in the root directory and name it `avatar.png` (512x512 resolution recommended)

## Usage

### Basic Usage

Run the application using the provided batch file:
```
run.bat --text "Hello, I am your AI Avatar!" --play
```

Or activate the virtual environment and run directly:
```
activate.bat
python main.py --text "Hello, I am your AI Avatar!" --play
```

### Command Line Arguments

- `--text`: Text input for the avatar to speak
- `--language`: Language code (default: en)
- `--mode`: Processing mode: standard or real-time (default: standard)
- `--play`: Play the generated video after creation
- `--use-gtts`: Use Google TTS instead of pyttsx3 (requires internet)

If no text is provided via command line, the program will prompt for input.

### Examples

1. Basic usage with English text:
   ```
   run.bat --text "Hello, I am your AI Avatar!" --play
   ```

2. Using Google TTS:
   ```
   run.bat --text "Hello world" --use-gtts --play
   ```

3. Generate video without playing:
   ```
   run.bat --text "Hello, this is a test"
   ```

4. Run the simplified demo version:
   ```
   run_simple_demo.bat --text "Hello world" --language en --play
   ```

## Output

Generated videos are saved to the `output/` directory. The default output file is `output/result.mp4`.

## Features Implemented

✅ **Text-to-Speech**: Multiple TTS engines (pyttsx3, gTTS)
✅ **Lip-Sync Animation**: Wav2Lip integration with face detection
✅ **High-Quality Output**: MoviePy integration for better video quality
✅ **Face Detection**: Automatic face detection and coordinate extraction
✅ **Error Handling**: Robust error handling and fallback mechanisms
✅ **Video Playback**: Built-in video player for preview
✅ **Multi-Language Support**: Support for various languages
✅ **Batch Processing**: Efficient batch processing for better performance

## Test Results

The system has been thoroughly tested and all components are working correctly:
- ✅ Basic functionality test: **PASSED**
- ✅ Output file generation: **PASSED** 
- ✅ Video playback test: **PASSED**

## Troubleshooting

- **CUDA/GPU Issues**: Make sure you have the latest NVIDIA drivers installed
- **Missing Models**: The application will attempt to download required models on first run
- **FFmpeg Errors**: Ensure FFmpeg is properly installed and available in your system PATH
- **Face Detection Issues**: Make sure your avatar image has a clear, frontal face
- **Virtual Environment Issues**: If you encounter problems with the virtual environment, try deleting the `venv` folder and running `setup.bat` again

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) for lip-sync technology
- [pyttsx3](https://github.com/nateshmbhat/pyttsx3) for offline text-to-speech
- [gTTS](https://github.com/pndurette/gTTS) for Google text-to-speech
- [MoviePy](https://github.com/Zulko/moviepy) for video processing