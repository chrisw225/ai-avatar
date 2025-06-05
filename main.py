import os
import sys
import time
import argparse
import subprocess
import numpy as np
from pathlib import Path
import torch
import cv2
import pygame
import tempfile

# TTS Imports - using alternative libraries compatible with Python 3.12
import pyttsx3
from gtts import gTTS

def setup_directories():
    """Ensure all required directories exist."""
    directories = ['tts', 'wav2lip', 'output']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Check if avatar.png exists
    avatar_path = Path('avatar.png')
    if not avatar_path.exists():
        print("Error: avatar.png not found. Please place your avatar image in the root directory.")
        sys.exit(1)

def download_models():
    """Download required models if they don't exist."""
    # Check if Wav2Lip model exists, if not download it
    wav2lip_model_path = Path('wav2lip/checkpoints/wav2lip_gan.pth')
    if not wav2lip_model_path.exists():
        print("Error: Wav2Lip model not found. Please run setup.py first.")
        sys.exit(1)
    
    # Check if face detection model exists
    face_detection_path = Path('wav2lip/face_detection/detection/sfd/s3fd.pth')
    if not face_detection_path.exists():
        print("Error: Face detection model not found. Please run setup.py first.")
        sys.exit(1)

def text_to_speech(text, output_path='tts/output.wav', language='en', use_gtts=False):
    """Convert text to speech using pyttsx3 or gTTS."""
    print("Converting text to speech...")
    start_time = time.time()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if use_gtts:
        # Use gTTS for online TTS (requires internet)
        try:
            # Map language codes
            lang_map = {
                'zh-cn': 'zh',
                'en': 'en',
                'es': 'es',
                'fr': 'fr',
                'de': 'de',
                'ja': 'ja',
                'ko': 'ko'
            }
            gtts_lang = lang_map.get(language, 'en')
            
            tts = gTTS(text=text, lang=gtts_lang, slow=False)
            tts.save(output_path)
            print(f"TTS completed using gTTS in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            print(f"gTTS failed: {e}. Falling back to pyttsx3...")
            use_gtts = False
    
    if not use_gtts:
        # Use pyttsx3 for offline TTS
        try:
            engine = pyttsx3.init()
            
            # Set properties
            voices = engine.getProperty('voices')
            if voices:
                # Try to find a suitable voice based on language
                for voice in voices:
                    if language == 'zh-cn' and ('chinese' in voice.name.lower() or 'mandarin' in voice.name.lower()):
                        engine.setProperty('voice', voice.id)
                        break
                    elif language == 'en' and 'english' in voice.name.lower():
                        engine.setProperty('voice', voice.id)
                        break
            
            # Set speech rate and volume
            engine.setProperty('rate', 150)  # Speed of speech
            engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)
            
            # Save to file
            engine.save_to_file(text, output_path)
            engine.runAndWait()
            print(f"TTS completed using pyttsx3 in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            print(f"pyttsx3 failed: {e}")
            raise
    
    elapsed = time.time() - start_time
    print(f"TTS completed in {elapsed:.2f} seconds")
    
    return output_path

def run_wav2lip(audio_path, output_path='output/result.mp4'):
    """Run Wav2Lip to generate lip-sync video"""
    print("Running Wav2Lip...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Use the avatar image
    face_path = "avatar.png"
    
    # Check if face image exists
    if not os.path.exists(face_path):
        raise FileNotFoundError(f"Face image not found: {face_path}")
    
    # Wav2Lip command with proper face detection enabled
    cmd = [
        sys.executable, "wav2lip/inference.py",
        "--checkpoint_path", "wav2lip/checkpoints/wav2lip_gan.pth",
        "--face", face_path,
        "--audio", audio_path,
        "--outfile", output_path,
        "--nosmooth",  # For faster processing
        "--resize_factor", "1",  # Keep original resolution
        "--face_det_batch_size", "4"  # Smaller batch size for better detection
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Wav2Lip completed successfully!")
        print(f"Video saved to: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error running Wav2Lip: {e}")
        print(f"Command output: {e.stdout}")
        print(f"Command error: {e.stderr}")
        return False

def play_video(video_path):
    """Play the generated video."""
    try:
        cap = cv2.VideoCapture(video_path)
        
        # Initialize pygame for audio
        pygame.init()
        pygame.mixer.init()
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create window
        cv2.namedWindow('AI Avatar', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('AI Avatar', width, height)
        
        print("Playing video. Press 'q' to quit.")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.imshow('AI Avatar', frame)
            
            # Maintain proper playback speed
            if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                break
        
        cap.release()
        pygame.quit()
        cv2.destroyAllWindows()
    
    except Exception as e:
        print(f"Error playing video: {e}")

def process_with_real_time_preview(text, language='en'):
    """Process text with near real-time preview (lower latency)."""
    print("Processing with real-time preview...")
    
    # Generate audio
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        audio_path = temp_audio.name
    
    # Use TTS to generate audio
    text_to_speech(text, audio_path, language)
    
    # Load avatar
    avatar = cv2.imread('avatar.png')
    
    # Play audio and animate
    pygame.init()
    pygame.mixer.init()
    sound = pygame.mixer.Sound(audio_path)
    
    # Create window
    cv2.namedWindow('AI Avatar (Real-time)', cv2.WINDOW_NORMAL)
    
    # Play audio
    channel = sound.play()
    
    print("Playing real-time preview. Press 'q' to quit.")
    
    # Simple animation loop - in a real implementation, this would use Wav2Lip's models
    # to generate the lip movement in sync with the audio
    while channel.get_busy():
        # Here you would use Wav2Lip to generate the frame
        # This is a placeholder for the actual frame generation
        frame = avatar.copy()
        
        cv2.imshow('AI Avatar (Real-time)', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    pygame.quit()
    cv2.destroyAllWindows()
    os.unlink(audio_path)  # Clean up temporary file

def main():
    """Main function to process text input and generate avatar animation."""
    parser = argparse.ArgumentParser(description='AI Avatar with lip-sync')
    parser.add_argument('--text', type=str, help='Text input for the avatar to speak')
    parser.add_argument('--language', type=str, default='en', help='Language code (default: en)')
    parser.add_argument('--mode', type=str, default='standard', choices=['standard', 'real-time'], 
                        help='Processing mode: standard or real-time')
    parser.add_argument('--play', action='store_true', help='Play the generated video after creation')
    parser.add_argument('--use-gtts', action='store_true', help='Use gTTS instead of pyttsx3 (requires internet)')
    args = parser.parse_args()
    
    # Setup directories and check models
    setup_directories()
    download_models()
    
    # If no text is provided, prompt for input
    text = args.text
    if not text:
        text = input("Enter text for the avatar to speak: ")
    
    # Process based on mode
    if args.mode == 'real-time':
        process_with_real_time_preview(text, args.language)
    else:
        # Standard processing
        start_time = time.time()
        
        # Generate speech from text
        audio_path = text_to_speech(text, language=args.language, use_gtts=args.use_gtts)
        
        # Generate lip-sync video
        video_path = run_wav2lip(audio_path)
        
        total_time = time.time() - start_time
        print(f"Total processing time: {total_time:.2f} seconds")
        
        # Play the video if requested
        if args.play:
            play_video(video_path)
        
        print(f"Output saved to {video_path}")

if __name__ == "__main__":
    main()