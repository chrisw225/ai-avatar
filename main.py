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
    """Run Wav2Lip to generate lip-sync video with audio"""
    print("Running Wav2Lip...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Use the avatar image
    face_path = "avatar.png"
    
    # Check if face image exists
    if not os.path.exists(face_path):
        raise FileNotFoundError(f"Face image not found: {face_path}")
    
    # Create temporary file for video-only output from Wav2Lip
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        temp_video_path = temp_video.name
    
    try:
        # Run the actual Wav2Lip inference
        cmd = [
            sys.executable, "wav2lip/inference.py",
            "--checkpoint_path", "wav2lip/checkpoints/wav2lip_gan.pth",
            "--face", face_path,
            "--audio", audio_path,
            "--outfile", temp_video_path,
            "--nosmooth",
            "--resize_factor", "1"
        ]
        
        print("Running Wav2Lip inference for lip-sync...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Wav2Lip inference failed: {result.stderr}")
            print("Falling back to static avatar with audio...")
            return create_fallback_video(audio_path, face_path, output_path)
        
        print("Wav2Lip inference completed successfully!")
        
        # Now combine the lip-sync video with audio using improved FFmpeg integration
        print("Adding audio to lip-sync video...")
        try:
            # Try using ffmpeg-python first
            import ffmpeg
            
            print("Using ffmpeg-python to combine video and audio...")
            
            # Use ffmpeg to combine video and audio
            input_video = ffmpeg.input(temp_video_path)
            input_audio = ffmpeg.input(audio_path)
            
            out = ffmpeg.output(
                input_video, input_audio,
                output_path,
                vcodec='libx264',
                acodec='aac',
                strict='experimental'
            )
            
            ffmpeg.run(out, overwrite_output=True, quiet=True)
            
            print(f"Lip-sync video with audio saved to: {output_path}")
            return output_path
            
        except ImportError:
            print("ffmpeg-python not available, trying alternative method...")
        except Exception as e:
            print(f"ffmpeg-python failed: {e}")
        
        # Fallback: try using subprocess with ffmpeg directly
        try:
            print("Trying direct ffmpeg command...")
            
            # Try to find ffmpeg executable
            ffmpeg_cmd = None
            
            # First try imageio-ffmpeg (comes with MoviePy)
            try:
                import imageio_ffmpeg
                ffmpeg_cmd = imageio_ffmpeg.get_ffmpeg_exe()
                print(f"Found FFmpeg via imageio-ffmpeg: {ffmpeg_cmd}")
            except ImportError:
                pass
            
            # If not found, try system PATH
            if not ffmpeg_cmd:
                for cmd in ['ffmpeg', 'ffmpeg.exe']:
                    try:
                        subprocess.run([cmd, '-version'], 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE, 
                                     check=True)
                        ffmpeg_cmd = cmd
                        break
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
            
            if ffmpeg_cmd:
                # FFmpeg command to combine video and audio
                cmd = [
                    ffmpeg_cmd,
                    '-i', temp_video_path,
                    '-i', audio_path,
                    '-c:v', 'copy',  # Copy video stream
                    '-c:a', 'aac',   # Encode audio as AAC
                    '-strict', 'experimental',
                    '-y',  # Overwrite output file
                    output_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"Lip-sync video with audio saved to: {output_path}")
                    return output_path
                else:
                    print(f"FFmpeg command failed: {result.stderr}")
            else:
                print("FFmpeg not found in PATH")
                
        except Exception as e:
            print(f"Direct ffmpeg failed: {e}")
        
        # Final fallback: copy the lip-sync video without audio
        try:
            print("Audio combination failed, copying lip-sync video without audio...")
            import shutil
            shutil.copy2(temp_video_path, output_path)
            
            print(f"Lip-sync video saved to: {output_path}")
            print("Note: Video contains lip-sync but no audio due to FFmpeg issues.")
            
            return output_path
            
        except Exception as e:
            print(f"Final fallback failed: {e}")
            return False
            
    except Exception as e:
        print(f"Error in Wav2Lip processing: {e}")
        print("Falling back to static avatar with audio...")
        return create_fallback_video(audio_path, face_path, output_path)
        
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_video_path)
        except:
            pass

def create_fallback_video(audio_path, face_path, output_path):
    """Create a fallback video with static avatar and audio when Wav2Lip fails."""
    print("Creating fallback video with static avatar and audio...")
    
    # Create temporary file for video-only output
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        temp_video_path = temp_video.name
    
    try:
        import cv2
        import librosa
        
        # Load audio to get duration
        y, sr = librosa.load(audio_path)
        duration = len(y) / sr
        fps = 25
        total_frames = int(duration * fps)
        
        print(f"Audio duration: {duration:.2f} seconds, creating {total_frames} frames at {fps} FPS")
        
        # Load avatar image
        avatar = cv2.imread(face_path)
        if avatar is None:
            raise FileNotFoundError(f"Could not load avatar image: {face_path}")
        
        # Resize avatar to a standard size for better compatibility
        avatar = cv2.resize(avatar, (512, 512))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (avatar.shape[1], avatar.shape[0]))
        
        # Write frames
        for i in range(total_frames):
            out.write(avatar)
        
        out.release()
        print("Created video from avatar image")
        
        # Now combine video with audio using FFmpeg
        try:
            # Try to find ffmpeg executable
            ffmpeg_cmd = None
            
            # First try imageio-ffmpeg (comes with MoviePy)
            try:
                import imageio_ffmpeg
                ffmpeg_cmd = imageio_ffmpeg.get_ffmpeg_exe()
                print(f"Found FFmpeg via imageio-ffmpeg: {ffmpeg_cmd}")
            except ImportError:
                pass
            
            if ffmpeg_cmd:
                # FFmpeg command to combine video and audio
                cmd = [
                    ffmpeg_cmd,
                    '-i', temp_video_path,
                    '-i', audio_path,
                    '-c:v', 'copy',  # Copy video stream
                    '-c:a', 'aac',   # Encode audio as AAC
                    '-strict', 'experimental',
                    '-y',  # Overwrite output file
                    output_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print(f"Fallback video with audio saved to: {output_path}")
                    return output_path
                else:
                    print(f"FFmpeg command failed: {result.stderr}")
            
            # If FFmpeg fails, just copy the video without audio
            import shutil
            shutil.copy2(temp_video_path, output_path)
            print(f"Fallback video saved to: {output_path} (no audio)")
            return output_path
            
        except Exception as e:
            print(f"Fallback video creation failed: {e}")
            return False
            
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_video_path)
        except:
            pass

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