import os
import sys
import argparse
import subprocess
import tempfile
import time
import pygame
import cv2
import numpy as np
from pathlib import Path

# Initialize pygame for audio playback
pygame.mixer.init()

def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = ['output', 'tts', 'wav2lip/checkpoints']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print("Directories set up successfully.")

def download_models():
    """Download required models if they don't exist."""
    models = {
        'wav2lip/checkpoints/wav2lip_gan.pth': 'https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?share=EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp2pgHDc0A'
    }
    
    for model_path, url in models.items():
        if not os.path.exists(model_path):
            print(f"Model {model_path} not found.")
            print("Please download the Wav2Lip model manually:")
            print(f"1. Download from: {url}")
            print(f"2. Save as: {model_path}")
            return False
    
    print("All models are available.")
    return True

def text_to_speech(text, output_path="tts/output.wav", voice_rate=150):
    """Convert text to speech using pyttsx3 or gTTS as fallback."""
    start_time = time.time()
    
    try:
        # Try pyttsx3 first (offline, faster)
        import pyttsx3
        
        engine = pyttsx3.init()
        engine.setProperty('rate', voice_rate)
        
        # Get available voices
        voices = engine.getProperty('voices')
        if voices:
            # Try to use a female voice if available
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
        
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        
        processing_time = time.time() - start_time
        print(f"TTS completed using pyttsx3 in {processing_time:.2f} seconds")
        
    except Exception as e:
        print(f"pyttsx3 failed: {e}")
        print("Trying gTTS as fallback...")
        
        try:
            # Fallback to gTTS (requires internet)
            from gtts import gTTS
            import pygame
            
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(output_path)
            
            processing_time = time.time() - start_time
            print(f"TTS completed using gTTS in {processing_time:.2f} seconds")
            
        except Exception as e2:
            print(f"Both TTS methods failed: pyttsx3: {e}, gTTS: {e2}")
            return False
    
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"Audio file created: {output_path} ({file_size} bytes)")
        return True
    else:
        print("Failed to create audio file")
        return False

def run_wav2lip(face_path, audio_path, output_path, checkpoint_path="wav2lip/checkpoints/wav2lip_gan.pth"):
    """Run Wav2Lip inference to generate lip-sync video."""
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return False
    
    if not os.path.exists(face_path):
        print(f"Face image not found: {face_path}")
        return False
    
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return False
    
    # Wav2Lip command
    cmd = [
        sys.executable, 'wav2lip/inference.py',
        '--checkpoint_path', checkpoint_path,
        '--face', face_path,
        '--audio', audio_path,
        '--outfile', output_path,
        '--nosmooth',
        '--resize_factor', '1',
        '--face_det_batch_size', '4'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Wav2Lip completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error running Wav2Lip: {e}")
        print(f"Command output: {e.stdout}")
        print(f"Command error: {e.stderr}")
        return False

def play_video(video_path):
    """Play video using OpenCV."""
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return False
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Error: Could not open video file")
            return False
        
        print("Playing video. Press 'q' to quit.")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25  # Default FPS
        
        frame_delay = 1.0 / fps
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Resize frame for display
            height, width = frame.shape[:2]
            if width > 800:
                scale = 800 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            cv2.imshow('AI Avatar', frame)
            
            # Control playback speed
            key = cv2.waitKey(int(frame_delay * 1000)) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return True
        
    except Exception as e:
        print(f"Error playing video: {e}")
        return False

def process_with_real_time_preview(text, face_path, play_video_flag=False):
    """Process text with real-time preview and fallback options."""
    
    # Check if face image exists
    if not os.path.exists(face_path):
        print(f"Error: {face_path} not found. Please place your avatar image in the root directory.")
        return False
    
    start_time = time.time()
    
    # Step 1: Text to Speech
    print("Converting text to speech...")
    audio_path = "tts/output.wav"
    if not text_to_speech(text, audio_path):
        print("Failed to generate speech")
        return False
    
    tts_time = time.time() - start_time
    print(f"TTS completed in {tts_time:.2f} seconds")
    
    # Step 2: Try Wav2Lip first
    print("Running Wav2Lip...")
    output_path = "output/result.mp4"
    
    if run_wav2lip(face_path, audio_path, output_path):
        print(f"Video saved to: {output_path}")
        
        # Step 3: Play video if requested
        if play_video_flag:
            play_video(output_path)
        
        total_time = time.time() - start_time
        print(f"Total processing time: {total_time:.2f} seconds")
        return output_path
    
    else:
        # Fallback: Create simple video with avatar image and audio
        print("Creating video with avatar image and audio...")
        
        try:
            # Load audio to get duration
            import librosa
            y, sr = librosa.load(audio_path)
            duration = len(y) / sr
            fps = 25
            total_frames = int(duration * fps)
            
            print(f"Audio duration: {duration:.2f} seconds, creating {total_frames} frames at {fps} FPS")
            
            # Load avatar image
            img = cv2.imread(face_path)
            if img is None:
                print(f"Could not load image: {face_path}")
                return False
            
            # Resize image to standard video size
            height, width = img.shape[:2]
            target_width = 512
            target_height = int(height * (target_width / width))
            img = cv2.resize(img, (target_width, target_height))
            
            # Create temporary video file without audio
            temp_video_path = tempfile.mktemp(suffix='.mp4')
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (target_width, target_height))
            
            # Write frames
            for i in range(total_frames):
                out.write(img)
            
            out.release()
            print("Created video from avatar image")
            
            # Now combine video with audio using a simpler approach
            print("Adding audio to video...")
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
                
                print(f"Video with audio saved to: {output_path}")
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
                        print(f"Video with audio saved to: {output_path}")
                        return output_path
                    else:
                        print(f"FFmpeg command failed: {result.stderr}")
                else:
                    print("FFmpeg not found in PATH")
                    
            except Exception as e:
                print(f"Direct ffmpeg failed: {e}")
            
            # Final fallback: create a simple solution using basic libraries
            try:
                print("Using basic audio-video combination...")
                
                # Read the video file and create a new one with audio
                import wave
                import struct
                
                # For now, just copy the video file and note that audio should be added manually
                import shutil
                shutil.copy2(temp_video_path, output_path)
                
                print(f"Video saved to: {output_path}")
                print("Note: Audio combination failed. The video contains the avatar image but no audio.")
                print("You may need to install FFmpeg or check your MoviePy installation.")
                
                return output_path
                
            except Exception as e:
                print(f"Final fallback failed: {e}")
                return False
                
        finally:
            # Clean up temporary file
            if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                except:
                    pass

def main():
    parser = argparse.ArgumentParser(description='AI Avatar with Lip Sync Animation')
    parser.add_argument('--text', type=str, required=True, help='Text to convert to speech')
    parser.add_argument('--face', type=str, default='avatar.png', help='Path to face image')
    parser.add_argument('--play', action='store_true', help='Play the generated video')
    parser.add_argument('--voice_rate', type=int, default=150, help='Speech rate (words per minute)')
    
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    # Check if models are available
    if not download_models():
        print("Please download the required models first.")
        return
    
    # Process the request
    start_time = time.time()
    
    result = process_with_real_time_preview(
        text=args.text,
        face_path=args.face,
        play_video_flag=args.play
    )
    
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds")
    
    if result:
        print(f"Output saved to {result}")
    else:
        print("Processing failed")

if __name__ == "__main__":
    main()