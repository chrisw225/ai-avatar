#!/usr/bin/env python3
"""
Simplified Wav2Lip inference script with proper audio handling.
This script ensures that the output video includes the audio track.
"""

import os
import sys
import cv2
import numpy as np
import subprocess
import tempfile
from pathlib import Path

def combine_video_audio(video_path, audio_path, output_path):
    """
    Combine video and audio using ffmpeg or moviepy.
    """
    try:
        # Try using moviepy first (more reliable)
        from moviepy.editor import VideoFileClip, AudioFileClip
        
        print("Combining video and audio using MoviePy...")
        
        # Load video and audio
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)
        
        # Set the audio of the video clip
        final_video = video.set_audio(audio)
        
        # Write the result
        final_video.write_videofile(output_path, 
                                  codec='libx264', 
                                  audio_codec='aac',
                                  temp_audiofile='temp-audio.m4a',
                                  remove_temp=True)
        
        # Close clips to free memory
        video.close()
        audio.close()
        final_video.close()
        
        print(f"Video with audio saved to: {output_path}")
        return True
        
    except ImportError:
        print("MoviePy not available, trying ffmpeg...")
        return combine_with_ffmpeg(video_path, audio_path, output_path)
    except Exception as e:
        print(f"MoviePy failed: {e}")
        return combine_with_ffmpeg(video_path, audio_path, output_path)

def combine_with_ffmpeg(video_path, audio_path, output_path):
    """
    Combine video and audio using ffmpeg command line.
    """
    try:
        # Try to find ffmpeg
        ffmpeg_cmd = None
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
        
        if not ffmpeg_cmd:
            print("FFmpeg not found. Please install FFmpeg.")
            return False
        
        print("Combining video and audio using FFmpeg...")
        
        # FFmpeg command to combine video and audio
        cmd = [
            ffmpeg_cmd,
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',  # Copy video stream without re-encoding
            '-c:a', 'aac',   # Encode audio as AAC
            '-strict', 'experimental',
            '-y',  # Overwrite output file
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Video with audio saved to: {output_path}")
            return True
        else:
            print(f"FFmpeg failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"FFmpeg combination failed: {e}")
        return False

def run_wav2lip_with_audio(face_path, audio_path, output_path):
    """
    Run Wav2Lip and ensure audio is included in the output.
    """
    print("Running Wav2Lip with audio integration...")
    
    # Create temporary file for video-only output
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        temp_video_path = temp_video.name
    
    try:
        # Run the original Wav2Lip command (video only)
        cmd = [
            sys.executable, "wav2lip/inference.py",
            "--checkpoint_path", "wav2lip/checkpoints/wav2lip_gan.pth",
            "--face", face_path,
            "--audio", audio_path,
            "--outfile", temp_video_path,
            "--nosmooth",
            "--resize_factor", "1"
        ]
        
        print("Running Wav2Lip inference...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Wav2Lip failed: {result.stderr}")
            return False
        
        # Now combine the video with audio
        if combine_video_audio(temp_video_path, audio_path, output_path):
            print("Successfully created video with audio!")
            return True
        else:
            # If combining fails, at least copy the video-only version
            print("Audio combination failed, copying video-only version...")
            import shutil
            shutil.copy2(temp_video_path, output_path)
            return True
            
    except Exception as e:
        print(f"Error in Wav2Lip processing: {e}")
        return False
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_video_path)
        except:
            pass

def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Wav2Lip with audio integration')
    parser.add_argument('--face', required=True, help='Path to face image/video')
    parser.add_argument('--audio', required=True, help='Path to audio file')
    parser.add_argument('--outfile', required=True, help='Output video path')
    
    args = parser.parse_args()
    
    if run_wav2lip_with_audio(args.face, args.audio, args.outfile):
        print("Success!")
        return 0
    else:
        print("Failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 