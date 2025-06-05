#!/usr/bin/env python3
"""
Test script to debug MoviePy import and functionality.
"""

print("Testing MoviePy imports...")

try:
    import moviepy
    print(f"✓ MoviePy base import successful, version: {moviepy.__version__}")
except ImportError as e:
    print(f"✗ MoviePy base import failed: {e}")
    exit(1)

try:
    from moviepy.editor import VideoFileClip, AudioFileClip
    print("✓ MoviePy editor imports successful")
except ImportError as e:
    print(f"✗ MoviePy editor imports failed: {e}")
    exit(1)

# Test basic functionality
try:
    print("Testing basic MoviePy functionality...")
    
    # Test with our actual files
    import os
    if os.path.exists("tts/output.wav"):
        audio = AudioFileClip("tts/output.wav")
        print(f"✓ Audio file loaded successfully, duration: {audio.duration:.2f}s")
        audio.close()
    else:
        print("⚠ No audio file found for testing")
    
    print("✓ MoviePy is working correctly!")
    
except Exception as e:
    print(f"✗ MoviePy functionality test failed: {e}")
    import traceback
    traceback.print_exc() 