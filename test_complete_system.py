#!/usr/bin/env python3
"""
Test script for the complete AI Avatar system.
This script tests both TTS and lip-sync functionality.
"""

import subprocess
import sys
import os

def test_basic_functionality():
    """Test basic functionality of the AI Avatar system."""
    print("Testing basic functionality...")
    
    # Run the AI Avatar with a simple phrase
    cmd = [sys.executable, "main.py", "--text", "Hello, this is a test of the AI Avatar system."]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
        print("✓ Basic functionality test passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Basic functionality test failed: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False
    except subprocess.TimeoutExpired:
        print("✗ Basic functionality test timed out")
        return False

def test_with_playback():
    """Test the AI Avatar with video playback."""
    print("Testing with playback (will timeout after 10 seconds)...")
    
    # Run the AI Avatar with playback enabled
    cmd = [sys.executable, "main.py", "--text", "This is a playback test.", "--play"]
    
    try:
        # Use a shorter timeout for playback test since it will run indefinitely
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=10)
        print("✓ Playback test completed")
        return True
    except subprocess.TimeoutExpired:
        print("✓ Playback test passed (timed out as expected)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Playback test failed: {e}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

def check_output_files():
    """Check if output files are created and have reasonable sizes."""
    print("Checking output files...")
    
    # Check for video output
    video_path = "output/result.mp4"
    audio_path = "tts/output.wav"
    
    success = True
    
    if os.path.exists(video_path):
        size = os.path.getsize(video_path)
        print(f"✓ Video output exists: {video_path} ({size} bytes)")
        if size < 1000:  # Less than 1KB is probably an error
            print("⚠ Warning: Video file is very small")
            success = False
    else:
        print(f"✗ Video output not found: {video_path}")
        success = False
    
    if os.path.exists(audio_path):
        size = os.path.getsize(audio_path)
        print(f"✓ Audio output exists: {audio_path} ({size} bytes)")
        if size < 1000:  # Less than 1KB is probably an error
            print("⚠ Warning: Audio file is very small")
            success = False
    else:
        print(f"✗ Audio output not found: {audio_path}")
        success = False
    
    return success

def main():
    """Run all tests."""
    print("Starting AI Avatar system tests...\n")
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Basic functionality
    if test_basic_functionality():
        tests_passed += 1
    print()
    
    # Test 2: Check output files
    if check_output_files():
        tests_passed += 1
    print()
    
    # Test 3: Playback test
    if test_with_playback():
        tests_passed += 1
    print()
    
    # Summary
    print(f"Tests completed: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The AI Avatar system is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())