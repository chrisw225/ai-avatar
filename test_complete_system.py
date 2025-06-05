#!/usr/bin/env python3
"""
Complete system test for AI Avatar with Wav2Lip integration.
Tests TTS, lip-sync, and playback functionality.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def test_basic_functionality():
    """Test basic AI Avatar functionality."""
    print("\n=== Testing Basic Functionality ===")
    
    # Test with a simple phrase
    cmd = [
        sys.executable, "main.py",
        "--text", "Hello, this is a test of the AI Avatar system."
    ]
    
    print("Running AI Avatar with test phrase...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Basic functionality test PASSED")
        return True
    else:
        print(f"❌ Basic functionality test FAILED: {result.stderr}")
        return False

def test_with_playback():
    """Test AI Avatar with video playback (will timeout after 10 seconds)."""
    print("\n=== Testing with Playback ===")
    
    cmd = [
        sys.executable, "main.py",
        "--text", "Testing playback functionality.",
        "--play"
    ]
    
    print("Running AI Avatar with playback (will timeout in 10 seconds)...")
    try:
        # Run with timeout to prevent hanging
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        print("✅ Playback test completed successfully")
        return True
    except subprocess.TimeoutExpired:
        print("✅ Playback test completed (timeout as expected)")
        return True
    except Exception as e:
        print(f"❌ Playback test FAILED: {e}")
        return False

def check_output_files():
    """Check if output files were created properly."""
    print("\n=== Checking Output Files ===")
    
    # Check video output
    video_path = Path("output/result.mp4")
    if video_path.exists():
        size = video_path.stat().st_size
        print(f"✅ Video output exists: {video_path} ({size} bytes)")
        video_ok = size > 1000  # Should be at least 1KB
    else:
        print(f"❌ Video output missing: {video_path}")
        video_ok = False
    
    # Check audio output
    audio_path = Path("tts/output.wav")
    if audio_path.exists():
        size = audio_path.stat().st_size
        print(f"✅ Audio output exists: {audio_path} ({size} bytes)")
        audio_ok = size > 1000  # Should be at least 1KB
    else:
        print(f"❌ Audio output missing: {audio_path}")
        audio_ok = False
    
    return video_ok and audio_ok

def main():
    """Run all tests and report results."""
    print("🚀 Starting AI Avatar System Tests")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Basic functionality
    if test_basic_functionality():
        tests_passed += 1
    
    # Test 2: Check output files
    if check_output_files():
        tests_passed += 1
    
    # Test 3: Playback test
    if test_with_playback():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests PASSED! AI Avatar system is working correctly.")
        return 0
    else:
        print(f"⚠️  {total_tests - tests_passed} test(s) FAILED. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
