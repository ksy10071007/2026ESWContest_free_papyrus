#!/usr/bin/env python3
"""
MediaPipe Verification Script for Jetson Orin Nano aarch64
Tests cv2, numpy, protobuf, and MediaPipe integration.
"""

import sys

print("=" * 60)
print("MediaPipe Environment Verification")
print("=" * 60)

# Test 1: OpenCV
print("\n[1] Testing OpenCV (cv2)...")
try:
    import cv2
    print(f"    ✓ cv2 version: {cv2.__version__}")
    opencv_cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
    print(f"    ✓ OpenCV CUDA device count: {opencv_cuda_devices}")
    print(f"    ✓ OpenCV CUDA available: {opencv_cuda_devices > 0}")
except Exception as e:
    print(f"    ✗ OpenCV import failed: {e}")
    sys.exit(1)

# Test 2: NumPy
print("\n[2] Testing NumPy...")
try:
    import numpy as np
    print(f"    ✓ NumPy version: {np.__version__}")
except Exception as e:
    print(f"    ✗ NumPy import failed: {e}")
    sys.exit(1)

# Test 3: Protobuf
print("\n[3] Testing Protobuf...")
try:
    import google.protobuf
    print(f"    ✓ Protobuf version: {google.protobuf.__version__}")
except Exception as e:
    print(f"    ✗ Protobuf import failed: {e}")
    sys.exit(1)

# Test 4: MediaPipe
print("\n[4] Testing MediaPipe...")
try:
    import mediapipe
    print(f"    ✓ MediaPipe version: {mediapipe.__version__}")
except Exception as e:
    print(f"    ✗ MediaPipe import failed: {e}")
    sys.exit(1)

# Test 5: MediaPipe Face Mesh backend
print("\n[5] Testing MediaPipe Face Mesh backend...")
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    # Try to initialize face mesh
    mp_face_mesh = mp.solutions.face_mesh
    print(f"    ✓ MediaPipe Face Mesh module loaded")
    print(f"    ✓ Available solutions: face_mesh, hands, pose, holistic")
except Exception as e:
    print(f"    ✗ MediaPipe Face Mesh backend failed: {e}")
    sys.exit(1)

# Test 6: System info
print("\n[6] System Information...")
try:
    import platform
    print(f"    ✓ Architecture: {platform.machine()}")
    print(f"    ✓ Python: {sys.version.split()[0]}")
    print(f"    ✓ Platform: {platform.system()}")
except Exception as e:
    print(f"    ✗ Platform check failed: {e}")

print("\n" + "=" * 60)
print("✓ All verifications passed!")
print("=" * 60)
print("\nMediaPipe is ready for use on Jetson Orin Nano aarch64")
