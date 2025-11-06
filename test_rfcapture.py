#!/usr/bin/env python3
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from cli.main import main
    print("✓ CLI module imported successfully")
    
    # Test basic functionality
    from hardware.sdr_controller import SDRController
    print("✓ SDR Controller imported successfully")
    
    from ml.signal_classifier import SignalClassifier
    print("✓ Signal Classifier imported successfully")
    
    from core.capture_manager import CaptureManager
    print("✓ Capture Manager imported successfully")
    
    print("\n✓ All modules imported successfully!")
    print("\nYou can now run: python -m src.cli.main --help")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nChecking project structure...")
    
    # Check project structure
    expected_dirs = ['src', 'src/cli', 'src/core', 'src/hardware', 'src/ml', 'src/utils']
    for dir_path in expected_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path} exists")
        else:
            print(f"✗ {dir_path} missing")
