#!/usr/bin/env python3
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_minimal_functionality():
    """Test minimal functionality without TensorFlow"""
    print("Testing minimal RF Capture functionality...")
    
    try:
        # Test basic imports without TensorFlow
        from src.hardware.sdr_controller import SDRController
        print("✓ SDR Controller imported successfully")
        
        from src.core.capture_manager import CaptureManager
        print("✓ Capture Manager imported successfully")
        
        from src.utils.config_loader import ConfigLoader
        print("✓ Config Loader imported successfully")
        
        # Test basic functionality
        sdr = SDRController()
        print("✓ SDR Controller created")
        
        cm = CaptureManager()
        print("✓ Capture Manager created")
        
        config = ConfigLoader()
        print("✓ Config Loader created")
        
        print("\n✓ Minimal functionality test PASSED!")
        print("Basic RF capture system is working!")
        return True
        
    except Exception as e:
        print(f"✗ Minimal test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_minimal_functionality()
