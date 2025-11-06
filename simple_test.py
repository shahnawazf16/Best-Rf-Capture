#!/usr/bin/env python3
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """Test basic functionality without hardware"""
    print("Testing RF Capture basic functionality...")
    
    try:
        # Try importing from src first
        try:
            from src.ml.signal_classifier import SignalClassifier
            print("✓ Imported from src.ml.signal_classifier")
        except ImportError:
            # Fallback to direct import
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
            from ml.signal_classifier import SignalClassifier
            print("✓ Imported from ml.signal_classifier")
        
        classifier = SignalClassifier()
        print("✓ Signal classifier created")
        
        # Test dummy data creation
        X_train, X_val, y_train, y_val = classifier.create_dummy_data()
        print(f"✓ Dummy data created: {X_train.shape}, {y_train.shape}")
        
        # Test model building
        model = classifier.build_cnn_model((1024,), 3)
        print("✓ CNN model built successfully")
        
        print("\n✓ Basic functionality test PASSED!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_basic_functionality()