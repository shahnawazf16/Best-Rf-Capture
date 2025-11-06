#!/usr/bin/env python3
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from src.cli.main import main
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternative import...")
    # Alternative import path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    from cli.main import main

if __name__ == "__main__":
    main()