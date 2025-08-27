#!/usr/bin/env python3
"""Test script to validate main.py imports correctly"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    print("Testing imports...")
    import src.main
    print("✓ Successfully imported src.main")
    
    app = src.main.app
    print("✓ Successfully accessed FastAPI app")
    
    print("✓ All imports successful - the service should start correctly")
    
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)