#!/usr/bin/env python3
"""
Test script to verify RCA import is working correctly.
"""

import sys
import os
from pathlib import Path

# Add the rca_python directory to Python path
rca_path = Path(__file__).parent.parent  # Go up to rca_python directory
sys.path.insert(0, str(rca_path))

print(f"Added to path: {rca_path}")
print("Testing RCA imports...")

try:
    # Import directly from the files
    from rca import ReliableComponentsAnalysis
    print("✓ ReliableComponentsAnalysis imported successfully")
    
    from rca_utils import run_rca_on_music_data
    print("✓ run_rca_on_music_data imported successfully")
    
    # Test creating an RCA instance
    rca = ReliableComponentsAnalysis(n_components=3)
    print("✓ RCA instance created successfully")
    
    print("\n🎉 RCA is ready to use!")
    print("\nTo use RCA in your scripts, add these lines:")
    print("```python")
    print("import sys")
    print("from pathlib import Path")
    print("rca_path = Path('code/analysis/rca_python')")
    print("sys.path.insert(0, str(rca_path))")
    print("from rca import ReliableComponentsAnalysis")
    print("from rca_utils import run_rca_on_music_data")
    print("```")
    
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("Check that all required dependencies are installed.")