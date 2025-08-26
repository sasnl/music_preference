#!/usr/bin/env python3
"""Test script for updated group visualization."""

import sys
import pandas as pd
from pathlib import Path

# Add the current directory to path to import the main analysis
sys.path.insert(0, str(Path(__file__).parent))

from trf_music_preference_analysis import TRFMusicPreferenceAnalysis

def test_group_visualization():
    """Test the updated group visualization."""
    # Initialize analysis
    trf_analysis = TRFMusicPreferenceAnalysis()
    
    # Load existing group summary
    group_csv = trf_analysis.output_dir / "group_trf_summary.csv"
    
    if group_csv.exists():
        group_df = pd.read_csv(group_csv)
        print(f"Loaded group data with {len(group_df)} participants")
        print(group_df.columns.tolist())
        
        # Create updated group visualization
        try:
            trf_analysis._create_group_visualization(group_df)
            print("✓ Successfully created updated group visualization!")
        except Exception as e:
            print(f"✗ Error creating group visualization: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Group summary file not found: {group_csv}")

if __name__ == "__main__":
    test_group_visualization()