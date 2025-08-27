#!/usr/bin/env python3
"""
Test script for RCA topographic plotting.

This script demonstrates the new topographic plotting functionality for RCA results,
showing proper EEG topographic maps of the reliable components.
"""

import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Add RCA to path
rca_path = Path(__file__).parent.parent  # Go up to rca_python directory
sys.path.insert(0, str(rca_path))

from rca_utils import run_rca_on_music_data, plot_music_rca_topographies


def test_topographic_plotting(subject_id='pilot_2'):
    """Test the new topographic plotting functionality."""
    print("🧠 Testing RCA Topographic Plotting")
    print("=" * 40)
    
    try:
        # Run RCA analysis
        print(f"Running RCA analysis for {subject_id}...")
        results = run_rca_on_music_data(
            subject_id=subject_id,
            data_dir='data/ica_cleaned',
            n_components=3,
            compare_conditions=True
        )
        print("✅ RCA analysis completed!")
        
        # Create topographic plots
        print("Creating topographic visualizations...")
        plt.ioff()  # Turn off interactive plotting
        
        fig = plot_music_rca_topographies(
            results, 
            save_path=f'rca_topography_{subject_id}.png'
        )
        
        plt.close(fig)  # Close to free memory
        print("✅ Topographic plots created and saved!")
        
        # Print interpretation guide
        print("\n📖 INTERPRETATION GUIDE:")
        print("-" * 25)
        print("🎯 Topographic Maps:")
        print("  • Red areas = positive activation")
        print("  • Blue areas = negative activation") 
        print("  • Contour lines show activation gradients")
        print("  • Spatial patterns reveal source locations")
        
        print("\n🧠 Typical RCA Patterns:")
        print("  • RC1: Often shows broad, frontal-central activation")
        print("  • RC2: May show more focal, posterior patterns")
        print("  • RC3+: Usually smaller, more specific patterns")
        
        print("\n🎵 Music Preference Analysis:")
        print("  • Compare preferred vs non-preferred time courses")
        print("  • Look for systematic differences between conditions")
        print("  • Green difference plots show preference effects")
        
        rca_model = results['rca_model']
        print(f"\n📊 RESULTS SUMMARY for {subject_id}:")
        print("-" * 30)
        for i in range(rca_model.n_components):
            eigenval = rca_model.eigenvalues_[i]
            max_weight = abs(rca_model.forward_models_[:, i]).max()
            max_channel_idx = abs(rca_model.forward_models_[:, i]).argmax()
            max_channel = results['channel_names'][max_channel_idx]
            
            print(f"RC{i+1}: λ={eigenval:.4f}, max at {max_channel} ({max_weight:.3f})")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def main():
    """Main function to test topographic plotting."""
    print("🚀 RCA Topographic Plotting Test")
    print()
    
    # Test with pilot_2 data
    results = test_topographic_plotting('pilot_2')
    
    if results is not None:
        print("\n🎉 SUCCESS! Topographic plotting is working!")
        print(f"   • Output saved as: rca_topography_pilot_2.png")
        print(f"   • {results['rca_model'].n_components} components analyzed")
        print(f"   • {len(results['channel_names'])} channels processed")
        print(f"   • {results['n_preferred_trials']} preferred + {results['n_nonpreferred_trials']} non-preferred trials")
    else:
        print("\n❌ Failed to create topographic plots.")
        print("   Check data availability and dependencies.")


if __name__ == "__main__":
    main()