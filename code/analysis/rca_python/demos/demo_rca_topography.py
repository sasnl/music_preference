#!/usr/bin/env python3
"""
Demo script showing how to create RCA topographic maps.

This script shows the enhanced RCA visualization with proper EEG topographic maps,
making it easy to interpret the spatial patterns of reliable components.
"""

import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Add RCA to path
rca_path = Path(__file__).parent.parent  # Go up to rca_python directory
sys.path.insert(0, str(rca_path))

from rca_utils import run_rca_on_music_data, plot_music_rca_topographies


def create_rca_topography_demo(subject_id='pilot_2'):
    """
    Create RCA analysis with topographic visualization.
    
    This is the main function to call for generating publication-quality
    RCA topographic plots for your music preference data.
    """
    print("🧠 Creating RCA Topographic Analysis")
    print("=" * 40)
    print(f"Analyzing subject: {subject_id}")
    print()
    
    # Step 1: Run RCA Analysis
    print("Step 1: Running RCA analysis...")
    results = run_rca_on_music_data(
        subject_id=subject_id,
        data_dir='data/ica_cleaned',
        n_components=3,  # Extract 3 components
        compare_conditions=True  # Compare preferred vs non-preferred
    )
    print("✅ RCA analysis complete!")
    print()
    
    # Step 2: Create Topographic Visualization
    print("Step 2: Creating topographic plots...")
    plt.ioff()  # Turn off interactive mode for cleaner output
    
    # Create the enhanced topographic plot
    fig = plot_music_rca_topographies(
        results,
        save_path=f'rca_final_analysis_{subject_id}.png'
    )
    plt.close(fig)  # Clean up
    
    print("✅ Topographic visualization created!")
    print()
    
    # Step 3: Provide Analysis Summary
    print("Step 3: Analysis Summary")
    print("-" * 25)
    
    rca = results['rca_model']
    
    print(f"📊 {subject_id.upper()} RESULTS:")
    print(f"  • {rca.n_components} reliable components extracted")
    print(f"  • {len(results['channel_names'])} EEG channels analyzed")
    print(f"  • {results['n_preferred_trials']} preferred music trials")
    print(f"  • {results['n_nonpreferred_trials']} non-preferred music trials")
    print()
    
    print("🧠 Component Details:")
    for i in range(rca.n_components):
        eigenval = rca.eigenvalues_[i]
        
        # Find channel with maximum activation
        max_activation = abs(rca.forward_models_[:, i]).max()
        max_channel_idx = abs(rca.forward_models_[:, i]).argmax()
        max_channel = results['channel_names'][max_channel_idx]
        
        print(f"  RC{i+1}: λ={eigenval:.4f} (reliability)")
        print(f"       Max activation at {max_channel}: {max_activation:.3f}")
        
        # Interpretation based on channel location
        interpretation = ""
        if max_channel in ['Fz', 'FC1', 'FC2', 'Cz']:
            interpretation = " → Frontal-central (attention/cognitive)"
        elif max_channel in ['Pz', 'P3', 'P4', 'Oz', 'O1', 'O2']:
            interpretation = " → Posterior (sensory/perceptual)"
        elif max_channel in ['T7', 'T8', 'TP9', 'TP10']:
            interpretation = " → Temporal (auditory processing)"
        elif max_channel in ['Fp1', 'Fp2']:
            interpretation = " → Frontal (executive/emotional)"
            
        print(f"       Location{interpretation}")
        print()
    
    print("📈 Interpretation Guide:")
    print("  • Higher eigenvalues (λ) = more reliable components")
    print("  • RC1 typically captures the dominant reliable response")
    print("  • Topographic maps show spatial distribution of neural activity")
    print("  • Red/warm colors = positive activation, Blue/cool = negative")
    print("  • Compare time courses between preferred/non-preferred conditions")
    print()
    
    print(f"📄 Output saved as: rca_final_analysis_{subject_id}.png")
    print("   This plot includes:")
    print("   • Topographic maps of spatial patterns")
    print("   • Channel weight profiles")
    print("   • Time course comparisons")
    print("   • Statistical difference plots")
    print("   • Eigenvalue spectrum")
    
    return results


def main():
    """Main demonstration function."""
    print("🎵 RCA TOPOGRAPHIC ANALYSIS DEMO")
    print("=" * 50)
    print("This demo creates publication-quality RCA topographic plots")
    print("for analyzing music preference EEG data.")
    print()
    
    try:
        # Run the analysis
        results = create_rca_topography_demo('pilot_2')
        
        print("\n" + "=" * 50)
        print("🎉 SUCCESS!")
        print("=" * 50)
        print("Your RCA topographic analysis is complete!")
        print()
        print("Key findings for pilot_2:")
        
        rca = results['rca_model']
        print(f"• Most reliable component: RC1 (λ={rca.eigenvalues_[0]:.4f})")
        
        if rca.eigenvalues_[0] > 0.01:
            print("• Strong reliability detected - good signal quality!")
        else:
            print("• Moderate reliability - consider more data or preprocessing")
            
        print("\nNext steps:")
        print("1. Examine the topographic plot for spatial patterns")
        print("2. Compare time courses between conditions")
        print("3. Look for preference-related differences")
        print("4. Repeat analysis for other subjects")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        print("Please check data availability and try again.")


if __name__ == "__main__":
    main()