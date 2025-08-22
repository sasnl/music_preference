#!/usr/bin/env python3
"""
Test script for TRF analysis with a single participant.

This script tests the TRF analysis pipeline on one participant to ensure 
everything works correctly before running the full analysis.
"""

import sys
sys.path.append('code/analysis')

from trf_music_preference_analysis import TRFMusicPreferenceAnalysis
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_single_participant():
    """Test TRF analysis on a single participant."""
    
    print("="*60)
    print("TESTING TRF ANALYSIS - SINGLE PARTICIPANT")
    print("="*60)
    
    # Initialize analysis
    trf_analysis = TRFMusicPreferenceAnalysis()
    
    # Test with pilot_2 (has complete data including song 2-1)
    test_participant = 'pilot_2'
    
    print(f"\nTesting with participant: {test_participant}")
    
    try:
        # Run analysis for single participant
        results = trf_analysis.analyze_participant(test_participant)
        
        if results is not None:
            print(f"\n✓ Analysis completed successfully for {test_participant}")
            
            # Save results
            trf_analysis.save_results(test_participant, results)
            print(f"✓ Results saved successfully")
            
            # Create visualizations
            trf_analysis.visualize_results(test_participant, results)
            print(f"✓ Visualizations created successfully")
            
            # Print summary
            print(f"\n{'-'*40}")
            print("ANALYSIS SUMMARY")
            print(f"{'-'*40}")
            print(f"Participant: {results['participant']}")
            print(f"Sampling rate: {results['sfreq']} Hz")
            print(f"Number of channels: {results['n_channels']}")
            print(f"Optimal lambda: {results['lambda_optimization']['best_lambda']:.2e}")
            
            for condition in ['preferred', 'nonpreferred']:
                if condition in results['performance']:
                    perf = results['performance'][condition]
                    print(f"{condition.capitalize()} CV score: {perf['cv_score']:.4f} "
                          f"(n_samples: {perf['n_samples']})")
            
            if 'statistical_comparison' in results:
                comp = results['statistical_comparison']
                print(f"Mean difference: {comp['mean_difference']:.4f}")
                print(f"t-test p-value: {comp['ttest']['p_value']:.4f}")
                print(f"Effect size: {comp['effect_size']:.4f}")
            
            print(f"\n✅ TEST PASSED - Single participant analysis working correctly!")
            return True
            
        else:
            print(f"✗ Analysis failed for {test_participant}")
            return False
            
    except Exception as e:
        print(f"✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_participant()
    
    if success:
        print(f"\n🎉 TRF analysis test completed successfully!")
        print(f"You can now run the full analysis with:")
        print(f"python code/analysis/trf_music_preference_analysis.py")
    else:
        print(f"\n❌ TRF analysis test failed. Please check the issues above.")
        sys.exit(1)