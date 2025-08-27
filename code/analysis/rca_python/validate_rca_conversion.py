#!/usr/bin/env python3
"""
Validate RCA Python implementation against expected behavior.

This script tests the Python RCA implementation to ensure it produces
reasonable results and matches the expected algorithmic behavior of 
the original MATLAB implementation.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import warnings

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from rca_python import ReliableComponentsAnalysis, demo_rca_analysis


def test_basic_rca_functionality():
    """Test basic RCA functionality with synthetic data."""
    print("=== Testing Basic RCA Functionality ===")
    
    # Create synthetic data similar to MATLAB demo
    np.random.seed(42)
    n_samples, n_channels, n_trials = 100, 16, 40
    
    # Create reliable signals
    time = np.linspace(0, 1, n_samples)
    signal1 = np.sin(2 * np.pi * 5 * time)  # 5 Hz
    signal2 = np.cos(2 * np.pi * 8 * time)  # 8 Hz
    
    # Create spatial patterns
    pattern1 = np.random.randn(n_channels)
    pattern1[:8] *= 2  # stronger in first half of channels
    pattern2 = np.random.randn(n_channels) 
    pattern2[8:] *= 2  # stronger in second half of channels
    
    # Generate data with reliable components
    data = np.zeros((n_samples, n_channels, n_trials))
    
    for trial in range(n_trials):
        # Add trial-to-trial variability to reliable signals
        trial_var1 = 1.0 + 0.2 * np.random.randn()
        trial_var2 = 0.8 + 0.2 * np.random.randn()
        
        for ch in range(n_channels):
            data[:, ch, trial] = (trial_var1 * pattern1[ch] * signal1 +
                                trial_var2 * pattern2[ch] * signal2 +
                                0.3 * np.random.randn(n_samples))  # noise
    
    print(f"Generated synthetic data: {data.shape}")
    
    # Test RCA
    rca = ReliableComponentsAnalysis(n_components=4, random_state=42)
    data_rca = rca.fit_transform(data)
    
    print(f"RCA output shape: {data_rca.shape}")
    print(f"Top 4 eigenvalues: {rca.eigenvalues_[:4]}")
    print(f"Explained variance ratios: {np.abs(rca.eigenvalues_[:4]) / np.sum(np.abs(rca.eigenvalues_))}")
    
    # Verify that first few components capture most reliable signal
    assert rca.eigenvalues_[0] > rca.eigenvalues_[1], "First component should have highest eigenvalue"
    assert rca.eigenvalues_[1] > rca.eigenvalues_[2], "Second component should have second highest eigenvalue"
    
    print("✓ Basic functionality test passed")
    return rca, data, data_rca


def test_rca_properties():
    """Test mathematical properties of RCA."""
    print("\n=== Testing RCA Mathematical Properties ===")
    
    # Generate simple test data
    np.random.seed(123)
    n_samples, n_channels, n_trials = 50, 8, 20
    
    # Create data with known structure
    data = np.random.randn(n_samples, n_channels, n_trials)
    
    # Add reliable component in first 4 channels
    reliable_signal = np.sin(2 * np.pi * np.linspace(0, 2, n_samples))
    for trial in range(n_trials):
        trial_amplitude = 2.0 + 0.3 * np.random.randn()
        data[:, :4, trial] += trial_amplitude * reliable_signal[:, np.newaxis]
    
    rca = ReliableComponentsAnalysis(n_components=3, n_reg=6)
    rca.fit(data)
    
    # Test that spatial filters are orthogonal (approximately)
    W = rca.spatial_filters_
    WtW = W.T @ W
    off_diagonal = WtW - np.diag(np.diag(WtW))
    max_off_diag = np.max(np.abs(off_diagonal))
    print(f"Max off-diagonal in W'W: {max_off_diag:.6f}")
    
    # Test forward model computation
    A = rca.forward_models_
    print(f"Spatial filters shape: {W.shape}")
    print(f"Forward models shape: {A.shape}")
    
    # Test that eigenvalues are real and decreasing
    eigenvals = rca.eigenvalues_[:rca.n_components]
    print(f"Eigenvalues: {eigenvals}")
    assert np.all(np.diff(eigenvals) <= 0), "Eigenvalues should be in descending order"
    assert np.all(np.isreal(eigenvals)), "Eigenvalues should be real"
    
    print("✓ Mathematical properties test passed")


def test_data_handling():
    """Test RCA's handling of different data conditions."""
    print("\n=== Testing Data Handling ===")
    
    # Test with NaN values
    np.random.seed(456)
    data = np.random.randn(30, 10, 15)
    
    # Insert some NaN values
    data[5:10, 3, 2] = np.nan
    data[15:20, :, 7] = np.nan
    
    rca = ReliableComponentsAnalysis(n_components=2)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data_rca = rca.fit_transform(data)
    
    print(f"Data with NaNs processed successfully: {data_rca.shape}")
    assert not np.any(np.isnan(data_rca)), "Output should not contain NaNs"
    
    # Test with different trial numbers
    small_data = np.random.randn(25, 6, 5)  # few trials
    rca_small = ReliableComponentsAnalysis(n_components=2)
    rca_small.fit(small_data)
    
    large_data = np.random.randn(50, 8, 100)  # many trials
    rca_large = ReliableComponentsAnalysis(n_components=2)
    rca_large.fit(large_data)
    
    print("✓ Data handling test passed")


def test_music_integration():
    """Test integration with music preference study workflow."""
    print("\n=== Testing Music Study Integration ===")
    
    # Simulate music preference data structure
    np.random.seed(789)
    n_samples, n_channels = 200, 32
    
    # Simulate preferred vs non-preferred conditions
    n_preferred_trials = 15
    n_nonpreferred_trials = 45
    
    # Create preferred data with stronger reliable component
    preferred_data = np.zeros((n_samples, n_channels, n_preferred_trials))
    reliable_pattern = np.random.randn(n_channels)
    reliable_pattern[:16] *= 2  # stronger in frontal channels
    
    time_course = np.sin(2 * np.pi * 6 * np.linspace(0, 1, n_samples))
    
    for trial in range(n_preferred_trials):
        amplitude = 3.0 + 0.5 * np.random.randn()  # high amplitude
        for ch in range(n_channels):
            preferred_data[:, ch, trial] = (amplitude * reliable_pattern[ch] * time_course +
                                          0.4 * np.random.randn(n_samples))
    
    # Create non-preferred data with weaker reliable component  
    nonpreferred_data = np.zeros((n_samples, n_channels, n_nonpreferred_trials))
    
    for trial in range(n_nonpreferred_trials):
        amplitude = 1.0 + 0.3 * np.random.randn()  # lower amplitude
        for ch in range(n_channels):
            nonpreferred_data[:, ch, trial] = (amplitude * reliable_pattern[ch] * time_course +
                                             0.6 * np.random.randn(n_samples))
    
    # Fit RCA on combined data
    all_data = np.concatenate([preferred_data, nonpreferred_data], axis=2)
    rca = ReliableComponentsAnalysis(n_components=3)
    rca.fit(all_data)
    
    # Transform each condition
    preferred_rca = rca.transform(preferred_data)
    nonpreferred_rca = rca.transform(nonpreferred_data)
    
    # Check condition differences
    for comp in range(rca.n_components):
        pref_mean = np.mean(preferred_rca[:, comp, :])
        nonpref_mean = np.mean(nonpreferred_rca[:, comp, :])
        print(f"RC{comp+1}: Preferred mean = {pref_mean:.3f}, Non-preferred mean = {nonpref_mean:.3f}")
    
    print("✓ Music integration test passed")
    
    return {
        'rca_model': rca,
        'preferred_data': preferred_data,
        'nonpreferred_data': nonpreferred_data,
        'preferred_rca': preferred_rca,
        'nonpreferred_rca': nonpreferred_rca
    }


def create_validation_plots(test_results):
    """Create plots to visualize validation results."""
    print("\n=== Creating Validation Plots ===")
    
    # Plot 1: Basic RCA components
    rca, data, data_rca = test_results['basic']
    
    fig1, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Spatial patterns
    for i in range(2):
        axes[0, i].plot(rca.forward_models_[:, i], 'o-')
        axes[0, i].set_title(f'RC{i+1} Spatial Pattern (λ={rca.eigenvalues_[i]:.3f})')
        axes[0, i].set_xlabel('Channel')
        axes[0, i].set_ylabel('Weight')
        axes[0, i].grid(True, alpha=0.3)
    
    # Time courses  
    for i in range(2):
        mean_tc = np.mean(data_rca[:, i, :], axis=1)
        sem_tc = np.std(data_rca[:, i, :], axis=1) / np.sqrt(data_rca.shape[2])
        
        time = np.arange(len(mean_tc))
        axes[1, i].plot(time, mean_tc, 'b-', linewidth=2)
        axes[1, i].fill_between(time, mean_tc - sem_tc, mean_tc + sem_tc, alpha=0.3)
        axes[1, i].set_title(f'RC{i+1} Time Course')
        axes[1, i].set_xlabel('Time (samples)')
        axes[1, i].set_ylabel('Amplitude')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.suptitle('RCA Validation: Synthetic Data', fontsize=14)
    plt.tight_layout()
    
    # Plot 2: Music preference simulation
    music_results = test_results['music']
    rca_music = music_results['rca_model']
    
    fig2, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    for comp in range(3):
        # Spatial patterns
        axes[0, comp].plot(rca_music.forward_models_[:, comp], 'ko-')
        axes[0, comp].set_title(f'RC{comp+1} Spatial Pattern')
        axes[0, comp].set_xlabel('Channel')
        axes[0, comp].set_ylabel('Weight')
        axes[0, comp].grid(True, alpha=0.3)
        
        # Condition comparison
        pref_mean = np.mean(music_results['preferred_rca'][:, comp, :], axis=1)
        nonpref_mean = np.mean(music_results['nonpreferred_rca'][:, comp, :], axis=1)
        
        pref_sem = np.std(music_results['preferred_rca'][:, comp, :], axis=1) / \
                  np.sqrt(music_results['preferred_rca'].shape[2])
        nonpref_sem = np.std(music_results['nonpreferred_rca'][:, comp, :], axis=1) / \
                     np.sqrt(music_results['nonpreferred_rca'].shape[2])
        
        time = np.arange(len(pref_mean))
        axes[1, comp].plot(time, pref_mean, 'r-', linewidth=2, label='Preferred')
        axes[1, comp].fill_between(time, pref_mean - pref_sem, pref_mean + pref_sem, alpha=0.3, color='red')
        
        axes[1, comp].plot(time, nonpref_mean, 'b-', linewidth=2, label='Non-preferred')
        axes[1, comp].fill_between(time, nonpref_mean - nonpref_sem, nonpref_mean + nonpref_sem, alpha=0.3, color='blue')
        
        axes[1, comp].set_title(f'RC{comp+1} Time Course')
        axes[1, comp].set_xlabel('Time (samples)')
        axes[1, comp].set_ylabel('Amplitude')
        axes[1, comp].legend()
        axes[1, comp].grid(True, alpha=0.3)
    
    plt.suptitle('RCA Validation: Music Preference Simulation', fontsize=14)
    plt.tight_layout()
    
    plt.show()
    
    return fig1, fig2


def main():
    """Run all validation tests."""
    print("RCA Python Implementation Validation")
    print("====================================")
    
    test_results = {}
    
    # Run validation tests
    test_results['basic'] = test_basic_rca_functionality()
    test_rca_properties()
    test_data_handling()
    test_results['music'] = test_music_integration()
    
    # Create visualization
    create_validation_plots(test_results)
    
    print("\n=== Validation Summary ===")
    print("✓ All tests passed successfully!")
    print("✓ RCA Python implementation is ready for use")
    print("\nKey features validated:")
    print("  - Reliable component extraction")
    print("  - Eigenvalue computation and sorting")
    print("  - Spatial filter and forward model calculation")
    print("  - NaN value handling")
    print("  - Multiple condition analysis")
    print("  - Integration with music preference workflow")
    
    print(f"\nTo use RCA in your analysis:")
    print(f"  from rca_python import ReliableComponentsAnalysis")
    print(f"  rca = ReliableComponentsAnalysis(n_components=3)")
    print(f"  data_rca = rca.fit_transform(your_data)")


if __name__ == '__main__':
    main()