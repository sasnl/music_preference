#!/usr/bin/env python3
"""
Test script for the new real-time ICA component selection interface.

This script creates mock ICA data to test the interactive selection functionality
without needing actual EEG data.
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from code.analysis.eeg_ica_artifact_removal import interactive_component_selection_realtime
except ImportError:
    # Alternative import method if above fails
    sys.path.insert(0, os.path.join(project_root, 'code', 'analysis'))
    from eeg_ica_artifact_removal import interactive_component_selection_realtime

def create_mock_ica_data(n_channels=32, n_components=20, duration=60, sfreq=1000):
    """
    Create mock ICA and EEG data for testing the interface.
    
    Parameters:
    -----------
    n_channels : int
        Number of EEG channels
    n_components : int
        Number of ICA components
    duration : float
        Duration in seconds
    sfreq : float
        Sampling frequency in Hz
    
    Returns:
    --------
    ica : mne.preprocessing.ICA
        Mock ICA object
    raw : mne.Raw
        Mock raw EEG data
    """
    print("Creating mock EEG and ICA data for testing...")
    
    # Create mock EEG data
    n_samples = int(duration * sfreq)
    
    # Create some realistic-looking EEG patterns
    times = np.arange(n_samples) / sfreq
    
    # Generate different types of mock components
    data = np.zeros((n_channels, n_samples))
    
    # Add some baseline EEG-like activity
    for ch in range(n_channels):
        # Alpha-like rhythm (8-12 Hz)
        alpha = np.sin(2 * np.pi * 10 * times + np.random.random() * 2 * np.pi)
        # Beta activity (13-30 Hz)
        beta = 0.5 * np.sin(2 * np.pi * 20 * times + np.random.random() * 2 * np.pi)
        # Add some noise
        noise = np.random.normal(0, 0.1, n_samples)
        
        data[ch] = alpha + beta + noise
    
    # Create standard 10-20 channel names for realistic topographies
    ch_names = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'T7', 'C3', 'Cz', 'C4', 'T8',
        'P7', 'P3', 'Pz', 'P4', 'P8',
        'O1', 'O2'
    ]
    
    # Add extra channels if needed
    while len(ch_names) < n_channels:
        ch_names.append(f'EEG{len(ch_names)+1:03d}')
    
    ch_names = ch_names[:n_channels]
    
    # Create MNE info structure
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    
    # Set standard montage for realistic topographies
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage, match_case=False, on_missing='ignore')
    
    # Create Raw object
    raw = mne.io.RawArray(data * 1e-6, info)  # Convert to Volts
    
    # Create and fit mock ICA
    ica = ICA(n_components=n_components, method='fastica', random_state=42)
    
    # Create mock ICA components with different patterns
    # This simulates what ICA.fit() would produce
    ica.info_ = info.copy()
    ica.ch_names = ch_names
    ica.n_components_ = n_components
    
    # Create mock mixing matrix (components x channels)
    np.random.seed(42)  # For reproducible results
    mixing_matrix = np.random.randn(n_components, n_channels)
    
    # Make some components look like typical artifacts:
    # Component 0: Eye blink (strong at Fp1, Fp2)
    if 'Fp1' in ch_names and 'Fp2' in ch_names:
        fp1_idx = ch_names.index('Fp1')
        fp2_idx = ch_names.index('Fp2')
        mixing_matrix[0, :] = np.random.randn(n_channels) * 0.1
        mixing_matrix[0, fp1_idx] = 1.5
        mixing_matrix[0, fp2_idx] = 1.2
    
    # Component 1: Lateral eye movement (Fp1 vs Fp2)
    if 'Fp1' in ch_names and 'Fp2' in ch_names:
        mixing_matrix[1, :] = np.random.randn(n_channels) * 0.1
        mixing_matrix[1, fp1_idx] = 1.0
        mixing_matrix[1, fp2_idx] = -1.0
    
    # Component 2: Muscle artifact (temporal electrodes)
    if 'T7' in ch_names and 'T8' in ch_names:
        t7_idx = ch_names.index('T7')
        t8_idx = ch_names.index('T8')
        mixing_matrix[2, :] = np.random.randn(n_channels) * 0.1
        mixing_matrix[2, t7_idx] = 1.0
        mixing_matrix[2, t8_idx] = 0.8
    
    # Store the mixing matrix in the ICA object
    ica.mixing_ = mixing_matrix.T  # MNE expects channels x components
    
    # Create mock explained variance
    ica.pca_explained_variance_ = np.linspace(0.8, 0.1, n_components)
    
    print(f"Created mock data:")
    print(f"  - {n_channels} channels: {ch_names[:5]}...")
    print(f"  - {n_components} ICA components")
    print(f"  - {duration}s duration at {sfreq}Hz")
    print(f"  - Simulated artifacts in components 0-2")
    
    return ica, raw

def test_realtime_selection():
    """Test the real-time component selection interface."""
    print("\n=== Testing Real-time ICA Component Selection ===")
    
    # Create mock data
    ica, raw = create_mock_ica_data(n_channels=19, n_components=15, duration=30)
    
    print("\nStarting interactive component selection test...")
    print("Look for these simulated artifacts:")
    print("  - Component 0: Eye blinks (strong at Fp1/Fp2)")
    print("  - Component 1: Horizontal eye movements (Fp1 vs Fp2)")
    print("  - Component 2: Muscle artifacts (temporal)")
    
    try:
        # Test the real-time selection
        selected_components = interactive_component_selection_realtime(ica, raw)
        
        print(f"\nTest completed!")
        print(f"You selected components: {selected_components}")
        
        if selected_components:
            print(f"Selected {len(selected_components)} components for removal")
            
            # Check if user found the simulated artifacts
            artifacts_found = []
            if 0 in selected_components:
                artifacts_found.append("Eye blinks (IC00)")
            if 1 in selected_components:
                artifacts_found.append("Horizontal eye movements (IC01)")
            if 2 in selected_components:
                artifacts_found.append("Muscle artifacts (IC02)")
            
            if artifacts_found:
                print(f"Great! You identified these simulated artifacts: {artifacts_found}")
            else:
                print("No simulated artifacts were selected. That's okay - this is just a test!")
        else:
            print("No components selected for removal")
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("ICA Real-time Selection Interface Test")
    print("=" * 50)
    
    # Check if we're in an environment that supports matplotlib interaction
    import matplotlib
    backend = matplotlib.get_backend()
    print(f"Matplotlib backend: {backend}")
    
    if 'inline' in backend.lower():
        print("Warning: Inline backend detected. Interactive features may not work.")
        print("Try running this in a regular Python session, not Jupyter notebook.")
    
    test_realtime_selection()
    
    print("\nTest completed!")