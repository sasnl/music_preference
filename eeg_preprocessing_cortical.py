#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG Preprocessing Script for Cortical Data

Preprocesses cortical EEG data following these steps:
1. Load EEG file (BrainVision format)
2. Pick only active electrode channels (32-channel)
3. Add back reference Cz channel
4. Bandpass filter: 0.5-30 Hz, zero-phase bidirectional
5. Re-reference: mean of TP9+TP10
6. Downsample: 128 Hz

Usage: 
  python eeg_preprocessing_cortical.py <subject_id> [options]

Examples:
  # Basic usage with defaults
  python eeg_preprocessing_cortical.py pilot_1
  
  # Custom filter settings
  python eeg_preprocessing_cortical.py pilot_1 --l_freq 1.0 --h_freq 40.0
  
  # Custom sampling rate and reference
  python eeg_preprocessing_cortical.py pilot_1 --target_sfreq 256 --ref_channels Cz
  
  # Skip plots and use custom paths
  python eeg_preprocessing_cortical.py pilot_1 --no_plots --data_root /path/to/data --output_root /path/to/output
"""

import numpy as np
import mne
import matplotlib.pyplot as plt
import os
import sys
import argparse

def load_eeg_file(subject_id, data_root='./data/'):
    """
    Step 1: Load EEG file (BrainVision format).
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier (e.g., 'pilot_1')
    data_root : str
        Root directory containing subject data
    
    Returns:
    --------
    raw : mne.Raw
        Loaded raw EEG data
    """
    print(f"\n=== Step 1: Loading EEG file for {subject_id} ===")
    
    subject_dir = os.path.join(data_root, subject_id)
    eeg_file = os.path.join(subject_dir, f"{subject_id}.vhdr")
    
    if not os.path.exists(eeg_file):
        raise FileNotFoundError(f"EEG file not found: {eeg_file}")
    
    print(f"Loading EEG file: {eeg_file}")
    raw = mne.io.read_raw_brainvision(eeg_file, preload=True, verbose=False)
    
    print(f"Original sampling rate: {raw.info['sfreq']} Hz")
    print(f"Original number of channels: {len(raw.ch_names)}")
    print(f"Duration: {raw.times[-1]:.1f} seconds")
    
    return raw

def pick_cortical_channels(raw):
    """
    Step 2: Pick only active electrode channels (32-channel cortical electrodes).
    
    Parameters:
    -----------
    raw : mne.Raw
        Raw EEG data
    
    Returns:
    --------
    raw : mne.Raw
        EEG data with only cortical channels
    cortical_channels : list
        List of selected cortical channel names
    """
    print(f"\n=== Step 2: Picking cortical channels ===")
    
    # Exclude ABR channels (Plus_R, Minus_R, Plus_L, Minus_L) and Audio channel
    cortical_channels = [ch for ch in raw.ch_names 
                        if ch not in ['Plus_R', 'Minus_R', 'Plus_L', 'Minus_L', 'Audio']]
    
    print(f"Picking {len(cortical_channels)} cortical channels:")
    print(f"Channels: {cortical_channels}")
    
    # Pick cortical channels (this excludes Cz reference for now)
    raw.pick_channels(cortical_channels)
    
    return raw, cortical_channels

def add_cz_reference(raw):
    """
    Step 3: Add back reference Cz channel.
    
    Parameters:
    -----------
    raw : mne.Raw
        EEG data without Cz
    
    Returns:
    --------
    raw : mne.Raw
        EEG data with Cz channel added back
    """
    print(f"\n=== Step 3: Adding back Cz reference channel ===")
    
    # For BrainVision data, we need to add Cz back as a channel
    # Since all channels were referenced to Cz during recording,
    # we can create Cz by setting it to zero (or estimate from average)
    
    # Create Cz channel as zeros (since all other channels are already referenced to it)
    cz_data = np.zeros((1, raw.n_times))
    
    # Add Cz channel info
    cz_info = mne.create_info(['Cz'], raw.info['sfreq'], ch_types='eeg')
    cz_raw = mne.io.RawArray(cz_data, cz_info)
    
    # Add Cz to the raw data
    raw.add_channels([cz_raw], force_update_info=True)
    
    # Set channel positions (standard 10-20 montage)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, on_missing='ignore')
    
    print(f"Channels after adding Cz: {len(raw.ch_names)}")
    
    return raw

def apply_bandpass_filter(raw, l_freq=0.5, h_freq=30.0):
    """
    Step 4: Apply bandpass filter (0.5-30 Hz, zero-phase bidirectional).
    
    Parameters:
    -----------
    raw : mne.Raw
        EEG data to filter
    l_freq : float
        Low cutoff frequency (default: 0.5 Hz)
    h_freq : float
        High cutoff frequency (default: 30.0 Hz)
    
    Returns:
    --------
    raw : mne.Raw
        Filtered EEG data
    """
    print(f"\n=== Step 4: Applying bandpass filter: {l_freq}-{h_freq} Hz (zero-phase) ===")
    
    raw.filter(l_freq=l_freq, h_freq=h_freq, method='fir', phase='zero-double', 
               fir_design='firwin', verbose=False)
    
    print(f"Filter applied: {l_freq}-{h_freq} Hz")
    
    return raw

def rereference_to_channels(raw, ref_channels=['TP9', 'TP10']):
    """
    Step 5: Re-reference to specified channels.
    
    Parameters:
    -----------
    raw : mne.Raw
        EEG data to re-reference
    ref_channels : list
        List of reference channels (default: ['TP9', 'TP10'])
    
    Returns:
    --------
    raw : mne.Raw
        Re-referenced EEG data
    """
    print(f"\n=== Step 5: Re-referencing to {ref_channels} ===")
    
    # Check if all reference channels are available
    missing_channels = [ch for ch in ref_channels if ch not in raw.ch_names]
    
    if not missing_channels:
        # Use MNE's set_eeg_reference to re-reference to the specified channels
        raw.set_eeg_reference(ref_channels=ref_channels, projection=False)
        print(f"Successfully re-referenced to mean of {ref_channels}")
    else:
        print(f"Warning: Reference channels {missing_channels} not found. Using average reference instead.")
        raw.set_eeg_reference('average', projection=False)
    
    return raw

def downsample_data(raw, target_sfreq=128.0):
    """
    Step 6: Downsample to target frequency.
    
    Parameters:
    -----------
    raw : mne.Raw
        EEG data to downsample
    target_sfreq : float
        Target sampling frequency (default: 128.0 Hz)
    
    Returns:
    --------
    raw : mne.Raw
        Downsampled EEG data
    original_sfreq : float
        Original sampling frequency
    """
    print(f"\n=== Step 6: Downsampling to {target_sfreq} Hz ===")
    
    original_sfreq = raw.info['sfreq']
    
    print(f"Downsampling from {original_sfreq} Hz to {target_sfreq} Hz...")
    raw.resample(sfreq=target_sfreq, verbose=False)
    
    print(f"Final sampling rate: {raw.info['sfreq']} Hz")
    print(f"Final number of samples: {raw.n_times}")
    print(f"Final duration: {raw.times[-1]:.1f} seconds")
    
    return raw, original_sfreq

def save_preprocessed_data(raw, subject_id, output_root='./output/'):
    """
    Save preprocessed data to file.
    
    Parameters:
    -----------
    raw : mne.Raw
        Preprocessed EEG data
    subject_id : str
        Subject identifier
    output_root : str
        Output directory
    
    Returns:
    --------
    output_file : str
        Path to saved file
    """
    print(f"\n=== Saving preprocessed data ===")
    
    os.makedirs(output_root, exist_ok=True)
    output_file = os.path.join(output_root, f"{subject_id}_cortical_preprocessed.fif")
    raw.save(output_file, overwrite=True, verbose=False)
    print(f"Preprocessed data saved to: {output_file}")
    
    return output_file

def generate_summary_plots(raw, subject_id, cortical_channels, original_sfreq, target_sfreq, output_root='./output/'):
    """
    Generate comprehensive summary plots.
    
    Parameters:
    -----------
    raw : mne.Raw
        Preprocessed EEG data
    subject_id : str
        Subject identifier
    cortical_channels : list
        Original cortical channel names
    original_sfreq : float
        Original sampling frequency
    target_sfreq : float
        Target sampling frequency
    output_root : str
        Output directory
    
    Returns:
    --------
    plot_file : str
        Path to saved plot
    """
    print(f"\n=== Generating summary plots ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Channel locations
    if raw.info['dig'] is not None:
        raw.plot_sensors(axes=axes[0, 0], show_names=True, show=False)
        axes[0, 0].set_title(f'{subject_id}: Channel Locations')
    else:
        axes[0, 0].text(0.5, 0.5, 'No channel locations available', 
                       ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Channel Locations (Not Available)')
    
    # Plot 2: Power spectral density
    raw.plot_psd(fmin=0.5, fmax=40, ax=axes[0, 1], show=False, verbose=False)
    axes[0, 1].set_title(f'{subject_id}: Power Spectral Density')
    
    # Plot 3: Sample of raw data (first 10 seconds)
    time_slice = slice(0, int(min(10 * raw.info['sfreq'], raw.n_times)))
    time_vec = raw.times[time_slice]
    data_slice = raw.get_data()[:8, time_slice]  # First 8 channels for visibility
    
    for i, ch_name in enumerate(raw.ch_names[:8]):
        axes[1, 0].plot(time_vec, data_slice[i] + i*50, label=ch_name)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Amplitude (μV)')
    axes[1, 0].set_title(f'{subject_id}: Sample Raw Data (First 10s)')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Processing summary
    axes[1, 1].text(0.1, 0.9, f'Subject: {subject_id}', transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.8, f'Original channels: {len(cortical_channels)} + Cz', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.7, f'Final channels: {len(raw.ch_names)}', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.6, f'Original fs: {original_sfreq} Hz', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.5, f'Final fs: {target_sfreq} Hz', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.4, f'Filter: 0.5-30 Hz', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.3, f'Reference: Mean(TP9+TP10)', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.2, f'Duration: {raw.times[-1]:.1f} s', transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Processing Summary')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plot_file = os.path.join(output_root, f"{subject_id}_cortical_preprocessing.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Summary plot saved to: {plot_file}")
    
    return plot_file

def print_summary_statistics(raw, subject_id):
    """
    Print final summary statistics.
    
    Parameters:
    -----------
    raw : mne.Raw
        Preprocessed EEG data
    subject_id : str
        Subject identifier
    """
    print(f"\n=== Preprocessing Summary for {subject_id} ===")
    print(f"Final channels: {len(raw.ch_names)}")
    print(f"Final sampling rate: {raw.info['sfreq']} Hz")
    print(f"Final duration: {raw.times[-1]:.1f} seconds")
    print(f"Data range: {np.min(raw.get_data()):.2f} to {np.max(raw.get_data()):.2f} μV")
    print(f"Data std: {np.std(raw.get_data()):.2f} μV")

def preprocess_cortical_eeg(subject_id, data_root='./data/', output_root='./output/',
                           l_freq=0.5, h_freq=30.0, target_sfreq=128.0, 
                           ref_channels=['TP9', 'TP10'], generate_plots=True):
    """
    Main preprocessing pipeline that aggregates all steps.
    
    Parameters:
    -----------
    subject_id : str
        Subject identifier (e.g., 'pilot_1')
    data_root : str
        Root directory containing subject data
    output_root : str
        Output directory for preprocessed data
    l_freq : float
        Low cutoff frequency for bandpass filter (default: 0.5 Hz)
    h_freq : float
        High cutoff frequency for bandpass filter (default: 30.0 Hz)
    target_sfreq : float
        Target sampling frequency for downsampling (default: 128.0 Hz)
    ref_channels : list
        Reference channels for re-referencing (default: ['TP9', 'TP10'])
    generate_plots : bool
        Whether to generate summary plots (default: True)
    
    Returns:
    --------
    raw : mne.Raw
        Preprocessed EEG data
    """
    
    print(f"\n=== Preprocessing Cortical EEG for {subject_id} ===")
    print(f"Parameters:")
    print(f"  Filter: {l_freq}-{h_freq} Hz")
    print(f"  Target sampling rate: {target_sfreq} Hz")
    print(f"  Reference channels: {ref_channels}")
    print(f"  Generate plots: {generate_plots}")
    
    # Step 1: Load EEG file
    raw = load_eeg_file(subject_id, data_root)
    
    # Step 2: Pick cortical channels
    raw, cortical_channels = pick_cortical_channels(raw)
    
    # Step 3: Add back Cz reference
    raw = add_cz_reference(raw)
    
    # Step 4: Apply bandpass filter
    raw = apply_bandpass_filter(raw, l_freq=l_freq, h_freq=h_freq)
    
    # Step 5: Re-reference to specified channels
    raw = rereference_to_channels(raw, ref_channels)
    
    # Step 6: Downsample
    raw, original_sfreq = downsample_data(raw, target_sfreq=target_sfreq)
    
    # Save preprocessed data
    _ = save_preprocessed_data(raw, subject_id, output_root)
    
    # Generate summary plots (optional)
    if generate_plots:
        _ = generate_summary_plots(raw, subject_id, cortical_channels, 
                                 original_sfreq, target_sfreq, output_root)
    
    # Print summary statistics
    print_summary_statistics(raw, subject_id)
    
    print(f"\n=== Preprocessing completed successfully for {subject_id} ===")
    
    return raw

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description='EEG Preprocessing Script for Cortical Data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required argument
    parser.add_argument('subject_id', 
                       help='Subject identifier (e.g., pilot_1)')
    
    # Optional arguments with defaults
    parser.add_argument('--data_root', 
                       default='./data/',
                       help='Root directory containing subject data')
    
    parser.add_argument('--output_root', 
                       default='./output/',
                       help='Output directory for preprocessed data')
    
    parser.add_argument('--l_freq', 
                       type=float, 
                       default=0.5,
                       help='Low cutoff frequency for bandpass filter (Hz)')
    
    parser.add_argument('--h_freq', 
                       type=float, 
                       default=30.0,
                       help='High cutoff frequency for bandpass filter (Hz)')
    
    parser.add_argument('--target_sfreq', 
                       type=float, 
                       default=128.0,
                       help='Target sampling frequency for downsampling (Hz)')
    
    parser.add_argument('--ref_channels', 
                       nargs='+', 
                       default=['TP9', 'TP10'],
                       help='Reference channels for re-referencing (space-separated)')
    
    parser.add_argument('--no_plots', 
                       action='store_true',
                       help='Skip generating summary plots')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Print configuration
    print(f"\n=== EEG Preprocessing Configuration ===")
    print(f"Subject ID: {args.subject_id}")
    print(f"Data root: {args.data_root}")
    print(f"Output root: {args.output_root}")
    print(f"Filter: {args.l_freq}-{args.h_freq} Hz")
    print(f"Target sampling rate: {args.target_sfreq} Hz")
    print(f"Reference channels: {args.ref_channels}")
    print(f"Generate plots: {not args.no_plots}")
    
    try:
        _ = preprocess_cortical_eeg(
            subject_id=args.subject_id,
            data_root=args.data_root,
            output_root=args.output_root,
            l_freq=args.l_freq,
            h_freq=args.h_freq,
            target_sfreq=args.target_sfreq,
            ref_channels=args.ref_channels,
            generate_plots=not args.no_plots
        )
        
    except Exception as e:
        print(f"\nError during preprocessing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()