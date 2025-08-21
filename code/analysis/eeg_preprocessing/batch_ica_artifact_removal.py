#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch ICA Artifact Removal Script

Fits ICA on a continuous EEG file, then applies the same artifact removal 
to multiple trial files. This ensures consistent artifact removal across 
all trials from the same subject.

Steps:
1. Load continuous preprocessed EEG data
2. Fit ICA on continuous data with interactive component selection
3. Apply the fitted ICA to all trial files automatically
4. Save cleaned trial files

Usage:
  python batch_ica_artifact_removal.py <continuous_file> <trial_directory> [options]

Examples:
  # Basic usage
  python batch_ica_artifact_removal.py data/preprocessed/pilot_1/pilot_1_cortical_preprocessed.fif data/preprocessed/pilot_1/preprocessed_trials/
  
  # Custom parameters
  python batch_ica_artifact_removal.py data/preprocessed/pilot_1/pilot_1_cortical_preprocessed.fif data/preprocessed/pilot_1/preprocessed_trials/ --n_components 25 --output_suffix "_ica_cleaned"
"""

import numpy as np
import mne
import matplotlib.pyplot as plt
import os
import sys
import argparse
import glob
from pathlib import Path

# Import the real-time interactive selection from the main script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from eeg_ica_artifact_removal import (
    load_preprocessed_eeg,
    prepare_data_for_ica, 
    fit_ica,
    interactive_component_selection,
    apply_ica_removal,
    save_cleaned_data
)

def find_trial_files(trial_directory, patterns=['*.fif'], exclude_patterns=['*_ica_cleaned*']):
    """
    Find trial files in the specified directory.
    
    Parameters:
    -----------
    trial_directory : str
        Directory containing trial files
    patterns : list
        File patterns to include (e.g., ['*.fif'])
    exclude_patterns : list
        File patterns to exclude (e.g., ['*_ica_cleaned*'])
    
    Returns:
    --------
    trial_files : list
        List of trial file paths
    """
    print(f"\n=== Finding trial files in {trial_directory} ===")
    
    trial_files = []
    trial_dir = Path(trial_directory)
    
    if not trial_dir.exists():
        raise FileNotFoundError(f"Trial directory not found: {trial_directory}")
    
    # Find files matching include patterns
    for pattern in patterns:
        files = list(trial_dir.glob(pattern))
        trial_files.extend(files)
    
    # Filter out excluded patterns
    filtered_files = []
    for file_path in trial_files:
        exclude = False
        for exclude_pattern in exclude_patterns:
            if file_path.match(exclude_pattern):
                exclude = True
                break
        if not exclude:
            filtered_files.append(str(file_path))
    
    trial_files = sorted(filtered_files)
    
    print(f"Found {len(trial_files)} trial files:")
    for i, file_path in enumerate(trial_files[:10]):  # Show first 10
        print(f"  {i+1:2d}. {os.path.basename(file_path)}")
    if len(trial_files) > 10:
        print(f"  ... and {len(trial_files) - 10} more files")
    
    return trial_files

def apply_ica_to_trial(ica, exclude_indices, trial_file, output_dir, output_suffix='_ica_cleaned'):
    """
    Apply fitted ICA to a single trial file.
    
    Parameters:
    -----------
    ica : mne.preprocessing.ICA
        Fitted ICA object
    exclude_indices : list
        Component indices to exclude
    trial_file : str
        Path to trial file
    output_dir : str
        Output directory
    output_suffix : str
        Suffix for cleaned files
    
    Returns:
    --------
    output_file : str
        Path to cleaned trial file
    """
    try:
        # Load trial data
        trial_raw = mne.io.read_raw_fif(trial_file, preload=True, verbose=False)
        
        # Apply ICA artifact removal
        if exclude_indices:
            trial_clean = trial_raw.copy()
            ica.apply(trial_clean, exclude=exclude_indices)
        else:
            trial_clean = trial_raw.copy()
        
        # Generate output filename
        trial_basename = os.path.basename(trial_file)
        trial_name = os.path.splitext(trial_basename)[0]
        output_file = os.path.join(output_dir, f"{trial_name}{output_suffix}.fif")
        
        # Save cleaned trial
        trial_clean.save(output_file, overwrite=True, verbose=False)
        
        return output_file
        
    except Exception as e:
        print(f"Error processing {trial_file}: {e}")
        return None

def batch_ica_artifact_removal(continuous_file, trial_directory, output_dir='./output_batch_ica/', 
                              n_components=25, method='fastica', output_suffix='_ica_cleaned',
                              use_realtime=True, file_patterns=['*.fif']):
    """
    Main function for batch ICA artifact removal.
    
    Parameters:
    -----------
    continuous_file : str
        Path to continuous preprocessed EEG file
    trial_directory : str
        Directory containing trial files
    output_dir : str
        Output directory for cleaned files
    n_components : int
        Number of ICA components
    method : str
        ICA method
    output_suffix : str
        Suffix for cleaned files
    use_realtime : bool
        Whether to use real-time interactive selection
    file_patterns : list
        File patterns to process
    
    Returns:
    --------
    results : dict
        Dictionary containing processing results
    """
    print(f"\n=== Batch ICA Artifact Removal Pipeline ===")
    print(f"Continuous file: {continuous_file}")
    print(f"Trial directory: {trial_directory}")
    print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load continuous data and fit ICA
    print(f"\n=== Step 1: Fitting ICA on continuous data ===")
    raw_continuous = load_preprocessed_eeg(continuous_file)
    
    # Prepare data for ICA
    raw_ica = prepare_data_for_ica(raw_continuous)
    
    # Fit ICA
    ica = fit_ica(raw_ica, n_components=n_components, method=method)
    
    # Interactive component selection
    print(f"\n=== Step 2: Interactive component selection ===")
    exclude_indices = interactive_component_selection(ica, raw_continuous, output_dir, use_realtime)
    
    print(f"\nSelected {len(exclude_indices)} components for removal: {exclude_indices}")
    
    # Step 3: Find trial files
    print(f"\n=== Step 3: Finding trial files ===")
    trial_files = find_trial_files(trial_directory, patterns=file_patterns)
    
    if not trial_files:
        print("No trial files found. Exiting.")
        return {'ica': ica, 'exclude_indices': exclude_indices, 'processed_files': []}
    
    # Step 4: Apply ICA to all trial files
    print(f"\n=== Step 4: Applying ICA to {len(trial_files)} trial files ===")
    
    processed_files = []
    failed_files = []
    
    for i, trial_file in enumerate(trial_files):
        print(f"Processing {i+1:2d}/{len(trial_files)}: {os.path.basename(trial_file)}")
        
        output_file = apply_ica_to_trial(
            ica, exclude_indices, trial_file, output_dir, output_suffix
        )
        
        if output_file:
            processed_files.append(output_file)
            print(f"  → Saved: {os.path.basename(output_file)}")
        else:
            failed_files.append(trial_file)
            print(f"  → Failed!")
    
    # Step 5: Save ICA object and summary
    print(f"\n=== Step 5: Saving ICA object and summary ===")
    
    # Extract subject ID from continuous file
    subject_id = os.path.splitext(os.path.basename(continuous_file))[0]
    subject_id = subject_id.replace('_cortical_preprocessed', '')
    
    # Save ICA object
    ica_file = os.path.join(output_dir, f'{subject_id}_batch_ica.fif')
    ica.save(ica_file, overwrite=True, verbose=False)
    print(f"ICA object saved: {ica_file}")
    
    # Save processing summary
    summary_file = os.path.join(output_dir, f'{subject_id}_batch_ica_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Batch ICA Artifact Removal Summary\n")
        f.write(f"==================================\n\n")
        f.write(f"Subject ID: {subject_id}\n")
        f.write(f"Continuous file: {continuous_file}\n")
        f.write(f"Trial directory: {trial_directory}\n")
        f.write(f"ICA method: {method}\n")
        f.write(f"Number of ICA components: {n_components}\n")
        f.write(f"Excluded components: {exclude_indices}\n")
        f.write(f"Number of excluded components: {len(exclude_indices)}\n")
        f.write(f"Processed files: {len(processed_files)}\n")
        f.write(f"Failed files: {len(failed_files)}\n\n")
        
        f.write(f"Processed files:\n")
        for pf in processed_files:
            f.write(f"  - {os.path.basename(pf)}\n")
        
        if failed_files:
            f.write(f"\nFailed files:\n")
            for ff in failed_files:
                f.write(f"  - {os.path.basename(ff)}\n")
    
    print(f"Summary saved: {summary_file}")
    
    # Compile results
    results = {
        'ica': ica,
        'exclude_indices': exclude_indices,
        'processed_files': processed_files,
        'failed_files': failed_files,
        'ica_file': ica_file,
        'summary_file': summary_file,
        'subject_id': subject_id
    }
    
    print(f"\n=== Batch processing completed ===")
    print(f"Subject: {subject_id}")
    print(f"Excluded components: {exclude_indices}")
    print(f"Successfully processed: {len(processed_files)} files")
    print(f"Failed: {len(failed_files)} files")
    
    return results

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description='Batch ICA Artifact Removal Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('continuous_file',
                       help='Path to continuous preprocessed EEG file (.fif format)')
    parser.add_argument('trial_directory',
                       help='Directory containing trial files')
    
    # Optional arguments
    parser.add_argument('--output_dir',
                       default='./output_batch_ica/',
                       help='Output directory for cleaned files')
    
    parser.add_argument('--n_components',
                       type=int,
                       default=25,
                       help='Number of ICA components')
    
    parser.add_argument('--method',
                       choices=['fastica', 'infomax', 'picard'],
                       default='fastica',
                       help='ICA method')
    
    parser.add_argument('--output_suffix',
                       default='_ica_cleaned',
                       help='Suffix for cleaned files')
    
    parser.add_argument('--file_patterns',
                       nargs='+',
                       default=['*.fif'],
                       help='File patterns to process (e.g., "*.fif" "*cortical_preproc.fif")')
    
    parser.add_argument('--no_realtime',
                       action='store_true',
                       help='Disable real-time interactive selection (use fallback methods)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Print configuration
    print(f"\n=== Batch ICA Configuration ===")
    print(f"Continuous file: {args.continuous_file}")
    print(f"Trial directory: {args.trial_directory}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of components: {args.n_components}")
    print(f"ICA method: {args.method}")
    print(f"Output suffix: {args.output_suffix}")
    print(f"File patterns: {args.file_patterns}")
    print(f"Real-time selection: {not args.no_realtime}")
    
    try:
        results = batch_ica_artifact_removal(
            continuous_file=args.continuous_file,
            trial_directory=args.trial_directory,
            output_dir=args.output_dir,
            n_components=args.n_components,
            method=args.method,
            output_suffix=args.output_suffix,
            use_realtime=not args.no_realtime,
            file_patterns=args.file_patterns
        )
        
        print(f"\n=== Batch processing completed successfully ===")
        
    except Exception as e:
        print(f"\nError during batch processing: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()