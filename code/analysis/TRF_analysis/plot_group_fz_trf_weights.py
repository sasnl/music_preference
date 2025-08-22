#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group-level Fz TRF Weights Analysis

Loads individual TRF results from all participants and plots the averaged 
Fz channel TRF weights to show group-level temporal dynamics of preference effects.

Usage: python plot_group_fz_trf_weights.py
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_participant_fz_weights(participant, output_dir):
    """
    Load Fz TRF weights for a specific participant.
    
    Parameters:
    -----------
    participant : str
        Participant ID (e.g., 'pilot_1')
    output_dir : Path
        Directory containing TRF results
        
    Returns:
    --------
    fz_weights : dict
        Dictionary with 'preferred', 'nonpreferred' TRF weights and 'times'
    """
    h5_file = output_dir / f"{participant}_trf_results.h5"
    
    if not h5_file.exists():
        logger.warning(f"No results file found for {participant}")
        return None
    
    try:
        with h5py.File(h5_file, 'r') as f:
            # Based on EEG data structure, Fz is at index 1
            # Channel order: ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'Cz']
            fz_channel = 1  # Fz channel index
            logger.info(f"{participant}: Using Fz channel at index {fz_channel}")
            
            fz_weights = {}
            
            # Load TRF weights for both conditions
            for condition in ['preferred', 'nonpreferred']:
                if condition in f:
                    weights = f[condition]['weights'][:]  # Shape: (1, 66, 32) = (features, time, channels)
                    times = f[condition]['times'][:]      # Shape: (66,)
                    
                    # Extract Fz channel weights from shape (1, 66, 32) -> (66,)
                    # weights[0, :, fz_channel] gives us the time series for Fz
                    fz_weights[condition] = weights[0, :, fz_channel]
                    fz_weights['times'] = times
                    
                    # Debug: print shapes
                    logger.info(f"{participant} {condition}: weights shape {weights.shape} -> "
                               f"fz_weights shape {fz_weights[condition].shape}")
                    
            return fz_weights
            
    except Exception as e:
        logger.error(f"Error loading {participant} data: {e}")
        return None

def plot_group_fz_weights(output_dir):
    """
    Plot group-averaged Fz TRF weights for preferred vs non-preferred conditions.
    
    Parameters:
    -----------
    output_dir : Path
        Directory containing TRF results
    """
    participants = ['pilot_1', 'pilot_2', 'pilot_3', 'pilot_4', 'pilot_5']
    
    all_weights_preferred = []
    all_weights_nonpreferred = []
    common_times = None
    valid_participants = []
    
    # Load data from all participants
    logger.info("Loading TRF weights from all participants...")
    
    for participant in participants:
        weights_data = load_participant_fz_weights(participant, output_dir)
        
        if weights_data is not None and 'preferred' in weights_data and 'nonpreferred' in weights_data:
            all_weights_preferred.append(weights_data['preferred'])
            all_weights_nonpreferred.append(weights_data['nonpreferred'])
            valid_participants.append(participant)
            
            if common_times is None:
                common_times = weights_data['times']
            
            logger.info(f"✓ Loaded {participant}: Preferred shape {weights_data['preferred'].shape}, "
                       f"Non-preferred shape {weights_data['nonpreferred'].shape}")
        else:
            logger.warning(f"✗ Failed to load {participant}")
    
    if len(all_weights_preferred) == 0:
        logger.error("No valid participant data found!")
        return
    
    # Convert to arrays and compute statistics
    all_weights_preferred = np.array(all_weights_preferred)
    all_weights_nonpreferred = np.array(all_weights_nonpreferred)
    
    # Compute mean and SEM across participants
    mean_preferred = np.mean(all_weights_preferred, axis=0)
    sem_preferred = np.std(all_weights_preferred, axis=0) / np.sqrt(len(all_weights_preferred))
    
    mean_nonpreferred = np.mean(all_weights_nonpreferred, axis=0)
    sem_nonpreferred = np.std(all_weights_nonpreferred, axis=0) / np.sqrt(len(all_weights_nonpreferred))
    
    # Convert times to milliseconds
    times_ms = common_times * 1000
    
    logger.info(f"Group analysis: {len(valid_participants)} participants")
    logger.info(f"Time range: {times_ms[0]:.1f} to {times_ms[-1]:.1f} ms")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot mean ± SEM with shaded error regions
    plt.plot(times_ms, mean_preferred, color='red', linewidth=3, label='Preferred', alpha=0.9)
    plt.fill_between(times_ms, mean_preferred - sem_preferred, mean_preferred + sem_preferred, 
                     color='red', alpha=0.2)
    
    plt.plot(times_ms, mean_nonpreferred, color='black', linewidth=3, label='Non-preferred', alpha=0.9)
    plt.fill_between(times_ms, mean_nonpreferred - sem_nonpreferred, mean_nonpreferred + sem_nonpreferred, 
                     color='black', alpha=0.2)
    
    # Customize plot
    plt.xlabel('Time (ms)', fontsize=14, fontweight='bold')
    plt.ylabel('TRF Weight (a.u.)', fontsize=14, fontweight='bold')
    plt.title(f'Group-Averaged Fz TRF Weights (n={len(valid_participants)})', 
              fontsize=16, fontweight='bold')
    
    # Add reference lines
    plt.axhline(0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add grid and legend
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12, loc='best')
    
    # Set limits for better visualization
    y_max = max(np.max(np.abs(mean_preferred)), np.max(np.abs(mean_nonpreferred))) * 2
    plt.ylim(-y_max, y_max)
    
    # Add participant information
    plt.text(0.02, 0.98, f'Participants: {", ".join(valid_participants)}', 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add statistical information
    # Compute t-test between conditions at each time point
    from scipy.stats import ttest_rel
    
    t_stats = []
    p_values = []
    
    for i in range(len(times_ms)):
        t_stat, p_val = ttest_rel(all_weights_preferred[:, i], all_weights_nonpreferred[:, i])
        t_stats.append(t_stat)
        p_values.append(p_val)
    
    p_values = np.array(p_values)
    
    # Find significant time points (p < 0.05)
    sig_times = times_ms[p_values < 0.05]
    
    if len(sig_times) > 0:
        plt.text(0.02, 0.02, f'Significant time points (p<0.05): {len(sig_times)} / {len(times_ms)}', 
                 transform=plt.gca().transAxes, fontsize=10, verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # Highlight significant regions
        sig_mask = p_values < 0.05
        if np.any(sig_mask):
            y_min, y_max = plt.ylim()
            plt.fill_between(times_ms, y_min, y_max, where=sig_mask, 
                           color='yellow', alpha=0.2, label='p < 0.05')
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_dir / "group_fz_trf_weights.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Saved group Fz TRF weights plot to {output_file}")
    
    # Save statistical results
    stats_df = pd.DataFrame({
        'time_ms': times_ms,
        'mean_preferred': mean_preferred,
        'sem_preferred': sem_preferred,
        'mean_nonpreferred': mean_nonpreferred,
        'sem_nonpreferred': sem_nonpreferred,
        't_statistic': t_stats,
        'p_value': p_values,
        'significant': p_values < 0.05
    })
    
    stats_file = output_dir / "group_fz_trf_statistics.csv"
    stats_df.to_csv(stats_file, index=False)
    logger.info(f"Saved statistical analysis to {stats_file}")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("GROUP Fz TRF WEIGHTS SUMMARY")
    print(f"{'='*60}")
    print(f"Participants analyzed: {len(valid_participants)}")
    print(f"Time range: {times_ms[0]:.1f} to {times_ms[-1]:.1f} ms")
    print(f"Total time points: {len(times_ms)}")
    print(f"Significant time points (p<0.05): {np.sum(p_values < 0.05)}")
    print(f"Percentage significant: {np.sum(p_values < 0.05) / len(p_values) * 100:.1f}%")
    
    # Peak analysis
    max_diff_idx = np.argmax(np.abs(mean_preferred - mean_nonpreferred))
    max_diff_time = times_ms[max_diff_idx]
    max_diff_value = mean_preferred[max_diff_idx] - mean_nonpreferred[max_diff_idx]
    
    print(f"\nLargest difference:")
    print(f"Time: {max_diff_time:.1f} ms")
    print(f"Difference: {max_diff_value:.6f} (Preferred - Non-preferred)")
    print(f"p-value at peak: {p_values[max_diff_idx]:.6f}")
    
    return valid_participants, stats_df

def main():
    """Main function to create group Fz TRF weights plot."""
    output_dir = Path("/Users/tongshan/Documents/music_preference/output/trf_analysis")
    
    if not output_dir.exists():
        logger.error(f"Output directory does not exist: {output_dir}")
        return
    
    logger.info("Starting group Fz TRF weights analysis...")
    
    valid_participants, stats_df = plot_group_fz_weights(output_dir)
    
    logger.info("Group Fz TRF weights analysis completed!")
    
    return valid_participants, stats_df

if __name__ == "__main__":
    main()