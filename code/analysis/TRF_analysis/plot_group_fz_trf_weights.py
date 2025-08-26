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
import mne
from scipy.stats import ttest_rel

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
            
            # Load Fisher z-scored TRF weights for both conditions
            for condition in ['preferred', 'nonpreferred']:
                if condition in f:
                    weights = f[condition]['weights_fisher_z'][:]  # Use Fisher z-scored weights
                    times = f[condition]['times'][:]      # Shape: (104,)
                    
                    # Extract Fz channel weights from shape (1, 104, 32) -> (104,)
                    # weights[0, :, fz_channel] gives us the time series for Fz
                    fz_weights[condition] = weights[0, :, fz_channel]
                    fz_weights['times'] = times
                    
                    # Debug: print shapes
                    logger.info(f"{participant} {condition}: Fisher z-scored weights shape {weights.shape} -> "
                               f"fz_weights shape {fz_weights[condition].shape}")
                    
            return fz_weights
            
    except Exception as e:
        logger.error(f"Error loading {participant} data: {e}")
        return None

def load_group_channel_cv_scores(output_dir):
    """
    Load per-channel CV scores from all participants for topographic analysis.
    
    Parameters:
    -----------
    output_dir : Path
        Directory containing TRF results
        
    Returns:
    --------
    group_data : dict
        Dictionary containing per-channel CV scores and channel info for all participants
    """
    participants = ['pilot_1', 'pilot_2', 'pilot_3', 'pilot_4', 'pilot_5']
    
    all_cv_preferred = []
    all_cv_nonpreferred = []
    valid_participants = []
    channel_names = None
    
    logger.info("Loading per-channel CV scores from all participants for topographic analysis...")
    
    for participant in participants:
        h5_file = output_dir / f"{participant}_trf_results.h5"
        
        if not h5_file.exists():
            logger.warning(f"No results file found for {participant}")
            continue
            
        try:
            with h5py.File(h5_file, 'r') as f:
                # Get per-channel CV scores from statistical_comparison
                if 'statistical_comparison' in f:
                    stat_group = f['statistical_comparison']
                    if 'performance_preferred' in stat_group and 'performance_nonpreferred' in stat_group:
                        cv_pref_channels = stat_group['performance_preferred'][:]
                        cv_nonpref_channels = stat_group['performance_nonpreferred'][:]
                        
                        all_cv_preferred.append(cv_pref_channels)
                        all_cv_nonpreferred.append(cv_nonpref_channels)
                        valid_participants.append(participant)
                        
                        # Get channel names from root attributes
                        if channel_names is None and 'channel_names' in f.attrs:
                            channel_names = f.attrs['channel_names']
                        
                        logger.info(f"✓ Loaded per-channel CV scores for {participant}: "
                                  f"Shape {cv_pref_channels.shape}, "
                                  f"Preferred mean={np.mean(cv_pref_channels):.4f}, "
                                  f"Non-preferred mean={np.mean(cv_nonpref_channels):.4f}")
                    else:
                        logger.warning(f"Missing per-channel performance data for {participant}")
                else:
                    logger.warning(f"Missing statistical_comparison data for {participant}")
                    
        except Exception as e:
            logger.error(f"Error loading {participant} per-channel CV scores: {e}")
            continue
    
    if len(all_cv_preferred) == 0:
        logger.error("No valid per-channel CV score data found!")
        return None
    
    # Convert to arrays: (participants, channels)
    all_cv_preferred = np.array(all_cv_preferred)
    all_cv_nonpreferred = np.array(all_cv_nonpreferred)
    
    # Average across participants: (channels,)
    group_cv_preferred = np.mean(all_cv_preferred, axis=0)
    group_cv_nonpreferred = np.mean(all_cv_nonpreferred, axis=0)
    
    return {
        'all_cv_preferred': all_cv_preferred,
        'all_cv_nonpreferred': all_cv_nonpreferred,
        'group_cv_preferred': group_cv_preferred,
        'group_cv_nonpreferred': group_cv_nonpreferred,
        'participants': valid_participants,
        'channel_names': channel_names
    }

def create_topographic_plot(ax, cv_scores, channel_names, title, cmap='RdBu_r'):
    """
    Create a topographic plot of CV scores.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    cv_scores : np.ndarray
        CV scores for each channel
    channel_names : list
        List of channel names
    title : str
        Plot title
    cmap : str
        Colormap name
    """
    try:
        # Create a simple montage for EEG channels
        ch_names_list = list(channel_names) if channel_names is not None else [f'CH{i}' for i in range(len(cv_scores))]
        
        # Create montage - try standard 10-20 first
        try:
            montage = mne.channels.make_standard_montage('standard_1020')
            # Filter to available channels
            available_channels = [ch for ch in ch_names_list if ch in montage.ch_names]
            if len(available_channels) < len(ch_names_list) // 2:
                # If less than half channels available, use biosemi64
                montage = mne.channels.make_standard_montage('biosemi64')
                available_channels = [ch for ch in ch_names_list if ch in montage.ch_names]
        except:
            # Fallback to a basic layout
            montage = None
            available_channels = ch_names_list
        
        # Create info object
        info = mne.create_info(available_channels, 1000, 'eeg')
        if montage is not None:
            info.set_montage(montage)
        
        # Filter cv_scores to available channels
        if len(available_channels) == len(cv_scores):
            scores_to_plot = cv_scores
        else:
            # Map available channels to original indices
            channel_indices = [ch_names_list.index(ch) for ch in available_channels if ch in ch_names_list]
            scores_to_plot = cv_scores[channel_indices] if len(channel_indices) > 0 else cv_scores
        
        # Create topographic plot
        im, _ = mne.viz.plot_topomap(scores_to_plot, info, axes=ax, show=False, 
                                   cmap=cmap, contours=6, show_names=True, 
                                   names=available_channels, size=3)
        
        ax.set_title(title, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('CV Score (R²)', rotation=270, labelpad=20)
        
    except Exception as e:
        # Fallback: simple text display
        logger.warning(f"Could not create topographic plot: {e}. Using fallback.")
        ax.text(0.5, 0.5, f'{title}\n\nMean CV: {np.mean(cv_scores):.4f}\nStd: {np.std(cv_scores):.4f}', 
                ha='center', va='center', transform=ax.transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax.set_title(title, fontweight='bold')
        ax.axis('off')

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
    
    # Load per-channel CV scores for topographic plots
    topo_data = load_group_channel_cv_scores(output_dir)
    
    # Create comprehensive figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Group TRF Analysis Summary (n={len(valid_participants)})', 
                 fontsize=18, fontweight='bold')
    
    # 1. Fz TRF Weights Time Series (top left)
    ax = axes[0, 0]
    ax.plot(times_ms, mean_preferred, color='red', linewidth=3, label='Preferred', alpha=0.9)
    ax.fill_between(times_ms, mean_preferred - sem_preferred, mean_preferred + sem_preferred, 
                    color='red', alpha=0.2)
    
    ax.plot(times_ms, mean_nonpreferred, color='black', linewidth=3, label='Non-preferred', alpha=0.9)
    ax.fill_between(times_ms, mean_nonpreferred - sem_nonpreferred, mean_nonpreferred + sem_nonpreferred, 
                    color='black', alpha=0.2)
    
    ax.set_xlabel('Time (ms)', fontweight='bold')
    ax.set_ylabel('Fisher z-score', fontweight='bold')
    ax.set_title('Fz Channel TRF Weights (Fisher z-scored)', fontweight='bold')
    ax.axhline(0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Set limits for Fz plot
    mean_range = max(np.max(np.abs(mean_preferred)), np.max(np.abs(mean_nonpreferred)))
    sem_range = max(np.max(sem_preferred), np.max(sem_nonpreferred))
    y_max = (mean_range + sem_range) * 1.3
    ax.set_ylim(-y_max, y_max)
    
    # 2. Topographic Plot - Preferred Condition CV Scores (top right)
    if topo_data is not None:
        ax = axes[0, 1]
        create_topographic_plot(ax, topo_data['group_cv_preferred'], 
                               topo_data['channel_names'], 
                               'Preferred Music CV Scores', cmap='Reds')
    else:
        ax = axes[0, 1]
        ax.text(0.5, 0.5, 'Per-channel CV scores\nnot available', ha='center', va='center', 
               transform=ax.transAxes, fontsize=12)
        ax.set_title('Preferred Music CV Scores')
    
    # 3. Topographic Plot - Non-preferred Condition CV Scores (bottom left)
    if topo_data is not None:
        ax = axes[1, 0]
        create_topographic_plot(ax, topo_data['group_cv_nonpreferred'], 
                               topo_data['channel_names'], 
                               'Non-preferred Music CV Scores', cmap='Blues')
    else:
        ax = axes[1, 0]
        ax.text(0.5, 0.5, 'Per-channel CV scores\nnot available', ha='center', va='center', 
               transform=ax.transAxes, fontsize=12)
        ax.set_title('Non-preferred Music CV Scores')
    
    # 4. Group Statistics Summary (bottom right)
    ax = axes[1, 1]
    ax.axis('off')  # Turn off axes for text summary
    
    # Add participant information
    summary_text = f'Participants: {", ".join(valid_participants)}\n'
    summary_text += f'Time range: {times_ms[0]:.1f} to {times_ms[-1]:.1f} ms\n'
    summary_text += f'Total time points: {len(times_ms)}\n\n'
    
    if topo_data is not None:
        mean_pref = np.mean(topo_data['group_cv_preferred'])
        mean_nonpref = np.mean(topo_data['group_cv_nonpreferred'])
        summary_text += f'Mean CV Score (Preferred): {mean_pref:.4f}\n'
        summary_text += f'Mean CV Score (Non-preferred): {mean_nonpref:.4f}\n'
        summary_text += f'Mean Difference: {mean_pref - mean_nonpref:.4f}\n\n'
        summary_text += f'Channels analyzed: {len(topo_data["group_cv_preferred"])}\n'
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Save comprehensive plot
    output_file = output_dir / "group_fz_trf_comprehensive.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Saved group Fz TRF weights plot to {output_file}")
    
    # Save Fz time series data
    fz_data_df = pd.DataFrame({
        'time_ms': times_ms,
        'mean_preferred': mean_preferred,
        'sem_preferred': sem_preferred,
        'mean_nonpreferred': mean_nonpreferred,
        'sem_nonpreferred': sem_nonpreferred
    })
    
    fz_file = output_dir / "group_fz_trf_timeseries.csv"
    fz_data_df.to_csv(fz_file, index=False)
    logger.info(f"Saved Fz time series data to {fz_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("GROUP Fz TRF WEIGHTS SUMMARY")
    print(f"{'='*60}")
    print(f"Participants analyzed: {len(valid_participants)}")
    print(f"Time range: {times_ms[0]:.1f} to {times_ms[-1]:.1f} ms")
    print(f"Total time points: {len(times_ms)}")
    
    if topo_data is not None:
        print(f"Channels analyzed: {len(topo_data['group_cv_preferred'])}")
        print(f"Mean CV Score (Preferred): {np.mean(topo_data['group_cv_preferred']):.4f}")
        print(f"Mean CV Score (Non-preferred): {np.mean(topo_data['group_cv_nonpreferred']):.4f}")
    
    # Peak analysis
    max_diff_idx = np.argmax(np.abs(mean_preferred - mean_nonpreferred))
    max_diff_time = times_ms[max_diff_idx]
    max_diff_value = mean_preferred[max_diff_idx] - mean_nonpreferred[max_diff_idx]
    
    print(f"\nLargest Fz difference:")
    print(f"Time: {max_diff_time:.1f} ms")
    print(f"Difference: {max_diff_value:.6f} (Preferred - Non-preferred)")
    
    return valid_participants, fz_data_df

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