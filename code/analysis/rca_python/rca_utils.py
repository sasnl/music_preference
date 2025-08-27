"""
Utility functions for RCA analysis in the music preference study.

This module provides helper functions for integrating RCA with EEG data,
handling MNE objects, and creating visualizations specific to the music preference study.
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
from typing import Union, List, Dict, Optional, Tuple
import pandas as pd
import json
from pathlib import Path

from rca import ReliableComponentsAnalysis


def load_music_preference_data(subject_id: str, data_dir: Union[str, Path], 
                             condition_type: str = 'preference') -> Dict[str, List[mne.Epochs]]:
    """
    Load preprocessed EEG data for music preference analysis.
    
    Parameters
    ----------
    subject_id : str
        Subject identifier (e.g., 'pilot_1')
    data_dir : str or Path
        Path to preprocessed data directory
    condition_type : str, optional (default='preference')
        Type of condition grouping ('preference', 'all')
        
    Returns
    -------
    data_dict : dict
        Dictionary with 'preferred' and 'nonpreferred' keys containing lists of Epochs
    """
    data_dir = Path(data_dir)
    subject_dir = data_dir / subject_id
    
    # Load behavioral data to determine preferred songs
    with open(data_dir.parent / 'beh_ratings.json', 'r') as f:
        ratings = json.load(f)
    
    subject_ratings = ratings['preference'][subject_id]
    
    # Determine preferred songs (top 3) vs non-preferred (bottom 12) 
    song_ratings = [(song, rating) for song, rating in subject_ratings.items()]
    song_ratings.sort(key=lambda x: x[1], reverse=True)
    
    preferred_songs = [song for song, _ in song_ratings[:3]]
    nonpreferred_songs = [song for song, _ in song_ratings[3:]]
    
    print(f"Preferred songs for {subject_id}: {preferred_songs}")
    print(f"Non-preferred songs: {len(nonpreferred_songs)} songs")
    
    # Load EEG data
    data_dict = {'preferred': [], 'nonpreferred': []}
    
    trial_files = list(subject_dir.glob(f"{subject_id}-trial*_proc_*.fif"))
    
    for trial_file in trial_files:
        # Extract song ID from filename
        filename = trial_file.stem
        # Format: pilot_2-trial1_4-1_proc_originalptp-4_cortical_preproc_ica_cleaned
        parts = filename.split('_')
        # Song ID is in the second position after the trial number
        song_id = None
        if len(parts) >= 2:
            # Look for pattern like "4-1" in the parts
            for part in parts[1:4]:  # Check a few parts after trial
                if '-' in part and len(part.split('-')) == 2:
                    # Check if it looks like song ID (number-number)
                    try:
                        nums = part.split('-')
                        int(nums[0])
                        int(nums[1])
                        song_id = part
                        break
                    except ValueError:
                        continue
        
        if song_id is None:
            continue
            
        try:
            # Try reading as epochs first (in case they exist)
            try:
                epochs = mne.read_epochs(trial_file, preload=True, verbose=False)
            except:
                # If no epochs, read as raw and create single epoch
                raw = mne.io.read_raw_fif(trial_file, preload=True, verbose=False)
                # Create a single epoch covering the entire data duration
                duration = raw.times[-1]  # Use full duration of the recording
                epochs = mne.make_fixed_length_epochs(raw, duration=duration, preload=True, verbose=False)
            
            if song_id in preferred_songs:
                data_dict['preferred'].append(epochs)
            elif song_id in nonpreferred_songs:
                data_dict['nonpreferred'].append(epochs)
                
        except Exception as e:
            print(f"Warning: Could not load {trial_file}: {e}")
            
    print(f"Loaded {len(data_dict['preferred'])} preferred trials, "
          f"{len(data_dict['nonpreferred'])} non-preferred trials")
    
    return data_dict


def epochs_to_rca_format(epochs_list: List[mne.Epochs]) -> np.ndarray:
    """
    Convert list of MNE Epochs objects to RCA input format.
    Handles variable-length trials by finding common minimum length.
    
    Parameters
    ----------
    epochs_list : list of mne.Epochs
        List of Epochs objects to concatenate
        
    Returns
    -------
    data_rca : array-like, shape (n_samples, n_channels, n_trials)
        Data in RCA format
    """
    if not epochs_list:
        raise ValueError("epochs_list cannot be empty")
    
    # Get data from each epochs object and find minimum length
    all_data = []
    min_length = float('inf')
    
    for epochs in epochs_list:
        # epochs.get_data() returns (n_epochs, n_channels, n_times)
        epoch_data = epochs.get_data()
        
        # For each epoch in this epochs object
        for epoch_idx in range(epoch_data.shape[0]):
            # Get single epoch: (n_channels, n_times)
            single_epoch = epoch_data[epoch_idx]
            min_length = min(min_length, single_epoch.shape[1])
            all_data.append(single_epoch)
    
    print(f"Found minimum trial length: {min_length} samples")
    
    # Truncate all trials to minimum length and stack
    truncated_data = []
    for epoch_data in all_data:
        # Truncate to minimum length: (n_channels, min_length)
        truncated = epoch_data[:, :min_length]
        truncated_data.append(truncated)
    
    # Stack into (n_trials, n_channels, n_times) then transpose to (n_times, n_channels, n_trials)
    data_stacked = np.stack(truncated_data, axis=0)  # (n_trials, n_channels, n_times)
    data_rca = data_stacked.transpose(2, 1, 0)       # (n_times, n_channels, n_trials)
    
    print(f"Converted {len(epochs_list)} epoch objects to RCA format: {data_rca.shape}")
    return data_rca


def epochs_to_rca_format_fixed_length(epochs_list: List[mne.Epochs], target_length: int) -> np.ndarray:
    """
    Convert list of MNE Epochs objects to RCA format with fixed length.
    
    Parameters
    ----------
    epochs_list : list of mne.Epochs
        List of Epochs objects to convert
    target_length : int
        Target length in samples to truncate all trials to
        
    Returns
    -------
    data_rca : array-like, shape (target_length, n_channels, n_trials)
        Data in RCA format
    """
    if not epochs_list:
        raise ValueError("epochs_list cannot be empty")
    
    # Get data from each epochs object
    all_data = []
    
    for epochs in epochs_list:
        # epochs.get_data() returns (n_epochs, n_channels, n_times)
        epoch_data = epochs.get_data()
        
        # For each epoch in this epochs object
        for epoch_idx in range(epoch_data.shape[0]):
            # Get single epoch: (n_channels, n_times)
            single_epoch = epoch_data[epoch_idx]
            # Truncate to target length
            truncated = single_epoch[:, :target_length]
            all_data.append(truncated)
    
    # Stack into (n_trials, n_channels, n_times) then transpose to (n_times, n_channels, n_trials)
    data_stacked = np.stack(all_data, axis=0)  # (n_trials, n_channels, n_times)
    data_rca = data_stacked.transpose(2, 1, 0)       # (n_times, n_channels, n_trials)
    
    print(f"Converted {len(epochs_list)} epoch objects to RCA format with fixed length {target_length}: {data_rca.shape}")
    return data_rca


def run_rca_on_music_data(subject_id: str, data_dir: Union[str, Path],
                         n_components: int = 3, n_reg: Optional[int] = None,
                         compare_conditions: bool = True) -> Dict[str, any]:
    """
    Run RCA analysis on music preference data for a single subject.
    
    Parameters
    ----------
    subject_id : str
        Subject identifier
    data_dir : str or Path  
        Path to data directory
    n_components : int, optional (default=3)
        Number of RCA components to extract
    n_reg : int or None, optional (default=None)
        Regularization parameter
    compare_conditions : bool, optional (default=True)
        Whether to compare preferred vs non-preferred conditions
        
    Returns
    -------
    results : dict
        Dictionary containing RCA results and analysis
    """
    print(f"=== Running RCA analysis for {subject_id} ===")
    
    # Load data
    data_dict = load_music_preference_data(subject_id, data_dir)
    
    if compare_conditions and len(data_dict['preferred']) > 0 and len(data_dict['nonpreferred']) > 0:
        # Find global minimum length across all trials
        all_epochs = data_dict['preferred'] + data_dict['nonpreferred']
        global_min_length = float('inf')
        
        for epochs in all_epochs:
            epoch_data = epochs.get_data()
            for epoch_idx in range(epoch_data.shape[0]):
                single_epoch = epoch_data[epoch_idx]
                global_min_length = min(global_min_length, single_epoch.shape[1])
        
        print(f"Using global minimum length: {global_min_length} samples")
        
        # Convert to RCA format with global minimum
        preferred_data = epochs_to_rca_format_fixed_length(data_dict['preferred'], global_min_length)
        nonpreferred_data = epochs_to_rca_format_fixed_length(data_dict['nonpreferred'], global_min_length)
        
        # Fit RCA on all data combined
        print("Fitting RCA on combined preferred and non-preferred data...")
        all_data = np.concatenate([preferred_data, nonpreferred_data], axis=2)
        rca = ReliableComponentsAnalysis(n_components=n_components, n_reg=n_reg)
        rca.fit(all_data)
        
        # Transform each condition separately
        preferred_rca = rca.transform(preferred_data)
        nonpreferred_rca = rca.transform(nonpreferred_data)
        
        results = {
            'subject_id': subject_id,
            'rca_model': rca,
            'preferred_data': preferred_data,
            'nonpreferred_data': nonpreferred_data, 
            'preferred_rca': preferred_rca,
            'nonpreferred_rca': nonpreferred_rca,
            'n_preferred_trials': preferred_data.shape[2],
            'n_nonpreferred_trials': nonpreferred_data.shape[2],
            'channel_names': data_dict['preferred'][0].ch_names if data_dict['preferred'] else None
        }
        
    else:
        # Combine all available data
        all_epochs = data_dict['preferred'] + data_dict['nonpreferred']
        if not all_epochs:
            raise ValueError(f"No valid data found for {subject_id}")
            
        all_data = epochs_to_rca_format(all_epochs)
        
        rca = ReliableComponentsAnalysis(n_components=n_components, n_reg=n_reg)
        all_data_rca = rca.fit_transform(all_data)
        
        results = {
            'subject_id': subject_id,
            'rca_model': rca,
            'all_data': all_data,
            'all_data_rca': all_data_rca,
            'n_trials': all_data.shape[2],
            'channel_names': all_epochs[0].ch_names
        }
    
    print(f"RCA analysis complete for {subject_id}")
    return results


def plot_music_rca_results(results: Dict[str, any], save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Plot RCA results specific to music preference analysis.
    
    Parameters
    ----------
    results : dict
        Results from run_rca_on_music_data
    save_path : str or Path, optional
        Path to save the figure
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    rca = results['rca_model']
    subject_id = results['subject_id']
    
    fig, axes = plt.subplots(2, rca.n_components, figsize=(4 * rca.n_components, 8))
    if rca.n_components == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot spatial patterns (forward models)
    for comp in range(rca.n_components):
        ax_spatial = axes[0, comp]
        
        # Simple line plot of spatial weights
        spatial_pattern = rca.forward_models_[:, comp]
        ax_spatial.plot(spatial_pattern, 'ko-', linewidth=2, markersize=6)
        ax_spatial.set_title(f'RC{comp+1} Spatial Pattern\n(λ={rca.eigenvalues_[comp]:.3f})')
        ax_spatial.set_xlabel('Channel')
        ax_spatial.set_ylabel('Weight')
        ax_spatial.grid(True, alpha=0.3)
        
        # Plot time courses if we have condition comparison
        ax_time = axes[1, comp]
        
        if 'preferred_rca' in results and 'nonpreferred_rca' in results:
            # Compute mean time courses
            preferred_mean = np.mean(results['preferred_rca'][:, comp, :], axis=1)
            nonpreferred_mean = np.mean(results['nonpreferred_rca'][:, comp, :], axis=1) 
            
            preferred_sem = np.std(results['preferred_rca'][:, comp, :], axis=1) / \
                           np.sqrt(results['preferred_rca'].shape[2])
            nonpreferred_sem = np.std(results['nonpreferred_rca'][:, comp, :], axis=1) / \
                              np.sqrt(results['nonpreferred_rca'].shape[2])
            
            time_samples = np.arange(len(preferred_mean))
            
            # Plot with error bars
            ax_time.plot(time_samples, preferred_mean, 'r-', linewidth=2, label='Preferred')
            ax_time.fill_between(time_samples, preferred_mean - preferred_sem, 
                               preferred_mean + preferred_sem, alpha=0.3, color='red')
            
            ax_time.plot(time_samples, nonpreferred_mean, 'b-', linewidth=2, label='Non-preferred')
            ax_time.fill_between(time_samples, nonpreferred_mean - nonpreferred_sem,
                               nonpreferred_mean + nonpreferred_sem, alpha=0.3, color='blue')
            
            ax_time.set_title(f'RC{comp+1} Time Course')
            ax_time.set_xlabel('Time (samples)')
            ax_time.set_ylabel('Amplitude')
            ax_time.legend()
            ax_time.grid(True, alpha=0.3)
            
        else:
            # Plot overall time course
            all_mean = np.mean(results['all_data_rca'][:, comp, :], axis=1)
            all_sem = np.std(results['all_data_rca'][:, comp, :], axis=1) / \
                     np.sqrt(results['all_data_rca'].shape[2])
            
            time_samples = np.arange(len(all_mean))
            ax_time.plot(time_samples, all_mean, 'k-', linewidth=2)
            ax_time.fill_between(time_samples, all_mean - all_sem, all_mean + all_sem, 
                               alpha=0.3, color='gray')
            
            ax_time.set_title(f'RC{comp+1} Time Course')
            ax_time.set_xlabel('Time (samples)')
            ax_time.set_ylabel('Amplitude')
            ax_time.grid(True, alpha=0.3)
    
    plt.suptitle(f'RCA Results: {subject_id}', fontsize=16, y=0.95)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        
    return fig


def plot_music_rca_topographies(results: Dict[str, any], save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """
    Plot RCA results with proper EEG topographic maps.
    
    Parameters
    ----------
    results : dict
        Results from run_rca_on_music_data
    save_path : str or Path, optional
        Path to save the figure
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """
    import mne
    from mne.viz import plot_topomap
    
    rca = results['rca_model']
    subject_id = results['subject_id']
    channel_names = results.get('channel_names', None)
    
    if channel_names is None:
        print("Warning: No channel names available, falling back to simple plotting")
        return plot_music_rca_results(results, save_path)
    
    # Create a dummy info object for topographic plotting
    info = mne.create_info(ch_names=channel_names, sfreq=1000, ch_types='eeg')
    info.set_montage('standard_1020')
    
    # Create figure with proper layout
    n_components = rca.n_components
    fig = plt.figure(figsize=(5 * n_components, 12))
    
    # Create grid: topographies on top, eigenvalue spectrum, and time courses below
    rows = 4 if 'preferred_rca' in results else 3
    
    for comp in range(n_components):
        # Topographic plot of forward model (spatial pattern)
        ax_topo = plt.subplot(rows, n_components, comp + 1)
        
        # Use forward model (spatial pattern) for topography
        spatial_pattern = rca.forward_models_[:, comp]
        
        # Create topographic plot
        im, _ = plot_topomap(spatial_pattern, info, axes=ax_topo, show=False, 
                           cmap='RdBu_r', contours=6)
        
        ax_topo.set_title(f'RC{comp+1} Topography\n(λ={rca.eigenvalues_[comp]:.4f})', 
                         fontsize=12, pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax_topo, shrink=0.8, aspect=20)
        cbar.set_label('Amplitude', fontsize=10)
        
        # Line plot of spatial weights for comparison
        ax_line = plt.subplot(rows, n_components, n_components + comp + 1)
        
        ax_line.plot(spatial_pattern, 'ko-', linewidth=2, markersize=4)
        ax_line.set_title(f'RC{comp+1} Channel Weights')
        ax_line.set_xlabel('Channel Index')
        ax_line.set_ylabel('Weight')
        ax_line.grid(True, alpha=0.3)
        
        # Add channel names as x-tick labels (rotate for readability)
        if len(channel_names) <= 32:  # Only for reasonable number of channels
            ax_line.set_xticks(range(len(channel_names)))
            ax_line.set_xticklabels(channel_names, rotation=45, ha='right', fontsize=8)
        
        # Plot time courses if we have condition comparison
        if 'preferred_rca' in results and 'nonpreferred_rca' in results:
            ax_time = plt.subplot(rows, n_components, 2 * n_components + comp + 1)
            
            # Compute mean time courses (downsample for visualization)
            downsample_factor = max(1, results['preferred_rca'].shape[0] // 1000)  # Max 1000 points
            time_indices = slice(0, None, downsample_factor)
            
            preferred_mean = np.mean(results['preferred_rca'][time_indices, comp, :], axis=1)
            nonpreferred_mean = np.mean(results['nonpreferred_rca'][time_indices, comp, :], axis=1)
            
            preferred_sem = np.std(results['preferred_rca'][time_indices, comp, :], axis=1) / \
                           np.sqrt(results['preferred_rca'].shape[2])
            nonpreferred_sem = np.std(results['nonpreferred_rca'][time_indices, comp, :], axis=1) / \
                              np.sqrt(results['nonpreferred_rca'].shape[2])
            
            time_samples = np.arange(len(preferred_mean)) * downsample_factor
            
            # Plot with error bars
            ax_time.plot(time_samples, preferred_mean, 'r-', linewidth=2, label='Preferred')
            ax_time.fill_between(time_samples, preferred_mean - preferred_sem, 
                               preferred_mean + preferred_sem, alpha=0.3, color='red')
            
            ax_time.plot(time_samples, nonpreferred_mean, 'b-', linewidth=2, label='Non-preferred')
            ax_time.fill_between(time_samples, nonpreferred_mean - nonpreferred_sem,
                               nonpreferred_mean + nonpreferred_sem, alpha=0.3, color='blue')
            
            ax_time.set_title(f'RC{comp+1} Time Course')
            ax_time.set_xlabel('Time (samples)')
            ax_time.set_ylabel('Amplitude')
            ax_time.legend(fontsize=10)
            ax_time.grid(True, alpha=0.3)
            
            # Statistical comparison plot
            ax_stats = plt.subplot(rows, n_components, 3 * n_components + comp + 1)
            
            # Compute difference and significance
            difference = preferred_mean - nonpreferred_mean
            pooled_sem = np.sqrt(preferred_sem**2 + nonpreferred_sem**2)
            t_stat = difference / (pooled_sem + 1e-10)  # Avoid division by zero
            
            ax_stats.plot(time_samples, difference, 'g-', linewidth=2, label='Preferred - Non-preferred')
            ax_stats.fill_between(time_samples, difference - pooled_sem, 
                                difference + pooled_sem, alpha=0.3, color='green')
            ax_stats.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            
            ax_stats.set_title(f'RC{comp+1} Condition Difference')
            ax_stats.set_xlabel('Time (samples)')
            ax_stats.set_ylabel('Difference')
            ax_stats.grid(True, alpha=0.3)
        
        else:
            # Plot overall time course if no condition comparison
            ax_time = plt.subplot(rows, n_components, 2 * n_components + comp + 1)
            
            downsample_factor = max(1, results['all_data_rca'].shape[0] // 1000)
            time_indices = slice(0, None, downsample_factor)
            
            all_mean = np.mean(results['all_data_rca'][time_indices, comp, :], axis=1)
            all_sem = np.std(results['all_data_rca'][time_indices, comp, :], axis=1) / \
                     np.sqrt(results['all_data_rca'].shape[2])
            
            time_samples = np.arange(len(all_mean)) * downsample_factor
            ax_time.plot(time_samples, all_mean, 'k-', linewidth=2)
            ax_time.fill_between(time_samples, all_mean - all_sem, all_mean + all_sem, 
                               alpha=0.3, color='gray')
            
            ax_time.set_title(f'RC{comp+1} Time Course')
            ax_time.set_xlabel('Time (samples)')
            ax_time.set_ylabel('Amplitude')
            ax_time.grid(True, alpha=0.3)
    
    # Add eigenvalue spectrum plot
    if n_components > 1:
        ax_eigenvals = plt.subplot(rows, n_components, n_components)  # Top right
        eigenvals_to_plot = rca.eigenvalues_[:min(10, len(rca.eigenvalues_))]
        bars = ax_eigenvals.bar(range(1, len(eigenvals_to_plot) + 1), eigenvals_to_plot)
        
        # Color the bars for the extracted components
        for i in range(min(n_components, len(bars))):
            bars[i].set_color('red')
            bars[i].set_alpha(0.8)
        
        ax_eigenvals.set_xlabel('Component')
        ax_eigenvals.set_ylabel('Eigenvalue')
        ax_eigenvals.set_title('Eigenvalue Spectrum')
        ax_eigenvals.grid(True, alpha=0.3)
    
    plt.suptitle(f'RCA Topographic Analysis: {subject_id}', fontsize=16, y=0.95)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Topographic plot saved to {save_path}")
    
    return fig


def compute_rca_reliability_metrics(results: Dict[str, any]) -> Dict[str, float]:
    """
    Compute reliability metrics for RCA components.
    
    Parameters
    ----------
    results : dict
        Results from run_rca_on_music_data
        
    Returns
    -------
    metrics : dict
        Dictionary of reliability metrics
    """
    rca = results['rca_model']
    
    metrics = {
        'eigenvalues': rca.eigenvalues_[:rca.n_components].tolist(),
        'explained_variance_ratio': [],
        'condition_separability': []
    }
    
    # Compute explained variance ratios
    total_eigenvals = np.sum(np.abs(rca.eigenvalues_))
    for comp in range(rca.n_components):
        evr = np.abs(rca.eigenvalues_[comp]) / total_eigenvals
        metrics['explained_variance_ratio'].append(float(evr))
    
    # Compute condition separability if we have both conditions
    if 'preferred_rca' in results and 'nonpreferred_rca' in results:
        for comp in range(rca.n_components):
            # Compute mean difference between conditions
            pref_mean = np.mean(results['preferred_rca'][:, comp, :])
            nonpref_mean = np.mean(results['nonpreferred_rca'][:, comp, :])
            
            # Compute pooled standard deviation
            pref_std = np.std(results['preferred_rca'][:, comp, :])
            nonpref_std = np.std(results['nonpreferred_rca'][:, comp, :])
            pooled_std = np.sqrt(0.5 * (pref_std**2 + nonpref_std**2))
            
            # Cohen's d as measure of separability
            cohens_d = (pref_mean - nonpref_mean) / pooled_std if pooled_std > 0 else 0
            metrics['condition_separability'].append(float(np.abs(cohens_d)))
    
    return metrics


def save_rca_results(results: Dict[str, any], output_path: Union[str, Path]):
    """
    Save RCA results to file for later analysis.
    
    Parameters
    ----------
    results : dict
        Results from run_rca_on_music_data
    output_path : str or Path
        Path to save results (will create .npz file)
    """
    output_path = Path(output_path)
    
    # Prepare data for saving
    save_dict = {
        'subject_id': results['subject_id'],
        'spatial_filters': results['rca_model'].spatial_filters_,
        'forward_models': results['rca_model'].forward_models_, 
        'eigenvalues': results['rca_model'].eigenvalues_,
        'n_components': results['rca_model'].n_components
    }
    
    if 'preferred_rca' in results:
        save_dict.update({
            'preferred_data_rca': results['preferred_rca'],
            'nonpreferred_data_rca': results['nonpreferred_rca'],
            'n_preferred_trials': results['n_preferred_trials'],
            'n_nonpreferred_trials': results['n_nonpreferred_trials']
        })
    else:
        save_dict.update({
            'all_data_rca': results['all_data_rca'],
            'n_trials': results['n_trials']
        })
    
    if 'channel_names' in results and results['channel_names']:
        save_dict['channel_names'] = np.array(results['channel_names'])
    
    # Compute and save metrics
    metrics = compute_rca_reliability_metrics(results)
    save_dict['metrics'] = metrics
    
    np.savez_compressed(output_path, **save_dict)
    print(f"RCA results saved to {output_path}")


def batch_rca_analysis(subject_ids: List[str], data_dir: Union[str, Path],
                      output_dir: Union[str, Path], **rca_kwargs) -> Dict[str, Dict]:
    """
    Run RCA analysis on multiple subjects.
    
    Parameters
    ----------
    subject_ids : list of str
        List of subject identifiers
    data_dir : str or Path
        Path to data directory
    output_dir : str or Path
        Path to save results
    **rca_kwargs
        Additional arguments for RCA analysis
        
    Returns
    -------
    all_results : dict
        Dictionary with results for each subject
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    all_results = {}
    
    for subject_id in subject_ids:
        print(f"\n=== Processing {subject_id} ===")
        
        try:
            # Run RCA analysis
            results = run_rca_on_music_data(subject_id, data_dir, **rca_kwargs)
            all_results[subject_id] = results
            
            # Save individual results
            save_path = output_dir / f"{subject_id}_rca_results.npz"
            save_rca_results(results, save_path)
            
            # Create and save plots
            plot_path = output_dir / f"{subject_id}_rca_plot.png"
            plot_music_rca_results(results, save_path=plot_path)
            
            plt.close('all')  # Clean up memory
            
        except Exception as e:
            print(f"Error processing {subject_id}: {e}")
            continue
    
    print(f"\nBatch analysis complete. Results saved to {output_dir}")
    return all_results