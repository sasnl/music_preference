#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG ICA Artifact Removal Script

Performs Independent Component Analysis (ICA) to identify and remove eye movement 
and other artifacts from preprocessed cortical EEG data.

Steps:
1. Load preprocessed EEG data
2. Apply high-pass filter for ICA (1Hz recommended)
3. Fit ICA on filtered data
4. Identify artifact components (EOG, muscle, cardiac)
5. Remove artifact components from original data
6. Save cleaned EEG data

Features:
- Real-time interactive component selection with click-to-select interface
- Live time series preview of selected components
- Keyboard shortcuts for efficient workflow
- Automatic artifact detection fallback
- Multiple visualization modes

Usage:
  python eeg_ica_artifact_removal.py <input_file> [options]

Examples:
  # Basic usage with real-time interactive selection (default)
  python eeg_ica_artifact_removal.py pilot_1-trial10_cortical_preprocessed_raw.fif
  
  # Custom ICA parameters
  python eeg_ica_artifact_removal.py pilot_1-trial10_cortical_preprocessed_raw.fif --n_components 25 --method fastica
  
  # Disable real-time mode (use fallback interactive methods)
  python eeg_ica_artifact_removal.py pilot_1-trial10_cortical_preprocessed_raw.fif --no_realtime
  
  # Automatic artifact detection only
  python eeg_ica_artifact_removal.py pilot_1-trial10_cortical_preprocessed_raw.fif --no_interactive

Interactive Controls (Real-time mode):
  • Click components to select/deselect for removal
  • 'h' - Show help
  • 'r' - Reset selection (clear all)
  • 'q' - Quit and proceed with current selection
"""

import numpy as np
import mne
import matplotlib.pyplot as plt
import os
import sys
import argparse
from scipy import stats
try:
    import tkinter as tk
    from tkinter import messagebox, simpledialog
    GUI_AVAILABLE = False  # Force disable GUI due to compatibility issues
    print("Note: GUI disabled for compatibility. Using matplotlib-based interaction.")
except ImportError:
    GUI_AVAILABLE = False
    print("Warning: tkinter not available. Interactive mode will use matplotlib instead.")

def load_preprocessed_eeg(input_file):
    """
    Load preprocessed EEG data.
    
    Parameters:
    -----------
    input_file : str
        Path to preprocessed EEG file (.fif format)
    
    Returns:
    --------
    raw : mne.Raw
        Loaded EEG data
    """
    print(f"\n=== Loading preprocessed EEG data ===")
    print(f"Input file: {input_file}")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    raw = mne.io.read_raw_fif(input_file, preload=True, verbose=False)
    
    print(f"Sampling rate: {raw.info['sfreq']} Hz")
    print(f"Number of channels: {len(raw.ch_names)}")
    print(f"Duration: {raw.times[-1]:.1f} seconds")
    print(f"Channel names: {raw.ch_names}")
    
    return raw

def prepare_data_for_ica(raw, l_freq=1.0):
    """
    Prepare data for ICA by applying high-pass filter.
    
    Parameters:
    -----------
    raw : mne.Raw
        Preprocessed EEG data
    l_freq : float
        High-pass filter frequency for ICA (default: 1.0 Hz)
    
    Returns:
    --------
    raw_ica : mne.Raw
        Data prepared for ICA (copy of original)
    """
    print(f"\n=== Preparing data for ICA ===")
    print(f"Applying high-pass filter: {l_freq} Hz")
    
    # Create a copy for ICA fitting (preserves original data)
    raw_ica = raw.copy()
    
    # Apply high-pass filter for ICA (helps with convergence)
    raw_ica.filter(l_freq=l_freq, h_freq=None, verbose=False)
    
    print(f"Data prepared for ICA fitting")
    return raw_ica

def fit_ica(raw_ica, n_components=25, method='fastica', max_iter=200, random_state=42):
    """
    Fit ICA on prepared EEG data.
    
    Parameters:
    -----------
    raw_ica : mne.Raw
        High-pass filtered EEG data for ICA
    n_components : int
        Number of ICA components (default: 25)
    method : str
        ICA method ('fastica', 'infomax', 'picard')
    max_iter : int
        Maximum iterations for ICA algorithm
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    ica : mne.preprocessing.ICA
        Fitted ICA object
    """
    print(f"\n=== Fitting ICA ===")
    print(f"Method: {method}")
    print(f"Number of components: {n_components}")
    print(f"Max iterations: {max_iter}")
    
    # Initialize ICA
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=method,
        max_iter=max_iter,
        random_state=random_state,
        verbose=False
    )
    
    # Fit ICA
    print("Fitting ICA... (this may take a few minutes)")
    ica.fit(raw_ica, verbose=False)
    
    print(f"ICA fitted successfully")
    print(f"Explained variance ratio: {ica.pca_explained_variance_[:5]}")  # First 5 components
    
    return ica

def detect_eog_components(ica, raw, eog_channels=['Fp1', 'Fp2'], threshold=0.3):
    """
    Automatically detect EOG (eye movement) components.
    
    Parameters:
    -----------
    ica : mne.preprocessing.ICA
        Fitted ICA object
    raw : mne.Raw
        Original EEG data
    eog_channels : list
        Channels to use for EOG detection
    threshold : float
        Correlation threshold for EOG detection
    
    Returns:
    --------
    eog_indices : list
        Indices of components correlating with EOG
    eog_scores : array
        Correlation scores for each component
    """
    print(f"\n=== Detecting EOG components ===")
    print(f"EOG channels: {eog_channels}")
    print(f"Correlation threshold: {threshold}")
    
    # Find available EOG channels
    available_eog = [ch for ch in eog_channels if ch in raw.ch_names]
    print(f"Available EOG channels: {available_eog}")
    
    if not available_eog:
        print("Warning: No EOG channels found. Using automated detection.")
        # Use automatic EOG detection
        eog_indices, eog_scores = ica.find_bads_eog(raw, threshold=threshold, verbose=False)
    else:
        # Use specified EOG channels
        eog_indices, eog_scores = ica.find_bads_eog(
            raw, 
            ch_name=available_eog,
            threshold=threshold,
            verbose=False
        )
    
    print(f"Detected EOG components: {eog_indices}")
    print(f"EOG scores: {eog_scores[eog_indices] if len(eog_indices) > 0 else 'None'}")
    
    return eog_indices, eog_scores

def detect_muscle_components(ica, raw, threshold=0.3, freq_range=(20, 40)):
    """
    Detect muscle artifact components based on high-frequency content.
    
    Parameters:
    -----------
    ica : mne.preprocessing.ICA
        Fitted ICA object
    raw : mne.Raw
        Original EEG data
    threshold : float
        Threshold for muscle detection
    freq_range : tuple
        Frequency range for muscle activity (default: 20-40 Hz)
    
    Returns:
    --------
    muscle_indices : list
        Indices of components with muscle artifacts
    muscle_scores : array
        Scores for muscle detection
    """
    print(f"\n=== Detecting muscle components ===")
    print(f"Frequency range: {freq_range[0]}-{freq_range[1]} Hz")
    print(f"Threshold: {threshold}")
    
    # Get ICA components
    ica_sources = ica.get_sources(raw)
    
    # Calculate power in muscle frequency range
    muscle_scores = []
    for i in range(ica.n_components_):
        component_data = ica_sources.get_data()[i]
        
        # Calculate PSD in muscle frequency range
        freqs, psd = mne.time_frequency.psd_array_welch(
            component_data.reshape(1, -1),
            sfreq=raw.info['sfreq'],
            fmin=freq_range[0],
            fmax=freq_range[1],
            verbose=False
        )
        
        # Calculate relative power in muscle band
        total_power = np.sum(psd)
        muscle_power = np.sum(psd)
        muscle_ratio = muscle_power / total_power if total_power > 0 else 0
        muscle_scores.append(muscle_ratio)
    
    muscle_scores = np.array(muscle_scores)
    muscle_indices = np.where(muscle_scores > threshold)[0].tolist()
    
    print(f"Detected muscle components: {muscle_indices}")
    print(f"Muscle scores: {muscle_scores[muscle_indices] if len(muscle_indices) > 0 else 'None'}")
    
    return muscle_indices, muscle_scores

def plot_ica_components(ica, raw, artifact_indices, output_dir='./output/'):
    """
    Generate comprehensive ICA component plots.
    
    Parameters:
    -----------
    ica : mne.preprocessing.ICA
        Fitted ICA object
    raw : mne.Raw
        Original EEG data
    artifact_indices : list
        Indices of detected artifact components
    output_dir : str
        Output directory for plots
    
    Returns:
    --------
    plot_files : list
        List of generated plot file paths
    """
    print(f"\n=== Generating ICA component plots ===")
    
    os.makedirs(output_dir, exist_ok=True)
    plot_files = []
    
    # Plot 1: ICA components overview
    figs = ica.plot_components(show=False, title='ICA Components Overview')
    if not isinstance(figs, list):
        figs = [figs]
    
    for i, fig in enumerate(figs):
        plot_file = os.path.join(output_dir, f'ica_components_overview_{i:02d}.png')
        fig.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plot_files.append(plot_file)
    
    # Plot 2: Artifact components (if any detected)
    if artifact_indices:
        figs = ica.plot_components(picks=artifact_indices, show=False, 
                                title='Detected Artifact Components')
        if not isinstance(figs, list):
            figs = [figs]
        
        for i, fig in enumerate(figs):
            plot_file = os.path.join(output_dir, f'ica_artifact_components_{i:02d}.png')
            fig.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            plot_files.append(plot_file)
        
        # Plot 3: Component time series for artifacts
        figs = ica.plot_sources(raw, picks=artifact_indices, show=False,
                             title='Artifact Component Time Series')
        if not isinstance(figs, list):
            figs = [figs]
        
        for i, fig in enumerate(figs):
            plot_file = os.path.join(output_dir, f'ica_artifact_timeseries_{i:02d}.png')
            fig.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            plot_files.append(plot_file)
    
    # Plot 4: Before/after comparison (if artifacts detected)
    if artifact_indices:
        # Create a temporary copy for comparison
        raw_clean = raw.copy()
        ica.apply(raw_clean, exclude=artifact_indices)
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot before (first 10 seconds)
        time_slice = slice(0, int(min(10 * raw.info['sfreq'], raw.n_times)))
        time_vec = raw.times[time_slice]
        data_orig = raw.get_data()[:8, time_slice]  # First 8 channels
        
        for i, ch_name in enumerate(raw.ch_names[:8]):
            axes[0].plot(time_vec, data_orig[i] + i*100e-6, label=ch_name)
        axes[0].set_title('Before ICA Artifact Removal')
        axes[0].set_ylabel('Amplitude (V)')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # Plot after
        data_clean = raw_clean.get_data()[:8, time_slice]
        for i, ch_name in enumerate(raw.ch_names[:8]):
            axes[1].plot(time_vec, data_clean[i] + i*100e-6, label=ch_name)
        axes[1].set_title('After ICA Artifact Removal')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Amplitude (V)')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(output_dir, 'ica_before_after_comparison.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(plot_file)
    
    print(f"Generated {len(plot_files)} plot files:")
    for pf in plot_files:
        print(f"  - {pf}")
    
    return plot_files

def apply_ica_removal(ica, raw, exclude_indices):
    """
    Apply ICA to remove artifact components from data.
    
    Parameters:
    -----------
    ica : mne.preprocessing.ICA
        Fitted ICA object
    raw : mne.Raw
        Original EEG data
    exclude_indices : list
        Indices of components to exclude
    
    Returns:
    --------
    raw_clean : mne.Raw
        EEG data with artifacts removed
    """
    print(f"\n=== Applying ICA artifact removal ===")
    print(f"Excluding components: {exclude_indices}")
    
    if not exclude_indices:
        print("No components to exclude. Returning original data.")
        return raw.copy()
    
    # Apply ICA to remove artifacts
    raw_clean = raw.copy()
    ica.apply(raw_clean, exclude=exclude_indices)
    
    print(f"ICA applied successfully")
    print(f"Removed {len(exclude_indices)} components")
    
    return raw_clean

def save_cleaned_data(raw_clean, ica, exclude_indices, output_dir='./output/', 
                     subject_id=None):
    """
    Save cleaned EEG data and ICA information.
    
    Parameters:
    -----------
    raw_clean : mne.Raw
        Cleaned EEG data
    ica : mne.preprocessing.ICA
        Fitted ICA object
    exclude_indices : list
        Excluded component indices
    output_dir : str
        Output directory
    subject_id : str
        Subject identifier for filename
    
    Returns:
    --------
    output_files : dict
        Dictionary of saved file paths
    """
    print(f"\n=== Saving cleaned data ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate base filename
    if subject_id is None:
        base_name = 'eeg_ica_cleaned'
    else:
        base_name = f'{subject_id}_ica_cleaned'
    
    output_files = {}
    
    # Save cleaned EEG data
    eeg_file = os.path.join(output_dir, f'{base_name}_raw.fif')
    raw_clean.save(eeg_file, overwrite=True, verbose=False)
    output_files['eeg'] = eeg_file
    print(f"Cleaned EEG saved: {eeg_file}")
    
    # Save ICA object
    ica_file = os.path.join(output_dir, f'{base_name}_ica.fif')
    ica.save(ica_file, overwrite=True, verbose=False)
    output_files['ica'] = ica_file
    print(f"ICA object saved: {ica_file}")
    
    # Save processing summary
    summary_file = os.path.join(output_dir, f'{base_name}_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"ICA Artifact Removal Summary\n")
        f.write(f"============================\n\n")
        f.write(f"Subject ID: {subject_id or 'Unknown'}\n")
        f.write(f"Number of channels: {len(raw_clean.ch_names)}\n")
        f.write(f"Sampling rate: {raw_clean.info['sfreq']} Hz\n")
        f.write(f"Duration: {raw_clean.times[-1]:.1f} seconds\n")
        f.write(f"ICA method: {ica.method}\n")
        f.write(f"Number of ICA components: {ica.n_components_}\n")
        f.write(f"Excluded components: {exclude_indices}\n")
        f.write(f"Number of excluded components: {len(exclude_indices)}\n")
        f.write(f"Explained variance (first 5 PCs): {ica.pca_explained_variance_[:5]}\n")
    
    output_files['summary'] = summary_file
    print(f"Summary saved: {summary_file}")
    
    return output_files

def interactive_component_selection_realtime(ica, raw):
    """
    Real-time interactive component selection using matplotlib with click-to-select interface.
    
    Parameters:
    -----------
    ica : mne.preprocessing.ICA
        Fitted ICA object
    raw : mne.Raw
        Original EEG data
    
    Returns:
    --------
    exclude_indices : list
        Manually selected component indices to exclude
    """
    print(f"\n=== Real-time Interactive Component Selection ===")
    print("Opening interactive component selection interface...")
    print("\nInstructions:")
    print("• Click on component topographies to select/deselect for removal")
    print("• Selected components will be highlighted in RED")
    print("• Press 'h' for help, 'r' to reset selection, 'q' to quit and proceed")
    print("• Look for artifacts: eye movements (frontal), muscle (temporal), cardiac (rhythmic)")
    
    # Set up the interactive plot
    plt.ion()  # Turn on interactive mode
    
    # Create the main figure with component topographies
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('ICA Component Selection - Click to Select/Deselect Components', fontsize=16, fontweight='bold')
    
    # Calculate grid layout for components
    n_components = ica.n_components_
    n_cols = min(5, n_components)
    n_rows = int(np.ceil(n_components / n_cols))
    
    # Store component information
    component_data = {
        'selected': set(),
        'axes': {},
        'artists': {},
        'texts': {}
    }
    
    # Create individual component plots
    for i in range(n_components):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        
        # Plot component topography
        im, _ = mne.viz.plot_topomap(
            ica.get_components()[:, i], 
            ica.info,
            axes=ax,
            show=False,
            contours=0,
            cmap='RdBu_r'
        )
        
        ax.set_title(f'IC{i:02d}', fontsize=10, pad=5)
        
        # Store references
        component_data['axes'][i] = ax
        component_data['artists'][i] = im
        
        # Add selection indicator text
        text = ax.text(0.5, -0.15, '', transform=ax.transAxes, 
                      ha='center', va='center', fontsize=8, fontweight='bold')
        component_data['texts'][i] = text
    
    plt.tight_layout()
    
    # Create a separate figure for time series of selected components
    fig_ts = plt.figure(figsize=(15, 8))
    fig_ts.suptitle('Selected Component Time Series (First 30 seconds)', fontsize=14)
    ax_ts = fig_ts.add_subplot(111)
    
    def update_time_series():
        """Update the time series plot with currently selected components."""
        ax_ts.clear()
        if component_data['selected']:
            # Get sources for selected components
            sources = ica.get_sources(raw)
            
            # Plot first 30 seconds
            duration = min(30.0, raw.times[-1])
            time_mask = raw.times <= duration
            times = raw.times[time_mask]
            
            # Plot each selected component
            for i, comp_idx in enumerate(sorted(component_data['selected'])):
                comp_data = sources.get_data()[comp_idx, time_mask]
                # Offset for visibility
                offset = i * 2 * np.std(comp_data)
                ax_ts.plot(times, comp_data + offset, 
                          label=f'IC{comp_idx:02d}', linewidth=1)
            
            ax_ts.set_xlabel('Time (s)')
            ax_ts.set_ylabel('Amplitude (offset for visibility)')
            ax_ts.set_title(f'Selected Components: {sorted(component_data["selected"])}')
            ax_ts.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax_ts.grid(True, alpha=0.3)
        else:
            ax_ts.text(0.5, 0.5, 'No components selected\nClick on component topographies to select', 
                      transform=ax_ts.transAxes, ha='center', va='center', fontsize=12)
            ax_ts.set_title('No Components Selected')
        
        fig_ts.canvas.draw()
    
    def update_component_display():
        """Update the visual display of component selection."""
        for i in range(n_components):
            if i in component_data['selected']:
                # Highlight selected components
                component_data['axes'][i].patch.set_facecolor('red')
                component_data['axes'][i].patch.set_alpha(0.3)
                component_data['texts'][i].set_text('SELECTED')
                component_data['texts'][i].set_color('red')
            else:
                # Reset unselected components
                component_data['axes'][i].patch.set_facecolor('white')
                component_data['axes'][i].patch.set_alpha(0.0)
                component_data['texts'][i].set_text('')
        
        fig.canvas.draw()
        update_time_series()
    
    def on_click(event):
        """Handle mouse clicks on component plots."""
        if event.inaxes is None:
            return
        
        # Find which component was clicked
        for comp_idx, ax in component_data['axes'].items():
            if event.inaxes == ax:
                if comp_idx in component_data['selected']:
                    component_data['selected'].remove(comp_idx)
                    print(f"Deselected component IC{comp_idx:02d}")
                else:
                    component_data['selected'].add(comp_idx)
                    print(f"Selected component IC{comp_idx:02d}")
                
                update_component_display()
                break
    
    def on_key(event):
        """Handle keyboard shortcuts."""
        if event.key == 'h':
            help_text = (
                "Keyboard Shortcuts:\n"
                "• h: Show this help\n"
                "• r: Reset selection (clear all)\n"
                "• q: Quit and proceed with current selection\n"
                "• Click on components to select/deselect\n\n"
                "Artifact Identification Tips:\n"
                "• Eye movements: Frontal topography (red at Fp1/Fp2)\n"
                "• Muscle artifacts: Temporal/edge activity, high-frequency noise\n"
                "• Cardiac artifacts: Regular rhythmic patterns (~1Hz)\n"
                "• Line noise: 50/60Hz oscillations"
            )
            print(f"\n{help_text}")
            
        elif event.key == 'r':
            component_data['selected'].clear()
            print("Selection reset - all components deselected")
            update_component_display()
            
        elif event.key == 'q':
            print(f"Quitting with selection: {sorted(component_data['selected'])}")
            plt.close('all')
    
    # Connect event handlers
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig_ts.canvas.mpl_connect('key_press_event', on_key)
    
    # Initial display update
    update_component_display()
    
    # Show plots
    plt.show(block=True)
    
    # Return selected components as sorted list
    exclude_indices = sorted(list(component_data['selected']))
    print(f"\nFinal selection: {exclude_indices}")
    return exclude_indices

def interactive_component_selection_matplotlib(ica, raw, output_dir='./output/'):
    """
    Fallback interactive component selection using file-based input (deprecated).
    Use interactive_component_selection_realtime for better experience.
    """
    print(f"\n=== Interactive Component Selection (file-based - DEPRECATED) ===")
    print("Using fallback file-based selection. Consider using real-time selection instead.")
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save component plots for examination
    figs1 = ica.plot_components(show=False, title='ICA Components - Topographies')
    if not isinstance(figs1, list):
        figs1 = [figs1]
    
    comp_topo_files = []
    for i, fig in enumerate(figs1):
        comp_topo_file = os.path.join(output_dir, f'ica_components_topographies_{i:02d}.png')
        fig.savefig(comp_topo_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        comp_topo_files.append(comp_topo_file)
    
    # Save component time series (first 60 seconds)
    figs2 = ica.plot_sources(raw, start=0, stop=60, show=False, 
                           title='ICA Components - Time Series (0-60s)')
    if not isinstance(figs2, list):
        figs2 = [figs2]
    
    comp_ts_files = []
    for i, fig in enumerate(figs2):
        comp_ts_file = os.path.join(output_dir, f'ica_components_timeseries_{i:02d}.png')
        fig.savefig(comp_ts_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        comp_ts_files.append(comp_ts_file)
    
    # Create input instruction file
    input_file = os.path.join(output_dir, 'component_selection_input.txt')
    with open(input_file, 'w') as f:
        f.write("# ICA Component Selection\n")
        f.write("# Instructions:\n")
        f.write("# 1. Open the component plots:\n")
        f.write(f"#    - Topographies: {', '.join(comp_topo_files)}\n")
        f.write(f"#    - Time series: {', '.join(comp_ts_files)}\n")
        f.write("# 2. Look for artifact patterns:\n")
        f.write("#    - Eye movements: Frontal (Fp1/Fp2) topography with step-like time series\n")
        f.write("#    - Muscle artifacts: Temporal/edge topography with high-frequency noise\n")
        f.write("#    - Cardiac artifacts: Regular rhythmic patterns (~1Hz)\n")
        f.write("#    - Line noise: 50/60Hz oscillations\n")
        f.write("# 3. Enter component numbers to exclude below (one per line or space-separated)\n")
        f.write(f"# Valid range: 0-{ica.n_components_-1}\n")
        f.write("# Example: 0 2 5 12\n")
        f.write("# Leave empty if no components should be excluded\n")
        f.write("\n")
        f.write("COMPONENTS_TO_EXCLUDE=\n")
    
    print(f"\nComponent plots saved:")
    print(f"  - Topographies: {comp_topo_files}")
    print(f"  - Time series: {comp_ts_files}")
    print(f"  - Input file: {input_file}")
    
    print(f"\n=== Manual Component Selection Required ===")
    print("1. Open the component plots to examine the components")
    print("2. Edit the input file and add component numbers to exclude")
    print("3. Press Enter when ready to continue...")
    
    input("Press Enter to continue after reviewing components and editing the input file...")
    
    # Read component selection from file
    exclude_indices = []
    try:
        with open(input_file, 'r') as f:
            for line in f:
                if line.startswith('COMPONENTS_TO_EXCLUDE='):
                    components_str = line.split('=', 1)[1].strip()
                    if components_str:
                        exclude_indices = [int(x.strip()) for x in components_str.split()]
                        # Validate indices
                        exclude_indices = [idx for idx in exclude_indices 
                                         if 0 <= idx < ica.n_components_]
                        exclude_indices = sorted(list(set(exclude_indices)))
                    break
    except (FileNotFoundError, ValueError) as e:
        print(f"Error reading component selection: {e}")
        exclude_indices = []
    
    print(f"Selected components for exclusion: {exclude_indices}")
    return exclude_indices

def interactive_component_selection_gui(ica, raw):
    """
    Interactive component selection using GUI dialogs.
    
    Parameters:
    -----------
    ica : mne.preprocessing.ICA
        Fitted ICA object
    raw : mne.Raw
        Original EEG data
    
    Returns:
    --------
    exclude_indices : list
        Manually selected component indices to exclude
    """
    print(f"\n=== Interactive Component Selection (GUI) ===")
    print("Opening component plots for visual inspection...")
    
    # Show components for manual inspection
    fig1 = ica.plot_components(show=True, title='ICA Components - Click to examine each component')
    fig2 = ica.plot_sources(raw, show=True, title='ICA Component Time Series - Look for artifacts')
    
    # Create a simple GUI for component selection
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Show instruction dialog
    instruction_msg = (
        "ICA Component Selection Instructions:\n\n"
        "1. Examine the component plots that just opened\n"
        "2. Look for artifact patterns:\n"
        "   • Eye movements: Frontal topography (Fp1/Fp2 area) with step-like time series\n"
        "   • Muscle artifacts: Temporal/edge topography with high-frequency noise\n"
        "   • Cardiac artifacts: Regular rhythmic patterns (~1Hz)\n"
        "   • Line noise: 50/60Hz oscillations\n\n"
        "3. Note the component numbers (IC000, IC001, etc.)\n"
        "4. In the next dialog, enter the component numbers to exclude\n\n"
        "Click OK when you're ready to select components"
    )
    
    messagebox.showinfo("Component Selection Instructions", instruction_msg)
    
    # Get component selection from user
    while True:
        user_input = simpledialog.askstring(
            "Component Selection",
            f"Enter component indices to exclude (0-{ica.n_components_-1}):\n"
            f"Separate multiple components with spaces (e.g., '0 2 5 12')\n"
            f"Leave empty if no components should be excluded:",
            initialvalue=""
        )
        
        if user_input is None:  # User clicked Cancel
            print("Component selection cancelled. No components will be excluded.")
            exclude_indices = []
            break
        
        user_input = user_input.strip()
        
        if not user_input:  # Empty input
            print("No components selected for exclusion.")
            exclude_indices = []
            break
        
        try:
            # Parse component indices
            exclude_indices = [int(x.strip()) for x in user_input.split()]
            
            # Validate indices
            invalid_indices = [idx for idx in exclude_indices if idx < 0 or idx >= ica.n_components_]
            if invalid_indices:
                messagebox.showerror(
                    "Invalid Component Indices",
                    f"Invalid component indices: {invalid_indices}\n"
                    f"Valid range is 0-{ica.n_components_-1}. Please try again."
                )
                continue
            
            # Remove duplicates and sort
            exclude_indices = sorted(list(set(exclude_indices)))
            
            # Confirm selection
            if exclude_indices:
                confirm_msg = f"You selected components: {exclude_indices}\n\nProceed with removing these components?"
                if messagebox.askyesno("Confirm Component Selection", confirm_msg):
                    print(f"Selected components for exclusion: {exclude_indices}")
                    break
                else:
                    continue  # Go back to selection
            else:
                print("No components selected for exclusion.")
                break
                
        except ValueError:
            messagebox.showerror(
                "Invalid Input",
                "Please enter only numbers separated by spaces.\n"
                "Example: '0 2 5 12'"
            )
            continue
    
    root.destroy()
    
    return exclude_indices

def interactive_component_selection(ica, raw, output_dir='./output/', use_realtime=True):
    """
    Interactive component selection with multiple interface options.
    
    Parameters:
    -----------
    ica : mne.preprocessing.ICA
        Fitted ICA object
    raw : mne.Raw
        Original EEG data
    output_dir : str
        Output directory for plots (used for fallback methods)
    use_realtime : bool
        Whether to use real-time interactive selection (default: True)
    
    Returns:
    --------
    exclude_indices : list
        Manually selected component indices to exclude
    """
    # Try real-time interactive selection first (best user experience)
    if use_realtime:
        try:
            return interactive_component_selection_realtime(ica, raw)
        except Exception as e:
            print(f"Real-time selection failed: {e}")
            print("Falling back to alternative methods...")
    
    # Fallback to GUI if available
    if GUI_AVAILABLE:
        try:
            return interactive_component_selection_gui(ica, raw)
        except Exception as e:
            print(f"GUI selection failed: {e}")
            print("Falling back to file-based selection...")
    
    # Final fallback to file-based selection
    return interactive_component_selection_matplotlib(ica, raw, output_dir)

def perform_ica_artifact_removal(input_file, output_dir='./output/', 
                                n_components=25, method='fastica',
                                eog_threshold=0.3, muscle_threshold=0.3,
                                interactive=True, generate_plots=True, 
                                use_realtime=True):
    """
    Main function to perform ICA artifact removal.
    
    Parameters:
    -----------
    input_file : str
        Path to preprocessed EEG file
    output_dir : str
        Output directory
    n_components : int
        Number of ICA components
    method : str
        ICA method
    eog_threshold : float
        EOG detection threshold
    muscle_threshold : float
        Muscle detection threshold
    interactive : bool
        Whether to use interactive component selection
    generate_plots : bool
        Whether to generate plots
    use_realtime : bool
        Whether to use real-time interactive selection (if interactive=True)
    
    Returns:
    --------
    results : dict
        Dictionary containing processing results
    """
    print(f"\n=== ICA Artifact Removal Pipeline ===")
    
    # Extract subject ID from filename
    subject_id = os.path.splitext(os.path.basename(input_file))[0]
    subject_id = subject_id.replace('_cortical_preprocessed_raw', '')
    
    # Step 1: Load data
    raw = load_preprocessed_eeg(input_file)
    
    # Step 2: Prepare data for ICA
    raw_ica = prepare_data_for_ica(raw)
    
    # Step 3: Fit ICA
    ica = fit_ica(raw_ica, n_components=n_components, method=method)
    
    # Step 4: Detect artifacts
    if interactive:
        exclude_indices = interactive_component_selection(ica, raw, output_dir, use_realtime)
    else:
        # Automatic detection
        eog_indices, _ = detect_eog_components(ica, raw, threshold=eog_threshold)
        muscle_indices, _ = detect_muscle_components(ica, raw, threshold=muscle_threshold)
        
        # Combine artifact indices
        exclude_indices = list(set(eog_indices + muscle_indices))
        exclude_indices.sort()
    
    # Step 5: Generate plots
    plot_files = []
    if generate_plots:
        plot_files = plot_ica_components(ica, raw, exclude_indices, output_dir)
    
    # Step 6: Apply ICA removal
    raw_clean = apply_ica_removal(ica, raw, exclude_indices)
    
    # Step 7: Save results
    output_files = save_cleaned_data(raw_clean, ica, exclude_indices, 
                                   output_dir, subject_id)
    
    # Compile results
    results = {
        'raw_clean': raw_clean,
        'ica': ica,
        'exclude_indices': exclude_indices,
        'output_files': output_files,
        'plot_files': plot_files,
        'subject_id': subject_id
    }
    
    print(f"\n=== ICA artifact removal completed successfully ===")
    print(f"Subject: {subject_id}")
    print(f"Excluded components: {exclude_indices}")
    print(f"Output files: {list(output_files.values())}")
    
    return results

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description='EEG ICA Artifact Removal Script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required argument
    parser.add_argument('input_file',
                       help='Path to preprocessed EEG file (.fif format)')
    
    # Optional arguments
    parser.add_argument('--output_dir',
                       default='./output/',
                       help='Output directory for cleaned data and plots')
    
    parser.add_argument('--n_components',
                       type=int,
                       default=25,
                       help='Number of ICA components')
    
    parser.add_argument('--method',
                       choices=['fastica', 'infomax', 'picard'],
                       default='fastica',
                       help='ICA method')
    
    parser.add_argument('--eog_threshold',
                       type=float,
                       default=0.3,
                       help='EOG detection threshold')
    
    parser.add_argument('--muscle_threshold',
                       type=float,
                       default=0.3,
                       help='Muscle artifact detection threshold')
    
    parser.add_argument('--no_interactive',
                       action='store_true',
                       help='Disable interactive component selection (use automatic detection)')
    
    parser.add_argument('--no_plots',
                       action='store_true',
                       help='Skip generating plots')
    
    parser.add_argument('--no_realtime',
                       action='store_true',
                       help='Disable real-time interactive selection (use fallback methods)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Print configuration
    print(f"\n=== ICA Artifact Removal Configuration ===")
    print(f"Input file: {args.input_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of components: {args.n_components}")
    print(f"ICA method: {args.method}")
    print(f"EOG threshold: {args.eog_threshold}")
    print(f"Muscle threshold: {args.muscle_threshold}")
    print(f"Interactive mode: {not args.no_interactive}")
    print(f"Real-time selection: {not args.no_realtime}")
    print(f"Generate plots: {not args.no_plots}")
    
    try:
        results = perform_ica_artifact_removal(
            input_file=args.input_file,
            output_dir=args.output_dir,
            n_components=args.n_components,
            method=args.method,
            eog_threshold=args.eog_threshold,
            muscle_threshold=args.muscle_threshold,
            interactive=not args.no_interactive,
            generate_plots=not args.no_plots,
            use_realtime=not args.no_realtime
        )
        
        print(f"\n=== Processing completed successfully ===")
        
    except Exception as e:
        print(f"\nError during ICA processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()