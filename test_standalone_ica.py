#!/usr/bin/env python3
"""
Standalone test script for the new real-time ICA component selection interface.
This version includes the interactive function directly to avoid import issues.
"""

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA

def interactive_component_selection_realtime(ica, raw):
    """
    Real-time interactive component selection using matplotlib with click-to-select interface.
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

def create_mock_ica_data(n_channels=32, n_components=20, duration=60, sfreq=1000):
    """
    Create mock ICA and EEG data for testing the interface.
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
    
    # Create and fit mock ICA properly
    ica = ICA(n_components=n_components, method='fastica', random_state=42)
    
    # Actually fit the ICA to create all necessary attributes
    print("Fitting ICA on mock data...")
    ica.fit(raw, verbose=False)
    
    # Now modify the mixing matrix to create realistic artifact patterns
    # Get the actual fitted mixing matrix
    mixing_matrix = ica.mixing_.copy()
    
    # Make some components look like typical artifacts:
    # Component 0: Eye blink (strong at Fp1, Fp2)
    if 'Fp1' in ch_names and 'Fp2' in ch_names:
        fp1_idx = ch_names.index('Fp1')
        fp2_idx = ch_names.index('Fp2')
        mixing_matrix[fp1_idx, 0] = 1.5
        mixing_matrix[fp2_idx, 0] = 1.2
        # Reduce activity at other channels for this component
        for i, ch in enumerate(ch_names):
            if ch not in ['Fp1', 'Fp2']:
                mixing_matrix[i, 0] *= 0.1
    
    # Component 1: Lateral eye movement (Fp1 vs Fp2)
    if len(ch_names) > 1 and 'Fp1' in ch_names and 'Fp2' in ch_names:
        fp1_idx = ch_names.index('Fp1')
        fp2_idx = ch_names.index('Fp2')
        mixing_matrix[fp1_idx, 1] = 1.0
        mixing_matrix[fp2_idx, 1] = -1.0
        # Reduce activity at other channels for this component
        for i, ch in enumerate(ch_names):
            if ch not in ['Fp1', 'Fp2']:
                mixing_matrix[i, 1] *= 0.1
    
    # Component 2: Muscle artifact (temporal electrodes)
    if len(ch_names) > 2 and 'T7' in ch_names and 'T8' in ch_names:
        t7_idx = ch_names.index('T7')
        t8_idx = ch_names.index('T8')
        mixing_matrix[t7_idx, 2] = 1.0
        mixing_matrix[t8_idx, 2] = 0.8
        # Reduce activity at other channels for this component
        for i, ch in enumerate(ch_names):
            if ch not in ['T7', 'T8']:
                mixing_matrix[i, 2] *= 0.1
    
    # Update the ICA mixing matrix
    ica.mixing_ = mixing_matrix
    
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