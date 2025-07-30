#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Music Presentation Script with Latin Square Randomization

This script demonstrates how to integrate the Latin square randomization
function into the music presentation experiment.

@author: Tong, Ariya
"""

import os
import sys
import random
import numpy as np

# Add the current directory to Python path to import our module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from music_randomization import (
    generate_latin_square_music_order,
    generate_balanced_music_order,
    get_music_presentation_info
)

# Import expyfun components (if available)
try:
    from expyfun import ExperimentController
    from expyfun.stimuli import window_edges
    from expyfun.io import read_wav
    EXPYFUN_AVAILABLE = True
except ImportError:
    EXPYFUN_AVAILABLE = False
    print("Warning: expyfun not available. Running in simulation mode.")


def setup_music_experiment(participant_id: str, 
                          preprocessed_dir: str = "../../music_stim/preprocesed"):
    """
    Setup music experiment with Latin square randomization.
    
    Args:
        participant_id: Identifier for the participant
        preprocessed_dir: Path to preprocessed music files
        
    Returns:
        Dictionary containing experiment setup information
    """
    print(f"Setting up music experiment for participant: {participant_id}")
    
    # Get randomized music presentation order
    try:
        presentation_info = get_music_presentation_info(participant_id, preprocessed_dir)
        
        if 'error' in presentation_info:
            print(f"Error setting up experiment: {presentation_info['error']}")
            return None
        
        print(f"Found {presentation_info['total_files']} music files")
        print(f"Randomization seed: {presentation_info['randomization_seed']}")
        
        return presentation_info
        
    except Exception as e:
        print(f"Error in setup_music_experiment: {e}")
        return None


def present_music_trial(ec, music_file_path: str, trial_number: int, total_trials: int):
    """
    Present a single music trial.
    
    Args:
        ec: ExperimentController instance
        music_file_path: Path to the music file
        trial_number: Current trial number
        total_trials: Total number of trials
    """
    if not EXPYFUN_AVAILABLE:
        # Simulation mode
        print(f"Trial {trial_number}/{total_trials}: Presenting {os.path.basename(music_file_path)}")
        return
    
    try:
        # Load and process the music file
        wave_temp, fs = read_wav(music_file_path)
        
        # Apply windowing for smooth onset/offset
        sig = window_edges(wave_temp, fs, dur=0.03)
        
        # Present the stimulus
        ec.screen_text(f'Trial {trial_number} of {total_trials}')
        ec.load_buffer(sig)
        
        # Start stimulus and wait
        trial_start_time = ec.start_stimulus()
        ec.wait_secs(10)  # Present for 10 seconds
        
        ec.stop()
        ec.trial_ok()
        
    except Exception as e:
        print(f"Error presenting trial {trial_number}: {e}")


def run_music_experiment(participant_id: str, 
                        preprocessed_dir: str = "../../music_stim/preprocesed",
                        simulation_mode: bool = False):
    """
    Run the complete music experiment with Latin square randomization.
    
    Args:
        participant_id: Identifier for the participant
        preprocessed_dir: Path to preprocessed music files
        simulation_mode: If True, run without expyfun (for testing)
    """
    print(f"Starting music experiment for {participant_id}")
    
    # Setup experiment
    experiment_info = setup_music_experiment(participant_id, preprocessed_dir)
    
    if experiment_info is None:
        print("Failed to setup experiment")
        return
    
    music_files = experiment_info['music_files']
    total_trials = len(music_files)
    
    print(f"\nPresentation order for {participant_id}:")
    for i, file_path in enumerate(music_files):
        file_name = os.path.basename(file_path)
        participant = os.path.basename(os.path.dirname(file_path))
        print(f"  {i+1:2d}. {file_name} ({participant})")
    
    if simulation_mode or not EXPYFUN_AVAILABLE:
        print("\nRunning in simulation mode...")
        for i, music_file in enumerate(music_files):
            present_music_trial(None, music_file, i+1, total_trials)
        print("Simulation completed!")
        return
    
    # Real experiment mode with expyfun
    try:
        with ExperimentController('music_preference_experiment', 
                               verbose=True, 
                               screen_num=0,
                               window_size=[1920, 1080], 
                               full_screen=False,
                               stim_db=65, 
                               stim_fs=48000,
                               session='study', 
                               version='dev',
                               check_rms='wholefile', 
                               n_channels=2,
                               force_quit=['end']) as ec:
            
            # Instructions
            ec.screen_prompt("You will now listen to music pieces.\n"
                           "Please listen carefully to each piece.\n"
                           "Click to start.", click=True)
            
            # Present each music trial
            for i, music_file in enumerate(music_files):
                present_music_trial(ec, music_file, i+1, total_trials)
                
                # Brief pause between trials
                if i < total_trials - 1:
                    ec.wait_secs(0.5)
            
            # End of experiment
            ec.screen_prompt("Experiment completed!\nThank you for participating.", click=True)
            
    except Exception as e:
        print(f"Error running experiment: {e}")


def test_different_participants():
    """
    Test randomization with different participants to show reproducibility.
    """
    participants = ["Pilot 1 Audrey", "Pilot 2  Raj", "Pilot 3 Ruiqi"]
    
    print("Testing randomization reproducibility across participants...\n")
    
    for participant in participants:
        print(f"=== {participant} ===")
        info = get_music_presentation_info(participant)
        
        if 'error' not in info:
            print(f"Seed: {info['randomization_seed']}")
            print("First 5 files:")
            for i in range(min(5, len(info['file_info']))):
                file_info = info['file_info'][i]
                print(f"  {file_info['presentation_order']}. {file_info['file_name']}")
            print()
        else:
            print(f"Error: {info['error']}\n")


if __name__ == "__main__":
    # Test the randomization function
    print("Testing music randomization integration...\n")
    
    # Test with different participants
    test_different_participants()
    
    # Run simulation for one participant
    print("Running simulation for Pilot 1 Audrey...")
    run_music_experiment("Pilot 1 Audrey", simulation_mode=True) 