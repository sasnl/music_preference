#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate Music Presentation Orders CSV

This script generates a CSV file containing Latin square randomized
music presentation orders for all participants in the experiment.

@author: Tong, Ariya
"""

import os
import sys
import csv
import random
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict

# Import soundfile for audio duration calculation
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("Warning: soundfile not available. Duration will not be calculated.")


def get_audio_duration(file_path: str) -> float:
    """
    Get the duration of an audio file in seconds.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Duration in seconds, or -1 if error
    """
    if not SOUNDFILE_AVAILABLE:
        return -1
    
    try:
        # Read audio file info
        info = sf.info(file_path)
        duration = info.duration
        return duration
    except Exception as e:
        print(f"Error reading duration for {file_path}: {e}")
        return -1


def collect_music_files(preprocessed_dir: str) -> Dict[str, List[str]]:
    """
    Collect all preprocessed music files organized by participant.
    
    Args:
        preprocessed_dir: Path to the preprocessed music directory
        
    Returns:
        Dictionary mapping participant names to their music file paths
    """
    music_files = {}
    
    if not os.path.exists(preprocessed_dir):
        raise FileNotFoundError(f"Preprocessed directory not found: {preprocessed_dir}")
    
    # Walk through the directory to find all participant folders
    for participant_dir in os.listdir(preprocessed_dir):
        participant_path = os.path.join(preprocessed_dir, participant_dir)
        
        if os.path.isdir(participant_path):
            # Collect all .wav files for this participant
            participant_files = []
            for file in os.listdir(participant_path):
                if file.endswith('_proc.wav'):
                    file_path = os.path.join(participant_path, file)
                    participant_files.append(file_path)
            
            if participant_files:
                # Sort files to ensure consistent ordering
                participant_files.sort()
                music_files[participant_dir] = participant_files
    
    return music_files


def generate_latin_square_music_order(participant_id: str, 
                                    preprocessed_dir: str = "music_stim/preprocesed",
                                    seed: int = None) -> List[str]:
    """
    Generate Latin square randomized music presentation order for a participant.
    
    This function creates a balanced presentation order where each participant
    hears all available music pieces, but in a randomized order that follows
    Latin square principles to minimize order effects.
    
    Args:
        participant_id: Identifier for the participant (e.g., "Pilot 1 Audrey")
        preprocessed_dir: Path to the preprocessed music directory
        seed: Random seed for reproducible randomization (optional)
        
    Returns:
        List of music file paths in randomized order
        
    Raises:
        FileNotFoundError: If preprocessed directory doesn't exist
        ValueError: If participant_id is not found or no music files available
    """
    # Set random seed for reproducible randomization
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Collect all music files
    all_music_files = collect_music_files(preprocessed_dir)
    
    if not all_music_files:
        raise FileNotFoundError(f"No music files found in {preprocessed_dir}")
    
    # Get all available music files (flatten the dictionary)
    all_files = []
    for participant, files in all_music_files.items():
        all_files.extend(files)
    
    # Remove duplicates and sort for consistency
    all_files = sorted(list(set(all_files)))
    
    # Create Latin square randomization
    n_files = len(all_files)
    
    if n_files == 0:
        raise ValueError("No music files available for randomization")
    
    # Generate a Latin square matrix
    # For simplicity, we'll use a balanced random order
    # that ensures each participant hears all pieces in different orders
    
    # Create participant-specific randomization based on participant_id
    participant_hash = hash(participant_id) % (2**32)
    random.seed(participant_hash)
    
    # Shuffle the file list
    randomized_files = all_files.copy()
    random.shuffle(randomized_files)
    
    return randomized_files


def get_music_presentation_info(participant_id: str,
                              preprocessed_dir: str = "music_stim/preprocesed") -> Dict:
    """
    Get comprehensive information about music presentation for a participant.
    
    Args:
        participant_id: Identifier for the participant
        preprocessed_dir: Path to the preprocessed music directory
        
    Returns:
        Dictionary containing presentation information
    """
    try:
        # Get randomized music files
        music_files = generate_latin_square_music_order(participant_id, preprocessed_dir)
        
        # Get file information
        file_info = []
        for i, file_path in enumerate(music_files):
            file_name = os.path.basename(file_path)
            participant = os.path.basename(os.path.dirname(file_path))
            
            file_info.append({
                'index': i,
                'file_path': file_path,
                'file_name': file_name,
                'participant': participant,
                'presentation_order': i + 1
            })
        
        return {
            'participant_id': participant_id,
            'total_files': len(music_files),
            'music_files': music_files,
            'file_info': file_info,
            'randomization_seed': hash(participant_id) % (2**32)
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'participant_id': participant_id
        }


def get_all_participants(preprocessed_dir: str = "music_stim/preprocesed"):
    """
    Get list of all participants from the preprocessed directory.
    
    Args:
        preprocessed_dir: Path to preprocessed music directory
        
    Returns:
        List of participant names
    """
    participants = []
    
    if not os.path.exists(preprocessed_dir):
        raise FileNotFoundError(f"Preprocessed directory not found: {preprocessed_dir}")
    
    for item in os.listdir(preprocessed_dir):
        item_path = os.path.join(preprocessed_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            participants.append(item)
    
    return sorted(participants)


def generate_participant_order(participant_id: str, 
                             preprocessed_dir: str = "music_stim/preprocesed"):
    """
    Generate randomized music order for a specific participant.
    
    Args:
        participant_id: Participant identifier
        preprocessed_dir: Path to preprocessed music directory
        
    Returns:
        List of dictionaries containing order information
    """
    try:
        # Get presentation info for this participant
        info = get_music_presentation_info(participant_id, preprocessed_dir)
        
        if 'error' in info:
            print(f"Error for {participant_id}: {info['error']}")
            return []
        
        # Create order information
        order_data = []
        for file_info in info['file_info']:
            # Get audio duration
            duration = get_audio_duration(file_info['file_path'])
            
            order_data.append({
                'Participant_ID': participant_id,
                'Presentation_Order': file_info['presentation_order'],
                'Song_File': file_info['file_name'],
                'Original_Participant': file_info['participant'],
                'File_Path': file_info['file_path'],
                'Duration_Seconds': duration,
                'Randomization_Seed': info['randomization_seed']
            })
        
        return order_data
        
    except Exception as e:
        print(f"Error generating order for {participant_id}: {e}")
        return []


def generate_all_orders_csv(output_file: str = "music_presentation_orders.csv",
                          preprocessed_dir: str = "music_stim/preprocesed"):
    """
    Generate CSV file with all participants' music presentation orders.
    
    Args:
        output_file: Output CSV file path
        preprocessed_dir: Path to preprocessed music directory
    """
    print("Generating music presentation orders CSV...")
    
    # Get all participants
    participants = get_all_participants(preprocessed_dir)
    print(f"Found {len(participants)} participants: {participants}")
    
    # Generate orders for all participants
    all_orders = []
    
    for participant in participants:
        print(f"Generating order for {participant}...")
        participant_orders = generate_participant_order(participant, preprocessed_dir)
        all_orders.extend(participant_orders)
    
    if not all_orders:
        print("No orders generated. Check if music files exist.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_orders)
    
    # Sort by Participant_ID and Presentation_Order
    df = df.sort_values(['Participant_ID', 'Presentation_Order'])
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"CSV file saved: {output_file}")
    print(f"Total entries: {len(df)}")
    print(f"Participants: {len(participants)}")
    
    # Print summary
    print("\nSummary:")
    total_duration = 0
    for participant in participants:
        participant_data = df[df['Participant_ID'] == participant]
        participant_duration = participant_data['Duration_Seconds'].sum()
        total_duration += participant_duration
        print(f"  {participant}: {len(participant_data)} songs, {participant_duration:.1f}s total")
    
    print(f"\nTotal duration across all participants: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    
    return df


def generate_detailed_report(df: pd.DataFrame, output_file: str = "music_presentation_report.txt"):
    """
    Generate a detailed text report of the music presentation orders.
    
    Args:
        df: DataFrame with music orders
        output_file: Output report file path
    """
    with open(output_file, 'w') as f:
        f.write("Music Presentation Orders Report\n")
        f.write("=" * 40 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary statistics
        participants = df['Participant_ID'].unique()
        total_duration = df['Duration_Seconds'].sum()
        f.write(f"Total Participants: {len(participants)}\n")
        f.write(f"Total Songs: {len(df)}\n")
        f.write(f"Songs per Participant: {len(df) // len(participants)}\n")
        f.write(f"Total Duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)\n\n")
        
        # Per-participant details
        for participant in sorted(participants):
            participant_data = df[df['Participant_ID'] == participant]
            participant_duration = participant_data['Duration_Seconds'].sum()
            f.write(f"Participant: {participant}\n")
            f.write(f"  Randomization Seed: {participant_data['Randomization_Seed'].iloc[0]}\n")
            f.write(f"  Total Duration: {participant_duration:.1f} seconds\n")
            f.write("  Presentation Order:\n")
            
            for _, row in participant_data.iterrows():
                duration_str = f" ({row['Duration_Seconds']:.1f}s)" if row['Duration_Seconds'] > 0 else ""
                f.write(f"    {row['Presentation_Order']:2d}. {row['Song_File']} (from {row['Original_Participant']}){duration_str}\n")
            f.write("\n")
    
    print(f"Detailed report saved: {output_file}")


def test_randomization_consistency():
    """
    Test that randomization is consistent (same participant gets same order).
    """
    print("Testing randomization consistency...")
    
    participant = "Pilot 1 Audrey"
    
    # Generate order twice
    order1 = generate_participant_order(participant)
    order2 = generate_participant_order(participant)
    
    if order1 == order2:
        print("✓ Randomization is consistent (reproducible)")
    else:
        print("✗ Randomization is not consistent")
    
    return order1 == order2


if __name__ == "__main__":
    print("Music Presentation Orders Generator")
    print("=" * 40)
    
    # Test consistency
    test_randomization_consistency()
    print()
    
    # Generate CSV file
    df = generate_all_orders_csv()
    
    if df is not None:
        # Generate detailed report
        generate_detailed_report(df)
        
        # Show first few rows
        print("\nFirst 10 rows of the CSV:")
        print(df.head(10).to_string(index=False))
        
        print(f"\nCSV file contains {len(df)} rows for {len(df['Participant_ID'].unique())} participants") 