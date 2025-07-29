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
import pandas as pd
from datetime import datetime

# Add the current directory to Python path to import our module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from music_randomization import (
    generate_latin_square_music_order,
    get_music_presentation_info
)


def get_all_participants(preprocessed_dir: str = "../../music_stim/preprocesed"):
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
                             preprocessed_dir: str = "../../music_stim/preprocesed"):
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
            order_data.append({
                'Participant_ID': participant_id,
                'Presentation_Order': file_info['presentation_order'],
                'Song_File': file_info['file_name'],
                'Original_Participant': file_info['participant'],
                'File_Path': file_info['file_path'],
                'Randomization_Seed': info['randomization_seed']
            })
        
        return order_data
        
    except Exception as e:
        print(f"Error generating order for {participant_id}: {e}")
        return []


def generate_all_orders_csv(output_file: str = "music_presentation_orders.csv",
                          preprocessed_dir: str = "../../music_stim/preprocesed"):
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
    for participant in participants:
        participant_data = df[df['Participant_ID'] == participant]
        print(f"  {participant}: {len(participant_data)} songs")
    
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
        f.write(f"Total Participants: {len(participants)}\n")
        f.write(f"Total Songs: {len(df)}\n")
        f.write(f"Songs per Participant: {len(df) // len(participants)}\n\n")
        
        # Per-participant details
        for participant in sorted(participants):
            participant_data = df[df['Participant_ID'] == participant]
            f.write(f"Participant: {participant}\n")
            f.write(f"  Randomization Seed: {participant_data['Randomization_Seed'].iloc[0]}\n")
            f.write("  Presentation Order:\n")
            
            for _, row in participant_data.iterrows():
                f.write(f"    {row['Presentation_Order']:2d}. {row['Song_File']} (from {row['Original_Participant']})\n")
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