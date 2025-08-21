#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV to JSON Reorganization Script for Music Preference Ratings

Converts the behavioral ratings CSV file from the current structure:
- Rows: Song IDs (1-1, 1-2, 1-3, etc.)
- Columns: Participant-Question combinations (P_1 Q1, P_1 Q2, etc.)
- Values: Rating scores (1-9 scale)

To a hierarchical JSON structure:
Question_number → Participants → SongID

Usage: python reorganize_ratings_csv_to_json.py
"""

import pandas as pd
import json
import os
from collections import defaultdict


def reorganize_ratings_to_json(csv_file_path, output_json_path):
    """
    Reorganize ratings CSV to nested JSON structure.
    
    Args:
        csv_file_path (str): Path to input CSV file
        output_json_path (str): Path for output JSON file
    """
    
    print(f"Reading CSV file: {csv_file_path}")
    
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Display basic info about the data
    print(f"Data shape: {df.shape}")
    print(f"Song IDs: {list(df.iloc[:, 0])}")
    print(f"Column headers: {list(df.columns[1:])}")
    
    # Initialize the nested structure: Question -> Participant -> SongID -> Rating
    reorganized_data = defaultdict(lambda: defaultdict(dict))
    
    # Get song IDs from the first column
    song_ids = df.iloc[:, 0].tolist()
    
    # Process each column (skip the first column which contains song IDs)
    for col_idx, column_name in enumerate(df.columns[1:], 1):
        # Parse column name to extract participant and question
        # Format: "P_X QY" where X is participant number, Y is question number
        try:
            parts = column_name.strip().split(' ')
            participant_part = parts[0]  # "P_X"
            question_part = parts[1]     # "QY"
            
            # Extract participant number
            participant_num = participant_part.split('_')[1]  # Extract X from "P_X"
            participant_id = f"pilot_{participant_num}"
            
            # Extract question number and map to meaningful labels
            question_num = question_part[1:]  # Extract Y from "QY"
            question_mapping = {
                "1": "preference",
                "2": "pleasantness", 
                "3": "arousal",
                "4": "chills"
            }
            question_id = question_mapping.get(question_num, f"Q{question_num}")
            
            print(f"Processing {column_name} -> Participant: {participant_id}, Question: {question_id}")
            
            # Get ratings for this participant-question combination
            ratings = df.iloc[:, col_idx].tolist()
            
            # Assign ratings to each song
            for song_idx, song_id in enumerate(song_ids):
                rating = ratings[song_idx]
                reorganized_data[question_id][participant_id][song_id] = rating
                
        except (IndexError, ValueError) as e:
            print(f"Warning: Could not parse column '{column_name}': {e}")
            continue
    
    # Convert defaultdict to regular dict for JSON serialization
    final_data = {}
    for question_id in reorganized_data:
        final_data[question_id] = {}
        for participant_id in reorganized_data[question_id]:
            final_data[question_id][participant_id] = dict(reorganized_data[question_id][participant_id])
    
    # Save to JSON file
    print(f"\nSaving reorganized data to: {output_json_path}")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    
    # Print summary statistics
    print(f"\nReorganization Summary:")
    print(f"Total questions: {len(final_data)}")
    print(f"Questions: {sorted(final_data.keys())}")
    
    if final_data:
        first_question = list(final_data.keys())[0]
        print(f"Participants per question: {len(final_data[first_question])}")
        print(f"Participants: {sorted(final_data[first_question].keys())}")
        
        first_participant = list(final_data[first_question].keys())[0]
        print(f"Songs per participant: {len(final_data[first_question][first_participant])}")
        print(f"Song IDs: {sorted(final_data[first_question][first_participant].keys())}")
    
    return final_data


def display_sample_data(data, max_songs=3):
    """Display a sample of the reorganized data for verification."""
    print(f"\n{'='*60}")
    print("SAMPLE OF REORGANIZED DATA:")
    print(f"{'='*60}")
    
    for question_id in sorted(data.keys())[:2]:  # Show first 2 questions
        print(f"\n{question_id}:")
        for participant_id in sorted(data[question_id].keys())[:3]:  # Show first 3 participants
            print(f"  {participant_id}:")
            songs_shown = 0
            for song_id in sorted(data[question_id][participant_id].keys()):
                if songs_shown >= max_songs:
                    print(f"    ... (and {len(data[question_id][participant_id]) - max_songs} more songs)")
                    break
                rating = data[question_id][participant_id][song_id]
                print(f"    {song_id}: {rating}")
                songs_shown += 1
        if len(data[question_id]) > 3:
            print(f"  ... (and {len(data[question_id]) - 3} more participants)")


def main():
    # File paths
    csv_file = "/Users/tongshan/Documents/music_preference/data/Organized Behavioral Folder - Ratings.csv"
    output_file = "/Users/tongshan/Documents/music_preference/data/beh_ratings.json"
    
    # Check if input file exists
    if not os.path.exists(csv_file):
        print(f"Error: Input CSV file not found: {csv_file}")
        return
    
    try:
        # Reorganize the data
        reorganized_data = reorganize_ratings_to_json(csv_file, output_file)
        
        # Display sample data for verification
        display_sample_data(reorganized_data)
        
        print(f"\n✓ Successfully converted CSV to JSON!")
        print(f"✓ Output saved to: {output_file}")
        
        # Verify the JSON can be read back
        print(f"\n✓ Verifying JSON file integrity...")
        with open(output_file, 'r') as f:
            test_load = json.load(f)
        print(f"✓ JSON file is valid and contains {len(test_load)} questions")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()