#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preference Ratings Scatter Plot Visualization

Creates scatter plots showing preference ratings across all songs, with each 
participant represented by a different color. Visualizes rating patterns and 
individual differences across the song collection.

Usage: python plot_preference_scatter.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from matplotlib.patches import Rectangle

def load_behavioral_data(data_path):
    """Load behavioral ratings from JSON file."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data

def create_preference_dataframe(data):
    """Convert preference data to pandas DataFrame for plotting."""
    preference_data = data['preference']
    
    # Create a list to store all ratings with metadata
    ratings_list = []
    
    for subject in preference_data:
        for song_id, rating in preference_data[subject].items():
            # Skip null/None ratings
            if rating is not None:
                ratings_list.append({
                    'subject': subject,
                    'song_id': song_id,
                    'rating': rating,
                    'artist': int(song_id.split('-')[0]),  # Extract artist number as int
                    'song_num': int(song_id.split('-')[1])  # Extract song number as int
                })
    
    return pd.DataFrame(ratings_list)

def create_song_order():
    """Create ordered list of all song IDs."""
    song_ids = []
    for artist in range(1, 6):  # Artists 1-5
        for song in range(1, 4):  # Songs 1-3 per artist
            song_ids.append(f"{artist}-{song}")
    return song_ids

def plot_preference_scatter_basic(df, output_dir):
    """Create basic scatter plot with all participants."""
    plt.figure(figsize=(16, 10))
    
    # Get ordered song list
    song_order = create_song_order()
    song_positions = {song: i for i, song in enumerate(song_order)}
    
    # Map song IDs to x-axis positions
    df['x_position'] = df['song_id'].map(song_positions)
    
    # Create scatter plot for each subject
    subjects = sorted(df['subject'].unique())
    colors = plt.cm.Set1(np.linspace(0, 1, len(subjects)))
    
    for i, subject in enumerate(subjects):
        subject_data = df[df['subject'] == subject]
        
        # Add small random jitter to avoid overlapping points
        jitter = np.random.normal(0, 0.05, len(subject_data))
        
        plt.scatter(subject_data['x_position'] + jitter, subject_data['rating'], 
                   c=[colors[i]], label=subject, s=80, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Customize plot
    plt.xlabel('Song ID', fontsize=14, fontweight='bold')
    plt.ylabel('Preference Rating', fontsize=14, fontweight='bold')
    plt.title('Preference Ratings by Song and Participant', fontsize=16, fontweight='bold')
    
    # Set x-axis labels
    plt.xticks(range(len(song_order)), song_order, rotation=45)
    plt.xlim(-0.5, len(song_order) - 0.5)
    
    # Set y-axis
    plt.ylim(0.5, 9.5)
    plt.yticks(range(1, 10))
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend(title='Participant', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add artist separators
    for i in range(1, 5):  # Between artists
        plt.axvline(x=i*3 - 0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Artist labels removed per user request
    
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, 'preference_scatter_basic.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def plot_preference_scatter_highlighted(df, output_dir):
    """Create scatter plot with preferred artists highlighted for each participant."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    # Get ordered song list
    song_order = create_song_order()
    song_positions = {song: i for i, song in enumerate(song_order)}
    df['x_position'] = df['song_id'].map(song_positions)
    
    subjects = sorted(df['subject'].unique())
    colors = plt.cm.Set1(np.linspace(0, 1, len(subjects)))
    
    for i, subject in enumerate(subjects):
        ax = axes[i]
        subject_data = df[df['subject'] == subject]
        
        # Get preferred artist for this subject
        subject_num = int(subject.split('_')[1])
        preferred_artist = subject_num
        
        # Plot all points in light gray first
        ax.scatter(subject_data['x_position'], subject_data['rating'], 
                  c='lightgray', s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Highlight preferred artist songs
        preferred_data = subject_data[subject_data['artist'] == preferred_artist]
        non_preferred_data = subject_data[subject_data['artist'] != preferred_artist]
        
        # Plot preferred songs in bright color
        ax.scatter(preferred_data['x_position'], preferred_data['rating'], 
                  c=[colors[i]], s=120, alpha=0.9, edgecolors='black', linewidth=1.5,
                  label=f'Preferred (Artist {preferred_artist})')
        
        # Plot non-preferred songs in darker gray
        ax.scatter(non_preferred_data['x_position'], non_preferred_data['rating'], 
                  c='darkgray', s=80, alpha=0.7, edgecolors='black', linewidth=0.5,
                  label='Non-preferred')
        
        # Customize subplot
        ax.set_xlabel('Song ID', fontsize=10)
        ax.set_ylabel('Preference Rating', fontsize=10)
        ax.set_title(f'{subject} (Prefers Artist {preferred_artist})', fontsize=12, fontweight='bold')
        
        # Set axis properties
        ax.set_xticks(range(len(song_order)))
        ax.set_xticklabels(song_order, rotation=45, fontsize=8)
        ax.set_xlim(-0.5, len(song_order) - 0.5)
        ax.set_ylim(0.5, 9.5)
        ax.set_yticks(range(1, 10))
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(fontsize=8, loc='upper right')
        
        # Add artist separators
        for j in range(1, 5):
            ax.axvline(x=j*3 - 0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Highlight preferred artist region
        preferred_start = (preferred_artist - 1) * 3 - 0.5
        preferred_end = preferred_artist * 3 - 0.5
        ax.add_patch(Rectangle((preferred_start, 0.5), preferred_end - preferred_start, 9, 
                              facecolor=colors[i], alpha=0.1, zorder=0))
    
    # Remove the empty subplot
    axes[-1].remove()
    
    plt.suptitle('Preference Ratings by Participant\n(Preferred Artists Highlighted)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, 'preference_scatter_highlighted.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def plot_preference_scatter_lines(df, output_dir):
    """Create scatter plot with connecting lines for each participant."""
    plt.figure(figsize=(16, 10))
    
    # Get ordered song list
    song_order = create_song_order()
    song_positions = {song: i for i, song in enumerate(song_order)}
    df['x_position'] = df['song_id'].map(song_positions)
    
    subjects = sorted(df['subject'].unique())
    colors = plt.cm.Set1(np.linspace(0, 1, len(subjects)))
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]  # Different line styles
    
    for i, subject in enumerate(subjects):
        subject_data = df[df['subject'] == subject].sort_values('x_position')
        
        # Plot line connecting all points for this subject
        plt.plot(subject_data['x_position'], subject_data['rating'], 
                color=colors[i], linestyle=line_styles[i % len(line_styles)], 
                linewidth=2, alpha=0.7, label=f'{subject} (line)')
        
        # Plot scatter points
        plt.scatter(subject_data['x_position'], subject_data['rating'], 
                   c=[colors[i]], s=100, alpha=0.9, edgecolors='black', 
                   linewidth=1, zorder=5)
    
    # Customize plot
    plt.xlabel('Song ID', fontsize=14, fontweight='bold')
    plt.ylabel('Preference Rating', fontsize=14, fontweight='bold')
    plt.title('Preference Rating Patterns Across Songs\n(Connected by Lines)', fontsize=16, fontweight='bold')
    
    # Set x-axis labels
    plt.xticks(range(len(song_order)), song_order, rotation=45)
    plt.xlim(-0.5, len(song_order) - 0.5)
    
    # Set y-axis
    plt.ylim(0.5, 9.5)
    plt.yticks(range(1, 10))
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend(title='Participant', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add artist separators
    for i in range(1, 5):
        plt.axvline(x=i*3 - 0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Add artist labels
    artist_centers = [1, 4, 7, 10, 13]
    for i, center in enumerate(artist_centers):
        plt.text(center, 9.7, f'Artist {i+1}', ha='center', va='bottom', 
                fontweight='bold', fontsize=12, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, 'preference_scatter_lines.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def plot_preference_heatmap(df, output_dir):
    """Create a heatmap showing all preference ratings."""
    # Pivot the data to create a matrix
    heatmap_data = df.pivot(index='subject', columns='song_id', values='rating')
    
    # Reorder columns to match song order
    song_order = create_song_order()
    heatmap_data = heatmap_data[song_order]
    
    plt.figure(figsize=(16, 8))
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, cmap='RdYlBu_r', center=5, 
                vmin=1, vmax=9, cbar_kws={'label': 'Preference Rating'},
                fmt='.0f', linewidths=0.5, linecolor='white')
    
    plt.xlabel('Song ID', fontsize=14, fontweight='bold')
    plt.ylabel('Participant', fontsize=14, fontweight='bold')
    plt.title('Preference Ratings Heatmap\n(Red = High Preference, Blue = Low Preference)', 
              fontsize=16, fontweight='bold')
    
    # Add artist separators
    for i in range(1, 5):
        plt.axvline(x=i*3, color='black', linewidth=2)
    
    # Add artist labels
    artist_centers = [1.5, 4.5, 7.5, 10.5, 13.5]
    for i, center in enumerate(artist_centers):
        plt.text(center, -0.5, f'Artist {i+1}', ha='center', va='top', 
                fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, 'preference_heatmap.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def generate_summary_statistics(df, output_dir):
    """Generate summary statistics for the scatter plot analysis."""
    print("=== PREFERENCE SCATTER PLOT ANALYSIS ===\n")
    
    # Song-wise statistics
    print("Song-wise Statistics:")
    print("-" * 60)
    song_stats = df.groupby('song_id')['rating'].agg(['mean', 'std', 'min', 'max']).round(2)
    song_stats = song_stats.reindex(create_song_order())  # Order by song sequence
    
    for song_id, stats in song_stats.iterrows():
        print(f"{song_id:>6} | Mean: {stats['mean']:5.2f} | Std: {stats['std']:5.2f} | "
              f"Range: {stats['min']:1.0f}-{stats['max']:1.0f}")
    
    # Artist-wise statistics
    print(f"\n{'='*60}")
    print("Artist-wise Statistics:")
    print("-" * 60)
    artist_stats = df.groupby('artist')['rating'].agg(['mean', 'std', 'count']).round(2)
    
    for artist, stats in artist_stats.iterrows():
        print(f"Artist {artist} | Mean: {stats['mean']:5.2f} | Std: {stats['std']:5.2f} | "
              f"Ratings: {stats['count']:2.0f}")
    
    # Preference validation
    print(f"\n{'='*60}")
    print("Preference Validation (Expected vs Actual):")
    print("-" * 60)
    
    for subject in sorted(df['subject'].unique()):
        subject_data = df[df['subject'] == subject]
        subject_num = int(subject.split('_')[1])
        expected_preferred = subject_num
        
        # Calculate mean rating for each artist
        artist_means = subject_data.groupby('artist')['rating'].mean()
        actual_preferred = artist_means.idxmax()
        
        status = "✓" if actual_preferred == expected_preferred else "✗"
        print(f"{subject} | Expected: Artist {expected_preferred} | "
              f"Actual highest: Artist {actual_preferred} | {status}")
    
    return True

def main():
    """Main function to generate all scatter plot visualizations."""
    # File paths
    data_file = "/Users/tongshan/Documents/music_preference/data/beh_ratings.json"
    output_dir = "/Users/tongshan/Documents/music_preference/output"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading behavioral data...")
    data = load_behavioral_data(data_file)
    df = create_preference_dataframe(data)
    
    print(f"Loaded {len(df)} preference ratings from {len(df['subject'].unique())} subjects")
    print(f"Songs: {len(df['song_id'].unique())} total")
    
    # Set plotting style
    plt.style.use('default')
    sns.set_palette("Set1")
    
    # Generate basic scatter plot only
    print("\nGenerating preference scatter plot...")
    
    plot_files = []
    
    # Basic scatter plot
    print("  Generating basic scatter plot...")
    plot_files.append(plot_preference_scatter_basic(df, output_dir))
    
    print(f"\n✓ Scatter plot analysis complete!")
    print(f"\nGenerated files:")
    for file_path in plot_files:
        print(f"  - {file_path}")
    
    return plot_files

if __name__ == "__main__":
    main()