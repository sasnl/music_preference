#!/usr/bin/env python3
"""
Individual RC1-Spectral Flux Correlation Matrix Plot
==================================================

Creates a focused, publication-quality visualization of the RC1-spectral flux 
correlation matrix showing correlations between each subject-song combination.

This script loads the correlation results and creates a detailed heatmap
with proper annotations and statistical information.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_correlation_matrix_plot():
    """Create individual correlation matrix plot"""
    
    # Paths
    base_path = Path('/Users/tongshan/Documents/music_preference')
    results_path = base_path / 'output' / 'rc1_spectral_flux_analysis'
    
    # Load correlation results
    summary_file = results_path / 'rc1_spectral_flux_summary.csv'
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")
    
    # Load summary data
    summary_df = pd.read_csv(summary_file)
    
    print(f"Loaded {len(summary_df)} correlation values")
    print(f"Mean correlation: {summary_df['Correlation'].mean():.4f}")
    print(f"Std correlation: {summary_df['Correlation'].std():.4f}")
    
    # Create pivot table for heatmap
    pivot_df = summary_df.pivot(index='Subject', columns='Song', values='Correlation')
    
    # Create figure
    plt.figure(figsize=(16, 10))
    
    # Create heatmap
    mask = pivot_df.isnull()  # Mask for missing values
    
    ax = sns.heatmap(pivot_df, 
                     annot=True, 
                     fmt='.3f',
                     cmap='RdBu_r', 
                     center=0,
                     square=True,
                     linewidths=0.5,
                     cbar_kws={
                         'shrink': 0.8,
                         'label': 'RC1-Spectral Flux Correlation'
                     },
                     annot_kws={'size': 10},
                     mask=mask)
    
    # Customize the plot
    plt.title('RC1-Spectral Flux Correlation Matrix\n' + 
              f'Mean: {summary_df["Correlation"].mean():.4f} ± {summary_df["Correlation"].std():.4f}',
              fontsize=16, fontweight='bold', pad=20)
    
    plt.xlabel('Song', fontsize=14, fontweight='bold')
    plt.ylabel('Subject', fontsize=14, fontweight='bold')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add grid for better readability
    ax.set_axisbelow(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_file = results_path / 'rc1_spectral_flux_correlation_matrix_individual.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Individual correlation matrix saved: {output_file}")
    
    plt.show()
    
    # Print statistics
    print("\n📊 Correlation Matrix Statistics:")
    print(f"   Shape: {pivot_df.shape}")
    print(f"   Total values: {pivot_df.notna().sum().sum()}")
    print(f"   Missing values: {pivot_df.isna().sum().sum()}")
    
    # Find strongest correlations
    print("\n🏆 Strongest Correlations:")
    summary_sorted = summary_df.nlargest(5, 'Correlation')
    for i, (_, row) in enumerate(summary_sorted.iterrows(), 1):
        print(f"   {i}. {row['Subject']}-{row['Song']}: r = {row['Correlation']:.4f}")
    
    print("\n🔻 Most Negative Correlations:")
    summary_sorted_neg = summary_df.nsmallest(5, 'Correlation')
    for i, (_, row) in enumerate(summary_sorted_neg.iterrows(), 1):
        print(f"   {i}. {row['Subject']}-{row['Song']}: r = {row['Correlation']:.4f}")
    
    # Subject averages
    print("\n👤 Subject Averages:")
    subject_means = summary_df.groupby('Subject')['Correlation'].mean().sort_values(ascending=False)
    for subject, mean_corr in subject_means.items():
        print(f"   {subject}: {mean_corr:.4f}")
    
    # Song averages
    print("\n🎵 Song Averages:")
    song_means = summary_df.groupby('Song')['Correlation'].mean().sort_values(ascending=False)
    for song, mean_corr in song_means.head(10).items():
        print(f"   {song}: {mean_corr:.4f}")
    
    return pivot_df

if __name__ == "__main__":
    print("🎨 Creating Individual RC1-Spectral Flux Correlation Matrix")
    print("=" * 60)
    
    correlation_matrix = create_correlation_matrix_plot()
    
    print("=" * 60)
    print("✅ Individual correlation matrix plot complete!")