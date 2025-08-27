#!/usr/bin/env python3
"""
Individual Song-Level Neural Coupling vs Preference Plot
======================================================

Creates a focused visualization of the relationship between RC1-spectral flux 
correlations and preference ratings at the song level, showing how different 
musical pieces relate neural-acoustic coupling to subjective preference.

This script generates a detailed scatter plot with song labels, statistics,
and interpretive annotations to highlight song-specific patterns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

def create_song_level_plot():
    """Create individual song-level neural coupling vs preference plot"""
    
    # Paths
    base_path = Path('/Users/tongshan/Documents/music_preference')
    results_path = base_path / 'output' / 'rc1_preference_analysis'
    
    # Load combined data
    data_file = results_path / 'rc1_spectral_flux_preference_combined.csv'
    if not data_file.exists():
        raise FileNotFoundError(f"Combined data not found: {data_file}")
    
    combined_data = pd.read_csv(data_file)
    
    print(f"📊 Loaded {len(combined_data)} subject-song observations")
    
    # Compute song-level means and statistics
    song_stats = combined_data.groupby('Song').agg({
        'Correlation': ['mean', 'std', 'count'],
        'Preference': ['mean', 'std']
    }).round(4)
    
    # Flatten column names
    song_stats.columns = ['_'.join(col).strip() for col in song_stats.columns]
    
    # Get mean values for plotting
    song_means = combined_data.groupby('Song')[['Correlation', 'Preference']].mean()
    
    print(f"📊 Analyzing {len(song_means)} songs")
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Create scatter plot
    scatter = plt.scatter(song_means['Correlation'], song_means['Preference'], 
                         s=120, alpha=0.7, c='steelblue', edgecolors='navy', linewidth=1.5)
    
    # Add song labels for all points
    for song, row in song_means.iterrows():
        plt.annotate(song, 
                    (row['Correlation'], row['Preference']), 
                    xytext=(8, 8), 
                    textcoords='offset points', 
                    fontsize=11,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.1', color='black', alpha=0.6))
    
    # Add trend line
    X = song_means['Correlation'].values.reshape(-1, 1)
    y = song_means['Preference'].values
    reg = LinearRegression().fit(X, y)
    
    x_range = np.linspace(song_means['Correlation'].min() * 1.1, 
                         song_means['Correlation'].max() * 1.1, 100)
    y_trend = reg.predict(x_range.reshape(-1, 1))
    
    plt.plot(x_range, y_trend, 'r--', linewidth=2.5, alpha=0.8, 
             label=f'Trend Line (slope = {reg.coef_[0]:.1f})')
    
    # Compute and display correlation
    r, p = pearsonr(song_means['Correlation'], song_means['Preference'])
    
    # Add vertical and horizontal reference lines
    plt.axhline(y=song_means['Preference'].mean(), color='gray', linestyle=':', alpha=0.6, 
                label=f'Mean Preference ({song_means["Preference"].mean():.1f})')
    plt.axvline(x=song_means['Correlation'].mean(), color='gray', linestyle=':', alpha=0.6,
                label=f'Mean Neural Coupling ({song_means["Correlation"].mean():.4f})')
    
    # Customize plot
    plt.title(f'Song-Level Analysis: RC1-Spectral Flux Coupling vs Music Preference\n' + 
              f'Correlation: r = {r:.3f}, p = {p:.3f} (n = {len(song_means)} songs)',
              fontsize=16, fontweight='bold', pad=20)
    
    plt.xlabel('Mean RC1-Spectral Flux Correlation\n(Neural-Acoustic Coupling)', 
               fontsize=14, fontweight='bold')
    plt.ylabel('Mean Preference Rating (1-9 scale)', 
               fontsize=14, fontweight='bold')
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='-')
    
    # Add legend
    plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Set axis limits with padding
    x_margin = (song_means['Correlation'].max() - song_means['Correlation'].min()) * 0.15
    y_margin = (song_means['Preference'].max() - song_means['Preference'].min()) * 0.1
    
    plt.xlim(song_means['Correlation'].min() - x_margin, 
             song_means['Correlation'].max() + x_margin)
    plt.ylim(song_means['Preference'].min() - y_margin, 
             song_means['Preference'].max() + y_margin)
    
    # Add quadrant labels for interpretation
    x_mid = song_means['Correlation'].mean()
    y_mid = song_means['Preference'].mean()
    
    # Quadrant annotations
    plt.text(song_means['Correlation'].max() - x_margin/2, y_mid + y_margin/2,
             'High Neural Coupling\n& High Preference', 
             fontsize=10, ha='right', va='center', 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.6))
    
    plt.text(song_means['Correlation'].min() + x_margin/2, y_mid + y_margin/2,
             'Low Neural Coupling\n& High Preference', 
             fontsize=10, ha='left', va='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.6))
    
    plt.text(song_means['Correlation'].max() - x_margin/2, y_mid - y_margin/2,
             'High Neural Coupling\n& Low Preference', 
             fontsize=10, ha='right', va='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.6))
    
    plt.text(song_means['Correlation'].min() + x_margin/2, y_mid - y_margin/2,
             'Low Neural Coupling\n& Low Preference', 
             fontsize=10, ha='left', va='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.6))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_file = results_path / 'song_level_neural_coupling_vs_preference.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Song-level plot saved: {output_file}")
    
    plt.show()
    
    # Print detailed song analysis
    print(f"\n📊 DETAILED SONG ANALYSIS:")
    print(f"{'='*60}")
    
    # Sort by neural coupling
    song_stats_sorted = song_stats.sort_values('Correlation_mean', ascending=False)
    
    print(f"\n🧠 Songs ranked by Neural Coupling (RC1-Spectral Flux Correlation):")
    print(f"{'Song':<6} {'Neural':<8} {'Preference':<10} {'N':<3} {'Category'}")
    print(f"{'='*50}")
    
    for song, row in song_stats_sorted.iterrows():
        neural = row['Correlation_mean']
        pref = row['Preference_mean']
        n = int(row['Correlation_count'])
        
        # Categorize songs
        if neural > x_mid and pref > y_mid:
            category = "High-High"
        elif neural > x_mid and pref <= y_mid:
            category = "High-Low"
        elif neural <= x_mid and pref > y_mid:
            category = "Low-High"
        else:
            category = "Low-Low"
        
        print(f"{song:<6} {neural:<8.4f} {pref:<10.2f} {n:<3} {category}")
    
    # Statistical summary
    print(f"\n📈 STATISTICAL SUMMARY:")
    print(f"{'='*40}")
    print(f"Song-level correlation: r = {r:.4f}, p = {p:.4f}")
    print(f"Significance: {'Yes' if p < 0.05 else 'No'} (α = 0.05)")
    print(f"Effect size: {'Small' if abs(r) < 0.3 else 'Medium' if abs(r) < 0.5 else 'Large'}")
    print(f"R² = {r**2:.4f} ({r**2*100:.1f}% variance explained)")
    
    # Identify interesting songs
    print(f"\n🎯 NOTABLE SONGS:")
    print(f"{'='*30}")
    
    # Highest neural coupling
    max_neural_song = song_stats_sorted.index[0]
    max_neural_stats = song_stats_sorted.iloc[0]
    print(f"Highest neural coupling: {max_neural_song} (r = {max_neural_stats['Correlation_mean']:.4f})")
    
    # Highest preference
    max_pref_song = song_stats_sorted.sort_values('Preference_mean', ascending=False).index[0]
    max_pref_stats = song_stats.loc[max_pref_song]
    print(f"Highest preference: {max_pref_song} (rating = {max_pref_stats['Preference_mean']:.2f})")
    
    # Most mismatched (high neural, low pref or vice versa)
    neural_pref_diff = song_stats_sorted['Correlation_mean'] - (song_stats_sorted['Preference_mean'] - song_means['Preference'].mean()) / (song_means['Preference'].max() - song_means['Preference'].min()) * (song_means['Correlation'].max() - song_means['Correlation'].min())
    
    return song_means, r, p

if __name__ == "__main__":
    print("🎨 Creating Individual Song-Level Neural Coupling vs Preference Plot")
    print("=" * 70)
    
    song_data, correlation, p_value = create_song_level_plot()
    
    print("=" * 70)
    print(f"✅ Song-level analysis complete!")
    print(f"📊 Key finding: r = {correlation:.4f}, p = {p_value:.4f}")