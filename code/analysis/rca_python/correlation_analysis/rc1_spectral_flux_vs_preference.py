#!/usr/bin/env python3
"""
RC1-Spectral Flux Correlation vs Music Preference Analysis
=========================================================

This script analyzes the relationship between RC1-spectral flux neural-acoustic 
coupling and behavioral preference ratings. It examines whether stronger 
neural tracking of spectral dynamics is associated with higher music preference.

Key analyses:
- Load RC1-spectral flux correlations and preference ratings
- Compute correlation between neural coupling and preference scores
- Create scatter plots and statistical visualizations
- Analyze patterns at subject and song levels
- Test for preference-dependent neural-acoustic coupling

Author: Music Preference Analysis Pipeline
Date: 2025-08-26
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

class RC1SpectralFluxVsPreference:
    """
    Analyze relationship between RC1-spectral flux correlations and preference ratings.
    """
    
    def __init__(self, base_path='/Users/tongshan/Documents/music_preference'):
        self.base_path = Path(base_path)
        self.results_path = self.base_path / 'output' / 'rc1_spectral_flux_analysis'
        self.output_path = self.base_path / 'output' / 'rc1_preference_analysis'
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.spectral_flux_correlations = None
        self.preference_ratings = None
        self.combined_data = None
        
    def load_behavioral_data(self):
        """Load behavioral preference ratings"""
        print("📊 Loading Behavioral Preference Data")
        print("=" * 40)
        
        # Load behavioral ratings
        behavioral_file = self.base_path / 'data' / 'beh_ratings.json'
        
        if not behavioral_file.exists():
            raise FileNotFoundError(f"Behavioral data not found: {behavioral_file}")
        
        with open(behavioral_file, 'r') as f:
            behavioral_data = json.load(f)
        
        # Extract preference ratings
        preference_data = behavioral_data['preference']
        
        # Convert to DataFrame format
        preference_rows = []
        for subject, ratings in preference_data.items():
            for song, rating in ratings.items():
                if rating is not None:  # Skip null ratings
                    preference_rows.append({
                        'Subject': subject,
                        'Song': song,
                        'Preference': rating
                    })
        
        self.preference_ratings = pd.DataFrame(preference_rows)
        
        print(f"✅ Loaded {len(self.preference_ratings)} preference ratings")
        print(f"   Subjects: {self.preference_ratings['Subject'].nunique()}")
        print(f"   Songs: {self.preference_ratings['Song'].nunique()}")
        print(f"   Preference range: {self.preference_ratings['Preference'].min()}-{self.preference_ratings['Preference'].max()}")
        print(f"   Mean preference: {self.preference_ratings['Preference'].mean():.2f}")
        
        return self.preference_ratings
    
    def load_spectral_flux_correlations(self):
        """Load RC1-spectral flux correlation data"""
        print("🧠 Loading RC1-Spectral Flux Correlations")
        print("=" * 45)
        
        # Load correlation summary
        corr_file = self.results_path / 'rc1_spectral_flux_summary.csv'
        
        if not corr_file.exists():
            raise FileNotFoundError(f"Correlation data not found: {corr_file}")
        
        self.spectral_flux_correlations = pd.read_csv(corr_file)
        
        print(f"✅ Loaded {len(self.spectral_flux_correlations)} RC1-spectral flux correlations")
        print(f"   Correlation range: {self.spectral_flux_correlations['Correlation'].min():.4f} to {self.spectral_flux_correlations['Correlation'].max():.4f}")
        print(f"   Mean correlation: {self.spectral_flux_correlations['Correlation'].mean():.4f}")
        
        return self.spectral_flux_correlations
    
    def merge_data(self):
        """Merge spectral flux correlations with preference ratings"""
        print("🔗 Merging Neural and Behavioral Data")
        print("=" * 40)
        
        if self.spectral_flux_correlations is None or self.preference_ratings is None:
            raise ValueError("Must load both spectral flux correlations and preference data first")
        
        # Merge on Subject and Song
        self.combined_data = pd.merge(
            self.spectral_flux_correlations,
            self.preference_ratings,
            on=['Subject', 'Song'],
            how='inner'
        )
        
        print(f"✅ Merged data: {len(self.combined_data)} subject-song pairs")
        print(f"   Lost {len(self.spectral_flux_correlations) - len(self.combined_data)} pairs due to missing preference data")
        
        # Print basic statistics
        print(f"\n📊 Combined Data Summary:")
        print(f"   RC1-Spectral Flux correlations: {self.combined_data['Correlation'].mean():.4f} ± {self.combined_data['Correlation'].std():.4f}")
        print(f"   Preference ratings: {self.combined_data['Preference'].mean():.2f} ± {self.combined_data['Preference'].std():.2f}")
        
        return self.combined_data
    
    def compute_preference_correlation(self):
        """Compute correlation between RC1-spectral flux correlation and preference"""
        print("📈 Computing Neural-Preference Correlations")
        print("=" * 45)
        
        if self.combined_data is None:
            raise ValueError("Must merge data first")
        
        # Overall correlation
        pearson_r, pearson_p = pearsonr(self.combined_data['Correlation'], self.combined_data['Preference'])
        spearman_r, spearman_p = spearmanr(self.combined_data['Correlation'], self.combined_data['Preference'])
        
        print(f"📊 Overall Neural-Preference Correlations:")
        print(f"   Pearson:  r = {pearson_r:.4f}, p = {pearson_p:.4f}")
        print(f"   Spearman: ρ = {spearman_r:.4f}, p = {spearman_p:.4f}")
        
        # Subject-level correlations
        print(f"\n👤 Subject-Level Correlations:")
        subject_correlations = {}
        
        for subject in self.combined_data['Subject'].unique():
            subject_data = self.combined_data[self.combined_data['Subject'] == subject]
            
            if len(subject_data) > 3:  # Need at least 4 points for meaningful correlation
                r, p = pearsonr(subject_data['Correlation'], subject_data['Preference'])
                subject_correlations[subject] = {'r': r, 'p': p, 'n': len(subject_data)}
                print(f"   {subject}: r = {r:.4f}, p = {p:.4f} (n = {len(subject_data)})")
            else:
                print(f"   {subject}: insufficient data (n = {len(subject_data)})")
        
        # Song-level analysis (across subjects)
        print(f"\n🎵 Song-Level Analysis:")
        song_stats = self.combined_data.groupby('Song').agg({
            'Correlation': ['mean', 'std', 'count'],
            'Preference': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        song_stats.columns = ['_'.join(col).strip() for col in song_stats.columns]
        
        # Show top songs by neural-acoustic coupling
        top_coupling_songs = song_stats.sort_values('Correlation_mean', ascending=False).head(5)
        print(f"   Top 5 songs by mean RC1-spectral flux correlation:")
        for song, row in top_coupling_songs.iterrows():
            print(f"     {song}: r = {row['Correlation_mean']:.4f}, pref = {row['Preference_mean']:.2f} (n = {row['Correlation_count']})")
        
        return {
            'overall': {'pearson_r': pearson_r, 'pearson_p': pearson_p, 
                       'spearman_r': spearman_r, 'spearman_p': spearman_p},
            'subjects': subject_correlations,
            'songs': song_stats
        }
    
    def create_preference_correlation_plots(self):
        """Create comprehensive visualization of neural-preference relationships"""
        print("🎨 Creating Neural-Preference Visualizations")
        print("=" * 45)
        
        if self.combined_data is None:
            raise ValueError("Must merge data first")
        
        # Create comprehensive figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 1. Overall scatter plot
        ax1 = axes[0]
        
        # Create scatter plot with color-coding by subject
        subjects = self.combined_data['Subject'].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(subjects)))
        
        for i, subject in enumerate(subjects):
            subject_data = self.combined_data[self.combined_data['Subject'] == subject]
            ax1.scatter(subject_data['Correlation'], subject_data['Preference'], 
                       c=[colors[i]], label=subject, alpha=0.7, s=60)
        
        # Add trend line
        X = self.combined_data['Correlation'].values.reshape(-1, 1)
        y = self.combined_data['Preference'].values
        reg = LinearRegression().fit(X, y)
        x_trend = np.linspace(self.combined_data['Correlation'].min(), 
                            self.combined_data['Correlation'].max(), 100)
        y_trend = reg.predict(x_trend.reshape(-1, 1))
        ax1.plot(x_trend, y_trend, 'k--', alpha=0.8, linewidth=2)
        
        # Compute and display correlation
        r, p = pearsonr(self.combined_data['Correlation'], self.combined_data['Preference'])
        ax1.set_title(f'RC1-Spectral Flux vs Preference\nr = {r:.4f}, p = {p:.4f}')
        ax1.set_xlabel('RC1-Spectral Flux Correlation')
        ax1.set_ylabel('Preference Rating (1-9)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Subject-level correlation strengths
        ax2 = axes[1]
        
        subject_correlations = {}
        for subject in subjects:
            subject_data = self.combined_data[self.combined_data['Subject'] == subject]
            if len(subject_data) > 3:
                r, _ = pearsonr(subject_data['Correlation'], subject_data['Preference'])
                subject_correlations[subject] = r
        
        if subject_correlations:
            subjects_list = list(subject_correlations.keys())
            correlations_list = list(subject_correlations.values())
            
            bars = ax2.bar(subjects_list, correlations_list)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.set_title('Subject-Level Neural-Preference Correlations')
            ax2.set_xlabel('Subject')
            ax2.set_ylabel('Correlation (r)')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Color bars by correlation strength
            for bar, corr in zip(bars, correlations_list):
                if corr > 0:
                    bar.set_color('red' if corr > 0.3 else 'pink')
                else:
                    bar.set_color('blue' if corr < -0.3 else 'lightblue')
        
        # 3. Preference distribution by correlation tertiles
        ax3 = axes[2]
        
        # Divide correlations into tertiles
        correlation_tertiles = pd.qcut(self.combined_data['Correlation'], 3, labels=['Low', 'Medium', 'High'])
        self.combined_data['Correlation_Tertile'] = correlation_tertiles
        
        # Box plot by tertiles
        sns.boxplot(data=self.combined_data, x='Correlation_Tertile', y='Preference', ax=ax3)
        ax3.set_title('Preference by Neural Coupling Tertiles')
        ax3.set_xlabel('RC1-Spectral Flux Correlation Tertile')
        ax3.set_ylabel('Preference Rating')
        
        # 4. Song-level analysis
        ax4 = axes[3]
        
        # Mean correlation vs mean preference by song
        song_means = self.combined_data.groupby('Song')[['Correlation', 'Preference']].mean()
        
        ax4.scatter(song_means['Correlation'], song_means['Preference'], s=80, alpha=0.7)
        
        # Add song labels for extremes
        for song, row in song_means.iterrows():
            if (abs(row['Correlation']) > song_means['Correlation'].std() or 
                abs(row['Preference'] - song_means['Preference'].mean()) > song_means['Preference'].std()):
                ax4.annotate(song, (row['Correlation'], row['Preference']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add trend line
        X_song = song_means['Correlation'].values.reshape(-1, 1)
        y_song = song_means['Preference'].values
        reg_song = LinearRegression().fit(X_song, y_song)
        x_song_trend = np.linspace(song_means['Correlation'].min(), 
                                 song_means['Correlation'].max(), 100)
        y_song_trend = reg_song.predict(x_song_trend.reshape(-1, 1))
        ax4.plot(x_song_trend, y_song_trend, 'r--', alpha=0.8)
        
        r_song, p_song = pearsonr(song_means['Correlation'], song_means['Preference'])
        ax4.set_title(f'Song-Level: Neural Coupling vs Preference\nr = {r_song:.4f}, p = {p_song:.4f}')
        ax4.set_xlabel('Mean RC1-Spectral Flux Correlation')
        ax4.set_ylabel('Mean Preference Rating')
        ax4.grid(True, alpha=0.3)
        
        # 5. Correlation vs preference heatmap by subject
        ax5 = axes[4]
        
        # Create pivot table for heatmap
        pivot_corr = self.combined_data.pivot_table(values='Correlation', 
                                                  index='Subject', columns='Song', 
                                                  aggfunc='mean')
        
        sns.heatmap(pivot_corr, annot=False, cmap='RdBu_r', center=0, ax=ax5,
                   cbar_kws={'label': 'RC1-Spectral Flux Correlation'})
        ax5.set_title('Neural Coupling Heatmap')
        ax5.set_xlabel('Song')
        ax5.set_ylabel('Subject')
        
        # 6. Statistical summary
        ax6 = axes[5]
        ax6.axis('off')
        
        # Create statistical summary table
        stats_text = [
            "STATISTICAL SUMMARY",
            "=" * 25,
            f"Total observations: {len(self.combined_data)}",
            f"Subjects: {self.combined_data['Subject'].nunique()}",
            f"Songs: {self.combined_data['Song'].nunique()}",
            "",
            "OVERALL CORRELATIONS:",
            f"Pearson r: {r:.4f} (p = {p:.4f})",
            f"{'Significant' if p < 0.05 else 'Not significant'} at α = 0.05",
            "",
            "NEURAL COUPLING:",
            f"Mean: {self.combined_data['Correlation'].mean():.4f}",
            f"Range: [{self.combined_data['Correlation'].min():.4f}, {self.combined_data['Correlation'].max():.4f}]",
            "",
            "PREFERENCE RATINGS:",
            f"Mean: {self.combined_data['Preference'].mean():.2f}",
            f"Range: [{self.combined_data['Preference'].min()}, {self.combined_data['Preference'].max()}]"
        ]
        
        ax6.text(0.1, 0.9, '\n'.join(stats_text), transform=ax6.transAxes, 
                fontfamily='monospace', fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        
        # Save the plot
        output_file = self.output_path / 'rc1_spectral_flux_vs_preference.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Preference correlation plots saved: {output_file}")
        
        plt.show()
        
        return fig
    
    def save_results(self):
        """Save analysis results"""
        print("💾 Saving Neural-Preference Analysis Results")
        print("=" * 45)
        
        # Save combined data
        if self.combined_data is not None:
            self.combined_data.to_csv(self.output_path / 'rc1_spectral_flux_preference_combined.csv', index=False)
        
        # Compute and save correlation statistics
        stats = self.compute_preference_correlation()
        
        # Save statistics as JSON
        stats_json = {}
        stats_json['overall'] = stats['overall']
        stats_json['subjects'] = stats['subjects']
        # Convert DataFrame to dict for JSON serialization
        stats_json['songs'] = stats['songs'].to_dict('index')
        
        with open(self.output_path / 'neural_preference_correlation_stats.json', 'w') as f:
            json.dump(stats_json, f, indent=2)
        
        print(f"✅ Results saved to {self.output_path}")
    
    def run_complete_analysis(self):
        """Run complete neural-preference correlation analysis"""
        print("=" * 60)
        print("RC1-SPECTRAL FLUX vs PREFERENCE ANALYSIS")
        print("=" * 60)
        
        # Step 1: Load behavioral data
        self.load_behavioral_data()
        
        # Step 2: Load spectral flux correlations
        self.load_spectral_flux_correlations()
        
        # Step 3: Merge data
        self.merge_data()
        
        # Step 4: Compute correlations
        correlation_stats = self.compute_preference_correlation()
        
        # Step 5: Create visualizations
        self.create_preference_correlation_plots()
        
        # Step 6: Save results
        self.save_results()
        
        print("=" * 60)
        print("NEURAL-PREFERENCE ANALYSIS COMPLETE!")
        print(f"Results saved to: {self.output_path}")
        print("=" * 60)
        
        return {
            'combined_data': self.combined_data,
            'correlation_stats': correlation_stats
        }

if __name__ == "__main__":
    # Run the complete analysis
    analyzer = RC1SpectralFluxVsPreference()
    results = analyzer.run_complete_analysis()
    
    # Print key findings
    if results['combined_data'] is not None:
        data = results['combined_data']
        r, p = pearsonr(data['Correlation'], data['Preference'])
        
        print(f"\nKEY FINDINGS:")
        print(f"Overall neural-preference correlation: r = {r:.4f}, p = {p:.4f}")
        
        if p < 0.05:
            direction = "positive" if r > 0 else "negative"
            print(f"*** SIGNIFICANT {direction.upper()} relationship found! ***")
        else:
            print("No significant relationship between neural coupling and preference.")
        
        print(f"Effect size: {'Small' if abs(r) < 0.3 else 'Medium' if abs(r) < 0.5 else 'Large'}")