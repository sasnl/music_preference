#!/usr/bin/env python3
"""
RC1-Spectral Flux Correlation Analysis
=====================================

This script analyzes the relationship between RC1-filtered neural responses and 
music spectral flux features. It applies the most reliable spatial component (RC1) 
from pooled RCA analysis to filter EEG data, then computes correlations with 
spectral flux to understand how neural responses track dynamic changes in music.

Key features:
- Load RC1 spatial filter from pooled RCA results
- Apply RC1 filter to extract reliable neural timecourses
- Load spectral flux features for all music stimuli
- Align neural and acoustic timeseries with proper temporal alignment
- Compute cross-correlations at multiple time lags
- Generate comprehensive visualizations and statistics

Author: Music Preference Analysis Pipeline
Date: 2025-08-26
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from pathlib import Path
import json
from scipy.stats import pearsonr
from scipy.signal import correlate, resample
import warnings
warnings.filterwarnings('ignore')

# Add RCA to path
import sys
rca_path = Path(__file__).parent.parent  # Go up to rca_python directory
sys.path.insert(0, str(rca_path))

from rca_utils import load_music_preference_data, epochs_to_rca_format_fixed_length

class RC1SpectralFluxCorrelation:
    """
    Analyze correlations between RC1-filtered neural responses and spectral flux.
    """
    
    def __init__(self, base_path='/Users/tongshan/Documents/music_preference'):
        self.base_path = Path(base_path)
        self.data_path = self.base_path / 'data' / 'ica_cleaned'
        self.features_path = self.base_path / 'music_stim' / 'music_features'
        self.output_path = self.base_path / 'output' / 'rc1_spectral_flux_analysis'
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Analysis parameters
        self.subjects = ['pilot_1', 'pilot_2', 'pilot_3', 'pilot_4', 'pilot_5']
        self.songs = [f"{i}-{j}" for i in range(1, 6) for j in range(1, 4)]
        self.eeg_fs = 1000  # EEG sampling rate
        self.feature_fs = 128  # Music feature sampling rate
        
        # Results storage
        self.rc1_filter = None
        self.rc1_timecourses = {}
        self.spectral_flux_features = {}
        self.correlation_results = {}
        
    def load_rc1_filter(self):
        """Load RC1 spatial filter from pooled RCA results"""
        print("🧠 Loading RC1 Spatial Filter")
        print("=" * 40)
        
        rca_results_path = self.base_path / 'output' / 'pooled_rca' / 'pooled_rca_results.npz'
        
        if not rca_results_path.exists():
            raise FileNotFoundError(f"RCA results not found at {rca_results_path}. Run pooled RCA analysis first.")
        
        # Load RCA results
        rca_data = np.load(rca_results_path)
        
        # Extract RC1 (first component)
        spatial_filters = rca_data['spatial_filters']
        eigenvalues = rca_data['eigenvalues']
        
        self.rc1_filter = spatial_filters[:, 0]  # First column = RC1
        rc1_eigenvalue = eigenvalues[0]
        
        print(f"✅ RC1 loaded successfully")
        print(f"   Eigenvalue: {rc1_eigenvalue:.6f}")
        print(f"   Filter shape: {self.rc1_filter.shape}")
        print(f"   Max coefficient: {np.abs(self.rc1_filter).max():.3f}")
        
        # Find electrode with maximum weight
        ch_names = rca_data['channel_names'] if 'channel_names' in rca_data else None
        if ch_names is not None:
            max_idx = np.argmax(np.abs(self.rc1_filter))
            print(f"   Strongest electrode: {ch_names[max_idx]} ({self.rc1_filter[max_idx]:.3f})")
        
        return self.rc1_filter
    
    def load_spectral_flux_features(self):
        """Load spectral flux features for all songs"""
        print("🎵 Loading Spectral Flux Features")
        print("=" * 40)
        
        spectral_flux_features = {}
        
        for song in self.songs:
            feature_file = self.features_path / f"{song}_proc_features.npz"
            
            if feature_file.exists():
                try:
                    feature_data = np.load(feature_file)
                    spectral_flux = feature_data['spectral_flux']
                    time_s = feature_data['time_s']
                    
                    spectral_flux_features[song] = {
                        'spectral_flux': spectral_flux,
                        'time': time_s,
                        'fs': feature_data['target_sr'].item(),
                        'duration': feature_data['duration_seconds'].item()
                    }
                    
                    print(f"   ✅ {song}: {len(spectral_flux)} samples, {spectral_flux_features[song]['duration']:.1f}s")
                    
                except Exception as e:
                    print(f"   ❌ Error loading {song}: {e}")
            else:
                print(f"   ⚠️  Missing features for {song}")
        
        self.spectral_flux_features = spectral_flux_features
        print(f"✅ Loaded features for {len(spectral_flux_features)}/{len(self.songs)} songs")
        
        return spectral_flux_features
    
    def apply_rc1_filter_to_eeg(self):
        """Apply RC1 spatial filter to EEG data for all subjects and songs"""
        print("🔬 Applying RC1 Filter to EEG Data")
        print("=" * 45)
        
        if self.rc1_filter is None:
            raise ValueError("RC1 filter not loaded. Run load_rc1_filter() first.")
        
        rc1_timecourses = {}
        
        for subject in self.subjects:
            print(f"📊 Processing {subject}...")
            rc1_timecourses[subject] = {}
            
            subject_path = self.data_path / subject
            if not subject_path.exists():
                print(f"   ❌ No data directory for {subject}")
                continue
            
            # Find trial files for this subject
            trial_files = list(subject_path.glob(f"{subject}-trial*_*_proc_*_ica_cleaned.fif"))
            song_count = 0
            
            for trial_file in trial_files:
                try:
                    # Extract song ID from filename
                    filename = trial_file.name
                    song_id = None
                    
                    # Look for song pattern in filename
                    parts = filename.split('_')
                    for part in parts:
                        if '-' in part and len(part) == 3 and part[1] == '-':
                            if part.replace('-', '').isdigit():
                                song_id = part
                                break
                    
                    if song_id is None or song_id not in self.songs:
                        continue
                    
                    # Load EEG data
                    raw = mne.io.read_raw_fif(trial_file, preload=True, verbose=False)
                    eeg_data = raw.get_data()  # Shape: (channels, timepoints)
                    
                    # Apply RC1 spatial filter
                    rc1_timecourse = self.rc1_filter @ eeg_data  # Matrix multiplication
                    
                    rc1_timecourses[subject][song_id] = {
                        'timecourse': rc1_timecourse,
                        'fs': raw.info['sfreq'],
                        'duration': len(rc1_timecourse) / raw.info['sfreq'],
                        'n_samples': len(rc1_timecourse)
                    }
                    
                    song_count += 1
                    
                except Exception as e:
                    print(f"   ❌ Error processing {trial_file.name}: {e}")
                    continue
            
            print(f"    ✅ Extracted RC1 for {song_count} songs")
        
        self.rc1_timecourses = rc1_timecourses
        
        # Print summary
        total_trials = sum(len(songs) for songs in rc1_timecourses.values())
        print(f"✅ RC1 extraction complete: {total_trials} trials across {len(rc1_timecourses)} subjects")
        
        return rc1_timecourses
    
    def compute_cross_correlations(self, max_lag_ms=2000, method='pearson'):
        """
        Compute cross-correlations between RC1 and spectral flux with multiple time lags.
        
        Parameters:
        -----------
        max_lag_ms : int
            Maximum lag in milliseconds to test
        method : str
            'pearson' for Pearson correlation, 'cross_corr' for cross-correlation
        """
        print("🔗 Computing RC1-Spectral Flux Correlations")
        print("=" * 50)
        
        if not self.rc1_timecourses or not self.spectral_flux_features:
            raise ValueError("Must load RC1 timecourses and spectral flux features first")
        
        correlation_results = {}
        max_lag_samples = int(max_lag_ms * self.eeg_fs / 1000)  # Convert to EEG samples
        
        for subject in self.subjects:
            if subject not in self.rc1_timecourses:
                continue
                
            print(f"📊 Processing {subject}...")
            correlation_results[subject] = {}
            
            for song in self.songs:
                if song not in self.rc1_timecourses[subject] or song not in self.spectral_flux_features:
                    continue
                
                try:
                    # Get RC1 timecourse and spectral flux
                    rc1_data = self.rc1_timecourses[subject][song]
                    flux_data = self.spectral_flux_features[song]
                    
                    rc1_timecourse = rc1_data['timecourse']
                    spectral_flux = flux_data['spectral_flux']
                    
                    # Resample spectral flux to match EEG sampling rate
                    n_eeg_samples = len(rc1_timecourse)
                    flux_resampled = resample(spectral_flux, n_eeg_samples)
                    
                    # Trim both to same length (minimum)
                    min_length = min(len(rc1_timecourse), len(flux_resampled))
                    rc1_trimmed = rc1_timecourse[:min_length]
                    flux_trimmed = flux_resampled[:min_length]
                    
                    if method == 'pearson':
                        # Simple Pearson correlation at zero lag
                        corr, p_value = pearsonr(rc1_trimmed, flux_trimmed)
                        correlation_results[subject][song] = {
                            'correlation': corr,
                            'p_value': p_value,
                            'n_samples': min_length,
                            'duration': min_length / self.eeg_fs
                        }
                        
                    elif method == 'cross_corr':
                        # Cross-correlation with multiple lags
                        cross_corr = correlate(rc1_trimmed, flux_trimmed, mode='full')
                        cross_corr = cross_corr / (np.std(rc1_trimmed) * np.std(flux_trimmed) * len(rc1_trimmed))
                        
                        # Create lag vector
                        lags = np.arange(-len(flux_trimmed) + 1, len(rc1_trimmed))
                        lag_ms = lags * 1000 / self.eeg_fs  # Convert to milliseconds
                        
                        # Find peak correlation and its lag
                        peak_idx = np.argmax(np.abs(cross_corr))
                        peak_corr = cross_corr[peak_idx]
                        peak_lag_ms = lag_ms[peak_idx]
                        
                        correlation_results[subject][song] = {
                            'cross_correlation': cross_corr,
                            'lags_ms': lag_ms,
                            'peak_correlation': peak_corr,
                            'peak_lag_ms': peak_lag_ms,
                            'n_samples': min_length,
                            'duration': min_length / self.eeg_fs
                        }
                    
                    print(f"    ✅ {song}: corr={correlation_results[subject][song].get('correlation', correlation_results[subject][song].get('peak_correlation', 0)):.3f}")
                    
                except Exception as e:
                    print(f"    ❌ Error processing {subject}-{song}: {e}")
                    continue
        
        self.correlation_results = correlation_results
        
        # Print summary
        total_correlations = sum(len(songs) for songs in correlation_results.values())
        print(f"✅ Computed correlations for {total_correlations} subject-song pairs")
        
        return correlation_results
    
    def create_correlation_summary(self):
        """Create summary statistics and visualizations"""
        print("📈 Creating Correlation Summary")
        print("=" * 35)
        
        if not self.correlation_results:
            raise ValueError("Must compute correlations first")
        
        # Collect all correlation values
        all_correlations = []
        summary_data = []
        
        for subject in self.subjects:
            if subject not in self.correlation_results:
                continue
                
            for song in self.songs:
                if song not in self.correlation_results[subject]:
                    continue
                
                result = self.correlation_results[subject][song]
                corr = result.get('correlation', result.get('peak_correlation', 0))
                
                all_correlations.append(corr)
                summary_data.append({
                    'Subject': subject,
                    'Song': song,
                    'Correlation': corr,
                    'P_value': result.get('p_value', np.nan),
                    'Duration': result.get('duration', np.nan),
                    'N_samples': result.get('n_samples', np.nan)
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Print statistics
        print(f"📊 Summary Statistics:")
        print(f"   Total correlations: {len(all_correlations)}")
        print(f"   Mean correlation: {np.mean(all_correlations):.4f}")
        print(f"   Std correlation: {np.std(all_correlations):.4f}")
        print(f"   Range: [{np.min(all_correlations):.4f}, {np.max(all_correlations):.4f}]")
        
        # Find strongest correlations
        if not summary_df.empty:
            top_correlations = summary_df.nlargest(5, 'Correlation')
            print(f"\n🏆 Strongest RC1-Spectral Flux Correlations:")
            for _, row in top_correlations.iterrows():
                print(f"   {row['Subject']}-{row['Song']}: r={row['Correlation']:.4f}")
        
        return summary_df, all_correlations
    
    def plot_correlation_results(self):
        """Create comprehensive visualization of correlation results"""
        print("🎨 Creating Correlation Visualizations")
        print("=" * 40)
        
        summary_df, all_correlations = self.create_correlation_summary()
        
        if summary_df.empty:
            print("❌ No correlation data to plot")
            return None
        
        # Create comprehensive figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 1. Distribution of all correlations
        ax1 = axes[0]
        ax1.hist(all_correlations, bins=25, alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(all_correlations), color='red', linestyle='--', 
                   label=f'Mean = {np.mean(all_correlations):.3f}')
        ax1.set_title('RC1-Spectral Flux Correlation Distribution')
        ax1.set_xlabel('Correlation Coefficient')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Correlations by subject
        ax2 = axes[1]
        subject_means = summary_df.groupby('Subject')['Correlation'].mean()
        subject_means.plot(kind='bar', ax=ax2)
        ax2.set_title('Mean RC1-Spectral Flux Correlation by Subject')
        ax2.set_xlabel('Subject')
        ax2.set_ylabel('Mean Correlation')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Correlations by song
        ax3 = axes[2]
        song_means = summary_df.groupby('Song')['Correlation'].mean()
        song_means.plot(kind='bar', ax=ax3)
        ax3.set_title('Mean RC1-Spectral Flux Correlation by Song')
        ax3.set_xlabel('Song')
        ax3.set_ylabel('Mean Correlation')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. Correlation heatmap (subjects × songs)
        ax4 = axes[3]
        pivot_df = summary_df.pivot(index='Subject', columns='Song', values='Correlation')
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                   ax=ax4, cbar_kws={'shrink': 0.8})
        ax4.set_title('RC1-Spectral Flux Correlation Matrix')
        ax4.set_xlabel('Song')
        ax4.set_ylabel('Subject')
        
        # 5. Scatter plot: correlation vs duration
        ax5 = axes[4]
        ax5.scatter(summary_df['Duration'], summary_df['Correlation'], alpha=0.6)
        ax5.set_title('Correlation vs Trial Duration')
        ax5.set_xlabel('Duration (s)')
        ax5.set_ylabel('Correlation')
        ax5.grid(True, alpha=0.3)
        
        # Add trend line
        from sklearn.linear_model import LinearRegression
        valid_mask = ~(np.isnan(summary_df['Duration']) | np.isnan(summary_df['Correlation']))
        if valid_mask.sum() > 1:
            X = summary_df.loc[valid_mask, 'Duration'].values.reshape(-1, 1)
            y = summary_df.loc[valid_mask, 'Correlation'].values
            reg = LinearRegression().fit(X, y)
            x_trend = np.linspace(summary_df['Duration'].min(), summary_df['Duration'].max(), 100)
            y_trend = reg.predict(x_trend.reshape(-1, 1))
            ax5.plot(x_trend, y_trend, 'r--', alpha=0.8)
        
        # 6. Example timecourse comparison
        ax6 = axes[5]
        
        # Find a representative example with good correlation
        if not summary_df.empty:
            best_example = summary_df.loc[summary_df['Correlation'].idxmax()]
            subject = best_example['Subject']
            song = best_example['Song']
            
            if subject in self.rc1_timecourses and song in self.rc1_timecourses[subject]:
                rc1_data = self.rc1_timecourses[subject][song]['timecourse']
                flux_data = self.spectral_flux_features[song]['spectral_flux']
                
                # Create time vectors
                rc1_time = np.arange(len(rc1_data)) / self.eeg_fs
                flux_time = np.linspace(0, len(rc1_data) / self.eeg_fs, len(flux_data))
                
                # Plot first 10 seconds
                mask_rc1 = rc1_time <= 10
                mask_flux = flux_time <= 10
                
                # Normalize for visualization
                rc1_norm = (rc1_data[mask_rc1] - np.mean(rc1_data[mask_rc1])) / np.std(rc1_data[mask_rc1])
                flux_norm = (flux_data[mask_flux] - np.mean(flux_data[mask_flux])) / np.std(flux_data[mask_flux])
                
                ax6.plot(rc1_time[mask_rc1], rc1_norm, 'b-', label='RC1 (normalized)', alpha=0.8)
                ax6.plot(flux_time[mask_flux], flux_norm, 'r-', label='Spectral Flux (normalized)', alpha=0.8)
                ax6.set_title(f'Example: {subject}-{song} (r={best_example["Correlation"]:.3f})')
                ax6.set_xlabel('Time (s)')
                ax6.set_ylabel('Normalized Amplitude')
                ax6.legend()
                ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_path / "rc1_spectral_flux_correlations.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def save_results(self):
        """Save correlation results to files"""
        print("💾 Saving Results")
        print("=" * 20)
        
        # Save correlation results
        np.savez(self.output_path / "rc1_spectral_flux_correlations.npz",
                correlation_results=self.correlation_results,
                subjects=self.subjects,
                songs=self.songs)
        
        # Save summary statistics
        summary_df, all_correlations = self.create_correlation_summary()
        if not summary_df.empty:
            summary_df.to_csv(self.output_path / "rc1_spectral_flux_summary.csv", index=False)
        
        # Save RC1 filter for reference
        if self.rc1_filter is not None:
            np.save(self.output_path / "rc1_spatial_filter.npy", self.rc1_filter)
        
        print(f"✅ Results saved to {self.output_path}")
    
    def run_complete_analysis(self, max_lag_ms=2000, method='pearson'):
        """Run the complete RC1-spectral flux correlation analysis"""
        print("=" * 60)
        print("RC1-SPECTRAL FLUX CORRELATION ANALYSIS")
        print("=" * 60)
        
        # Step 1: Load RC1 filter
        self.load_rc1_filter()
        
        # Step 2: Load spectral flux features
        self.load_spectral_flux_features()
        
        # Step 3: Apply RC1 filter to EEG
        self.apply_rc1_filter_to_eeg()
        
        # Step 4: Compute correlations
        self.compute_cross_correlations(max_lag_ms=max_lag_ms, method=method)
        
        # Step 5: Create visualizations
        self.plot_correlation_results()
        
        # Step 6: Save results
        self.save_results()
        
        print("=" * 60)
        print("RC1-SPECTRAL FLUX ANALYSIS COMPLETE!")
        print(f"Results saved to: {self.output_path}")
        print("=" * 60)
        
        return {
            'rc1_filter': self.rc1_filter,
            'correlation_results': self.correlation_results,
            'spectral_flux_features': self.spectral_flux_features,
            'rc1_timecourses': self.rc1_timecourses
        }

if __name__ == "__main__":
    # Run the complete analysis
    analyzer = RC1SpectralFluxCorrelation()
    results = analyzer.run_complete_analysis(method='pearson')
    
    # Print final summary
    summary_df, all_correlations = analyzer.create_correlation_summary()
    
    if not summary_df.empty:
        print("\nFINAL SUMMARY:")
        print(f"Mean RC1-Spectral Flux correlation: {np.mean(all_correlations):.4f} ± {np.std(all_correlations):.4f}")
        
        # Top correlations
        top_correlations = summary_df.nlargest(3, 'Correlation')
        print("\nTop 3 RC1-Spectral Flux correlations:")
        for i, (_, row) in enumerate(top_correlations.iterrows(), 1):
            print(f"  {i}. {row['Subject']}-{row['Song']}: r={row['Correlation']:.4f}")