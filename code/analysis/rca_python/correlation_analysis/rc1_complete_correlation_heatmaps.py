#!/usr/bin/env python3
"""
Complete RC1 Inter-Subject Correlation Heatmaps for All Songs.

This script creates correlation heatmaps for all 15 songs, showing RC1 time-course
correlations between all 5 subjects. Missing data is shown as NA/white in the matrices.

Features:
- Process all 15 songs systematically
- Create 5x5 correlation matrices for each song
- Handle missing subject data with NA values
- Generate comprehensive heatmap visualization
- Statistical summary of correlation patterns
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from typing import Dict, List, Any, Optional

# Add RCA to path
rca_path = Path(__file__).parent.parent  # Go up to rca_python directory
sys.path.insert(0, str(rca_path))

from rca_utils import load_music_preference_data, epochs_to_rca_format_fixed_length


class CompleteRC1CorrelationAnalysis:
    """
    Complete RC1 inter-subject correlation analysis for all 15 songs.
    """
    
    def __init__(self, pooled_results_path: str = "output/pooled_rca/pooled_rca_results.npz",
                 data_dir: str = "data/ica_cleaned", output_dir: str = "output/rc1_complete_analysis"):
        self.pooled_results_path = Path(pooled_results_path)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Define all subjects and songs systematically
        self.subjects = ['pilot_1', 'pilot_2', 'pilot_3', 'pilot_4', 'pilot_5']
        self.songs = [f'{artist}-{song}' for artist in range(1, 6) for song in range(1, 4)]
        
        print(f"Initialized for {len(self.subjects)} subjects and {len(self.songs)} songs:")
        print(f"  Subjects: {self.subjects}")
        print(f"  Songs: {self.songs}")
        
        # Data storage
        self.rc1_filter = None
        self.channel_names = None
        self.song_timecourses = {}  # song_id -> {subject_id: timecourse or None}
        self.correlation_matrices = {}  # song_id -> 5x5 correlation matrix
        
    def load_rc1_spatial_filter(self):
        """Load RC1 spatial filter from pooled analysis."""
        print("\n🧠 Loading RC1 Spatial Filter")
        print("=" * 35)
        
        pooled_data = np.load(self.pooled_results_path)
        spatial_filters = pooled_data['spatial_filters']
        self.rc1_filter = spatial_filters[:, 0]  # First component (RC1)
        self.channel_names = pooled_data['channel_names'].tolist()
        
        eigenvalue = pooled_data['eigenvalues'][0]
        max_channel_idx = np.argmax(np.abs(self.rc1_filter))
        max_channel = self.channel_names[max_channel_idx]
        
        print(f"✅ RC1 loaded: λ={eigenvalue:.6f}, max at {max_channel}")
        
    def extract_all_rc1_timecourses(self):
        """Extract RC1 timecourses for all subjects and all songs."""
        print("\n🎵 Extracting RC1 Timecourses for All Songs")
        print("=" * 50)
        
        global_min_length = 25489  # Use same as pooled analysis
        
        # Initialize data structure
        song_timecourses = {}
        for song_id in self.songs:
            song_timecourses[song_id] = {}
            for subject_id in self.subjects:
                song_timecourses[song_id][subject_id] = None
        
        # Process each subject
        for subject_id in self.subjects:
            print(f"\n📊 Processing {subject_id}...")
            
            try:
                # Load all trial files for this subject
                subject_dir = self.data_dir / subject_id
                if not subject_dir.exists():
                    print(f"    ⚠️ Directory not found")
                    continue
                
                trial_files = list(subject_dir.glob(f"{subject_id}-trial*_proc_*.fif"))
                print(f"    Found {len(trial_files)} trial files")
                
                subject_songs_found = []
                
                for trial_file in trial_files:
                    # Extract song ID from filename
                    filename = trial_file.stem
                    parts = filename.split('_')
                    song_id = None
                    
                    # Look for song ID pattern
                    for part in parts[1:4]:
                        if '-' in part and len(part.split('-')) == 2:
                            try:
                                nums = part.split('-')
                                int(nums[0])
                                int(nums[1])
                                song_id = part
                                break
                            except ValueError:
                                continue
                    
                    if song_id is None or song_id not in self.songs:
                        continue
                    
                    try:
                        # Load the trial data
                        import mne
                        
                        # Try reading as epochs first
                        try:
                            epochs = mne.read_epochs(trial_file, preload=True, verbose=False)
                        except:
                            # Read as raw and create single epoch
                            raw = mne.io.read_raw_fif(trial_file, preload=True, verbose=False)
                            duration = raw.times[-1]
                            epochs = mne.make_fixed_length_epochs(raw, duration=duration, preload=True, verbose=False)
                        
                        # Extract data
                        epoch_data = epochs.get_data()  # (n_epochs, n_channels, n_times)
                        
                        # Process each epoch (should typically be 1 epoch per file)
                        trial_timecourses = []
                        for epoch_idx in range(epoch_data.shape[0]):
                            trial_data = epoch_data[epoch_idx]  # (n_channels, n_times)
                            
                            # Truncate to global minimum length
                            if trial_data.shape[1] >= global_min_length:
                                trial_data_truncated = trial_data[:, :global_min_length]
                                
                                # Apply RC1 spatial filter
                                rc1_timecourse = self.rc1_filter @ trial_data_truncated
                                trial_timecourses.append(rc1_timecourse)
                        
                        # Average multiple epochs if present
                        if trial_timecourses:
                            if len(trial_timecourses) > 1:
                                final_timecourse = np.mean(trial_timecourses, axis=0)
                            else:
                                final_timecourse = trial_timecourses[0]
                            
                            # Store in data structure
                            song_timecourses[song_id][subject_id] = final_timecourse
                            subject_songs_found.append(song_id)
                    
                    except Exception as e:
                        print(f"    ⚠️ Error processing {trial_file.name}: {e}")
                        continue
                
                print(f"    ✅ Extracted RC1 for {len(subject_songs_found)} songs: {subject_songs_found}")
                
            except Exception as e:
                print(f"    ❌ Failed to process {subject_id}: {e}")
                continue
        
        self.song_timecourses = song_timecourses
        
        # Print summary
        print(f"\n📊 EXTRACTION SUMMARY:")
        for song_id in self.songs:
            available_subjects = [s for s in self.subjects if song_timecourses[song_id][s] is not None]
            print(f"  {song_id}: {len(available_subjects)}/5 subjects ({', '.join(available_subjects)})")
        
        return song_timecourses
    
    def compute_all_correlation_matrices(self):
        """Compute 5x5 correlation matrices for all 15 songs."""
        print("\n🔗 Computing Correlation Matrices for All Songs")
        print("=" * 50)
        
        correlation_matrices = {}
        
        for song_id in self.songs:
            print(f"  Processing {song_id}...")
            
            # Create 5x5 correlation matrix
            corr_matrix = np.full((5, 5), np.nan)  # Initialize with NaN
            p_value_matrix = np.full((5, 5), np.nan)
            
            # Fill diagonal with 1.0 for subjects with data
            for i, subject_i in enumerate(self.subjects):
                if self.song_timecourses[song_id][subject_i] is not None:
                    corr_matrix[i, i] = 1.0
                    p_value_matrix[i, i] = 0.0
            
            # Compute pairwise correlations
            for i, subject_i in enumerate(self.subjects):
                for j, subject_j in enumerate(self.subjects):
                    if i != j:  # Skip diagonal
                        timecourse_i = self.song_timecourses[song_id][subject_i]
                        timecourse_j = self.song_timecourses[song_id][subject_j]
                        
                        if timecourse_i is not None and timecourse_j is not None:
                            try:
                                corr, p_val = pearsonr(timecourse_i, timecourse_j)
                                corr_matrix[i, j] = corr
                                p_value_matrix[i, j] = p_val
                            except Exception as e:
                                print(f"    ⚠️ Error computing correlation for {subject_i}-{subject_j}: {e}")
            
            # Store results
            available_subjects = [s for s in self.subjects if self.song_timecourses[song_id][s] is not None]
            n_available = len(available_subjects)
            
            # Calculate mean ISC (excluding diagonal and NaN values)
            mask = ~np.isnan(corr_matrix) & ~np.eye(5, dtype=bool)
            if np.any(mask):
                mean_isc = np.mean(corr_matrix[mask])
                std_isc = np.std(corr_matrix[mask])
            else:
                mean_isc = np.nan
                std_isc = np.nan
            
            correlation_matrices[song_id] = {
                'correlation_matrix': corr_matrix,
                'p_value_matrix': p_value_matrix,
                'available_subjects': available_subjects,
                'n_subjects': n_available,
                'mean_isc': mean_isc,
                'std_isc': std_isc
            }
            
            if not np.isnan(mean_isc):
                print(f"    ✅ {n_available}/5 subjects, ISC = {mean_isc:.3f}±{std_isc:.3f}")
            else:
                print(f"    ⚠️ {n_available}/5 subjects, insufficient data for ISC")
        
        self.correlation_matrices = correlation_matrices
        return correlation_matrices
    
    def create_comprehensive_heatmap_visualization(self):
        """Create comprehensive heatmap visualization for all 15 songs."""
        print("\n🎨 Creating Comprehensive Heatmap Visualization")
        print("=" * 55)
        
        # Create figure with subplots for all 15 songs (5x3 grid)
        fig, axes = plt.subplots(5, 3, figsize=(15, 20))
        axes = axes.flatten()
        
        # Color settings
        cmap = 'RdBu_r'
        vmin, vmax = -0.5, 0.5  # Correlation range
        
        for idx, song_id in enumerate(self.songs):
            ax = axes[idx]
            
            corr_matrix = self.correlation_matrices[song_id]['correlation_matrix']
            available_subjects = self.correlation_matrices[song_id]['available_subjects']
            n_subjects = self.correlation_matrices[song_id]['n_subjects']
            mean_isc = self.correlation_matrices[song_id]['mean_isc']
            
            # Create heatmap
            im = ax.imshow(corr_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
            
            # Set labels
            ax.set_xticks(range(5))
            ax.set_yticks(range(5))
            ax.set_xticklabels(self.subjects, rotation=45, ha='right')
            ax.set_yticklabels(self.subjects)
            
            # Add correlation values as text
            for i in range(5):
                for j in range(5):
                    value = corr_matrix[i, j]
                    if not np.isnan(value):
                        # Choose text color based on background
                        text_color = 'white' if abs(value) > 0.3 else 'black'
                        ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                               color=text_color, fontsize=8, weight='bold')
                    else:
                        # Show NA for missing data
                        ax.text(j, i, 'NA', ha='center', va='center', 
                               color='gray', fontsize=8, weight='bold')
            
            # Title with ISC information
            if not np.isnan(mean_isc):
                title = f'{song_id}\n({n_subjects}/5 subj, ISC={mean_isc:.3f})'
            else:
                title = f'{song_id}\n({n_subjects}/5 subj, ISC=NA)'
            
            ax.set_title(title, fontsize=10, pad=10)
            
            # Add border around available subjects
            # Highlight cells where data exists
            for i in range(5):
                for j in range(5):
                    if not np.isnan(corr_matrix[i, j]):
                        # Add subtle border to indicate data availability
                        rect = plt.Rectangle((j-0.4, i-0.4), 0.8, 0.8, 
                                           fill=False, edgecolor='black', linewidth=0.5)
                        ax.add_patch(rect)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=axes, shrink=0.6, aspect=30, pad=0.02)
        cbar.set_label('RC1 Time-Course Correlation', rotation=270, labelpad=20)
        
        # Main title
        plt.suptitle('RC1 Inter-Subject Correlation Matrices for All Songs\n' + 
                    'Complete Analysis Across All 5 Subjects', fontsize=16, y=0.98)
        
        plt.tight_layout()
        
        # Save the comprehensive heatmap
        heatmap_path = self.output_dir / 'rc1_complete_correlation_heatmaps.png'
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        print(f"📊 Complete heatmaps saved: {heatmap_path}")
        
        return fig
    
    def create_summary_statistics_plot(self):
        """Create summary statistics visualization."""
        print("\n📈 Creating Summary Statistics")
        print("=" * 35)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # 1. ISC by song
        ax1 = axes[0]
        songs = self.songs
        isc_values = [self.correlation_matrices[song]['mean_isc'] for song in songs]
        n_subjects = [self.correlation_matrices[song]['n_subjects'] for song in songs]
        
        # Color bars by number of subjects
        colors = []
        for n in n_subjects:
            if n >= 4:
                colors.append('darkgreen')
            elif n >= 3:
                colors.append('orange')
            elif n >= 2:
                colors.append('lightcoral')
            else:
                colors.append('gray')
        
        bars = ax1.bar(range(len(songs)), isc_values, color=colors)
        ax1.set_xlabel('Song')
        ax1.set_ylabel('Mean ISC')
        ax1.set_title('RC1 Inter-Subject Correlation by Song')
        ax1.set_xticks(range(len(songs)))
        ax1.set_xticklabels(songs, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add text labels for number of subjects
        for i, (bar, n) in enumerate(zip(bars, n_subjects)):
            if not np.isnan(isc_values[i]):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                        f'{n}', ha='center', va='bottom', fontsize=8)
        
        # 2. Distribution of all correlations
        ax2 = axes[1]
        all_correlations = []
        for song_data in self.correlation_matrices.values():
            corr_matrix = song_data['correlation_matrix']
            mask = ~np.isnan(corr_matrix) & ~np.eye(5, dtype=bool)  # Exclude diagonal and NaN
            all_correlations.extend(corr_matrix[mask])
        
        if all_correlations:
            ax2.hist(all_correlations, bins=20, alpha=0.7, edgecolor='black')
            ax2.axvline(np.mean(all_correlations), color='red', linestyle='--', 
                       label=f'Mean = {np.mean(all_correlations):.3f}')
            ax2.set_xlabel('Correlation Value')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of All Pairwise Correlations')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Subject availability heatmap
        ax3 = axes[2]
        availability_matrix = np.zeros((len(self.subjects), len(self.songs)))
        
        for j, song_id in enumerate(self.songs):
            for i, subject_id in enumerate(self.subjects):
                if self.song_timecourses[song_id][subject_id] is not None:
                    availability_matrix[i, j] = 1
        
        im = ax3.imshow(availability_matrix, cmap='RdYlGn', aspect='auto')
        ax3.set_xlabel('Song')
        ax3.set_ylabel('Subject')
        ax3.set_title('Data Availability Matrix')
        ax3.set_xticks(range(len(self.songs)))
        ax3.set_yticks(range(len(self.subjects)))
        ax3.set_xticklabels(self.songs, rotation=45, ha='right')
        ax3.set_yticklabels(self.subjects)
        
        # Add text annotations
        for i in range(len(self.subjects)):
            for j in range(len(self.songs)):
                value = availability_matrix[i, j]
                text = '✓' if value == 1 else '✗'
                color = 'white' if value == 1 else 'black'
                ax3.text(j, i, text, ha='center', va='center', color=color, fontsize=12)
        
        # 4. Summary statistics table
        ax4 = axes[3]
        ax4.axis('off')
        
        # Calculate summary statistics
        total_possible_pairs = len(self.songs) * 5 * 4 / 2  # 15 songs * 5 subjects * 4 other subjects / 2
        actual_pairs = len(all_correlations)
        coverage_percent = actual_pairs / total_possible_pairs * 100 if total_possible_pairs > 0 else 0
        
        summary_stats = [
            ['Total Songs', f'{len(self.songs)}'],
            ['Total Subjects', f'{len(self.subjects)}'],
            ['Data Coverage', f'{coverage_percent:.1f}%'],
            ['Total Correlations', f'{len(all_correlations)}'],
            ['Mean ISC', f'{np.mean(all_correlations):.3f}' if all_correlations else 'NA'],
            ['Std ISC', f'{np.std(all_correlations):.3f}' if all_correlations else 'NA'],
            ['Max ISC', f'{np.max(all_correlations):.3f}' if all_correlations else 'NA'],
            ['Min ISC', f'{np.min(all_correlations):.3f}' if all_correlations else 'NA']
        ]
        
        # Create table
        table = ax4.table(cellText=summary_stats,
                         colLabels=['Statistic', 'Value'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        ax4.set_title('Summary Statistics', pad=20)
        
        plt.tight_layout()
        
        # Save summary plot
        summary_path = self.output_dir / 'rc1_correlation_summary.png'
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        print(f"📊 Summary statistics saved: {summary_path}")
        
        return fig
    
    def save_correlation_data(self):
        """Save correlation matrices and summary data."""
        print("\n💾 Saving Correlation Data")
        print("=" * 30)
        
        # Save as NPZ file
        save_dict = {
            'songs': np.array(self.songs),
            'subjects': np.array(self.subjects),
            'rc1_filter': self.rc1_filter,
            'channel_names': np.array(self.channel_names)
        }
        
        # Add correlation matrices for each song
        for song_id in self.songs:
            save_dict[f'{song_id}_correlation_matrix'] = self.correlation_matrices[song_id]['correlation_matrix']
            save_dict[f'{song_id}_p_values'] = self.correlation_matrices[song_id]['p_value_matrix']
            save_dict[f'{song_id}_mean_isc'] = self.correlation_matrices[song_id]['mean_isc']
            save_dict[f'{song_id}_n_subjects'] = self.correlation_matrices[song_id]['n_subjects']
        
        data_path = self.output_dir / 'rc1_complete_correlations.npz'
        np.savez_compressed(data_path, **save_dict)
        print(f"📄 Correlation data saved: {data_path}")
        
        # Save summary as CSV
        summary_data = []
        for song_id in self.songs:
            result = self.correlation_matrices[song_id]
            summary_data.append({
                'Song': song_id,
                'N_Subjects': result['n_subjects'],
                'Available_Subjects': '; '.join(result['available_subjects']),
                'Mean_ISC': result['mean_isc'],
                'Std_ISC': result['std_isc']
            })
        
        summary_df = pd.DataFrame(summary_data)
        csv_path = self.output_dir / 'rc1_complete_summary.csv'
        summary_df.to_csv(csv_path, index=False)
        print(f"📊 Summary CSV saved: {csv_path}")
    
    def run_complete_analysis(self):
        """Run the complete correlation analysis for all songs."""
        print("🧠 COMPLETE RC1 CORRELATION ANALYSIS")
        print("=" * 55)
        print("Creating correlation heatmaps for all 15 songs across 5 subjects...")
        
        try:
            # Step 1: Load RC1 filter
            self.load_rc1_spatial_filter()
            
            # Step 2: Extract all RC1 timecourses
            self.extract_all_rc1_timecourses()
            
            # Step 3: Compute correlation matrices
            self.compute_all_correlation_matrices()
            
            # Step 4: Create comprehensive visualization
            plt.ioff()
            heatmap_fig = self.create_comprehensive_heatmap_visualization()
            plt.close(heatmap_fig)
            
            # Step 5: Create summary statistics
            summary_fig = self.create_summary_statistics_plot()
            plt.close(summary_fig)
            
            # Step 6: Save data
            self.save_correlation_data()
            
            # Step 7: Print final summary
            self.print_final_summary()
            
            return True
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def print_final_summary(self):
        """Print final analysis summary."""
        print("\n" + "=" * 55)
        print("🎉 COMPLETE RC1 CORRELATION ANALYSIS COMPLETE!")
        print("=" * 55)
        
        # Calculate overall statistics
        total_songs = len(self.songs)
        songs_with_data = sum(1 for result in self.correlation_matrices.values() if result['n_subjects'] > 1)
        
        all_correlations = []
        for song_data in self.correlation_matrices.values():
            corr_matrix = song_data['correlation_matrix']
            mask = ~np.isnan(corr_matrix) & ~np.eye(5, dtype=bool)
            all_correlations.extend(corr_matrix[mask])
        
        print(f"📊 ANALYSIS SUMMARY:")
        print(f"  • Total songs analyzed: {total_songs}/15")
        print(f"  • Songs with 2+ subjects: {songs_with_data}")
        print(f"  • Total pairwise correlations: {len(all_correlations)}")
        if all_correlations:
            print(f"  • Overall mean ISC: {np.mean(all_correlations):.3f}±{np.std(all_correlations):.3f}")
        
        # Find best and worst songs
        valid_songs = [(song, result['mean_isc']) for song, result in self.correlation_matrices.items() 
                      if not np.isnan(result['mean_isc'])]
        
        if valid_songs:
            valid_songs.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n🏆 HIGHEST ISC SONGS:")
            for i, (song, isc) in enumerate(valid_songs[:3]):
                n_subj = self.correlation_matrices[song]['n_subjects']
                print(f"  {i+1}. {song}: ISC = {isc:.3f} ({n_subj} subjects)")
            
            print(f"\n📉 LOWEST ISC SONGS:")
            for i, (song, isc) in enumerate(valid_songs[-3:]):
                n_subj = self.correlation_matrices[song]['n_subjects']
                print(f"  {len(valid_songs)-2+i}. {song}: ISC = {isc:.3f} ({n_subj} subjects)")
        
        print(f"\n📁 OUTPUT FILES:")
        print(f"  • {self.output_dir}/rc1_complete_correlation_heatmaps.png")
        print(f"  • {self.output_dir}/rc1_correlation_summary.png")
        print(f"  • {self.output_dir}/rc1_complete_correlations.npz")
        print(f"  • {self.output_dir}/rc1_complete_summary.csv")
        
        print(f"\n🧠 KEY INSIGHTS:")
        print("  • White/NA cells indicate missing subject data for that song")
        print("  • Red colors indicate negative correlations, Blue indicates positive")
        print("  • Higher correlations suggest more synchronized neural responses")
        print("  • Pattern reveals which songs evoke consistent vs. variable responses")


def main():
    """Main execution function."""
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    analyzer = CompleteRC1CorrelationAnalysis()
    success = analyzer.run_complete_analysis()
    
    if success:
        print("\n🧠 Complete RC1 correlation analysis finished! Check output directory.")
    else:
        print("\n❌ Analysis failed. Check data and try again.")


if __name__ == "__main__":
    main()