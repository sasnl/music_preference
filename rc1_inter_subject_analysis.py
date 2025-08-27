#!/usr/bin/env python3
"""
RC1 Inter-Subject Analysis for Music Preference Study.

This script applies the RC1 spatial filter (derived from pooled analysis) to extract
the most reliable neural component from each subject's data for each song, then
computes inter-subject correlations to measure neural synchrony.

Features:
- Load RC1 spatial filter from pooled analysis
- Apply RC1 filter to each subject's individual song data
- Compute inter-subject correlations (ISC) for each song
- Visualize RC1 time courses and correlations
- Analyze preference-related patterns in neural synchrony
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import json
from scipy.stats import pearsonr
from itertools import combinations
import seaborn as sns

# Add RCA to path
rca_path = Path(__file__).parent / 'code' / 'analysis' / 'rca_python'
sys.path.insert(0, str(rca_path))

from rca_utils import load_music_preference_data, epochs_to_rca_format_fixed_length


class RC1InterSubjectAnalysis:
    """
    Apply RC1 spatial filter and compute inter-subject correlations for each song.
    """
    
    def __init__(self, pooled_results_path: str = "output/pooled_rca/pooled_rca_results.npz",
                 data_dir: str = "data/ica_cleaned", output_dir: str = "output/rc1_isc_analysis"):
        self.pooled_results_path = Path(pooled_results_path)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.subjects = ['pilot_1', 'pilot_2', 'pilot_3', 'pilot_4', 'pilot_5']
        self.songs = [f'{artist}-{song}' for artist in range(1, 6) for song in range(1, 4)]  # 15 songs total
        
        # Analysis results storage
        self.rc1_filter = None
        self.channel_names = None
        self.song_data = {}  # song_id -> {subject_id: rc1_timecourse}
        self.isc_results = {}  # song_id -> correlation_matrix
        self.preference_data = {}
        
    def load_rc1_spatial_filter(self):
        """Load RC1 spatial filter from pooled analysis results."""
        print("🧠 Loading RC1 Spatial Filter from Pooled Analysis")
        print("=" * 55)
        
        if not self.pooled_results_path.exists():
            raise FileNotFoundError(f"Pooled results not found: {self.pooled_results_path}")
        
        # Load pooled results
        pooled_data = np.load(self.pooled_results_path)
        
        # Extract RC1 (first component) spatial filter
        spatial_filters = pooled_data['spatial_filters']
        self.rc1_filter = spatial_filters[:, 0]  # First component
        self.channel_names = pooled_data['channel_names'].tolist()
        
        eigenvalues = pooled_data['eigenvalues']
        rc1_eigenvalue = eigenvalues[0]
        
        # Find max activation channel
        max_channel_idx = np.argmax(np.abs(self.rc1_filter))
        max_channel = self.channel_names[max_channel_idx]
        max_weight = self.rc1_filter[max_channel_idx]
        
        print(f"✅ RC1 spatial filter loaded successfully!")
        print(f"  • Eigenvalue (reliability): {rc1_eigenvalue:.6f}")
        print(f"  • Channels: {len(self.channel_names)}")
        print(f"  • Max activation: {max_channel} ({max_weight:.3f})")
        print(f"  • Filter shape: {self.rc1_filter.shape}")
        print()
        
        return self.rc1_filter
    
    def load_behavioral_preferences(self):
        """Load behavioral preference data."""
        print("📊 Loading Behavioral Preference Data")
        print("=" * 40)
        
        beh_file = self.data_dir.parent / 'beh_ratings.json'
        if not beh_file.exists():
            raise FileNotFoundError(f"Behavioral data not found: {beh_file}")
        
        with open(beh_file, 'r') as f:
            ratings = json.load(f)
        
        self.preference_data = ratings['preference']
        
        # Create preference categories for each subject
        subject_preferences = {}
        for subject_id in self.subjects:
            if subject_id in self.preference_data:
                subject_ratings = {k: v for k, v in self.preference_data[subject_id].items() 
                                 if v is not None}  # Filter out null values
                
                # Sort songs by preference rating
                sorted_songs = sorted(subject_ratings.items(), key=lambda x: x[1], reverse=True)
                
                subject_preferences[subject_id] = {
                    'top_3': [song for song, _ in sorted_songs[:3]],
                    'bottom_3': [song for song, _ in sorted_songs[-3:]],
                    'all_ratings': subject_ratings
                }
                
                print(f"  {subject_id}: Top 3: {subject_preferences[subject_id]['top_3']}")
        
        self.subject_preferences = subject_preferences
        print(f"✅ Loaded preferences for {len(subject_preferences)} subjects")
        print()
        
        return subject_preferences
    
    def apply_rc1_filter_to_all_data(self):
        """Apply RC1 spatial filter to each subject's data for each song."""
        print("🎵 Applying RC1 Filter to Individual Song Data")
        print("=" * 50)
        
        if self.rc1_filter is None:
            raise ValueError("RC1 filter not loaded. Run load_rc1_spatial_filter() first.")
        
        # Find global minimum length (use same as pooled analysis)
        global_min_length = 25489  # From pooled analysis
        print(f"Using global minimum length: {global_min_length} samples")
        
        song_data = {}
        processing_summary = {}
        
        for subject_id in self.subjects:
            print(f"\n📊 Processing {subject_id}...")
            
            try:
                # Load all data for this subject
                data_dict = load_music_preference_data(subject_id, self.data_dir)
                
                if not data_dict['preferred'] and not data_dict['nonpreferred']:
                    print(f"    ⚠️ No data available")
                    continue
                
                # Combine all epochs and extract song IDs
                all_epochs = data_dict['preferred'] + data_dict['nonpreferred']
                subject_song_data = {}
                
                for epochs in all_epochs:
                    # Extract song ID from filename
                    # epochs comes from specific trial files, need to map back to song IDs
                    # We'll use a different approach - process all available trial files
                    pass
                
                # Alternative approach: directly load trial files and extract song IDs
                subject_dir = self.data_dir / subject_id
                trial_files = list(subject_dir.glob(f"{subject_id}-trial*_proc_*.fif"))
                
                for trial_file in trial_files:
                    # Extract song ID from filename
                    filename = trial_file.stem
                    parts = filename.split('_')
                    song_id = None
                    
                    # Look for song ID pattern (e.g., "2-1", "3-2")
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
                    
                    if song_id is None:
                        continue
                    
                    try:
                        # Load this specific trial
                        import mne
                        
                        # Try reading as epochs first
                        try:
                            epochs = mne.read_epochs(trial_file, preload=True, verbose=False)
                        except:
                            # If no epochs, read as raw and create single epoch
                            raw = mne.io.read_raw_fif(trial_file, preload=True, verbose=False)
                            duration = raw.times[-1]
                            epochs = mne.make_fixed_length_epochs(raw, duration=duration, preload=True, verbose=False)
                        
                        # Convert to RCA format and truncate
                        epoch_data = epochs.get_data()  # (n_epochs, n_channels, n_times)
                        
                        for epoch_idx in range(epoch_data.shape[0]):
                            trial_data = epoch_data[epoch_idx]  # (n_channels, n_times)
                            
                            # Truncate to global minimum length
                            if trial_data.shape[1] >= global_min_length:
                                trial_data_truncated = trial_data[:, :global_min_length]
                                
                                # Apply RC1 spatial filter: RC1_timecourse = W_RC1^T * data
                                rc1_timecourse = self.rc1_filter @ trial_data_truncated  # (n_samples,)
                                
                                # Store by song ID
                                if song_id not in subject_song_data:
                                    subject_song_data[song_id] = []
                                subject_song_data[song_id].append(rc1_timecourse)
                        
                    except Exception as e:
                        print(f"    ⚠️ Could not process {trial_file.name}: {e}")
                        continue
                
                # Average multiple trials for same song (if any)
                subject_song_final = {}
                for song_id, timecourses in subject_song_data.items():
                    if len(timecourses) > 1:
                        # Average multiple trials for same song
                        subject_song_final[song_id] = np.mean(timecourses, axis=0)
                        print(f"    ✓ {song_id}: averaged {len(timecourses)} trials")
                    else:
                        subject_song_final[song_id] = timecourses[0]
                        print(f"    ✓ {song_id}: single trial")
                
                processing_summary[subject_id] = {
                    'n_songs': len(subject_song_final),
                    'songs': list(subject_song_final.keys())
                }
                
                # Store in main data structure
                for song_id, timecourse in subject_song_final.items():
                    if song_id not in song_data:
                        song_data[song_id] = {}
                    song_data[song_id][subject_id] = timecourse
                
            except Exception as e:
                print(f"    ❌ Error processing {subject_id}: {e}")
                continue
        
        self.song_data = song_data
        
        # Print summary
        print(f"\n📊 RC1 APPLICATION SUMMARY:")
        print(f"  • Subjects processed: {len(processing_summary)}")
        print(f"  • Unique songs found: {len(song_data)}")
        
        # Show song coverage
        print(f"\n🎵 SONG COVERAGE:")
        for song_id in sorted(song_data.keys()):
            subjects_with_song = list(song_data[song_id].keys())
            print(f"  {song_id}: {len(subjects_with_song)} subjects ({', '.join(subjects_with_song)})")
        
        return song_data
    
    def compute_inter_subject_correlations(self):
        """Compute inter-subject correlations for each song."""
        print("\n🔗 Computing Inter-Subject Correlations")
        print("=" * 45)
        
        if not self.song_data:
            raise ValueError("No song data available. Run apply_rc1_filter_to_all_data() first.")
        
        isc_results = {}
        
        for song_id, subject_data in self.song_data.items():
            print(f"  Processing {song_id}...")
            
            # Get subjects with data for this song
            available_subjects = list(subject_data.keys())
            n_subjects = len(available_subjects)
            
            if n_subjects < 2:
                print(f"    ⚠️ Only {n_subjects} subject(s), skipping ISC")
                continue
            
            # Create correlation matrix
            correlation_matrix = np.ones((n_subjects, n_subjects))
            p_values = np.ones((n_subjects, n_subjects))
            
            # Compute pairwise correlations
            for i, subj1 in enumerate(available_subjects):
                for j, subj2 in enumerate(available_subjects):
                    if i != j:
                        timecourse1 = subject_data[subj1]
                        timecourse2 = subject_data[subj2]
                        
                        # Compute correlation
                        corr, p_val = pearsonr(timecourse1, timecourse2)
                        correlation_matrix[i, j] = corr
                        p_values[i, j] = p_val
            
            # Compute summary statistics
            # ISC is typically the mean of all pairwise correlations (excluding diagonal)
            mask = ~np.eye(n_subjects, dtype=bool)
            isc_values = correlation_matrix[mask]
            
            isc_summary = {
                'subjects': available_subjects,
                'n_subjects': n_subjects,
                'correlation_matrix': correlation_matrix,
                'p_values': p_values,
                'mean_isc': np.mean(isc_values),
                'std_isc': np.std(isc_values),
                'min_isc': np.min(isc_values),
                'max_isc': np.max(isc_values),
                'significant_pairs': np.sum(p_values[mask] < 0.05)
            }
            
            isc_results[song_id] = isc_summary
            
            print(f"    ✓ {n_subjects} subjects, ISC = {isc_summary['mean_isc']:.3f}±{isc_summary['std_isc']:.3f}")
        
        self.isc_results = isc_results
        
        print(f"\n📊 ISC COMPUTATION SUMMARY:")
        print(f"  • Songs analyzed: {len(isc_results)}")
        
        # Find songs with highest ISC
        if isc_results:
            isc_values = [(song_id, result['mean_isc']) for song_id, result in isc_results.items()]
            isc_values.sort(key=lambda x: x[1], reverse=True)
            
            print(f"  • Highest ISC: {isc_values[0][0]} ({isc_values[0][1]:.3f})")
            print(f"  • Lowest ISC: {isc_values[-1][0]} ({isc_values[-1][1]:.3f})")
        
        return isc_results
    
    def create_isc_visualization(self):
        """Create comprehensive ISC visualization."""
        print("\n📊 Creating ISC Visualizations")
        print("=" * 35)
        
        if not self.isc_results:
            raise ValueError("No ISC results available.")
        
        # Create comprehensive figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 1. ISC values by song
        ax1 = axes[0]
        songs = list(self.isc_results.keys())
        isc_means = [self.isc_results[song]['mean_isc'] for song in songs]
        isc_stds = [self.isc_results[song]['std_isc'] for song in songs]
        
        bars = ax1.bar(range(len(songs)), isc_means, yerr=isc_stds, capsize=3)
        ax1.set_xlabel('Song')
        ax1.set_ylabel('Inter-Subject Correlation')
        ax1.set_title('RC1 Inter-Subject Correlation by Song')
        ax1.set_xticks(range(len(songs)))
        ax1.set_xticklabels(songs, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Color bars by ISC strength
        for bar, isc_val in zip(bars, isc_means):
            if isc_val > 0.3:
                bar.set_color('darkgreen')
            elif isc_val > 0.1:
                bar.set_color('orange')
            else:
                bar.set_color('gray')
        
        # 2. ISC distribution
        ax2 = axes[1]
        all_isc_values = []
        for result in self.isc_results.values():
            mask = ~np.eye(result['n_subjects'], dtype=bool)
            isc_vals = result['correlation_matrix'][mask]
            all_isc_values.extend(isc_vals)
        
        ax2.hist(all_isc_values, bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(all_isc_values), color='red', linestyle='--', 
                   label=f'Mean = {np.mean(all_isc_values):.3f}')
        ax2.set_xlabel('ISC Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of ISC Values')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Number of subjects per song
        ax3 = axes[2]
        n_subjects_per_song = [self.isc_results[song]['n_subjects'] for song in songs]
        
        ax3.bar(range(len(songs)), n_subjects_per_song)
        ax3.set_xlabel('Song')
        ax3.set_ylabel('Number of Subjects')
        ax3.set_title('Subject Coverage per Song')
        ax3.set_xticks(range(len(songs)))
        ax3.set_xticklabels(songs, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # 4. Correlation matrix for best ISC song
        ax4 = axes[3]
        
        # Find song with highest mean ISC
        best_song = max(self.isc_results.keys(), 
                       key=lambda x: self.isc_results[x]['mean_isc'])
        best_result = self.isc_results[best_song]
        
        im = ax4.imshow(best_result['correlation_matrix'], cmap='RdBu_r', vmin=-1, vmax=1)
        ax4.set_title(f'ISC Matrix: {best_song}\n(Mean ISC = {best_result["mean_isc"]:.3f})')
        ax4.set_xlabel('Subject')
        ax4.set_ylabel('Subject')
        
        # Add subject labels
        ax4.set_xticks(range(len(best_result['subjects'])))
        ax4.set_yticks(range(len(best_result['subjects'])))
        ax4.set_xticklabels(best_result['subjects'])
        ax4.set_yticklabels(best_result['subjects'])
        
        plt.colorbar(im, ax=ax4)
        
        # 5. ISC vs preference analysis
        ax5 = axes[4]
        
        if hasattr(self, 'subject_preferences'):
            # Analyze if preferred songs have higher ISC
            preferred_isc = []
            nonpreferred_isc = []
            
            for song_id, result in self.isc_results.items():
                # Check if this song is preferred by any subject with data
                is_preferred_by_someone = False
                for subject_id in result['subjects']:
                    if (subject_id in self.subject_preferences and 
                        song_id in self.subject_preferences[subject_id]['top_3']):
                        is_preferred_by_someone = True
                        break
                
                if is_preferred_by_someone:
                    preferred_isc.append(result['mean_isc'])
                else:
                    nonpreferred_isc.append(result['mean_isc'])
            
            # Box plot comparison
            data_to_plot = []
            labels = []
            if preferred_isc:
                data_to_plot.append(preferred_isc)
                labels.append('Preferred\nby Someone')
            if nonpreferred_isc:
                data_to_plot.append(nonpreferred_isc)
                labels.append('Not Top\nPreferred')
            
            if data_to_plot:
                box_plot = ax5.boxplot(data_to_plot, labels=labels, patch_artist=True)
                ax5.set_ylabel('ISC Value')
                ax5.set_title('ISC by Preference Status')
                ax5.grid(True, alpha=0.3)
                
                # Color boxes
                colors = ['lightcoral', 'lightblue']
                for patch, color in zip(box_plot['boxes'], colors):
                    patch.set_facecolor(color)
        
        # 6. Time course examples
        ax6 = axes[5]
        
        # Show RC1 time courses for the best ISC song
        best_result = self.isc_results[best_song]
        song_data = self.song_data[best_song]
        
        # Plot first 1000 samples for visualization
        for i, subject_id in enumerate(best_result['subjects'][:4]):  # Show max 4 subjects
            timecourse = song_data[subject_id][:1000]  # First 1000 samples
            ax6.plot(timecourse, alpha=0.7, label=subject_id)
        
        ax6.set_xlabel('Time (samples)')
        ax6.set_ylabel('RC1 Amplitude')
        ax6.set_title(f'RC1 Time Courses: {best_song}')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'RC1 Inter-Subject Correlation Analysis\n'
                    f'{len(self.isc_results)} songs analyzed', 
                    fontsize=16)
        
        plt.tight_layout()
        
        # Save the figure
        isc_plot_path = self.output_dir / 'rc1_isc_analysis.png'
        plt.savefig(isc_plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 ISC visualization saved: {isc_plot_path}")
        
        return fig
    
    def save_isc_results(self):
        """Save ISC analysis results."""
        print("\n💾 Saving ISC Results")
        print("=" * 20)
        
        # Save detailed results as npz
        save_data = {
            'rc1_filter': self.rc1_filter,
            'channel_names': np.array(self.channel_names),
            'songs_analyzed': list(self.isc_results.keys())
        }
        
        # Add ISC results for each song
        for song_id, result in self.isc_results.items():
            save_data[f'{song_id}_subjects'] = np.array(result['subjects'])
            save_data[f'{song_id}_correlation_matrix'] = result['correlation_matrix']
            save_data[f'{song_id}_p_values'] = result['p_values']
            save_data[f'{song_id}_mean_isc'] = result['mean_isc']
            save_data[f'{song_id}_std_isc'] = result['std_isc']
        
        results_path = self.output_dir / 'rc1_isc_results.npz'
        np.savez_compressed(results_path, **save_data)
        print(f"📄 Detailed results saved: {results_path}")
        
        # Save summary as CSV
        summary_data = []
        for song_id, result in self.isc_results.items():
            summary_data.append({
                'Song': song_id,
                'N_Subjects': result['n_subjects'],
                'Mean_ISC': result['mean_isc'],
                'Std_ISC': result['std_isc'],
                'Min_ISC': result['min_isc'],
                'Max_ISC': result['max_isc'],
                'Significant_Pairs': result['significant_pairs'],
                'Subjects': '; '.join(result['subjects'])
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = self.output_dir / 'rc1_isc_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"📊 Summary table saved: {summary_path}")
    
    def run_complete_isc_analysis(self):
        """Run the complete RC1 inter-subject correlation analysis."""
        print("🧠 RC1 INTER-SUBJECT CORRELATION ANALYSIS")
        print("=" * 60)
        print("Applying RC1 spatial filter and computing neural synchrony across subjects...")
        print()
        
        try:
            # Step 1: Load RC1 spatial filter
            self.load_rc1_spatial_filter()
            
            # Step 2: Load behavioral preferences
            self.load_behavioral_preferences()
            
            # Step 3: Apply RC1 filter to all data
            self.apply_rc1_filter_to_all_data()
            
            # Step 4: Compute inter-subject correlations
            self.compute_inter_subject_correlations()
            
            # Step 5: Create visualization
            plt.ioff()
            fig = self.create_isc_visualization()
            plt.close(fig)
            
            # Step 6: Save results
            self.save_isc_results()
            
            # Step 7: Print final summary
            self.print_final_summary()
            
            return True
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def print_final_summary(self):
        """Print final summary of ISC analysis."""
        print("\n" + "=" * 60)
        print("🎉 RC1 INTER-SUBJECT ANALYSIS COMPLETE!")
        print("=" * 60)
        
        print(f"📊 ANALYSIS SUMMARY:")
        print(f"  • RC1 spatial filter applied to individual song data")
        print(f"  • Songs analyzed: {len(self.isc_results)}")
        print(f"  • Total subject-song combinations: {sum(r['n_subjects'] for r in self.isc_results.values())}")
        
        if self.isc_results:
            all_isc_values = [r['mean_isc'] for r in self.isc_results.values()]
            print(f"  • Mean ISC across all songs: {np.mean(all_isc_values):.3f}±{np.std(all_isc_values):.3f}")
            
            # Top ISC songs
            sorted_songs = sorted(self.isc_results.items(), key=lambda x: x[1]['mean_isc'], reverse=True)
            
            print(f"\n🏆 TOP 3 SONGS BY NEURAL SYNCHRONY:")
            for i, (song_id, result) in enumerate(sorted_songs[:3]):
                print(f"  {i+1}. {song_id}: ISC = {result['mean_isc']:.3f} ({result['n_subjects']} subjects)")
            
            print(f"\n📉 LOWEST 3 SONGS BY NEURAL SYNCHRONY:")
            for i, (song_id, result) in enumerate(sorted_songs[-3:]):
                print(f"  {len(sorted_songs)-2+i}. {song_id}: ISC = {result['mean_isc']:.3f} ({result['n_subjects']} subjects)")
        
        print(f"\n📁 OUTPUT FILES:")
        print(f"  • {self.output_dir}/rc1_isc_analysis.png")
        print(f"  • {self.output_dir}/rc1_isc_results.npz")
        print(f"  • {self.output_dir}/rc1_isc_summary.csv")
        
        print(f"\n🧠 INTERPRETATION:")
        print("  • Higher ISC = more synchronized neural responses across subjects")
        print("  • RC1 represents the most reliable component from pooled analysis")
        print("  • Songs with high ISC may have universal neural processing patterns")
        print("  • Low ISC songs may reflect individual differences in processing")


def main():
    """Main execution function."""
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    # Run ISC analysis
    analyzer = RC1InterSubjectAnalysis()
    success = analyzer.run_complete_isc_analysis()
    
    if success:
        print("\n🧠 RC1 inter-subject analysis complete! Check output directory for results.")
    else:
        print("\n❌ Analysis failed. Check data and try again.")


if __name__ == "__main__":
    main()