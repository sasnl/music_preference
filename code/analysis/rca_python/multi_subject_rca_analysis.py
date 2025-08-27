#!/usr/bin/env python3
"""
Multi-Subject RCA Analysis for Music Preference Study.

This script performs comprehensive RCA analysis across all subjects in the music preference study,
identifies common reliable components, and creates group-level topographic visualizations.

Features:
- Individual subject RCA analysis
- Group-level component identification
- Cross-subject reliability assessment
- Comprehensive topographic visualizations
- Statistical summary across subjects
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import json
from typing import Dict, List, Any, Optional
import warnings

# Add RCA to path
rca_path = Path(__file__).parent / 'code' / 'analysis' / 'rca_python'
sys.path.insert(0, str(rca_path))

from rca_utils import (run_rca_on_music_data, plot_music_rca_topographies, 
                      compute_rca_reliability_metrics, save_rca_results)


class MultiSubjectRCAAnalysis:
    """
    Comprehensive multi-subject RCA analysis for music preference study.
    """
    
    def __init__(self, data_dir: str = "data/ica_cleaned", output_dir: str = "output/multi_subject_rca"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Analysis parameters
        self.n_components = 5  # Extract more components for group analysis
        self.subjects = ['pilot_1', 'pilot_2', 'pilot_3', 'pilot_4', 'pilot_5']
        
        # Results storage
        self.individual_results = {}
        self.group_analysis = {}
        
    def run_individual_analyses(self) -> Dict[str, Any]:
        """Run RCA analysis for each subject individually."""
        print("🧠 Running Individual Subject RCA Analyses")
        print("=" * 50)
        
        individual_results = {}
        successful_subjects = []
        
        for subject_id in self.subjects:
            print(f"\n📊 Analyzing {subject_id}...")
            
            try:
                # Run RCA analysis
                results = run_rca_on_music_data(
                    subject_id=subject_id,
                    data_dir=self.data_dir,
                    n_components=self.n_components,
                    compare_conditions=True
                )
                
                # Compute metrics
                metrics = compute_rca_reliability_metrics(results)
                results['metrics'] = metrics
                
                # Store results
                individual_results[subject_id] = results
                successful_subjects.append(subject_id)
                
                print(f"✅ {subject_id}: {results['rca_model'].n_components} components, "
                      f"λ_max={results['rca_model'].eigenvalues_[0]:.4f}")
                      
                # Save individual results
                individual_output_dir = self.output_dir / subject_id
                individual_output_dir.mkdir(exist_ok=True)
                
                # Save numerical results
                save_rca_results(results, individual_output_dir / f"{subject_id}_rca_results.npz")
                
                # Create individual topographic plot
                plt.ioff()
                fig = plot_music_rca_topographies(
                    results, 
                    save_path=individual_output_dir / f"{subject_id}_topography.png"
                )
                plt.close(fig)
                
            except Exception as e:
                print(f"❌ {subject_id}: Failed - {e}")
                continue
        
        print(f"\n✅ Individual analysis complete: {len(successful_subjects)}/{len(self.subjects)} subjects successful")
        
        self.individual_results = individual_results
        self.successful_subjects = successful_subjects
        
        return individual_results
    
    def compute_group_statistics(self) -> Dict[str, Any]:
        """Compute group-level statistics across subjects."""
        print("\n📈 Computing Group-Level Statistics")
        print("=" * 40)
        
        if not self.individual_results:
            raise ValueError("No individual results available. Run individual analyses first.")
        
        # Collect eigenvalues across subjects
        all_eigenvalues = []
        all_metrics = []
        component_data = {f'RC{i+1}': {'eigenvalues': [], 'max_channels': [], 'activations': []} 
                         for i in range(self.n_components)}
        
        for subject_id, results in self.individual_results.items():
            rca = results['rca_model']
            metrics = results['metrics']
            
            # Store eigenvalues
            eigenvals = rca.eigenvalues_[:self.n_components]
            all_eigenvalues.append(eigenvals)
            all_metrics.append(metrics)
            
            # Store component-specific data
            for comp in range(min(self.n_components, len(eigenvals))):
                component_data[f'RC{comp+1}']['eigenvalues'].append(eigenvals[comp])
                
                # Find maximum activation channel
                max_idx = np.argmax(np.abs(rca.forward_models_[:, comp]))
                max_channel = results['channel_names'][max_idx]
                component_data[f'RC{comp+1}']['max_channels'].append(max_channel)
                component_data[f'RC{comp+1}']['activations'].append(rca.forward_models_[max_idx, comp])
        
        # Convert to arrays for statistics
        eigenvalue_matrix = np.array(all_eigenvalues)  # (n_subjects, n_components)
        
        # Compute statistics
        group_stats = {
            'n_subjects': len(self.individual_results),
            'eigenvalue_stats': {
                'mean': np.mean(eigenvalue_matrix, axis=0),
                'std': np.std(eigenvalue_matrix, axis=0),
                'min': np.min(eigenvalue_matrix, axis=0),
                'max': np.max(eigenvalue_matrix, axis=0)
            },
            'component_data': component_data,
            'consistent_components': []
        }
        
        # Identify consistent components (present in most subjects)
        reliability_threshold = 0.005  # Minimum eigenvalue for reliable component
        for comp in range(self.n_components):
            reliable_count = np.sum(eigenvalue_matrix[:, comp] > reliability_threshold)
            consistency = reliable_count / len(self.individual_results)
            
            if consistency >= 0.6:  # Present in at least 60% of subjects
                group_stats['consistent_components'].append({
                    'component': f'RC{comp+1}',
                    'consistency': consistency,
                    'mean_eigenvalue': group_stats['eigenvalue_stats']['mean'][comp],
                    'std_eigenvalue': group_stats['eigenvalue_stats']['std'][comp]
                })
        
        self.group_analysis = group_stats
        
        # Print summary
        print(f"Subjects analyzed: {group_stats['n_subjects']}")
        print(f"Consistent components found: {len(group_stats['consistent_components'])}")
        
        for comp_info in group_stats['consistent_components']:
            print(f"  {comp_info['component']}: {comp_info['consistency']:.1%} consistency, "
                  f"λ={comp_info['mean_eigenvalue']:.4f}±{comp_info['std_eigenvalue']:.4f}")
        
        return group_stats
    
    def create_group_topographic_analysis(self) -> plt.Figure:
        """Create group-level topographic analysis."""
        print("\n🗺️ Creating Group Topographic Analysis")
        print("=" * 40)
        
        if not self.individual_results or not self.group_analysis:
            raise ValueError("Need individual results and group analysis first.")
        
        # Use subject with best overall reliability for reference topography
        best_subject = None
        best_reliability = -1
        
        for subject_id, results in self.individual_results.items():
            overall_reliability = np.mean(results['rca_model'].eigenvalues_[:3])
            if overall_reliability > best_reliability:
                best_reliability = overall_reliability
                best_subject = subject_id
        
        print(f"Using {best_subject} as reference (highest reliability: {best_reliability:.4f})")
        
        # Create comprehensive group visualization
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Group eigenvalue statistics
        ax_eigen = plt.subplot(3, 4, 1)
        eigenval_stats = self.group_analysis['eigenvalue_stats']
        components = range(1, self.n_components + 1)
        
        ax_eigen.errorbar(components, eigenval_stats['mean'], yerr=eigenval_stats['std'], 
                         fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8)
        ax_eigen.set_xlabel('Component')
        ax_eigen.set_ylabel('Eigenvalue')
        ax_eigen.set_title('Group Average Eigenvalues\n(Mean ± SD)')
        ax_eigen.grid(True, alpha=0.3)
        
        # 2. Individual subject eigenvalue comparison
        ax_indiv = plt.subplot(3, 4, 2)
        
        for i, (subject_id, results) in enumerate(self.individual_results.items()):
            eigenvals = results['rca_model'].eigenvalues_[:self.n_components]
            ax_indiv.plot(components, eigenvals, 'o-', alpha=0.7, label=subject_id)
        
        ax_indiv.set_xlabel('Component')
        ax_indiv.set_ylabel('Eigenvalue')
        ax_indiv.set_title('Individual Subject Eigenvalues')
        ax_indiv.legend(fontsize=10)
        ax_indiv.grid(True, alpha=0.3)
        
        # 3. Consistency analysis
        ax_consist = plt.subplot(3, 4, 3)
        
        if self.group_analysis['consistent_components']:
            consistent_comps = [c['component'] for c in self.group_analysis['consistent_components']]
            consistencies = [c['consistency'] for c in self.group_analysis['consistent_components']]
            
            bars = ax_consist.bar(consistent_comps, consistencies)
            ax_consist.set_ylabel('Consistency (proportion)')
            ax_consist.set_title('Component Consistency\nAcross Subjects')
            ax_consist.set_ylim(0, 1)
            
            # Color bars by consistency
            for bar, consistency in zip(bars, consistencies):
                if consistency >= 0.8:
                    bar.set_color('green')
                elif consistency >= 0.6:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
        
        # 4. Reference topographies (from best subject)
        if best_subject:
            reference_results = self.individual_results[best_subject]
            rca_ref = reference_results['rca_model']
            
            # Import topographic plotting
            import mne
            from mne.viz import plot_topomap
            
            info = mne.create_info(ch_names=reference_results['channel_names'], 
                                 sfreq=1000, ch_types='eeg')
            info.set_montage('standard_1020')
            
            # Plot top 3 components as reference
            for comp in range(min(3, rca_ref.n_components)):
                ax_topo = plt.subplot(3, 4, 5 + comp)
                
                spatial_pattern = rca_ref.forward_models_[:, comp]
                im, _ = plot_topomap(spatial_pattern, info, axes=ax_topo, show=False, 
                                   cmap='RdBu_r', contours=4)
                
                eigenval = rca_ref.eigenvalues_[comp]
                ax_topo.set_title(f'RC{comp+1} Reference\n(λ={eigenval:.4f})', fontsize=11)
                
                # Add small colorbar
                cbar = plt.colorbar(im, ax=ax_topo, shrink=0.6)
                cbar.ax.tick_params(labelsize=8)
        
        # 5. Channel activation frequency analysis
        ax_channels = plt.subplot(3, 4, 8)
        
        # Count which channels are most frequently the maximum for each component
        channel_counts = {}
        for comp_data in self.group_analysis['component_data'].values():
            for channel in comp_data['max_channels']:
                if channel not in channel_counts:
                    channel_counts[channel] = 0
                channel_counts[channel] += 1
        
        # Plot most frequent channels
        if channel_counts:
            sorted_channels = sorted(channel_counts.items(), key=lambda x: x[1], reverse=True)
            top_channels = sorted_channels[:8]  # Top 8 channels
            
            channels, counts = zip(*top_channels)
            bars = ax_channels.bar(channels, counts)
            ax_channels.set_ylabel('Frequency')
            ax_channels.set_title('Most Active Channels\nAcross Components')
            ax_channels.tick_params(axis='x', rotation=45)
        
        # 6-8. Preference comparison statistics
        preference_stats = self._compute_preference_statistics()
        
        if preference_stats:
            # 6. Preference effect sizes
            ax_effect = plt.subplot(3, 4, 9)
            
            effect_sizes = [preference_stats[subj]['effect_sizes'] for subj in preference_stats.keys()]
            if effect_sizes:
                effect_matrix = np.array(effect_sizes)
                mean_effects = np.mean(effect_matrix, axis=0)
                std_effects = np.std(effect_matrix, axis=0)
                
                components = range(1, len(mean_effects) + 1)
                ax_effect.errorbar(components, mean_effects, yerr=std_effects, 
                                 fmt='s-', capsize=4, linewidth=2, markersize=6)
                ax_effect.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                ax_effect.set_xlabel('Component')
                ax_effect.set_ylabel('Effect Size (Cohen\'s d)')
                ax_effect.set_title('Preference Effect Sizes\n(Preferred vs Non-preferred)')
                ax_effect.grid(True, alpha=0.3)
            
            # 7. Trial counts comparison
            ax_trials = plt.subplot(3, 4, 10)
            
            trial_data = []
            for subj_data in preference_stats.values():
                trial_data.append([subj_data['n_preferred'], subj_data['n_nonpreferred']])
            
            if trial_data:
                trial_matrix = np.array(trial_data)
                subjects_list = list(preference_stats.keys())
                
                x_pos = np.arange(len(subjects_list))
                width = 0.35
                
                ax_trials.bar(x_pos - width/2, trial_matrix[:, 0], width, 
                            label='Preferred', alpha=0.8)
                ax_trials.bar(x_pos + width/2, trial_matrix[:, 1], width, 
                            label='Non-preferred', alpha=0.8)
                
                ax_trials.set_xlabel('Subject')
                ax_trials.set_ylabel('Number of Trials')
                ax_trials.set_title('Trial Counts by Condition')
                ax_trials.set_xticks(x_pos)
                ax_trials.set_xticklabels(subjects_list, rotation=45)
                ax_trials.legend()
        
        # Add overall title and layout
        plt.suptitle(f'Multi-Subject RCA Analysis: Music Preference Study\n'
                    f'{len(self.individual_results)} Subjects, {self.n_components} Components', 
                    fontsize=16, y=0.98)
        
        plt.tight_layout()
        
        # Save the group analysis plot
        group_plot_path = self.output_dir / 'group_rca_analysis.png'
        plt.savefig(group_plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Group analysis plot saved: {group_plot_path}")
        
        return fig
    
    def _compute_preference_statistics(self) -> Dict[str, Dict]:
        """Compute preference-related statistics across subjects."""
        preference_stats = {}
        
        for subject_id, results in self.individual_results.items():
            if 'preferred_rca' in results and 'nonpreferred_rca' in results:
                n_components = results['rca_model'].n_components
                effect_sizes = []
                
                for comp in range(n_components):
                    # Compute Cohen's d for each component
                    pref_data = results['preferred_rca'][:, comp, :].flatten()
                    nonpref_data = results['nonpreferred_rca'][:, comp, :].flatten()
                    
                    mean_diff = np.mean(pref_data) - np.mean(nonpref_data)
                    pooled_std = np.sqrt(0.5 * (np.std(pref_data)**2 + np.std(nonpref_data)**2))
                    
                    if pooled_std > 0:
                        cohens_d = mean_diff / pooled_std
                    else:
                        cohens_d = 0
                    
                    effect_sizes.append(cohens_d)
                
                preference_stats[subject_id] = {
                    'effect_sizes': effect_sizes,
                    'n_preferred': results['n_preferred_trials'],
                    'n_nonpreferred': results['n_nonpreferred_trials']
                }
        
        return preference_stats
    
    def save_group_results(self):
        """Save comprehensive group results to files."""
        print("\n💾 Saving Group Results")
        print("=" * 25)
        
        # Save group statistics as JSON
        group_stats_path = self.output_dir / 'group_statistics.json'
        
        # Convert numpy arrays to lists for JSON serialization
        json_stats = {}
        for key, value in self.group_analysis.items():
            if key == 'eigenvalue_stats':
                json_stats[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                 for k, v in value.items()}
            else:
                json_stats[key] = value
        
        with open(group_stats_path, 'w') as f:
            json.dump(json_stats, f, indent=2)
        
        print(f"📄 Group statistics saved: {group_stats_path}")
        
        # Create summary DataFrame
        summary_data = []
        for subject_id, results in self.individual_results.items():
            rca = results['rca_model']
            summary_data.append({
                'Subject': subject_id,
                'RC1_Eigenvalue': rca.eigenvalues_[0],
                'RC2_Eigenvalue': rca.eigenvalues_[1] if len(rca.eigenvalues_) > 1 else np.nan,
                'RC3_Eigenvalue': rca.eigenvalues_[2] if len(rca.eigenvalues_) > 2 else np.nan,
                'N_Preferred_Trials': results.get('n_preferred_trials', 0),
                'N_NonPreferred_Trials': results.get('n_nonpreferred_trials', 0),
                'Overall_Reliability': np.mean(rca.eigenvalues_[:3])
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = self.output_dir / 'subject_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        
        print(f"📊 Subject summary saved: {summary_path}")
        
    def run_complete_analysis(self):
        """Run the complete multi-subject RCA analysis pipeline."""
        print("🎵 MULTI-SUBJECT RCA ANALYSIS")
        print("=" * 60)
        print("Analyzing music preference data across all subjects...")
        print()
        
        # Step 1: Individual analyses
        self.run_individual_analyses()
        
        # Step 2: Group statistics
        if self.individual_results:
            self.compute_group_statistics()
            
            # Step 3: Group visualization
            plt.ioff()
            fig = self.create_group_topographic_analysis()
            plt.close(fig)
            
            # Step 4: Save results
            self.save_group_results()
            
            # Step 5: Print final summary
            self.print_final_summary()
            
            return True
        else:
            print("❌ No successful individual analyses. Cannot proceed with group analysis.")
            return False
    
    def print_final_summary(self):
        """Print comprehensive final summary."""
        print("\n" + "=" * 60)
        print("🎉 MULTI-SUBJECT RCA ANALYSIS COMPLETE!")
        print("=" * 60)
        
        print(f"📊 SUMMARY:")
        print(f"  • Subjects analyzed: {len(self.individual_results)}/{len(self.subjects)}")
        print(f"  • Components extracted: {self.n_components}")
        print(f"  • Consistent components: {len(self.group_analysis['consistent_components'])}")
        
        if self.group_analysis['consistent_components']:
            print("\n🧠 RELIABLE COMPONENTS FOUND:")
            for comp_info in self.group_analysis['consistent_components']:
                print(f"  {comp_info['component']}: {comp_info['consistency']:.0%} consistency, "
                      f"λ={comp_info['mean_eigenvalue']:.4f}±{comp_info['std_eigenvalue']:.4f}")
        
        print(f"\n📁 OUTPUT DIRECTORY: {self.output_dir}")
        print("  • Individual subject analyses and topographies")
        print("  • group_rca_analysis.png - Comprehensive group visualization")
        print("  • group_statistics.json - Detailed numerical results")
        print("  • subject_summary.csv - Subject-wise summary table")
        
        print(f"\n🎯 NEXT STEPS:")
        print("  1. Examine group_rca_analysis.png for overall patterns")
        print("  2. Check individual topographies for subject-specific patterns")
        print("  3. Focus on consistent components for further analysis")
        print("  4. Compare preference effects across subjects")


def main():
    """Main analysis function."""
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    # Initialize and run analysis
    analyzer = MultiSubjectRCAAnalysis()
    success = analyzer.run_complete_analysis()
    
    if success:
        print("\n🎵 Analysis complete! Check the output directory for results.")
    else:
        print("\n❌ Analysis failed. Check data availability and try again.")


if __name__ == "__main__":
    main()