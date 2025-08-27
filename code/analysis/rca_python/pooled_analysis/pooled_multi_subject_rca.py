#!/usr/bin/env python3
"""
Pooled Multi-Subject RCA Analysis for Music Preference Study.

This script pools data from ALL subjects and runs a single RCA analysis on the 
combined dataset to find reliable components that are common across all participants.

Key features:
- Pool all subject data into single large dataset
- Run single RCA analysis on combined data
- Find reliable components shared across all subjects
- Create comprehensive topographic visualizations
- Compare pooled preferred vs non-preferred responses
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Any

# Add RCA to path
rca_path = Path(__file__).parent.parent  # Go up to rca_python directory
sys.path.insert(0, str(rca_path))

from rca import ReliableComponentsAnalysis
from rca_utils import (load_music_preference_data, epochs_to_rca_format_fixed_length, 
                      plot_music_rca_topographies, compute_rca_reliability_metrics)


class PooledMultiSubjectRCA:
    """
    Pooled multi-subject RCA analysis - combines all subject data before analysis.
    """
    
    def __init__(self, data_dir: str = "data/ica_cleaned", output_dir: str = "output/pooled_rca"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.subjects = ['pilot_1', 'pilot_2', 'pilot_3', 'pilot_4', 'pilot_5']
        self.n_components = 5
        
        # Data storage
        self.pooled_data = None
        self.pooled_results = None
        self.subject_contributions = {}
        
    def load_and_pool_all_data(self) -> Dict[str, Any]:
        """Load data from all subjects and pool into single dataset."""
        print("🔗 Pooling Data from All Subjects")
        print("=" * 40)
        
        all_preferred_data = []
        all_nonpreferred_data = []
        all_channel_names = None
        subject_trial_info = {}
        
        # Global minimum length across ALL subjects and trials
        global_min_length = float('inf')
        
        print("Step 1: Finding global minimum trial length...")
        
        # First pass: find global minimum length
        for subject_id in self.subjects:
            print(f"  Checking {subject_id}...")
            
            try:
                data_dict = load_music_preference_data(subject_id, self.data_dir)
                
                if not data_dict['preferred'] and not data_dict['nonpreferred']:
                    print(f"    ⚠️ No data available for {subject_id}")
                    continue
                
                # Check all trials for minimum length
                all_epochs = data_dict['preferred'] + data_dict['nonpreferred']
                for epochs in all_epochs:
                    epoch_data = epochs.get_data()
                    for epoch_idx in range(epoch_data.shape[0]):
                        trial_length = epoch_data[epoch_idx].shape[1]
                        global_min_length = min(global_min_length, trial_length)
                
                print(f"    ✓ {len(data_dict['preferred'])} preferred, {len(data_dict['nonpreferred'])} non-preferred trials")
                
            except Exception as e:
                print(f"    ❌ Error loading {subject_id}: {e}")
                continue
        
        print(f"📏 Global minimum length: {global_min_length} samples")
        print()
        
        print("Step 2: Loading and pooling all data...")
        
        # Second pass: load and pool data with global minimum length
        for subject_id in self.subjects:
            print(f"  Processing {subject_id}...")
            
            try:
                data_dict = load_music_preference_data(subject_id, self.data_dir)
                
                if not data_dict['preferred'] and not data_dict['nonpreferred']:
                    continue
                
                # Store channel names (should be consistent across subjects)
                if all_channel_names is None and data_dict['preferred']:
                    all_channel_names = data_dict['preferred'][0].ch_names
                
                # Convert to RCA format with global minimum length
                preferred_trials = 0
                nonpreferred_trials = 0
                
                if data_dict['preferred']:
                    preferred_data = epochs_to_rca_format_fixed_length(
                        data_dict['preferred'], global_min_length)
                    all_preferred_data.append(preferred_data)
                    preferred_trials = preferred_data.shape[2]
                
                if data_dict['nonpreferred']:
                    nonpreferred_data = epochs_to_rca_format_fixed_length(
                        data_dict['nonpreferred'], global_min_length)
                    all_nonpreferred_data.append(nonpreferred_data)
                    nonpreferred_trials = nonpreferred_data.shape[2]
                
                subject_trial_info[subject_id] = {
                    'preferred_trials': preferred_trials,
                    'nonpreferred_trials': nonpreferred_trials
                }
                
                print(f"    ✓ Added {preferred_trials} preferred, {nonpreferred_trials} non-preferred trials")
                
            except Exception as e:
                print(f"    ❌ Error processing {subject_id}: {e}")
                continue
        
        # Combine all data
        print("\nStep 3: Combining data across subjects...")
        
        pooled_preferred = None
        pooled_nonpreferred = None
        
        if all_preferred_data:
            pooled_preferred = np.concatenate(all_preferred_data, axis=2)
            print(f"  ✓ Combined preferred data: {pooled_preferred.shape}")
        
        if all_nonpreferred_data:
            pooled_nonpreferred = np.concatenate(all_nonpreferred_data, axis=2)
            print(f"  ✓ Combined non-preferred data: {pooled_nonpreferred.shape}")
        
        if pooled_preferred is None or pooled_nonpreferred is None:
            raise ValueError("Could not pool data - no valid data found!")
        
        # Store pooled data
        self.pooled_data = {
            'preferred_data': pooled_preferred,
            'nonpreferred_data': pooled_nonpreferred,
            'channel_names': all_channel_names,
            'global_min_length': global_min_length,
            'subject_contributions': subject_trial_info
        }
        
        print(f"\n🎉 Data pooling complete!")
        print(f"  • Total preferred trials: {pooled_preferred.shape[2]}")
        print(f"  • Total non-preferred trials: {pooled_nonpreferred.shape[2]}")
        print(f"  • Data length: {global_min_length} samples")
        print(f"  • Channels: {len(all_channel_names)}")
        print(f"  • Contributing subjects: {len(subject_trial_info)}")
        
        return self.pooled_data
    
    def run_pooled_rca_analysis(self) -> Dict[str, Any]:
        """Run RCA analysis on the pooled dataset."""
        print("\n🧠 Running RCA Analysis on Pooled Data")
        print("=" * 45)
        
        if self.pooled_data is None:
            raise ValueError("No pooled data available. Run load_and_pool_all_data() first.")
        
        preferred_data = self.pooled_data['preferred_data']
        nonpreferred_data = self.pooled_data['nonpreferred_data']
        
        print(f"Input data shapes:")
        print(f"  Preferred: {preferred_data.shape}")
        print(f"  Non-preferred: {nonpreferred_data.shape}")
        
        # Combine all data for RCA fitting
        print(f"\nCombining all trials for RCA analysis...")
        all_data = np.concatenate([preferred_data, nonpreferred_data], axis=2)
        print(f"  Combined shape: {all_data.shape}")
        
        # Fit RCA model
        print(f"\nFitting RCA model with {self.n_components} components...")
        rca = ReliableComponentsAnalysis(n_components=self.n_components, random_state=42)
        rca.fit(all_data)
        
        print("✅ RCA fitting complete!")
        
        # Transform each condition separately
        print("Transforming data for each condition...")
        preferred_rca = rca.transform(preferred_data)
        nonpreferred_rca = rca.transform(nonpreferred_data)
        
        # Create results structure
        results = {
            'subject_id': 'pooled_all_subjects',
            'rca_model': rca,
            'preferred_data': preferred_data,
            'nonpreferred_data': nonpreferred_data,
            'preferred_rca': preferred_rca,
            'nonpreferred_rca': nonpreferred_rca,
            'n_preferred_trials': preferred_data.shape[2],
            'n_nonpreferred_trials': nonpreferred_data.shape[2],
            'channel_names': self.pooled_data['channel_names'],
            'pooling_info': self.pooled_data['subject_contributions']
        }
        
        # Compute reliability metrics
        print("Computing reliability metrics...")
        metrics = compute_rca_reliability_metrics(results)
        results['metrics'] = metrics
        
        print(f"\n📊 POOLED RCA RESULTS:")
        print(f"  • Components extracted: {rca.n_components}")
        print(f"  • Total trials analyzed: {all_data.shape[2]}")
        print(f"  • Preferred trials: {preferred_data.shape[2]}")
        print(f"  • Non-preferred trials: {nonpreferred_data.shape[2]}")
        print(f"  • Data dimensions: {all_data.shape[0]} samples × {all_data.shape[1]} channels")
        
        print(f"\n🧠 COMPONENT RELIABILITY:")
        for i in range(rca.n_components):
            eigenval = rca.eigenvalues_[i]
            max_channel_idx = np.argmax(np.abs(rca.forward_models_[:, i]))
            max_channel = self.pooled_data['channel_names'][max_channel_idx]
            max_weight = rca.forward_models_[max_channel_idx, i]
            
            print(f"  RC{i+1}: λ={eigenval:.6f}, max at {max_channel} ({max_weight:.3f})")
        
        self.pooled_results = results
        return results
    
    def create_pooled_topographic_visualization(self) -> plt.Figure:
        """Create comprehensive topographic visualization for pooled analysis."""
        print("\n🗺️ Creating Pooled Topographic Visualization")
        print("=" * 45)
        
        if self.pooled_results is None:
            raise ValueError("No pooled results available. Run RCA analysis first.")
        
        # Use the enhanced topographic plotting
        fig = plot_music_rca_topographies(
            self.pooled_results,
            save_path=self.output_dir / 'pooled_rca_topographies.png'
        )
        
        print("✅ Topographic visualization created!")
        return fig
    
    def create_pooled_summary_analysis(self) -> plt.Figure:
        """Create additional summary analysis specific to pooled data."""
        print("\n📊 Creating Pooled Summary Analysis")
        print("=" * 40)
        
        results = self.pooled_results
        rca = results['rca_model']
        
        # Create summary figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # 1. Eigenvalue spectrum
        ax1 = axes[0]
        eigenvals_to_plot = rca.eigenvalues_[:min(10, len(rca.eigenvalues_))]
        bars = ax1.bar(range(1, len(eigenvals_to_plot) + 1), eigenvals_to_plot)
        
        # Highlight extracted components
        for i in range(min(self.n_components, len(bars))):
            bars[i].set_color('red')
            bars[i].set_alpha(0.8)
        
        ax1.set_xlabel('Component')
        ax1.set_ylabel('Eigenvalue (Reliability)')
        ax1.set_title('Pooled RCA Eigenvalue Spectrum')
        ax1.grid(True, alpha=0.3)
        
        # 2. Subject contributions
        ax2 = axes[1]
        subjects = list(results['pooling_info'].keys())
        preferred_counts = [results['pooling_info'][s]['preferred_trials'] for s in subjects]
        nonpreferred_counts = [results['pooling_info'][s]['nonpreferred_trials'] for s in subjects]
        
        x_pos = np.arange(len(subjects))
        width = 0.35
        
        ax2.bar(x_pos - width/2, preferred_counts, width, label='Preferred', alpha=0.8)
        ax2.bar(x_pos + width/2, nonpreferred_counts, width, label='Non-preferred', alpha=0.8)
        
        ax2.set_xlabel('Subject')
        ax2.set_ylabel('Number of Trials')
        ax2.set_title('Trial Contributions by Subject')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(subjects, rotation=45)
        ax2.legend()
        
        # 3. Component activations comparison
        ax3 = axes[2]
        component_means_pref = []
        component_means_nonpref = []
        
        for comp in range(rca.n_components):
            pref_mean = np.mean(np.abs(results['preferred_rca'][:, comp, :]))
            nonpref_mean = np.mean(np.abs(results['nonpreferred_rca'][:, comp, :]))
            component_means_pref.append(pref_mean)
            component_means_nonpref.append(nonpref_mean)
        
        comp_nums = range(1, rca.n_components + 1)
        ax3.plot(comp_nums, component_means_pref, 'ro-', label='Preferred', linewidth=2, markersize=8)
        ax3.plot(comp_nums, component_means_nonpref, 'bo-', label='Non-preferred', linewidth=2, markersize=8)
        
        ax3.set_xlabel('Component')
        ax3.set_ylabel('Mean |Activation|')
        ax3.set_title('Component Activation Levels')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Preference effect sizes
        ax4 = axes[3]
        effect_sizes = []
        
        for comp in range(rca.n_components):
            pref_data = results['preferred_rca'][:, comp, :].flatten()
            nonpref_data = results['nonpreferred_rca'][:, comp, :].flatten()
            
            mean_diff = np.mean(pref_data) - np.mean(nonpref_data)
            pooled_std = np.sqrt(0.5 * (np.std(pref_data)**2 + np.std(nonpref_data)**2))
            
            if pooled_std > 0:
                cohens_d = mean_diff / pooled_std
            else:
                cohens_d = 0
            
            effect_sizes.append(cohens_d)
        
        bars = ax4.bar(comp_nums, effect_sizes)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Component')
        ax4.set_ylabel('Effect Size (Cohen\'s d)')
        ax4.set_title('Preference Effects (Pooled Analysis)')
        ax4.grid(True, alpha=0.3)
        
        # Color bars by effect size magnitude
        for bar, effect in zip(bars, effect_sizes):
            if abs(effect) > 0.5:
                bar.set_color('red')
            elif abs(effect) > 0.2:
                bar.set_color('orange')
            else:
                bar.set_color('gray')
        
        # 5. Spatial pattern correlation matrix
        ax5 = axes[4]
        spatial_patterns = rca.forward_models_[:, :rca.n_components]
        correlation_matrix = np.corrcoef(spatial_patterns.T)
        
        im = ax5.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax5.set_xlabel('Component')
        ax5.set_ylabel('Component')
        ax5.set_title('Spatial Pattern Correlations')
        
        # Add correlation values as text
        for i in range(rca.n_components):
            for j in range(rca.n_components):
                text = ax5.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black" if abs(correlation_matrix[i, j]) < 0.5 else "white")
        
        plt.colorbar(im, ax=ax5, shrink=0.8)
        
        # 6. Channel importance (most frequently max activation)
        ax6 = axes[5]
        
        max_channels = []
        for comp in range(rca.n_components):
            max_idx = np.argmax(np.abs(rca.forward_models_[:, comp]))
            max_channels.append(results['channel_names'][max_idx])
        
        # Count channel occurrences
        from collections import Counter
        channel_counts = Counter(max_channels)
        
        channels = list(channel_counts.keys())
        counts = list(channel_counts.values())
        
        ax6.bar(channels, counts)
        ax6.set_xlabel('Channel')
        ax6.set_ylabel('Frequency as Max')
        ax6.set_title('Most Important Channels')
        ax6.tick_params(axis='x', rotation=45)
        
        plt.suptitle(f'Pooled Multi-Subject RCA Analysis Summary\n'
                    f'{results["n_preferred_trials"] + results["n_nonpreferred_trials"]} total trials from '
                    f'{len(results["pooling_info"])} subjects', 
                    fontsize=14)
        
        plt.tight_layout()
        
        # Save summary
        summary_path = self.output_dir / 'pooled_rca_summary.png'
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        print(f"📊 Summary analysis saved: {summary_path}")
        
        return fig
    
    def save_pooled_results(self):
        """Save pooled analysis results."""
        print("\n💾 Saving Pooled Results")
        print("=" * 25)
        
        if self.pooled_results is None:
            raise ValueError("No results to save.")
        
        # Save main results
        results_path = self.output_dir / 'pooled_rca_results.npz'
        
        save_data = {
            'spatial_filters': self.pooled_results['rca_model'].spatial_filters_,
            'forward_models': self.pooled_results['rca_model'].forward_models_,
            'eigenvalues': self.pooled_results['rca_model'].eigenvalues_,
            'preferred_rca': self.pooled_results['preferred_rca'],
            'nonpreferred_rca': self.pooled_results['nonpreferred_rca'],
            'channel_names': np.array(self.pooled_results['channel_names']),
            'n_preferred_trials': self.pooled_results['n_preferred_trials'],
            'n_nonpreferred_trials': self.pooled_results['n_nonpreferred_trials'],
            'metrics': self.pooled_results['metrics']
        }
        
        np.savez_compressed(results_path, **save_data)
        print(f"📄 Results saved: {results_path}")
        
        # Save pooling info as JSON
        pooling_info_path = self.output_dir / 'subject_contributions.json'
        with open(pooling_info_path, 'w') as f:
            json.dump(self.pooled_results['pooling_info'], f, indent=2)
        
        print(f"📄 Subject contributions saved: {pooling_info_path}")
    
    def run_complete_pooled_analysis(self):
        """Run the complete pooled multi-subject RCA analysis."""
        print("🎵 POOLED MULTI-SUBJECT RCA ANALYSIS")
        print("=" * 50)
        print("Pooling all subject data and finding common reliable components...")
        print()
        
        try:
            # Step 1: Pool all data
            self.load_and_pool_all_data()
            
            # Step 2: Run RCA on pooled data
            self.run_pooled_rca_analysis()
            
            # Step 3: Create visualizations
            plt.ioff()
            
            # Main topographic visualization
            topo_fig = self.create_pooled_topographic_visualization()
            plt.close(topo_fig)
            
            # Summary analysis
            summary_fig = self.create_pooled_summary_analysis()
            plt.close(summary_fig)
            
            # Step 4: Save results
            self.save_pooled_results()
            
            # Step 5: Print final summary
            self.print_final_summary()
            
            return True
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def print_final_summary(self):
        """Print final summary of pooled analysis."""
        print("\n" + "=" * 50)
        print("🎉 POOLED RCA ANALYSIS COMPLETE!")
        print("=" * 50)
        
        results = self.pooled_results
        rca = results['rca_model']
        
        print(f"📊 POOLED ANALYSIS SUMMARY:")
        print(f"  • Contributing subjects: {len(results['pooling_info'])}")
        print(f"  • Total trials: {results['n_preferred_trials'] + results['n_nonpreferred_trials']}")
        print(f"  • Preferred trials: {results['n_preferred_trials']}")
        print(f"  • Non-preferred trials: {results['n_nonpreferred_trials']}")
        print(f"  • Data length: {self.pooled_data['global_min_length']} samples")
        print(f"  • Components extracted: {rca.n_components}")
        
        print(f"\n🧠 RELIABLE COMPONENTS FOUND:")
        for i in range(rca.n_components):
            eigenval = rca.eigenvalues_[i]
            max_idx = np.argmax(np.abs(rca.forward_models_[:, i]))
            max_channel = results['channel_names'][max_idx]
            
            reliability_level = "High" if eigenval > 0.01 else "Moderate" if eigenval > 0.005 else "Low"
            print(f"  RC{i+1}: λ={eigenval:.6f} ({reliability_level}), max at {max_channel}")
        
        print(f"\n📁 OUTPUT FILES:")
        print(f"  • {self.output_dir}/pooled_rca_topographies.png")
        print(f"  • {self.output_dir}/pooled_rca_summary.png") 
        print(f"  • {self.output_dir}/pooled_rca_results.npz")
        print(f"  • {self.output_dir}/subject_contributions.json")
        
        print(f"\n🎯 INTERPRETATION:")
        print("  • These components represent neural patterns reliable across ALL subjects")
        print("  • Higher eigenvalues indicate stronger cross-subject consistency")
        print("  • Compare topographic patterns to identify brain regions involved")
        print("  • Examine preference effects to understand music processing differences")


def main():
    """Main execution function."""
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    # Run pooled analysis
    analyzer = PooledMultiSubjectRCA()
    success = analyzer.run_complete_pooled_analysis()
    
    if success:
        print("\n🎵 Pooled analysis complete! Check output directory for results.")
    else:
        print("\n❌ Analysis failed. Check data and try again.")


if __name__ == "__main__":
    main()