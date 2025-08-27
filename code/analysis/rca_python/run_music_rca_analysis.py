#!/usr/bin/env python3
"""
Run RCA analysis on music preference EEG data.

This script demonstrates how to use the Python RCA implementation
on the music preference study data, comparing preferred vs non-preferred
music responses.

Usage:
    python run_music_rca_analysis.py [--subject SUBJECT_ID] [--all-subjects]
    
Examples:
    # Analyze single subject
    python run_music_rca_analysis.py --subject pilot_2
    
    # Analyze all subjects  
    python run_music_rca_analysis.py --all-subjects
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to path to import rca_python
sys.path.append(str(Path(__file__).parent.parent.parent))

from rca_python import (
    ReliableComponentsAnalysis,
    run_rca_on_music_data,
    batch_rca_analysis,
    plot_music_rca_results,
    compute_rca_reliability_metrics
)


def main():
    parser = argparse.ArgumentParser(description='Run RCA analysis on music preference data')
    parser.add_argument('--subject', type=str, help='Subject ID to analyze (e.g., pilot_2)')
    parser.add_argument('--all-subjects', action='store_true', 
                       help='Analyze all subjects (pilot_1 through pilot_5)')
    parser.add_argument('--n-components', type=int, default=3,
                       help='Number of RCA components to extract (default: 3)')
    parser.add_argument('--n-reg', type=int, default=None,
                       help='Regularization parameter (default: auto-detect)')
    parser.add_argument('--data-dir', type=str, 
                       default='../../../data/ica_cleaned',
                       help='Path to preprocessed data directory')
    parser.add_argument('--output-dir', type=str,
                       default='../../../output/rca_analysis',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Set up paths
    script_dir = Path(__file__).parent
    data_dir = Path(script_dir / args.data_dir).resolve()
    output_dir = Path(script_dir / args.output_dir).resolve()
    
    print("=== Music Preference RCA Analysis ===")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Components: {args.n_components}, Regularization: {args.n_reg}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.all_subjects:
        # Analyze all subjects
        subject_ids = ['pilot_1', 'pilot_2', 'pilot_3', 'pilot_4', 'pilot_5']
        print(f"Running batch analysis on subjects: {subject_ids}")
        
        results = batch_rca_analysis(
            subject_ids=subject_ids,
            data_dir=data_dir,
            output_dir=output_dir,
            n_components=args.n_components,
            n_reg=args.n_reg
        )
        
        # Create summary plot
        create_group_summary_plot(results, output_dir)
        
    elif args.subject:
        # Analyze single subject
        print(f"Analyzing subject: {args.subject}")
        
        try:
            results = run_rca_on_music_data(
                subject_id=args.subject,
                data_dir=data_dir,
                n_components=args.n_components,
                n_reg=args.n_reg
            )
            
            # Create plots
            fig = plot_music_rca_results(results)
            plot_path = output_dir / f"{args.subject}_rca_analysis.png"
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {plot_path}")
            
            # Compute and print metrics
            metrics = compute_rca_reliability_metrics(results)
            print("\n=== RCA Reliability Metrics ===")
            print(f"Eigenvalues: {metrics['eigenvalues']}")
            print(f"Explained variance ratio: {metrics['explained_variance_ratio']}")
            if 'condition_separability' in metrics:
                print(f"Condition separability (Cohen's d): {metrics['condition_separability']}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error analyzing {args.subject}: {e}")
            sys.exit(1)
            
    else:
        print("Error: Must specify either --subject or --all-subjects")
        parser.print_help()
        sys.exit(1)


def create_group_summary_plot(all_results: dict, output_dir: Path):
    """Create a summary plot across all subjects."""
    if not all_results:
        return
        
    subjects = list(all_results.keys())
    n_subjects = len(subjects)
    n_components = all_results[subjects[0]]['rca_model'].n_components
    
    fig, axes = plt.subplots(2, n_components, figsize=(4 * n_components, 8))
    if n_components == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot eigenvalues across subjects
    for comp in range(n_components):
        ax_eigen = axes[0, comp]
        
        eigenvals = []
        for subject in subjects:
            eigenvals.append(all_results[subject]['rca_model'].eigenvalues_[comp])
            
        ax_eigen.bar(range(n_subjects), eigenvals, alpha=0.7)
        ax_eigen.set_title(f'RC{comp+1} Eigenvalues')
        ax_eigen.set_xlabel('Subject')
        ax_eigen.set_ylabel('Eigenvalue')
        ax_eigen.set_xticks(range(n_subjects))
        ax_eigen.set_xticklabels(subjects, rotation=45)
        ax_eigen.grid(True, alpha=0.3)
        
        # Plot condition separability if available
        ax_sep = axes[1, comp]
        
        separabilities = []
        for subject in subjects:
            metrics = compute_rca_reliability_metrics(all_results[subject])
            if 'condition_separability' in metrics and len(metrics['condition_separability']) > comp:
                separabilities.append(metrics['condition_separability'][comp])
            else:
                separabilities.append(0)
        
        ax_sep.bar(range(n_subjects), separabilities, alpha=0.7, color='orange')
        ax_sep.set_title(f'RC{comp+1} Condition Separability')
        ax_sep.set_xlabel('Subject')
        ax_sep.set_ylabel("Cohen's d")
        ax_sep.set_xticks(range(n_subjects))
        ax_sep.set_xticklabels(subjects, rotation=45)
        ax_sep.grid(True, alpha=0.3)
    
    plt.suptitle('RCA Group Summary: Music Preference Study', fontsize=16)
    plt.tight_layout()
    
    summary_path = output_dir / 'group_rca_summary.png'
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"Group summary plot saved to: {summary_path}")
    
    return fig


if __name__ == '__main__':
    main()