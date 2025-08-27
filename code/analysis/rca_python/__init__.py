"""
RCA Python Package

A Python implementation of Reliable Components Analysis (RCA) for neural data analysis.
Specifically designed for integration with the music preference EEG study.

Main classes:
- ReliableComponentsAnalysis: Core RCA implementation
- Music-specific utility functions for loading and analyzing EEG data

Based on the MATLAB toolbox by Jacek P. Dmochowski (2015).
"""

from .rca import ReliableComponentsAnalysis, demo_rca_analysis
from .rca_utils import (
    load_music_preference_data, 
    epochs_to_rca_format,
    run_rca_on_music_data,
    plot_music_rca_results,
    compute_rca_reliability_metrics,
    save_rca_results,
    batch_rca_analysis
)

__version__ = "1.0.0"
__author__ = "Music Preference Study Team"

__all__ = [
    'ReliableComponentsAnalysis',
    'demo_rca_analysis',
    'load_music_preference_data',
    'epochs_to_rca_format', 
    'run_rca_on_music_data',
    'plot_music_rca_results',
    'compute_rca_reliability_metrics',
    'save_rca_results',
    'batch_rca_analysis'
]