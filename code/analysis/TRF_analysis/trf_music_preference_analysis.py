#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TRF Analysis for Music Preference Study

This script performs Temporal Response Function (TRF) analysis to compare neural 
responses to preferred vs non-preferred music using mTRF (multivariate Temporal 
Response Function) modeling.

The analysis compares how well acoustic features (primarily spectral flux) can 
predict EEG responses for each participant's most and least preferred songs.

Key Features:
- Lambda optimization using concatenated data
- Separate TRF models for preferred vs non-preferred songs  
- Cross-validation with statistical comparison
- Comprehensive output saving and visualization

Usage: python trf_music_preference_analysis.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import mtrf
from scipy import signal
from scipy.stats import ttest_rel, wilcoxon
import h5py
import warnings
from pathlib import Path
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress some warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
mne.set_log_level('WARNING')

class TRFMusicPreferenceAnalysis:
    """
    Temporal Response Function analysis for music preference study.
    
    Compares neural responses to preferred vs non-preferred music using TRF modeling.
    """
    
    def __init__(self, base_dir="/Users/tongshan/Documents/music_preference"):
        """Initialize analysis with base directory paths."""
        self.base_dir = Path(base_dir)
        self.behavioral_file = self.base_dir / "data" / "beh_ratings.json"
        self.eeg_dir = self.base_dir / "data" / "ica_cleaned"
        self.features_dir = self.base_dir / "music_stim" / "music_features"
        self.output_dir = self.base_dir / "output" / "trf_analysis"
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # TRF parameters
        self.tmin = -0.1  # -100ms
        self.tmax = 0.7   # 700ms  
        self.lambda_range = np.logspace(-6, 6, 25)  # 10^-6 to 10^6
        self.n_folds = 5  # Cross-validation folds
        
        # Load behavioral data
        self.behavioral_data = self._load_behavioral_data()
        self.participants = list(self.behavioral_data['preference'].keys())
        
        logger.info(f"Initialized TRF analysis for {len(self.participants)} participants")
        logger.info(f"Output directory: {self.output_dir}")

    def _load_behavioral_data(self):
        """Load behavioral ratings from JSON file."""
        with open(self.behavioral_file, 'r') as f:
            data = json.load(f)
        logger.info("Loaded behavioral data successfully")
        return data

    def _get_participant_song_preferences(self, participant):
        """
        Get top 5 preferred and bottom 5 non-preferred songs for a participant.
        
        Parameters:
        -----------
        participant : str
            Participant ID (e.g., 'pilot_1')
            
        Returns:
        --------
        preferred_songs : list
            List of top 5 preferred song IDs
        nonpreferred_songs : list  
            List of bottom 5 non-preferred song IDs
        """
        ratings = self.behavioral_data['preference'][participant]
        
        # Filter out None ratings and sort by preference
        valid_ratings = {song: rating for song, rating in ratings.items() 
                        if rating is not None}
        
        sorted_songs = sorted(valid_ratings.items(), key=lambda x: x[1], reverse=True)
        
        # Get top 5 and bottom 5
        preferred_songs = [song for song, _ in sorted_songs[:5]]
        nonpreferred_songs = [song for song, _ in sorted_songs[-5:]]
        
        logger.info(f"{participant}: Preferred songs {preferred_songs}, Non-preferred {nonpreferred_songs}")
        return preferred_songs, nonpreferred_songs

    def _load_eeg_data(self, participant, song_id):
        """
        Load EEG data for a specific participant and song.
        
        Parameters:
        -----------
        participant : str
            Participant ID  
        song_id : str
            Song ID (e.g., '1-1')
            
        Returns:
        --------
        eeg_data : np.ndarray
            EEG data array (channels x timepoints)
        sfreq : float
            Sampling frequency
        channel_names : list
            List of EEG channel names
        """
        # Find the corresponding EEG trial file
        eeg_files = list((self.eeg_dir / participant).glob(f"*{song_id}_*_ica_cleaned.fif"))
        
        if not eeg_files:
            raise FileNotFoundError(f"No EEG file found for {participant} song {song_id}")
        
        eeg_file = eeg_files[0]
        raw = mne.io.read_raw_fif(eeg_file, preload=True, verbose=False)
        
        # Get EEG channels only (exclude non-EEG channels)
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
        eeg_data = raw.get_data(picks=eeg_picks)
        
        # Get channel names for EEG channels only
        channel_names = [raw.ch_names[i] for i in eeg_picks]
        
        return eeg_data, raw.info['sfreq'], channel_names

    def _load_music_features(self, song_id, feature_name='spectral_flux'):
        """
        Load music features for a specific song.
        
        Parameters:
        -----------
        song_id : str
            Song ID (e.g., '1-1')
        feature_name : str
            Name of feature to extract
            
        Returns:
        --------
        features : np.ndarray
            Feature time series
        feature_sr : float
            Feature sampling rate
        """
        feature_file = self.features_dir / f"{song_id}_proc_features.npz"
        
        if not feature_file.exists():
            raise FileNotFoundError(f"No feature file found for song {song_id}")
        
        features_data = np.load(feature_file)
        
        if feature_name not in features_data:
            raise ValueError(f"Feature '{feature_name}' not found. Available: {list(features_data.keys())}")
        
        features = features_data[feature_name]
        
        # Calculate feature sampling rate from time vector
        time_s = features_data['time_s']
        feature_sr = 1.0 / (time_s[1] - time_s[0])
        
        # Validate that feature sampling rate matches expected 128 Hz
        expected_sr = 128.0
        if abs(feature_sr - expected_sr) > 0.1:  # Allow small floating point tolerance
            logger.warning(f"Feature sampling rate {feature_sr:.2f} Hz does not match expected {expected_sr} Hz for song {song_id}")
        else:
            logger.debug(f"Feature sampling rate validated: {feature_sr:.2f} Hz for song {song_id}")
        
        return features, feature_sr

    def _align_eeg_features(self, eeg_data, eeg_sr, features, feature_sr):
        """
        Align EEG and feature data in time and sampling rate.
        
        Parameters:
        -----------
        eeg_data : np.ndarray
            EEG data (channels x timepoints)
        eeg_sr : float
            EEG sampling rate
        features : np.ndarray
            Feature data
        feature_sr : float
            Feature sampling rate
            
        Returns:
        --------
        eeg_aligned : np.ndarray
            Aligned EEG data
        features_aligned : np.ndarray
            Aligned feature data
        aligned_sr : float
            Common sampling rate
        """
        # Use EEG sampling rate as target
        target_sr = eeg_sr
        
        # Resample features to match EEG sampling rate
        if feature_sr != target_sr:
            n_samples_target = int(len(features) * target_sr / feature_sr)
            features_resampled = signal.resample(features, n_samples_target)
        else:
            features_resampled = features
        
        # Trim to shortest length
        min_length = min(eeg_data.shape[1], len(features_resampled))
        eeg_aligned = eeg_data[:, :min_length]
        features_aligned = features_resampled[:min_length]
        
        # Ensure features are 2D (features x timepoints)
        if features_aligned.ndim == 1:
            features_aligned = features_aligned.reshape(1, -1)
        
        logger.debug(f"Aligned data: EEG shape {eeg_aligned.shape}, Features shape {features_aligned.shape}")
        return eeg_aligned, features_aligned, target_sr

    def _prepare_participant_data(self, participant):
        """
        Load and prepare all data for a participant.
        
        Parameters:
        -----------
        participant : str
            Participant ID
            
        Returns:
        --------
        data_dict : dict
            Dictionary containing prepared data for TRF analysis
        """
        preferred_songs, nonpreferred_songs = self._get_participant_song_preferences(participant)
        
        data_dict = {
            'preferred': {'eeg_trials': [], 'features_trials': []},
            'nonpreferred': {'eeg_trials': [], 'features_trials': []},
            'all_trials': {'eeg_trials': [], 'features_trials': []},
            'sfreq': None,
            'channel_names': None
        }
        
        # Process preferred songs
        logger.info(f"Loading preferred songs for {participant}")
        for song_id in preferred_songs:
            try:
                eeg_data, eeg_sr, channel_names = self._load_eeg_data(participant, song_id)
                features, feature_sr = self._load_music_features(song_id)
                
                eeg_aligned, features_aligned, aligned_sr = self._align_eeg_features(
                    eeg_data, eeg_sr, features, feature_sr)
                
                # Store as individual trials (transpose for mtrf format: samples x channels/features)
                data_dict['preferred']['eeg_trials'].append(eeg_aligned.T)
                data_dict['preferred']['features_trials'].append(features_aligned.T)
                data_dict['all_trials']['eeg_trials'].append(eeg_aligned.T)
                data_dict['all_trials']['features_trials'].append(features_aligned.T)
                
                if data_dict['sfreq'] is None:
                    data_dict['sfreq'] = aligned_sr
                if data_dict['channel_names'] is None:
                    data_dict['channel_names'] = channel_names
                    
            except Exception as e:
                logger.warning(f"Failed to load {participant} song {song_id}: {e}")
        
        # Process non-preferred songs
        logger.info(f"Loading non-preferred songs for {participant}")
        for song_id in nonpreferred_songs:
            try:
                eeg_data, eeg_sr, channel_names = self._load_eeg_data(participant, song_id)
                features, feature_sr = self._load_music_features(song_id)
                
                eeg_aligned, features_aligned, aligned_sr = self._align_eeg_features(
                    eeg_data, eeg_sr, features, feature_sr)
                
                # Store as individual trials (transpose for mtrf format: samples x channels/features)
                data_dict['nonpreferred']['eeg_trials'].append(eeg_aligned.T)
                data_dict['nonpreferred']['features_trials'].append(features_aligned.T)
                data_dict['all_trials']['eeg_trials'].append(eeg_aligned.T)
                data_dict['all_trials']['features_trials'].append(features_aligned.T)
                
            except Exception as e:
                logger.warning(f"Failed to load {participant} song {song_id}: {e}")
        
        n_preferred_trials = len(data_dict['preferred']['eeg_trials'])
        n_nonpreferred_trials = len(data_dict['nonpreferred']['eeg_trials'])
        total_trials = len(data_dict['all_trials']['eeg_trials'])
        
        logger.info(f"Prepared data for {participant}: "
                   f"Preferred {n_preferred_trials} trials, "
                   f"Non-preferred {n_nonpreferred_trials} trials, "
                   f"Total {total_trials} trials")
        
        return data_dict

    def _optimize_lambda(self, all_trials_data, sfreq):
        """
        Optimize lambda (regularization parameter) using TRF's built-in cross-validation.
        
        Parameters:
        -----------
        all_trials_data : dict
            Dictionary containing 'eeg_trials' and 'features_trials' lists
        sfreq : float
            Sampling frequency
            
        Returns:
        --------
        best_lambda : float
            Optimal lambda value
        cv_scores : np.ndarray
            Cross-validation scores for each lambda
        """
        logger.info("Optimizing lambda parameter using TRF built-in cross-validation...")
        
        stim_trials = all_trials_data['features_trials']
        resp_trials = all_trials_data['eeg_trials']
        
        logger.info(f"Using {len(stim_trials)} trials for lambda optimization")
        
        # Create TRF model
        model = mtrf.TRF(direction=1)
        
        # Train with lambda range - TRF will automatically do cross-validation
        cv_scores = model.train(stim_trials, resp_trials, sfreq, 
                               tmin=self.tmin, tmax=self.tmax, 
                               regularization=self.lambda_range)
        
        # Get the optimal regularization parameter
        best_lambda = model.regularization
        
        # Find the best CV score
        best_idx = np.argmax(cv_scores)
        best_cv_score = cv_scores[best_idx]
        
        logger.info(f"Optimal lambda (TRF built-in CV): {best_lambda:.2e} (CV score: {best_cv_score:.4f})")
        
        return best_lambda, cv_scores

    def _fit_trf_model(self, trials_data, sfreq, lambda_val):
        """
        Fit TRF model with specified lambda using trial data.
        
        Parameters:
        -----------
        trials_data : dict
            Dictionary containing 'eeg_trials' and 'features_trials' lists
        sfreq : float
            Sampling frequency
        lambda_val : float
            Regularization parameter
            
        Returns:
        --------
        model : mtrf.TRF
            Fitted TRF model
        cv_score : float
            Cross-validation score
        """
        stim_trials = trials_data['features_trials']
        resp_trials = trials_data['eeg_trials']
        
        # Use mtrf.stats.crossval for proper trial-based cross-validation
        try:
            from mtrf.stats import crossval
            
            # Create TRF model
            model = mtrf.TRF(direction=1)
            
            # Perform cross-validation
            cv_scores = crossval(
                model, stim_trials, resp_trials, sfreq,
                tmin=self.tmin, tmax=self.tmax,
                regularization=lambda_val,
                k=self.n_folds,
                average=True,
                verbose=False
            )
            
            cv_score = np.mean(cv_scores)
            
            # Fit final model on all data
            stim_all = np.concatenate(stim_trials, axis=0)
            resp_all = np.concatenate(resp_trials, axis=0)
            final_model = mtrf.TRF(direction=1)
            final_model.train(stim_all, resp_all, sfreq, self.tmin, self.tmax, lambda_val)
            
        except Exception as e:
            logger.warning(f"Trial-based crossval failed: {e}. Using concatenated data.")
            
            # Fallback: concatenate trials and use traditional CV
            stim_concat = np.concatenate(stim_trials, axis=0)
            resp_concat = np.concatenate(resp_trials, axis=0)
            
            # Fit model with cross-validation
            final_model = mtrf.TRF(direction=1)
            final_model.train(stim_concat, resp_concat, sfreq, self.tmin, self.tmax, lambda_val)
            
            # Evaluate with cross-validation
            scores_fold = []
            n_samples = resp_concat.shape[0]
            fold_size = n_samples // self.n_folds
            
            for fold in range(self.n_folds):
                test_start = fold * fold_size
                test_end = test_start + fold_size if fold < self.n_folds - 1 else n_samples
                
                test_indices = np.arange(test_start, test_end)
                train_indices = np.concatenate([np.arange(0, test_start), np.arange(test_end, n_samples)])
                
                stim_train = stim_concat[train_indices]
                resp_train = resp_concat[train_indices]
                stim_test = stim_concat[test_indices] 
                resp_test = resp_concat[test_indices]
                
                # Fit and test
                trf_fold = mtrf.TRF(direction=1)
                trf_fold.train(stim_train, resp_train, sfreq, self.tmin, self.tmax, lambda_val)
                _, r = trf_fold.predict(stim_test, resp_test)
                scores_fold.append(np.mean(r))
            
            cv_score = np.mean(scores_fold)
        
        logger.debug(f"TRF model CV score: {cv_score:.4f}")
        
        return final_model, cv_score

    def analyze_participant(self, participant):
        """
        Run complete TRF analysis for a single participant.
        
        Parameters:
        -----------
        participant : str
            Participant ID
            
        Returns:
        --------
        results : dict
            Analysis results
        """
        logger.info(f"Starting TRF analysis for {participant}")
        
        # Prepare data
        data = self._prepare_participant_data(participant)
        
        if not data['all_trials']['eeg_trials']:
            logger.error(f"No valid data found for {participant}")
            return None
        
        # Get number of channels from first trial
        n_channels = data['all_trials']['eeg_trials'][0].shape[1]
        
        results = {
            'participant': participant,
            'sfreq': data['sfreq'],
            'n_channels': n_channels,
            'channel_names': data['channel_names']
        }
        
        # Step 1: Optimize lambda using all trials
        logger.info(f"Step 1: Lambda optimization for {participant}")
        best_lambda, cv_scores = self._optimize_lambda(
            data['all_trials'], data['sfreq'])
        
        results['lambda_optimization'] = {
            'lambda_range': self.lambda_range,
            'cv_scores': cv_scores,
            'best_lambda': best_lambda
        }
        
        # Step 2: Fit models for preferred and non-preferred conditions
        results['models'] = {}
        results['performance'] = {}
        
        for condition in ['preferred', 'nonpreferred']:
            if not data[condition]['eeg_trials']:
                logger.warning(f"No {condition} data for {participant}")
                continue
                
            logger.info(f"Step 2: Fitting {condition} model for {participant}")
            
            model, cv_score = self._fit_trf_model(
                data[condition], data['sfreq'], best_lambda)
            
            results['models'][condition] = {
                'weights': model.weights,
                'bias': model.bias,
                'times': model.times
            }
            
            # Calculate total samples for this condition
            total_samples = sum(trial.shape[0] for trial in data[condition]['eeg_trials'])
            
            results['performance'][condition] = {
                'cv_score': cv_score,
                'n_samples': total_samples
            }
        
        # Step 3: Statistical comparison
        if 'preferred' in results['performance'] and 'nonpreferred' in results['performance']:
            logger.info(f"Step 3: Statistical comparison for {participant}")
            results['statistical_comparison'] = self._compare_conditions(
                data, best_lambda, results)
        
        logger.info(f"Completed TRF analysis for {participant}")
        return results

    def _compare_conditions(self, data, lambda_val, results):
        """
        Compare TRF performance between preferred and nonpreferred conditions.
        
        Parameters:
        -----------
        data : dict
            Participant data
        lambda_val : float
            Regularization parameter
        results : dict
            Current results
            
        Returns:
        --------
        comparison : dict
            Statistical comparison results
        """
        # Get channel-wise performance for both conditions
        perf_preferred = []
        perf_nonpreferred = []
        
        # Cross-validation comparison
        n_channels = data['preferred']['eeg_trials'][0].shape[1]
        
        for ch in range(n_channels):
            # Single-channel analysis for statistical comparison
            # Extract single channel from all trials
            pref_ch_trials = {'eeg_trials': [trial[:, ch:ch+1] for trial in data['preferred']['eeg_trials']],
                             'features_trials': data['preferred']['features_trials']}
            nonpref_ch_trials = {'eeg_trials': [trial[:, ch:ch+1] for trial in data['nonpreferred']['eeg_trials']],
                                'features_trials': data['nonpreferred']['features_trials']}
            
            # Fit models
            _, score_pref = self._fit_trf_model(pref_ch_trials, data['sfreq'], lambda_val)
            _, score_nonpref = self._fit_trf_model(nonpref_ch_trials, data['sfreq'], lambda_val)
            
            perf_preferred.append(score_pref)
            perf_nonpreferred.append(score_nonpref)
        
        perf_preferred = np.array(perf_preferred)
        perf_nonpreferred = np.array(perf_nonpreferred)
        
        # Statistical tests
        stat_t, p_val_t = ttest_rel(perf_preferred, perf_nonpreferred)
        stat_w, p_val_w = wilcoxon(perf_preferred, perf_nonpreferred, alternative='two-sided')
        
        comparison = {
            'performance_preferred': perf_preferred,
            'performance_nonpreferred': perf_nonpreferred,
            'mean_difference': np.mean(perf_preferred - perf_nonpreferred),
            'ttest': {'statistic': stat_t, 'p_value': p_val_t},
            'wilcoxon': {'statistic': stat_w, 'p_value': p_val_w},
            'effect_size': np.mean(perf_preferred - perf_nonpreferred) / np.std(perf_preferred - perf_nonpreferred)
        }
        
        logger.info(f"Statistical comparison: Mean difference = {comparison['mean_difference']:.4f}, "
                   f"t-test p = {p_val_t:.4f}, Wilcoxon p = {p_val_w:.4f}")
        
        return comparison

    def save_results(self, participant, results):
        """
        Save analysis results to HDF5 and CSV files.
        
        Parameters:
        -----------
        participant : str
            Participant ID
        results : dict
            Analysis results
        """
        if results is None:
            return
        
        # Save to HDF5
        h5_file = self.output_dir / f"{participant}_trf_results.h5"
        
        with h5py.File(h5_file, 'w') as f:
            # Basic info
            f.attrs['participant'] = participant
            f.attrs['sfreq'] = results['sfreq']
            f.attrs['tmin'] = self.tmin
            f.attrs['tmax'] = self.tmax
            f.attrs['n_channels'] = results['n_channels']
            f.attrs['analysis_date'] = datetime.now().isoformat()
            
            # Lambda optimization
            lam_group = f.create_group('lambda_optimization')
            lam_group.create_dataset('lambda_range', data=results['lambda_optimization']['lambda_range'])
            lam_group.create_dataset('cv_scores', data=results['lambda_optimization']['cv_scores'])
            lam_group.attrs['best_lambda'] = results['lambda_optimization']['best_lambda']
            
            # Models and performance
            for condition in ['preferred', 'nonpreferred']:
                if condition in results['models']:
                    cond_group = f.create_group(condition)
                    cond_group.create_dataset('weights', data=results['models'][condition]['weights'])
                    cond_group.create_dataset('times', data=results['models'][condition]['times'])
                    cond_group.attrs['cv_score'] = results['performance'][condition]['cv_score']
                    cond_group.attrs['n_samples'] = results['performance'][condition]['n_samples']
            
            # Statistical comparison
            if 'statistical_comparison' in results:
                stat_group = f.create_group('statistical_comparison')
                comp = results['statistical_comparison']
                stat_group.create_dataset('performance_preferred', data=comp['performance_preferred'])
                stat_group.create_dataset('performance_nonpreferred', data=comp['performance_nonpreferred'])
                stat_group.attrs['mean_difference'] = comp['mean_difference']
                stat_group.attrs['ttest_statistic'] = comp['ttest']['statistic']
                stat_group.attrs['ttest_p_value'] = comp['ttest']['p_value']
                stat_group.attrs['wilcoxon_statistic'] = comp['wilcoxon']['statistic']
                stat_group.attrs['wilcoxon_p_value'] = comp['wilcoxon']['p_value']
                stat_group.attrs['effect_size'] = comp['effect_size']
        
        logger.info(f"Saved results to {h5_file}")
        
        # Save summary to CSV
        csv_file = self.output_dir / f"{participant}_trf_summary.csv"
        
        summary_data = {
            'participant': [participant],
            'best_lambda': [results['lambda_optimization']['best_lambda']],
            'n_channels': [results['n_channels']]
        }
        
        for condition in ['preferred', 'nonpreferred']:
            if condition in results['performance']:
                summary_data[f'{condition}_cv_score'] = [results['performance'][condition]['cv_score']]
                summary_data[f'{condition}_n_samples'] = [results['performance'][condition]['n_samples']]
            else:
                summary_data[f'{condition}_cv_score'] = [np.nan]
                summary_data[f'{condition}_n_samples'] = [0]
        
        if 'statistical_comparison' in results:
            comp = results['statistical_comparison']
            summary_data['mean_difference'] = [comp['mean_difference']]
            summary_data['ttest_p_value'] = [comp['ttest']['p_value']]
            summary_data['wilcoxon_p_value'] = [comp['wilcoxon']['p_value']]
            summary_data['effect_size'] = [comp['effect_size']]
        
        pd.DataFrame(summary_data).to_csv(csv_file, index=False)
        logger.info(f"Saved summary to {csv_file}")

    def visualize_results(self, participant, results):
        """
        Create visualizations for TRF analysis results.
        
        Parameters:
        -----------
        participant : str
            Participant ID
        results : dict
            Analysis results
        """
        if results is None:
            return
        
        # Create figure with subplots - 2x3 layout for topography
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Lambda optimization curve
        ax = axes[0, 0]
        lam_opt = results['lambda_optimization']
        
        # Check if we have meaningful CV scores (from manual CV) or just placeholder
        if np.count_nonzero(lam_opt['cv_scores']) > 1:
            # Full lambda curve from manual CV
            ax.semilogx(lam_opt['lambda_range'], lam_opt['cv_scores'], 'o-')
            ax.set_title('Lambda Optimization (Manual CV)')
        else:
            # Nested CV result - show just the optimal point
            ax.semilogx(lam_opt['best_lambda'], np.max(lam_opt['cv_scores']), 'ro', markersize=10)
            ax.set_title('Lambda Optimization (Nested CV)')
            ax.text(lam_opt['best_lambda'], np.max(lam_opt['cv_scores']) + 0.001, 
                   f"Optimal λ", ha='center', va='bottom')
        
        ax.axvline(lam_opt['best_lambda'], color='red', linestyle='--', 
                  label=f"Best λ = {lam_opt['best_lambda']:.2e}")
        ax.set_xlabel('Lambda (Regularization)')
        ax.set_ylabel('CV Score (R²)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. TRF weights comparison
        ax = axes[0, 1]
        for condition in ['preferred', 'nonpreferred']:
            if condition in results['models']:
                times = results['models'][condition]['times']
                weights = results['models'][condition]['weights']
                
                # Handle different weight shapes: (features, time, channels) or (time, channels)
                if weights.ndim == 3:
                    # Shape: (features, time, channels) - average across features and channels
                    mean_weights = np.mean(weights, axis=(0, 2))
                elif weights.ndim == 2:
                    # Shape: (time, channels) - average across channels
                    mean_weights = np.mean(weights, axis=1)
                else:
                    # Shape: (time,) - use as is
                    mean_weights = weights
                
                ax.plot(times * 1000, mean_weights, label=condition.capitalize(), linewidth=2)
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('TRF Weight (a.u.)')
        ax.set_title('TRF Weights (Channel Average)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        
        # 3. Topography - Preferred condition TRF scores
        ax = axes[0, 2]
        if 'statistical_comparison' in results and 'performance_preferred' in results['statistical_comparison']:
            scores_pref = results['statistical_comparison']['performance_preferred']
            self._plot_topography(ax, scores_pref, results['channel_names'], 'Preferred TRF Scores', 'Reds')
        else:
            ax.text(0.5, 0.5, 'Channel-wise scores\nnot available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Preferred TRF Scores')
        
        # 4. Topography - Non-preferred condition TRF scores  
        ax = axes[1, 0]
        if 'statistical_comparison' in results and 'performance_nonpreferred' in results['statistical_comparison']:
            scores_nonpref = results['statistical_comparison']['performance_nonpreferred']
            self._plot_topography(ax, scores_nonpref, results['channel_names'], 'Non-preferred TRF Scores', 'Greys')
        else:
            ax.text(0.5, 0.5, 'Channel-wise scores\nnot available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Non-preferred TRF Scores')
        
        # 5. Performance comparison - Bar plot with preferred (red) and non-preferred (black)
        ax = axes[1, 1]
        conditions = []
        scores = []
        colors = []
        
        for condition in ['preferred', 'nonpreferred']:
            if condition in results['performance']:
                conditions.append(condition.capitalize())
                scores.append(results['performance'][condition]['cv_score'])
                colors.append('red' if condition == 'preferred' else 'black')
        
        if len(conditions) == 2:
            bars = ax.bar(conditions, scores, color=colors, alpha=0.8)
            ax.set_ylabel('TRF Score (R²)')
            ax.set_title('TRF Performance Comparison')
            
            # Add value labels on bars with white text
            for bar, score in zip(bars, scores):
                # Place text in the center of each bar
                bar_height = bar.get_height()
                y_pos = bar_height / 2
                ax.text(bar.get_x() + bar.get_width()/2, y_pos, 
                       f'{score:.4f}', ha='center', va='center', 
                       fontweight='bold', color='white', fontsize=11)
            
            ax.grid(True, alpha=0.3, axis='y')
        
        # 6. FCz Channel TRF weights - preferred (red) and non-preferred (black)
        ax = axes[1, 2]
        
        # Find FCz channel by name
        fcz_channel = None
        fcz_channel_name = 'FCz'
        
        if 'channel_names' in results and results['channel_names']:
            channel_names = results['channel_names']
            
            # Try to find FCz channel
            if 'FCz' in channel_names:
                fcz_channel = channel_names.index('FCz')
                fcz_channel_name = 'FCz'
            elif 'Fz' in channel_names:  # Alternative: use Fz if FCz not available
                fcz_channel = channel_names.index('Fz')
                fcz_channel_name = 'Fz'
            else:
                # Look for similar channels
                for i, ch_name in enumerate(channel_names):
                    if 'fcz' in ch_name.lower() or 'fz' in ch_name.lower():
                        fcz_channel = i
                        fcz_channel_name = ch_name
                        break
                
                # If still not found, use middle channel as fallback
                if fcz_channel is None:
                    fcz_channel = len(channel_names) // 2
                    fcz_channel_name = f"{channel_names[fcz_channel]} (fallback)"
            
            logger.debug(f"Using channel {fcz_channel_name} (index {fcz_channel}) for FCz visualization")
        
        if fcz_channel is not None:
            for condition in ['preferred', 'nonpreferred']:
                if condition in results['models']:
                    times = results['models'][condition]['times']
                    weights = results['models'][condition]['weights']
                    
                    # Extract FCz channel weights
                    if weights.ndim == 3:
                        # Shape: (features, time, channels)
                        fcz_weights = weights[0, :, fcz_channel]
                    elif weights.ndim == 2:
                        # Shape: (time, channels)
                        fcz_weights = weights[:, fcz_channel]
                    else:
                        # Shape: (time,) - single channel case
                        fcz_weights = weights
                    
                    color = 'red' if condition == 'preferred' else 'black'
                    ax.plot(times * 1000, fcz_weights, color=color, 
                           label=condition.capitalize(), linewidth=2)
            
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('TRF Weight (a.u.)')
            ax.set_title(f'{fcz_channel_name} TRF Weights')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
            ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        else:
            ax.text(0.5, 0.5, 'FCz channel\nnot available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('FCz Channel TRF Weights')
        
        plt.suptitle(f'TRF Analysis Results - {participant}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        fig_file = self.output_dir / f"{participant}_trf_analysis.png"
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization to {fig_file}")
    
    def _plot_topography(self, ax, scores, channel_names, title, colormap):
        """
        Plot topographic map of TRF scores using MNE.
        
        Parameters:
        -----------
        ax : matplotlib.axes
            Axes to plot on
        scores : np.ndarray
            TRF scores for each channel
        channel_names : list
            List of channel names
        title : str
            Plot title
        colormap : str
            Matplotlib colormap name
        """
        try:
            # Create a dummy MNE info structure for topography
            info = mne.create_info(ch_names=channel_names, sfreq=128, ch_types='eeg')
            
            # Set standard montage for EEG channel positions
            montage = mne.channels.make_standard_montage('standard_1020')
            info.set_montage(montage, match_case=False, on_missing='ignore')
            
            # Plot topography
            im, _ = mne.viz.plot_topomap(scores, info, axes=ax, show=False, 
                                       cmap=colormap, contours=6, 
                                       names=None, size=3)
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('TRF Score (R²)', fontsize=10)
            
        except Exception as e:
            logger.warning(f"Failed to create topography plot: {e}")
            # Fallback: simple channel bar plot
            ax.bar(range(len(scores)), scores, color=plt.cm.get_cmap(colormap)(0.7))
            ax.set_xlabel('Channel Index')
            ax.set_ylabel('TRF Score (R²)')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

    def run_analysis(self, participants=None):
        """
        Run TRF analysis for specified participants.
        
        Parameters:
        -----------
        participants : list or None
            List of participant IDs. If None, analyze all participants.
        """
        if participants is None:
            participants = self.participants
        
        all_results = {}
        
        logger.info(f"Starting TRF analysis for {len(participants)} participants")
        
        for participant in participants:
            logger.info(f"\n{'='*50}")
            logger.info(f"ANALYZING {participant.upper()}")
            logger.info(f"{'='*50}")
            
            try:
                # Run analysis
                results = self.analyze_participant(participant)
                
                if results is not None:
                    # Save results
                    self.save_results(participant, results)
                    
                    # Create visualizations
                    self.visualize_results(participant, results)
                    
                    all_results[participant] = results
                    logger.info(f"✓ Successfully completed analysis for {participant}")
                else:
                    logger.error(f"✗ Failed to analyze {participant}")
                    
            except Exception as e:
                logger.error(f"✗ Error analyzing {participant}: {e}")
                import traceback
                traceback.print_exc()
        
        # Create group summary
        self.create_group_summary(all_results)
        
        logger.info(f"\n{'='*50}")
        logger.info("TRF ANALYSIS COMPLETED")
        logger.info(f"{'='*50}")
        logger.info(f"Results saved to: {self.output_dir}")
        
        return all_results

    def create_group_summary(self, all_results):
        """
        Create summary analysis across all participants.
        
        Parameters:
        -----------
        all_results : dict
            Results from all participants
        """
        if not all_results:
            return
        
        logger.info("Creating group summary...")
        
        # Compile group statistics
        group_data = []
        
        for participant, results in all_results.items():
            row = {
                'participant': participant,
                'best_lambda': results['lambda_optimization']['best_lambda'],
                'n_channels': results['n_channels']
            }
            
            for condition in ['preferred', 'nonpreferred']:
                if condition in results['performance']:
                    row[f'{condition}_cv_score'] = results['performance'][condition]['cv_score']
                    row[f'{condition}_n_samples'] = results['performance'][condition]['n_samples']
                else:
                    row[f'{condition}_cv_score'] = np.nan
                    row[f'{condition}_n_samples'] = 0
            
            if 'statistical_comparison' in results:
                comp = results['statistical_comparison']
                row['mean_difference'] = comp['mean_difference']
                row['ttest_p_value'] = comp['ttest']['p_value']
                row['wilcoxon_p_value'] = comp['wilcoxon']['p_value']
                row['effect_size'] = comp['effect_size']
            
            group_data.append(row)
        
        # Save group summary
        group_df = pd.DataFrame(group_data)
        group_csv = self.output_dir / "group_trf_summary.csv"
        group_df.to_csv(group_csv, index=False)
        
        # Create group visualization
        self._create_group_visualization(group_df)
        
        logger.info(f"Saved group summary to {group_csv}")

    def _create_group_visualization(self, group_df):
        """Create group-level visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Individual performance scores
        ax = axes[0, 0]
        x = np.arange(len(group_df))
        width = 0.35
        
        pref_scores = group_df['preferred_cv_score'].fillna(0)
        nonpref_scores = group_df['nonpreferred_cv_score'].fillna(0)
        
        ax.bar(x - width/2, pref_scores, width, label='Preferred', alpha=0.7)
        ax.bar(x + width/2, nonpref_scores, width, label='Non-preferred', alpha=0.7)
        
        ax.set_xlabel('Participant')
        ax.set_ylabel('CV Score (R²)')
        ax.set_title('TRF Performance by Participant')
        ax.set_xticks(x)
        ax.set_xticklabels(group_df['participant'], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Performance difference distribution
        ax = axes[0, 1]
        if 'mean_difference' in group_df.columns:
            differences = group_df['mean_difference'].dropna()
            ax.hist(differences, bins=10, alpha=0.7, edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', label='No difference')
            ax.axvline(np.mean(differences), color='blue', linestyle='-', 
                      label=f'Mean = {np.mean(differences):.4f}')
            ax.set_xlabel('Performance Difference (Preferred - Non-preferred)')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Performance Differences')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 3. Lambda values distribution
        ax = axes[1, 0]
        lambda_vals = group_df['best_lambda']
        ax.hist(np.log10(lambda_vals), bins=10, alpha=0.7, edgecolor='black')
        ax.set_xlabel('log₁₀(Best Lambda)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Optimal Lambda Values')
        ax.grid(True, alpha=0.3)
        
        # 4. Statistical significance summary
        ax = axes[1, 1]
        if 'ttest_p_value' in group_df.columns:
            p_vals = group_df['ttest_p_value'].dropna()
            significant = (p_vals < 0.05).sum()
            non_significant = (p_vals >= 0.05).sum()
            
            ax.pie([significant, non_significant], 
                  labels=[f'Significant (p<0.05)\nn={significant}', 
                         f'Not Significant\nn={non_significant}'],
                  autopct='%1.1f%%', startangle=90)
            ax.set_title('Statistical Significance Summary\n(t-test for preference effect)')
        
        plt.suptitle('Group TRF Analysis Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        fig_file = self.output_dir / "group_trf_summary.png"
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved group visualization to {fig_file}")


def main():
    """Main function to run TRF analysis."""
    # Initialize analysis
    trf_analysis = TRFMusicPreferenceAnalysis()
    
    # Run analysis for all participants
    results = trf_analysis.run_analysis()
    
    return results


if __name__ == "__main__":
    main()