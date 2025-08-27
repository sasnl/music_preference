#!/usr/bin/env python3
"""
Test RCA integration with actual music preference EEG data.

This script tests the RCA implementation with the actual preprocessed EEG data
from the music preference study, including:
- Loading behavioral ratings
- Processing real EEG trial files
- Running preference-based RCA analysis
- Generating visualizations
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
import warnings
from typing import Dict, Any, Optional

# Add RCA to path
rca_path = Path(__file__).parent.parent  # Go up to rca_python directory
sys.path.insert(0, str(rca_path))

from rca import ReliableComponentsAnalysis
from rca_utils import (load_music_preference_data, run_rca_on_music_data, 
                      plot_music_rca_results, compute_rca_reliability_metrics)

class MusicRCAIntegrationTest:
    """Test suite for RCA music preference integration."""
    
    def __init__(self, data_dir: str = "data/ica_cleaned"):
        self.data_dir = Path(data_dir)
        self.results = {}
        
    def test_behavioral_data_loading(self) -> Dict[str, Any]:
        """Test loading and parsing of behavioral preference data."""
        print("Testing behavioral data loading...")
        
        # Check if behavioral data exists
        beh_file = self.data_dir.parent / 'beh_ratings.json'
        if not beh_file.exists():
            return {'status': 'SKIPPED', 'reason': 'No behavioral data file found'}
        
        with open(beh_file, 'r') as f:
            ratings = json.load(f)
        
        # Check structure
        expected_questions = ['preference', 'pleasantness', 'arousal', 'chills']
        expected_subjects = ['pilot_1', 'pilot_2', 'pilot_3', 'pilot_4', 'pilot_5']
        
        results = {
            'file_exists': True,
            'has_expected_questions': all(q in ratings for q in expected_questions),
            'has_expected_subjects': all(subj in ratings['preference'] for subj in expected_subjects),
            'n_subjects': len(ratings['preference']),
            'n_songs_per_subject': {},
            'rating_ranges': {}
        }
        
        # Check each subject's data
        for subject in expected_subjects:
            if subject in ratings['preference']:
                subject_ratings = ratings['preference'][subject]
                results['n_songs_per_subject'][subject] = len(subject_ratings)
                
                # Check rating range (filter out null values)
                rating_values = [v for v in subject_ratings.values() if v is not None]
                if rating_values:
                    results['rating_ranges'][subject] = {
                        'min': min(rating_values),
                        'max': max(rating_values),
                        'mean': np.mean(rating_values),
                        'std': np.std(rating_values),
                        'n_valid': len(rating_values),
                        'n_null': len(subject_ratings) - len(rating_values)
                    }
                else:
                    results['rating_ranges'][subject] = {
                        'error': 'All ratings are null',
                        'n_null': len(subject_ratings)
                    }
        
        return results
    
    def test_eeg_data_availability(self) -> Dict[str, Any]:
        """Test availability and structure of EEG data files."""
        print("Testing EEG data availability...")
        
        results = {
            'data_dir_exists': self.data_dir.exists(),
            'subjects': {},
            'total_trial_files': 0
        }
        
        if not self.data_dir.exists():
            results['status'] = 'FAILED'
            results['reason'] = f'Data directory {self.data_dir} does not exist'
            return results
        
        # Check each subject directory
        for subject_dir in self.data_dir.iterdir():
            if subject_dir.is_dir() and subject_dir.name.startswith('pilot_'):
                subject_id = subject_dir.name
                trial_files = list(subject_dir.glob(f"{subject_id}-trial*_ica_cleaned.fif"))
                click_files = list(subject_dir.glob(f"{subject_id}_click_trial*_ica_cleaned.fif"))
                
                results['subjects'][subject_id] = {
                    'n_trial_files': len(trial_files),
                    'n_click_files': len(click_files),
                    'trial_files_exist': len(trial_files) > 0,
                    'example_files': [f.name for f in trial_files[:3]]  # First 3 examples
                }
                
                results['total_trial_files'] += len(trial_files)
        
        results['n_subjects_with_data'] = len([s for s in results['subjects'].values() 
                                             if s['trial_files_exist']])
        results['status'] = 'PASSED' if results['n_subjects_with_data'] > 0 else 'FAILED'
        
        return results
    
    def test_single_subject_rca_analysis(self, subject_id: str = 'pilot_2') -> Dict[str, Any]:
        """Test full RCA analysis pipeline on a single subject."""
        print(f"Testing RCA analysis for {subject_id}...")
        
        try:
            # Run RCA analysis
            results = run_rca_on_music_data(
                subject_id=subject_id,
                data_dir=self.data_dir,
                n_components=3,
                compare_conditions=True
            )
            
            # Compute reliability metrics
            metrics = compute_rca_reliability_metrics(results)
            
            # Check results structure
            analysis_results = {
                'status': 'PASSED',
                'subject_id': results['subject_id'],
                'rca_fitted': results['rca_model'].is_fitted_,
                'n_components': results['rca_model'].n_components,
                'eigenvalues': results['rca_model'].eigenvalues_[:3].tolist(),
                'spatial_filters_shape': results['rca_model'].spatial_filters_.shape,
                'has_condition_comparison': 'preferred_rca' in results,
                'metrics': metrics
            }
            
            if 'preferred_rca' in results:
                analysis_results.update({
                    'n_preferred_trials': results['n_preferred_trials'],
                    'n_nonpreferred_trials': results['n_nonpreferred_trials'],
                    'preferred_data_shape': results['preferred_rca'].shape,
                    'nonpreferred_data_shape': results['nonpreferred_rca'].shape
                })
            
            # Test visualization (without showing)
            try:
                plt.ioff()  # Turn off interactive plotting
                fig = plot_music_rca_results(results)
                plt.close(fig)
                analysis_results['visualization_works'] = True
            except Exception as e:
                analysis_results['visualization_works'] = False
                analysis_results['visualization_error'] = str(e)
            
            return analysis_results
            
        except Exception as e:
            return {
                'status': 'FAILED',
                'subject_id': subject_id,
                'error': str(e)
            }
    
    def test_multiple_subjects(self, max_subjects: int = 3) -> Dict[str, Any]:
        """Test RCA analysis on multiple subjects."""
        print(f"Testing RCA analysis on multiple subjects (max {max_subjects})...")
        
        # Get available subjects
        available_subjects = [d.name for d in self.data_dir.iterdir() 
                            if d.is_dir() and d.name.startswith('pilot_')]
        test_subjects = available_subjects[:max_subjects]
        
        results = {
            'available_subjects': available_subjects,
            'test_subjects': test_subjects,
            'subject_results': {},
            'success_count': 0,
            'failure_count': 0
        }
        
        for subject_id in test_subjects:
            print(f"  Processing {subject_id}...")
            try:
                # Run lightweight analysis (fewer components)
                subject_results = run_rca_on_music_data(
                    subject_id=subject_id,
                    data_dir=self.data_dir,
                    n_components=2,  # Fewer components for speed
                    compare_conditions=True
                )
                
                results['subject_results'][subject_id] = {
                    'status': 'SUCCESS',
                    'eigenvalues': subject_results['rca_model'].eigenvalues_[:2].tolist(),
                    'n_preferred': subject_results.get('n_preferred_trials', 0),
                    'n_nonpreferred': subject_results.get('n_nonpreferred_trials', 0)
                }
                results['success_count'] += 1
                
            except Exception as e:
                results['subject_results'][subject_id] = {
                    'status': 'FAILED',
                    'error': str(e)
                }
                results['failure_count'] += 1
        
        results['overall_status'] = 'PASSED' if results['success_count'] > 0 else 'FAILED'
        results['success_rate'] = results['success_count'] / len(test_subjects) if test_subjects else 0
        
        return results
    
    def test_data_format_compatibility(self) -> Dict[str, Any]:
        """Test that the data formats are compatible with RCA expectations."""
        print("Testing data format compatibility...")
        
        results = {'status': 'PASSED', 'checks': {}}
        
        # Try to load data for one subject
        try:
            subject_id = 'pilot_2'
            data_dict = load_music_preference_data(subject_id, self.data_dir)
            
            results['checks']['data_loading'] = 'PASSED'
            results['checks']['has_preferred'] = len(data_dict['preferred']) > 0
            results['checks']['has_nonpreferred'] = len(data_dict['nonpreferred']) > 0
            
            if data_dict['preferred']:
                # Check data properties
                example_epochs = data_dict['preferred'][0]
                
                results['checks']['epochs_structure'] = {
                    'n_epochs': len(example_epochs),
                    'n_channels': len(example_epochs.ch_names),
                    'n_times': len(example_epochs.times),
                    'sampling_rate': example_epochs.info['sfreq']
                }
                
                # Test conversion to RCA format
                from rca_utils import epochs_to_rca_format
                rca_data = epochs_to_rca_format(data_dict['preferred'][:1])  # Just first file
                
                results['checks']['rca_conversion'] = {
                    'shape': rca_data.shape,
                    'has_nan': bool(np.any(np.isnan(rca_data))),
                    'data_range': [float(np.nanmin(rca_data)), float(np.nanmax(rca_data))]
                }
            
        except Exception as e:
            results['status'] = 'FAILED'
            results['error'] = str(e)
        
        return results
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all music integration tests."""
        print("=" * 60)
        print("RCA MUSIC PREFERENCE INTEGRATION TESTS")
        print("=" * 60)
        
        all_results = {}
        
        # Run tests
        all_results['behavioral_data'] = self.test_behavioral_data_loading()
        all_results['eeg_data_availability'] = self.test_eeg_data_availability()
        all_results['data_format_compatibility'] = self.test_data_format_compatibility()
        
        # Only run analysis tests if data is available
        if (all_results['eeg_data_availability'].get('status') == 'PASSED' and 
            all_results['data_format_compatibility'].get('status') == 'PASSED'):
            
            all_results['single_subject_analysis'] = self.test_single_subject_rca_analysis()
            all_results['multiple_subjects'] = self.test_multiple_subjects()
        else:
            print("Skipping analysis tests due to data availability issues")
            all_results['single_subject_analysis'] = {'status': 'SKIPPED', 'reason': 'No data available'}
            all_results['multiple_subjects'] = {'status': 'SKIPPED', 'reason': 'No data available'}
        
        # Print summary
        print("\n" + "=" * 60)
        print("INTEGRATION TEST SUMMARY") 
        print("=" * 60)
        
        passed_tests = sum(1 for result in all_results.values() 
                          if result.get('status') == 'PASSED' or 
                             result.get('overall_status') == 'PASSED')
        
        failed_tests = sum(1 for result in all_results.values() 
                          if result.get('status') == 'FAILED' or
                             result.get('overall_status') == 'FAILED')
        
        skipped_tests = sum(1 for result in all_results.values() 
                           if result.get('status') == 'SKIPPED')
        
        total_tests = len(all_results)
        
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}") 
        print(f"Skipped: {skipped_tests}")
        
        if failed_tests == 0:
            print("\n🎉 ALL INTEGRATION TESTS PASSED! RCA is ready for music preference analysis.")
        elif passed_tests > 0:
            print(f"\n✅ {passed_tests} tests passed, {failed_tests} failed. RCA partially functional.")
        else:
            print(f"\n❌ All tests failed. Check data availability and setup.")
        
        return all_results


def main():
    """Main test execution."""
    # Suppress some warnings for cleaner output
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    # Initialize and run tests
    test_suite = MusicRCAIntegrationTest()
    results = test_suite.run_all_tests()
    
    # Print detailed results for failures
    print("\nDETAILED TEST RESULTS:")
    print("-" * 40)
    
    for test_name, result in results.items():
        status = result.get('status', result.get('overall_status', 'UNKNOWN'))
        print(f"\n{test_name.upper()}: {status}")
        
        if status == 'FAILED' and 'error' in result:
            print(f"  Error: {result['error']}")
        elif status == 'SKIPPED' and 'reason' in result:
            print(f"  Reason: {result['reason']}")
        
        # Print key metrics for successful tests
        if status == 'PASSED':
            if test_name == 'single_subject_analysis':
                print(f"  Subject: {result.get('subject_id', 'N/A')}")
                print(f"  Components: {result.get('n_components', 'N/A')}")
                print(f"  Eigenvalues: {result.get('eigenvalues', 'N/A')}")
                if result.get('has_condition_comparison'):
                    print(f"  Preferred trials: {result.get('n_preferred_trials', 'N/A')}")
                    print(f"  Non-preferred trials: {result.get('n_nonpreferred_trials', 'N/A')}")
            elif test_name == 'multiple_subjects':
                print(f"  Success rate: {result.get('success_rate', 0):.1%}")
                print(f"  Processed: {result.get('success_count', 0)}/{len(result.get('test_subjects', []))}")
    
    return results


if __name__ == "__main__":
    main()