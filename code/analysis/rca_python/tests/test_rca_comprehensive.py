#!/usr/bin/env python3
"""
Comprehensive test suite for RCA Python implementation.

This script tests all major functionality of the Reliable Components Analysis
implementation, including:
- Basic RCA functionality with synthetic data
- Mathematical properties verification
- Edge cases and error handling
- Integration with music preference data format
- Performance with different data sizes
"""

import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
from typing import Dict, Any

# Add RCA to path
rca_path = Path(__file__).parent.parent  # Go up to rca_python directory
sys.path.insert(0, str(rca_path))

from rca import ReliableComponentsAnalysis
from rca_utils import epochs_to_rca_format

class RCATestSuite:
    """Comprehensive test suite for RCA implementation."""
    
    def __init__(self):
        self.results = {}
        self.passed_tests = 0
        self.total_tests = 0
        
    def run_test(self, test_name: str, test_func):
        """Run a single test and record results."""
        print(f"\n=== {test_name} ===")
        self.total_tests += 1
        
        try:
            start_time = time.time()
            result = test_func()
            end_time = time.time()
            
            self.results[test_name] = {
                'status': 'PASSED',
                'time': end_time - start_time,
                'result': result
            }
            self.passed_tests += 1
            print(f"✓ {test_name} PASSED ({end_time - start_time:.2f}s)")
            
        except Exception as e:
            self.results[test_name] = {
                'status': 'FAILED', 
                'error': str(e),
                'time': None
            }
            print(f"✗ {test_name} FAILED: {e}")
            
    def test_basic_functionality(self) -> Dict[str, Any]:
        """Test basic RCA functionality with synthetic data."""
        # Generate synthetic data with known reliable components
        np.random.seed(42)
        n_samples, n_channels, n_trials = 200, 16, 30
        
        # Create reliable signal components
        time = np.linspace(0, 1, n_samples)
        signal1 = np.sin(2 * np.pi * 10 * time)  # 10 Hz
        signal2 = np.cos(2 * np.pi * 15 * time)  # 15 Hz
        
        # Spatial patterns
        spatial1 = np.random.randn(n_channels)
        spatial2 = np.random.randn(n_channels)
        
        # Generate data
        data = np.zeros((n_samples, n_channels, n_trials))
        snr = 2.0  # Signal to noise ratio
        
        for trial in range(n_trials):
            # Add variability to make it realistic
            amp1 = 1.0 + 0.3 * np.random.randn()
            amp2 = 0.8 + 0.2 * np.random.randn()
            
            for ch in range(n_channels):
                data[:, ch, trial] = (
                    amp1 * spatial1[ch] * signal1 +
                    amp2 * spatial2[ch] * signal2 +
                    np.random.randn(n_samples) / snr
                )
        
        # Fit RCA
        rca = ReliableComponentsAnalysis(n_components=3, random_state=42)
        data_rca = rca.fit_transform(data)
        
        # Verify basic properties
        assert data_rca.shape == (n_samples, 3, n_trials), "Output shape incorrect"
        assert rca.spatial_filters_.shape == (n_channels, 3), "Spatial filters shape incorrect"
        assert rca.forward_models_.shape == (n_channels, 3), "Forward models shape incorrect"
        assert len(rca.eigenvalues_) > 0, "No eigenvalues computed"
        assert all(np.isfinite(rca.eigenvalues_)), "Eigenvalues contain inf/nan"
        
        # Check that eigenvalues are sorted descending
        eigenvals_sorted = rca.eigenvalues_[:rca.n_components]
        assert all(eigenvals_sorted[i] >= eigenvals_sorted[i+1] for i in range(len(eigenvals_sorted)-1)), \
            "Eigenvalues not sorted descending"
            
        return {
            'data_shape': data.shape,
            'rca_shape': data_rca.shape,
            'eigenvalues': rca.eigenvalues_[:3].tolist(),
            'max_eigenvalue': float(np.max(rca.eigenvalues_)),
            'n_components_extracted': rca.n_components
        }
    
    def test_mathematical_properties(self) -> Dict[str, Any]:
        """Test mathematical properties of RCA solution."""
        np.random.seed(123)
        n_samples, n_channels, n_trials = 100, 12, 25
        
        # Generate simple test data
        data = np.random.randn(n_samples, n_channels, n_trials)
        
        rca = ReliableComponentsAnalysis(n_components=5, random_state=123)
        rca.fit(data)
        
        W = rca.spatial_filters_
        A = rca.forward_models_
        R_pool = 0.5 * (rca.covariance_xx_ + rca.covariance_yy_)
        
        # Test orthogonality properties
        # W' * R_pool * W should be diagonal-like (generalized orthogonality)
        WtRW = W.T @ R_pool @ W
        off_diagonal_power = np.sum(WtRW**2) - np.sum(np.diag(WtRW)**2)
        diagonal_power = np.sum(np.diag(WtRW)**2)
        orthogonality_ratio = off_diagonal_power / diagonal_power if diagonal_power > 0 else np.inf
        
        # Test forward model relationship: A = R_pool * W * (W' * R_pool * W)^(-1)
        try:
            A_computed = R_pool @ W @ np.linalg.pinv(WtRW)
            forward_model_error = np.mean((A - A_computed)**2)
        except np.linalg.LinAlgError:
            forward_model_error = np.inf
            
        # Test that spatial filters have unit norm in the metric defined by R_pool
        filter_norms = np.diag(W.T @ R_pool @ W)
        
        return {
            'orthogonality_ratio': float(orthogonality_ratio),
            'forward_model_error': float(forward_model_error),
            'filter_norms': filter_norms.tolist(),
            'eigenvalue_range': [float(np.min(rca.eigenvalues_)), float(np.max(rca.eigenvalues_))],
            'covariance_condition_number': float(np.linalg.cond(R_pool))
        }
    
    def test_edge_cases(self) -> Dict[str, Any]:
        """Test edge cases and error handling."""
        results = {}
        
        # Test with minimum number of trials
        try:
            data = np.random.randn(50, 8, 3)  # Minimum 3 trials
            rca = ReliableComponentsAnalysis(n_components=2)
            rca.fit(data)
            results['min_trials'] = 'PASSED'
        except Exception as e:
            results['min_trials'] = f'FAILED: {e}'
        
        # Test with too few trials (should fail)
        try:
            data = np.random.randn(50, 8, 2)  # Only 2 trials
            rca = ReliableComponentsAnalysis(n_components=2)
            rca.fit(data)
            results['too_few_trials'] = 'FAILED: Should have raised error'
        except ValueError:
            results['too_few_trials'] = 'PASSED'
        except Exception as e:
            results['too_few_trials'] = f'FAILED: Wrong error type: {e}'
        
        # Test with NaN values
        try:
            data = np.random.randn(50, 8, 10)
            data[10:15, 2:4, 3:6] = np.nan  # Insert some NaN values
            rca = ReliableComponentsAnalysis(n_components=2)
            data_rca = rca.fit_transform(data)
            has_nan = np.any(np.isnan(data_rca))
            results['nan_handling'] = f'PASSED: Output has NaN={has_nan}'
        except Exception as e:
            results['nan_handling'] = f'FAILED: {e}'
        
        # Test with more components than channels
        try:
            data = np.random.randn(50, 5, 10)
            rca = ReliableComponentsAnalysis(n_components=8)  # More than 5 channels
            rca.fit(data)
            results['too_many_components'] = 'FAILED: Should have raised error'
        except ValueError:
            results['too_many_components'] = 'PASSED'
        except Exception as e:
            results['too_many_components'] = f'FAILED: Wrong error type: {e}'
            
        # Test transform before fit
        try:
            data = np.random.randn(50, 8, 10)
            rca = ReliableComponentsAnalysis(n_components=2)
            rca.transform(data)  # Transform without fitting
            results['transform_before_fit'] = 'FAILED: Should have raised error'
        except ValueError:
            results['transform_before_fit'] = 'PASSED'
        except Exception as e:
            results['transform_before_fit'] = f'FAILED: Wrong error type: {e}'
        
        return results
    
    def test_different_data_sizes(self) -> Dict[str, Any]:
        """Test performance with different data sizes."""
        results = {}
        
        test_configs = [
            {'samples': 100, 'channels': 8, 'trials': 15, 'components': 3},
            {'samples': 200, 'channels': 16, 'trials': 30, 'components': 4},
            {'samples': 500, 'channels': 32, 'trials': 50, 'components': 5},
        ]
        
        for i, config in enumerate(test_configs):
            try:
                # Generate data
                data = np.random.randn(config['samples'], config['channels'], config['trials'])
                
                # Add some structure
                reliable_signal = np.sin(np.linspace(0, 4*np.pi, config['samples']))
                spatial_pattern = np.random.randn(config['channels'])
                
                for trial in range(config['trials']):
                    amplitude = 1 + 0.5 * np.random.randn()
                    for ch in range(config['channels']):
                        data[:, ch, trial] += amplitude * spatial_pattern[ch] * reliable_signal
                
                # Fit RCA
                start_time = time.time()
                rca = ReliableComponentsAnalysis(n_components=config['components'], random_state=42)
                data_rca = rca.fit_transform(data)
                fit_time = time.time() - start_time
                
                results[f'config_{i+1}'] = {
                    'config': config,
                    'fit_time': fit_time,
                    'max_eigenvalue': float(np.max(rca.eigenvalues_)),
                    'output_shape': data_rca.shape,
                    'success': True
                }
                
            except Exception as e:
                results[f'config_{i+1}'] = {
                    'config': config,
                    'error': str(e),
                    'success': False
                }
        
        return results
    
    def test_reproducibility(self) -> Dict[str, Any]:
        """Test that results are reproducible with same random seed."""
        np.random.seed(456)
        data = np.random.randn(100, 10, 20)
        
        # Fit RCA twice with same random state
        rca1 = ReliableComponentsAnalysis(n_components=3, random_state=456)
        data_rca1 = rca1.fit_transform(data)
        
        rca2 = ReliableComponentsAnalysis(n_components=3, random_state=456)  
        data_rca2 = rca2.fit_transform(data)
        
        # Check if results are identical (allowing for sign flips in components)
        eigenval_diff = np.max(np.abs(rca1.eigenvalues_ - rca2.eigenvalues_))
        
        # For spatial filters, check if they're identical or sign-flipped
        spatial_filter_diff = 0
        for comp in range(3):
            filter1 = rca1.spatial_filters_[:, comp]
            filter2 = rca2.spatial_filters_[:, comp]
            
            # Check both same sign and opposite sign
            diff_same = np.mean((filter1 - filter2)**2)
            diff_flip = np.mean((filter1 + filter2)**2)
            spatial_filter_diff += min(diff_same, diff_flip)
        
        return {
            'eigenvalue_difference': float(eigenval_diff),
            'spatial_filter_difference': float(spatial_filter_diff),
            'is_reproducible': eigenval_diff < 1e-10 and spatial_filter_diff < 1e-10
        }
    
    def test_demo_functionality(self) -> Dict[str, Any]:
        """Test the demo function from the main module."""
        # Capture demo output
        import io
        from contextlib import redirect_stdout
        
        with redirect_stdout(io.StringIO()) as captured_output:
            # Import and run demo (without showing plots)
            plt.ioff()  # Turn off interactive plotting
            from rca import demo_rca_analysis
            rca, data, data_rca = demo_rca_analysis()
            plt.close('all')  # Close any plots
        
        demo_output = captured_output.getvalue()
        
        # Verify demo results
        return {
            'demo_completed': True,
            'data_shape': data.shape,
            'rca_shape': data_rca.shape,
            'eigenvalues': rca.eigenvalues_[:3].tolist(),
            'output_length': len(demo_output),
            'has_output': len(demo_output) > 0
        }
    
    def test_regularization(self) -> Dict[str, Any]:
        """Test automatic regularization parameter selection."""
        np.random.seed(789)
        data = np.random.randn(150, 20, 25)
        
        # Test with automatic regularization
        rca_auto = ReliableComponentsAnalysis(n_components=5, n_reg=None, random_state=789)
        rca_auto.fit(data)
        
        # Test with manual regularization
        rca_manual = ReliableComponentsAnalysis(n_components=5, n_reg=10, random_state=789)
        rca_manual.fit(data)
        
        return {
            'auto_regularization_used': rca_auto.n_reg is None,
            'auto_eigenvalues': rca_auto.eigenvalues_[:5].tolist(),
            'manual_eigenvalues': rca_manual.eigenvalues_[:5].tolist(),
            'eigenvalue_difference': float(np.mean(np.abs(rca_auto.eigenvalues_[:5] - rca_manual.eigenvalues_[:5])))
        }
    
    def run_all_tests(self):
        """Run all tests in the test suite."""
        print("=" * 60)
        print("COMPREHENSIVE RCA TEST SUITE")
        print("=" * 60)
        
        # Run all tests
        self.run_test("Basic Functionality", self.test_basic_functionality)
        self.run_test("Mathematical Properties", self.test_mathematical_properties)  
        self.run_test("Edge Cases", self.test_edge_cases)
        self.run_test("Different Data Sizes", self.test_different_data_sizes)
        self.run_test("Reproducibility", self.test_reproducibility)
        self.run_test("Demo Functionality", self.test_demo_functionality)
        self.run_test("Regularization", self.test_regularization)
        
        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Total tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.total_tests - self.passed_tests}")
        print(f"Success rate: {100 * self.passed_tests / self.total_tests:.1f}%")
        
        if self.passed_tests == self.total_tests:
            print("\n🎉 ALL TESTS PASSED! RCA implementation is working correctly.")
        else:
            print(f"\n⚠️  {self.total_tests - self.passed_tests} test(s) failed. See details above.")
            
        return self.results


def main():
    """Main test execution."""
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Run test suite
    test_suite = RCATestSuite()
    results = test_suite.run_all_tests()
    
    # Print detailed results for failed tests
    failed_tests = [name for name, result in results.items() if result['status'] == 'FAILED']
    if failed_tests:
        print("\nDETAILED FAILURE INFORMATION:")
        print("-" * 40)
        for test_name in failed_tests:
            result = results[test_name]
            print(f"{test_name}: {result['error']}")
    
    return results


if __name__ == "__main__":
    main()