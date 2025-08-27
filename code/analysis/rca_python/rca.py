"""
Reliable Components Analysis (RCA) - Python Implementation

This module provides a Python implementation of Reliable Components Analysis,
a dimensionality reduction technique for neural data that maximizes trial-to-trial
reliability by finding spatial filters that capture consistent patterns across trials.

Based on the MATLAB toolbox by Jacek P. Dmochowski (2015).
Key paper: Dmochowski et al., Front Hum Neurosci (2012)

Author: Converted to Python for music preference study
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.stats import zscore
from itertools import combinations
import warnings
from typing import Union, Optional, Tuple, List, Dict, Any
import joblib
from joblib import Parallel, delayed

class ReliableComponentsAnalysis:
    """
    Reliable Components Analysis (RCA) for neural data dimensionality reduction.
    
    RCA finds spatial filters that maximize trial-to-trial reliability by solving
    a generalized eigenvalue problem on covariance matrices computed from trial pairs.
    
    Parameters
    ----------
    n_components : int, optional (default=3)
        Number of reliable components to extract
    n_reg : int or None, optional (default=None)
        Regularization parameter for autocovariance diagonalization.
        If None, automatically determined using knee point detection.
    random_state : int or None, optional (default=None)
        Random seed for reproducibility
    n_jobs : int, optional (default=1)
        Number of parallel jobs for covariance computation
    """
    
    def __init__(self, n_components: int = 3, n_reg: Optional[int] = None, 
                 random_state: Optional[int] = None, n_jobs: int = 1):
        self.n_components = n_components
        self.n_reg = n_reg
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        # Fitted parameters
        self.spatial_filters_ = None  # W matrix
        self.forward_models_ = None   # A matrix
        self.eigenvalues_ = None      # generalized eigenvalues
        self.covariance_xx_ = None    # Rxx
        self.covariance_yy_ = None    # Ryy  
        self.covariance_xy_ = None    # Rxy
        self.is_fitted_ = False
        
    def _validate_input(self, data: np.ndarray) -> np.ndarray:
        """Validate and prepare input data."""
        data = np.asarray(data, dtype=np.float64)
        
        if data.ndim != 3:
            raise ValueError(f"Data must be 3D array (samples x channels x trials), "
                           f"got shape {data.shape}")
                           
        n_samples, n_channels, n_trials = data.shape
        
        if n_samples < n_channels:
            warnings.warn("Number of samples is less than number of channels. "
                         "Consider using more data points.")
                         
        if n_trials < 3:
            raise ValueError("Need at least 3 trials for RCA analysis")
            
        return data
        
    def _knee_point_detection(self, eigenvalues: np.ndarray) -> int:
        """
        Detect knee point in eigenvalue spectrum for automatic regularization.
        
        Parameters
        ----------
        eigenvalues : array-like
            Eigenvalues in ascending order
            
        Returns
        -------
        knee_idx : int
            Index of the knee point
        """
        n_vals = len(eigenvalues)
        if n_vals < 3:
            return max(0, n_vals - 2)
            
        # Use simple method: find point with maximum curvature
        # This approximates the more complex knee detection from MATLAB
        x = np.arange(n_vals)
        
        # Compute second derivative as curvature measure
        if n_vals >= 3:
            second_deriv = np.diff(eigenvalues, 2)
            knee_idx = np.argmax(np.abs(second_deriv)) + 1
            # Ensure we use a reasonable number of components
            knee_idx = max(knee_idx, n_vals // 4)
            knee_idx = min(knee_idx, n_vals - 2)
        else:
            knee_idx = n_vals - 2
            
        return knee_idx
        
    def _compute_covariances_vectorized(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute covariance matrices using vectorized operations for efficiency.
        
        Parameters
        ----------
        data : array-like, shape (n_samples, n_channels, n_trials)
            Input neural data
            
        Returns
        -------
        Rxx, Ryy, Rxy : arrays
            Covariance matrices
        """
        n_samples, n_channels, n_trials = data.shape
        
        # Generate all trial pairs (including reverse pairs for symmetry)
        pairs = list(combinations(range(n_trials), 2))
        pairs_symmetric = pairs + [(j, i) for i, j in pairs]
        n_pairs = len(pairs_symmetric)
        
        print(f"Computing covariances for {n_trials} trials ({n_pairs} pairs)...")
        
        # Center data by removing mean across time and trials
        data_centered = data - np.nanmean(data, axis=(0, 2), keepdims=True)
        
        # Reshape for efficient computation: (n_samples * n_trials, n_channels)
        data_2d = data_centered.reshape(-1, n_channels)
        
        # Handle NaN values by setting to zero (contribution tracked separately)
        valid_mask = ~np.isnan(data_2d)
        data_clean = np.where(valid_mask, data_2d, 0)
        
        # For large numbers of trials, use batch processing
        if n_trials >= 30:
            return self._compute_covariances_batch(data_centered, pairs_symmetric)
        else:
            return self._compute_covariances_direct(data_centered, pairs_symmetric)
            
    def _compute_covariances_batch(self, data_centered: np.ndarray, 
                                 pairs: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute covariances in batches for memory efficiency."""
        n_samples, n_channels, n_trials = data_centered.shape
        n_pairs = len(pairs)
        
        # Initialize covariance accumulators
        sum_xx = np.zeros((n_channels, n_channels))
        sum_yy = np.zeros((n_channels, n_channels))  
        sum_xy = np.zeros((n_channels, n_channels))
        n_points = 0
        
        # Process in batches
        batch_size = min(1000, n_pairs)
        
        for batch_start in range(0, n_pairs, batch_size):
            batch_end = min(batch_start + batch_size, n_pairs)
            batch_pairs = pairs[batch_start:batch_end]
            
            for trial_i, trial_j in batch_pairs:
                # Extract trial data: (n_samples, n_channels)
                x_trial = data_centered[:, :, trial_i] 
                y_trial = data_centered[:, :, trial_j]
                
                # Handle NaNs
                x_clean = np.where(np.isnan(x_trial), 0, x_trial)
                y_clean = np.where(np.isnan(y_trial), 0, y_trial)
                
                # Accumulate covariances: (n_channels, n_channels)
                sum_xx += x_clean.T @ x_clean
                sum_yy += y_clean.T @ y_clean
                sum_xy += x_clean.T @ y_clean
                n_points += n_samples
                
        # Normalize by number of data points
        Rxx = sum_xx / n_points
        Ryy = sum_yy / n_points  
        Rxy = sum_xy / n_points
        
        return Rxx, Ryy, Rxy
        
    def _compute_covariances_direct(self, data_centered: np.ndarray,
                                  pairs: List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute covariances directly for smaller datasets."""
        n_samples, n_channels, n_trials = data_centered.shape
        
        # Concatenate all trial pairs
        x_data = []
        y_data = []
        
        for trial_i, trial_j in pairs:
            x_trial = data_centered[:, :, trial_i].reshape(-1, n_channels)
            y_trial = data_centered[:, :, trial_j].reshape(-1, n_channels) 
            
            x_data.append(x_trial)
            y_data.append(y_trial)
            
        # Stack all pairs: (n_pairs * n_samples, n_channels)
        x_concat = np.vstack(x_data)
        y_concat = np.vstack(y_data)
        
        # Remove pair-wise means
        x_concat -= np.nanmean(x_concat, axis=0, keepdims=True)
        y_concat -= np.nanmean(y_concat, axis=0, keepdims=True)
        
        # Handle NaNs
        x_concat = np.where(np.isnan(x_concat), 0, x_concat)
        y_concat = np.where(np.isnan(y_concat), 0, y_concat)
        
        # Compute covariances
        n_total = x_concat.shape[0]
        Rxx = (x_concat.T @ x_concat) / n_total
        Ryy = (y_concat.T @ y_concat) / n_total
        Rxy = (x_concat.T @ y_concat) / n_total
        
        return Rxx, Ryy, Rxy
        
    def _solve_generalized_eigenvalue_problem(self, Rxx: np.ndarray, Ryy: np.ndarray, 
                                            Rxy: np.ndarray, n_reg: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve the generalized eigenvalue problem for RCA.
        
        This follows the approach in rcaTrain.m:
        1. Compute regularized pooled covariance 
        2. Solve generalized eigenvalue problem
        3. Sort by eigenvalue magnitude
        """
        n_channels = Rxx.shape[0]
        
        # Handle NaN values in covariance matrices
        if np.any(np.isnan(Rxx)) or np.any(np.isnan(Ryy)) or np.any(np.isnan(Rxy)):
            warnings.warn("Covariance matrices contain NaNs, setting to zero")
            Rxx = np.nan_to_num(Rxx)
            Ryy = np.nan_to_num(Ryy)
            Rxy = np.nan_to_num(Rxy)
            
        # Compute pooled autocovariance matrix
        R_pool = Rxx + Ryy
        
        # Eigendecomposition of pooled covariance for regularization
        eigenvals_pool, eigenvecs_pool = linalg.eigh(R_pool)
        
        # Determine regularization if not provided
        if self.n_reg is None:
            knee_idx = self._knee_point_detection(eigenvals_pool)
            n_reg = len(eigenvals_pool) - knee_idx
            print(f"Using {n_reg} bases for autocovariance diagonalization")
        else:
            n_reg = self.n_reg
            
        if n_reg >= n_channels:
            n_reg = n_channels - 1
            warnings.warn(f"Regularization parameter too large, setting to {n_reg}")
            
        # Use top n_reg eigenvectors for regularization  
        eigenvals_reg = eigenvals_pool[-n_reg:]
        eigenvecs_reg = eigenvecs_pool[:, -n_reg:]
        
        # Compute regularized cross-covariance matrix
        # Rw = V * D^(-1) * V' * (Rxy + Rxy')
        cross_cov_sym = Rxy + Rxy.T
        inv_eigenvals = 1.0 / eigenvals_reg
        Rw = eigenvecs_reg @ np.diag(inv_eigenvals) @ eigenvecs_reg.T @ cross_cov_sym
        
        # Solve generalized eigenvalue problem
        eigenvals_gen, eigenvecs_gen = linalg.eigh(Rw)
        
        # Sort by eigenvalue magnitude (descending)
        sort_indices = np.argsort(np.abs(eigenvals_gen))[::-1]
        eigenvals_sorted = eigenvals_gen[sort_indices]
        eigenvecs_sorted = eigenvecs_gen[:, sort_indices]
        
        return eigenvecs_sorted, eigenvals_sorted, eigenvals_pool
        
    def fit(self, data: Union[np.ndarray, List[np.ndarray]]) -> 'ReliableComponentsAnalysis':
        """
        Fit RCA model to neural data.
        
        Parameters
        ----------
        data : array-like or list of arrays
            Neural data with shape (n_samples, n_channels, n_trials)
            If list, each element should be data for one condition/subject
            
        Returns
        -------
        self : ReliableComponentsAnalysis
            The fitted estimator
        """
        if isinstance(data, list):
            # Concatenate multiple conditions/subjects
            data = np.concatenate(data, axis=2)
            
        data = self._validate_input(data)
        n_samples, n_channels, n_trials = data.shape
        
        if self.n_components > n_channels:
            raise ValueError(f"n_components ({self.n_components}) cannot exceed "
                           f"n_channels ({n_channels})")
        
        print(f"Fitting RCA on data with shape {data.shape}")
        
        # Compute covariance matrices
        print("Computing covariance matrices...")
        Rxx, Ryy, Rxy = self._compute_covariances_vectorized(data)
        
        # Store covariance matrices
        self.covariance_xx_ = Rxx
        self.covariance_yy_ = Ryy  
        self.covariance_xy_ = Rxy
        
        # Solve generalized eigenvalue problem
        print("Solving generalized eigenvalue problem...")
        W_full, eigenvals, eigenvals_pool = self._solve_generalized_eigenvalue_problem(
            Rxx, Ryy, Rxy, self.n_reg)
            
        # Extract top components
        self.spatial_filters_ = W_full[:, :self.n_components]
        self.eigenvalues_ = eigenvals
        
        # Compute forward models (spatial patterns)
        R_pool = 0.5 * (Rxx + Ryy)
        try:
            # A = R_pool * W * (W' * R_pool * W)^(-1)
            WtRW = self.spatial_filters_.T @ R_pool @ self.spatial_filters_
            self.forward_models_ = R_pool @ self.spatial_filters_ @ linalg.pinv(WtRW)
        except linalg.LinAlgError:
            warnings.warn("Could not compute forward models due to numerical issues")
            self.forward_models_ = self.spatial_filters_.copy()
            
        self.is_fitted_ = True
        
        print(f"RCA fitting complete. Extracted {self.n_components} components.")
        print(f"Eigenvalues: {self.eigenvalues_[:self.n_components]}")
        
        return self
        
    def transform(self, data: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Project data into RCA component space.
        
        Parameters
        ----------
        data : array-like or list of arrays
            Neural data with shape (n_samples, n_channels, n_trials)
            
        Returns
        -------
        data_rca : array-like or list of arrays
            Transformed data with shape (n_samples, n_components, n_trials)
        """
        if not self.is_fitted_:
            raise ValueError("RCA model must be fitted before transform")
            
        if isinstance(data, list):
            return [self._project_single(d) for d in data]
        else:
            return self._project_single(data)
            
    def _project_single(self, data: np.ndarray) -> np.ndarray:
        """Project single data array to RCA space."""
        data = self._validate_input(data)
        n_samples, n_channels, n_trials = data.shape
        
        if n_channels != self.spatial_filters_.shape[0]:
            raise ValueError(f"Number of channels ({n_channels}) does not match "
                           f"fitted model ({self.spatial_filters_.shape[0]})")
                           
        # Project data: Y = W' * X for each trial
        data_rca = np.zeros((n_samples, self.n_components, n_trials))
        
        for trial in range(n_trials):
            trial_data = data[:, :, trial]  # (n_samples, n_channels)
            
            # Handle NaN values
            valid_mask = ~np.isnan(trial_data)
            
            for comp in range(self.n_components):
                spatial_filter = self.spatial_filters_[:, comp]  # (n_channels,)
                
                # Compute weighted sum, handling NaNs
                weighted_data = trial_data * spatial_filter[np.newaxis, :]
                data_rca[:, comp, trial] = np.nansum(weighted_data, axis=1)
                
        return data_rca
        
    def fit_transform(self, data: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
        """Fit RCA model and transform data in one step."""
        return self.fit(data).transform(data)
        
    def plot_components(self, show_timecourses: bool = True, 
                       electrode_positions: Optional[np.ndarray] = None,
                       figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot RCA components (topographies and time courses).
        
        Parameters
        ----------
        show_timecourses : bool, optional (default=True)
            Whether to show component time courses
        electrode_positions : array-like, optional
            Electrode positions for topographic plotting
        figsize : tuple, optional (default=(12, 8))
            Figure size
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure
        """
        if not self.is_fitted_:
            raise ValueError("RCA model must be fitted before plotting")
            
        n_rows = 2 if show_timecourses else 1
        fig, axes = plt.subplots(n_rows, self.n_components, figsize=figsize)
        
        if self.n_components == 1:
            axes = axes.reshape(-1, 1)
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
            
        # Plot forward models (topographies)
        for comp in range(self.n_components):
            ax = axes[0, comp] if n_rows > 1 else axes[0, comp]
            
            if electrode_positions is not None:
                # Simple scatter plot of electrode weights
                scatter = ax.scatter(electrode_positions[:, 0], electrode_positions[:, 1], 
                                   c=self.forward_models_[:, comp], 
                                   cmap='RdBu_r', s=50)
                plt.colorbar(scatter, ax=ax)
            else:
                # Line plot of spatial weights
                ax.plot(self.forward_models_[:, comp], 'k-o')
                ax.set_xlabel('Channel')
                ax.set_ylabel('Weight')
                
            ax.set_title(f'RC{comp + 1} (λ={self.eigenvalues_[comp]:.3f})')
            
        # Plot eigenvalue spectrum
        if show_timecourses and len(axes) > 1:
            ax_spectrum = axes[1, -1]
            ax_spectrum.plot(self.eigenvalues_[:min(20, len(self.eigenvalues_))], 'ko-')
            ax_spectrum.set_xlabel('Component')
            ax_spectrum.set_ylabel('Eigenvalue')
            ax_spectrum.set_title('Eigenvalue Spectrum')
            ax_spectrum.grid(True)
            
        plt.tight_layout()
        return fig


def demo_rca_analysis():
    """Demo function showing RCA usage with synthetic data."""
    print("=== RCA Demo with Synthetic Data ===")
    
    # Generate synthetic neural data
    np.random.seed(42)
    n_samples, n_channels, n_trials = 200, 32, 50
    
    # Create synthetic signal with reliable components
    time = np.linspace(0, 1, n_samples)
    reliable_signal1 = np.sin(2 * np.pi * 10 * time)  # 10 Hz
    reliable_signal2 = np.cos(2 * np.pi * 15 * time)  # 15 Hz
    
    # Spatial patterns
    spatial_pattern1 = np.random.randn(n_channels)
    spatial_pattern2 = np.random.randn(n_channels)
    
    # Generate data
    data = np.zeros((n_samples, n_channels, n_trials))
    
    for trial in range(n_trials):
        # Add reliable components
        signal1_var = 1.0 + 0.3 * np.random.randn()  # trial-to-trial variability
        signal2_var = 0.8 + 0.2 * np.random.randn()
        
        for ch in range(n_channels):
            data[:, ch, trial] = (signal1_var * spatial_pattern1[ch] * reliable_signal1 +
                                signal2_var * spatial_pattern2[ch] * reliable_signal2 +
                                0.5 * np.random.randn(n_samples))  # noise
    
    # Apply RCA
    rca = ReliableComponentsAnalysis(n_components=3, random_state=42)
    data_rca = rca.fit_transform(data)
    
    print(f"Original data shape: {data.shape}")
    print(f"RCA transformed shape: {data_rca.shape}")
    print(f"Top 3 eigenvalues: {rca.eigenvalues_[:3]}")
    
    # Plot results
    rca.plot_components()
    plt.show()
    
    return rca, data, data_rca


if __name__ == "__main__":
    demo_rca_analysis()