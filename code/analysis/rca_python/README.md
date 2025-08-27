# RCA Python Implementation

A Python implementation of Reliable Components Analysis (RCA) for neural data dimensionality reduction, specifically designed for the music preference EEG study.

## Overview

Reliable Components Analysis (RCA) is a technique that reduces dimensionality and increases interpretability of neural data by finding spatial filters that maximize trial-to-trial reliability. Unlike PCA which maximizes variance, RCA maximizes the correlation between repeated measurements, making it ideal for identifying consistent neural responses across trials.

This Python implementation is converted from the MATLAB toolbox by Jacek P. Dmochowski (2015) and optimized for integration with the music preference study pipeline.

## Key Features

- **Trial-to-trial reliability maximization**: Finds components that are most consistent across repeated measurements
- **Robust to noise**: Focuses on reliable signal patterns rather than high-variance noise
- **Physiologically interpretable**: Spatial patterns have clear neurophysiological meaning
- **Music study integration**: Specialized functions for comparing preferred vs non-preferred music responses
- **MNE compatibility**: Direct integration with MNE-Python Epochs objects
- **Efficient computation**: Vectorized operations and parallel processing for large datasets

## Algorithm Overview

RCA works by:

1. **Covariance Computation**: Calculate auto- and cross-covariance matrices from trial pairs
   - Rxx: covariance of "data record 1" (odd trials)
   - Ryy: covariance of "data record 2" (even trials)  
   - Rxy: cross-covariance between records
   
2. **Regularization**: Use eigendecomposition of pooled covariance (Rxx + Ryy) to regularize the solution

3. **Generalized Eigenvalue Problem**: Solve for spatial filters W that maximize:
   ```
   W' * (Rxy + Rxy') * W / W' * (Rxx + Ryy) * W
   ```

4. **Component Extraction**: Extract top components with highest eigenvalues (reliability)

5. **Forward Models**: Compute spatial patterns A that show how reliable sources project to sensors

## Installation

The package requires:
```bash
pip install numpy scipy matplotlib mne joblib pandas
```

Add to your Python path:
```python
import sys
sys.path.append('/path/to/music_preference/code/analysis')
from rca_python import ReliableComponentsAnalysis
```

## Basic Usage

### Simple RCA Analysis

```python
import numpy as np
from rca_python import ReliableComponentsAnalysis

# Your data: (n_samples, n_channels, n_trials)
data = np.random.randn(200, 32, 50)

# Fit RCA
rca = ReliableComponentsAnalysis(n_components=3)
data_rca = rca.fit_transform(data)

print(f"Original shape: {data.shape}")
print(f"RCA shape: {data_rca.shape}")
print(f"Eigenvalues: {rca.eigenvalues_[:3]}")

# Plot results
rca.plot_components()
```

### Music Preference Analysis

```python
from rca_python import run_rca_on_music_data, plot_music_rca_results

# Analyze single subject
results = run_rca_on_music_data(
    subject_id='pilot_2',
    data_dir='../../../data/ica_cleaned',
    n_components=3
)

# Visualize results
plot_music_rca_results(results)
```

### Batch Analysis

```python
from rca_python import batch_rca_analysis

# Process multiple subjects
subjects = ['pilot_1', 'pilot_2', 'pilot_3', 'pilot_4', 'pilot_5']
all_results = batch_rca_analysis(
    subject_ids=subjects,
    data_dir='../../../data/ica_cleaned',
    output_dir='../../../output/rca_analysis'
)
```

## Command Line Usage

Run analysis from command line:

```bash
# Single subject
python run_music_rca_analysis.py --subject pilot_2

# All subjects  
python run_music_rca_analysis.py --all-subjects

# Custom parameters
python run_music_rca_analysis.py --subject pilot_2 --n-components 5 --n-reg 10
```

## API Reference

### ReliableComponentsAnalysis

Main RCA class implementing the algorithm.

**Parameters:**
- `n_components` (int): Number of reliable components to extract (default: 3)
- `n_reg` (int, optional): Regularization parameter. If None, auto-detected using knee point method
- `random_state` (int, optional): Random seed for reproducibility
- `n_jobs` (int): Number of parallel jobs for computation (default: 1)

**Attributes:**
- `spatial_filters_`: Spatial filters W (channels × components)
- `forward_models_`: Forward models A (channels × components)  
- `eigenvalues_`: Generalized eigenvalues (reliability measures)
- `covariance_xx_`, `covariance_yy_`, `covariance_xy_`: Computed covariance matrices

**Methods:**
- `fit(data)`: Fit RCA model to data
- `transform(data)`: Project data into RCA space
- `fit_transform(data)`: Fit and transform in one step
- `plot_components()`: Visualize spatial patterns and eigenvalues

### Utility Functions

**Data Loading:**
- `load_music_preference_data()`: Load EEG data organized by preference
- `epochs_to_rca_format()`: Convert MNE Epochs to RCA format

**Analysis:**
- `run_rca_on_music_data()`: Complete RCA analysis pipeline for single subject
- `batch_rca_analysis()`: Process multiple subjects
- `compute_rca_reliability_metrics()`: Calculate reliability metrics

**Visualization:**
- `plot_music_rca_results()`: Create comprehensive results plots
- Group summary plots for multi-subject analysis

## Output Interpretation

### Eigenvalues
- Higher eigenvalues indicate more reliable components
- Values range from 0 to 1 (correlation coefficient)
- Typically only first few components are meaningful

### Spatial Patterns (Forward Models)
- Show how reliable sources project to sensor space
- Positive/negative values indicate source orientation
- Can be interpreted similarly to ERP topographies

### Time Courses  
- Projected neural activity for each reliable component
- Higher amplitude indicates stronger activation
- Compare between conditions (preferred vs non-preferred)

## Differences from MATLAB Version

**Improvements:**
- Modern Python syntax and error handling
- Better memory management for large datasets
- Integrated plotting with matplotlib
- Direct MNE-Python compatibility
- Batch processing capabilities
- Automated parameter selection

**Maintained Compatibility:**
- Same mathematical algorithm
- Equivalent output format
- Similar parameter naming

## Music Preference Study Integration

The implementation is specifically designed for the music preference study:

### Data Organization
- Automatically loads behavioral ratings to determine preferred/non-preferred songs
- Handles subject-specific trial files
- Integrates with existing preprocessing pipeline

### Analysis Workflow
1. Load preprocessed EEG data (`pilot_X-trial*_ica_cleaned.fif`)
2. Group trials by preference rating
3. Fit RCA on combined data to find common reliable components
4. Compare component activations between preferred/non-preferred conditions
5. Generate plots showing spatial patterns and condition differences

### Expected Results
- **RC1**: Should capture dominant neural response to music
- **Higher eigenvalues** for preferred music components indicate stronger reliability
- **Spatial patterns** may show fronto-central activation (attention/preference)
- **Time course differences** between conditions indicate preference-related modulation

## Validation

Run the validation script to test the implementation:

```bash
python validate_rca_conversion.py
```

This tests:
- Basic RCA functionality
- Mathematical properties (eigenvalue sorting, orthogonality)
- Data handling (NaN values, different trial numbers)
- Music study integration workflow

## Troubleshooting

**Common Issues:**

1. **"Number of samples less than channels" warning**
   - Use more time points or reduce channels
   - Consider temporal downsampling

2. **Low eigenvalues**
   - Data may lack reliable components
   - Try reducing n_components or n_reg
   - Check data preprocessing quality

3. **Memory errors with large datasets**
   - Reduce batch size in covariance computation
   - Use fewer parallel jobs (n_jobs=1)
   - Process subjects individually

4. **NaN in results**
   - Check input data for excessive NaN values
   - Verify channel consistency across trials

## References

**Primary Reference:**
Dmochowski, J. P., Sajda, P., Dias, J., & Parra, L. C. (2012). Correlated components of ongoing EEG point to emotionally laden attention–a possible marker of engagement?. *Frontiers in Human Neuroscience*, 6.

**Related Papers:**
- Dmochowski, J. P., et al. (2014). Audience preferences are predicted by temporal reliability of neural processing. *Nature Communications*, 5.
- Dmochowski, J. P., Greaves, A. S., & Norcia, A. M. (2015). Maximally reliable spatial filtering of steady state visual evoked potentials. *NeuroImage*, 109, 63-72.

## License

Converted for the music preference study. Original MATLAB implementation by Jacek P. Dmochowski (2015).

## Contact

For issues specific to this Python implementation, please refer to the music preference study documentation or create an issue in the project repository.