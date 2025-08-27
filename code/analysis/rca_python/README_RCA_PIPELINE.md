# RCA Analysis Pipeline for Music Preference Study

This directory contains a complete implementation of Reliable Components Analysis (RCA) for the music preference EEG study, organized into a coherent pipeline with tests, demos, and analysis tools.

## Directory Structure

```
rca_python/
├── README.md                    # Original RCA documentation
├── README_RCA_PIPELINE.md      # This pipeline documentation
├── __init__.py                 # Package initialization
├── rca.py                      # Core RCA implementation
├── rca_utils.py               # Utility functions for music integration
├── run_music_rca_analysis.py  # Main analysis runner
├── setup.py                   # Package setup
├── validate_rca_conversion.py # Data validation utilities
│
├── tests/                     # Test suite for RCA functionality
│   ├── test_rca_import.py          # Import validation tests
│   ├── test_rca_comprehensive.py   # Comprehensive functionality tests
│   ├── test_rca_music_integration.py # Music data integration tests
│   └── test_rca_topography.py      # Topographic plotting tests
│
├── demos/                     # Demonstration scripts
│   ├── demo_rca_music_analysis.py  # Basic RCA music analysis demo
│   └── demo_rca_topography.py      # Topographic visualization demo
│
├── pooled_analysis/          # Multi-subject pooled analysis
│   └── pooled_multi_subject_rca.py # Pooled RCA across all subjects
│
└── correlation_analysis/     # Inter-subject correlation analysis
    └── rc1_complete_correlation_heatmaps.py # Complete ISC analysis
```

## Core Components

### 1. Core RCA Implementation (`rca.py`)
- **ReliableComponentsAnalysis** class implementing the RCA algorithm
- Finds neural components that are reliable across trials
- Uses generalized eigenvalue decomposition approach
- Methods: `fit()`, `transform()`, `plot_components()`

### 2. Music Integration Utilities (`rca_utils.py`)
- `load_music_preference_data()`: Load and organize EEG trials by preference
- `epochs_to_rca_format()`: Convert MNE epochs to RCA format
- `plot_music_rca_topographies()`: Create topographic maps of RCA components
- `compute_rca_reliability_metrics()`: Calculate component reliability statistics

## Analysis Pipeline

### Stage 1: Basic Testing
Run the test suite to verify functionality:
```bash
cd code/analysis/rca_python/tests
python test_rca_import.py              # Test imports
python test_rca_comprehensive.py       # Test core functionality
python test_rca_music_integration.py   # Test music data integration
python test_rca_topography.py         # Test topographic plotting
```

### Stage 2: Demo Analysis
Explore RCA capabilities with demo scripts:
```bash
cd code/analysis/rca_python/demos
python demo_rca_music_analysis.py     # Basic RCA on music data
python demo_rca_topography.py         # Topographic visualization demo
```

### Stage 3: Pooled Multi-Subject Analysis
Combine all subject data for robust component extraction:
```bash
cd code/analysis/rca_python/pooled_analysis
python pooled_multi_subject_rca.py
```

**Outputs:**
- `output/pooled_rca/pooled_rca_topographies.png` - Component topographies
- `output/pooled_rca/pooled_rca_results.npz` - RCA results
- `output/pooled_rca/subject_contributions.json` - Subject metadata

### Stage 4: Inter-Subject Correlation Analysis
Apply RC1 spatial filter and compute correlations:
```bash
cd code/analysis/rca_python/correlation_analysis
python rc1_complete_correlation_heatmaps.py
```

**Outputs:**
- `output/rc1_complete_analysis/rc1_complete_correlation_heatmaps.png` - 5x5 correlation matrices for all songs
- `output/rc1_complete_analysis/rc1_correlation_summary.png` - Summary statistics
- `output/rc1_complete_analysis/rc1_complete_correlations.npz` - Correlation data
- `output/rc1_complete_analysis/rc1_complete_summary.csv` - ISC summary table

## Key Analysis Features

### Pooled RCA Analysis
- **Data Pooling**: Combines trials from all subjects before RCA computation
- **Robust Components**: Finds components reliable across the entire subject pool
- **Preference Integration**: Separates preferred vs. non-preferred trials
- **Quality Control**: Handles missing data and variable trial lengths

### Inter-Subject Correlation (ISC)
- **RC1 Spatial Filter**: Applies the most reliable component as spatial filter
- **Complete Coverage**: Analyzes all 15 songs across 5 subjects
- **Missing Data Handling**: Properly handles NA values for incomplete data
- **Correlation Matrices**: Creates 5x5 subject correlation matrices per song

### Topographic Visualization
- **EEG Montage**: Uses standard 10-20 electrode system
- **Component Maps**: Shows spatial distribution of reliable components
- **Publication Quality**: High-resolution plots with proper scaling

## Data Requirements

### Input Data Structure
```
data/ica_cleaned/
├── pilot_1/
│   ├── pilot_1-trial1_4-1_proc_*_ica_cleaned.fif
│   ├── pilot_1-trial2_3-3_proc_*_ica_cleaned.fif
│   └── ...
├── pilot_2/ ... pilot_5/
```

### Behavioral Data
- `data/beh_ratings.json`: Preference ratings for each subject-song combination
- Organized by question type (preference, pleasantness, arousal, chills)

## Key Parameters

### RCA Parameters
- **n_components**: Number of components to extract (default: 5)
- **n_subjects**: Expected number of subjects (5)
- **trial_length**: Automatically determined from minimum across subjects

### Analysis Parameters
- **ISC computation**: Pearson correlation between subject RC1 timecourses
- **Data coverage**: Reports available vs. missing subject-song combinations
- **Quality metrics**: Component eigenvalues and reliability measures

## Results Interpretation

### Component Reliability
- **Eigenvalues**: Higher values indicate stronger cross-trial reliability
- **RC1**: Most reliable component, typically used for further analysis
- **Topography**: Spatial pattern showing electrode contributions

### Inter-Subject Correlations
- **High ISC songs**: Indicate shared neural responses across subjects
- **Low ISC songs**: Suggest more variable individual responses
- **Coverage matrix**: Shows data availability patterns

### Expected Output Patterns
- **FC2 dominance**: RC1 often peaks at frontocentral electrodes
- **ISC range**: Typically -0.01 to +0.03 for music stimuli
- **Subject coverage**: Most songs should have 4-5 subjects available

## Troubleshooting

### Common Issues
1. **Import errors**: Ensure you're running from correct directory
2. **Missing data**: Check `data/ica_cleaned/` directory structure
3. **Memory issues**: Large datasets may require chunking for processing
4. **Path problems**: Scripts automatically handle relative paths

### Performance Notes
- **Pooled analysis**: May take 5-10 minutes for full dataset
- **Memory usage**: ~2-4GB for complete analysis
- **Visualization**: High-resolution plots may take additional time

## Citations and Methods

This implementation is based on:
- Dmochowski et al. (2012). "Extracting multidimensional stimulus-response correlations using hybrid encoding-decoding of neural activity"
- RCA methodology for finding reliable neural components across trials
- Inter-subject correlation analysis for measuring neural synchrony

## Contact and Support

For questions about the RCA implementation or analysis pipeline:
- Check test scripts for usage examples
- Review demo scripts for analysis patterns
- Examine utility functions for data handling