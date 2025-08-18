# ABR (Auditory Brainstem Response) Derivation Module

This module provides functionality to derive ABR responses from click stimuli using cross-correlation analysis in the frequency domain. It includes both the original modular script for BrainVision files and a new single-subject script optimized for the music preference project with .fif files.

## Features

### Original Script (`derive_click_ABR.py`)
- **Modular Design**: Separate functions for each processing step
- **BrainVision Support**: Works with .vhdr files
- **Configurable Parameters**: Flexible EEG and stimulus parameters
- **Command-line Interface**: Easy to use with different datasets
- **Error Handling**: Robust validation and error reporting
- **Logging**: Detailed progress tracking

### Single Subject Script (`derive_click_ABR_single_subject.py`)
- **Music Preference Project Optimized**: Designed for pilot study structure
- **FIF File Support**: Direct processing of .fif files (no .vhdr needed)
- **Trial-to-Stimulus Mapping**: Automatic mapping of trials to click stimuli
- **Enhanced Visualization**: Comprehensive plots including individual trials
- **ExpyFun Integration**: Consistent with music preference analysis pipeline
- **25kHz Sampling**: Optimized for high-resolution EEG data

## Requirements

### Python Dependencies
```
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0
pandas>=1.3.0
mne>=1.5.0
expyfun>=0.8.0
h5py>=3.7.0
glob2
pyglet<1.6
PyOpenGL
```

### Environment Management

The project includes environment setup files for easy dependency management:

- **`environment.yml`**: Conda environment with all dependencies
- **`requirements.txt`**: pip requirements for virtual environments

Choose the installation method that best fits your workflow:
- **Conda**: Use `environment.yml` for a complete environment setup
- **pip**: Use `requirements.txt` for pip-based installations
- **Manual**: Install packages individually as needed

### Data Requirements

#### Original Script
- **EEG Data**: BrainVision (.vhdr) files with ABR channels (Plus_R, Minus_R, Plus_L, Minus_L)
- **Click Stimuli**: WAV files named `click000.wav`, `click001.wav`, etc.

#### Single Subject Script (Music Preference Project)
- **EEG Data**: .fif files named `{subject}_click_trial1.fif` through `{subject}_click_trial5.fif`
- **ABR Channels**: Plus_R, Minus_R, Plus_L, Minus_L (automatically referenced)
- **Click Stimuli**: WAV files `click000.wav` through `click004.wav` in `./click_stim/` directory
- **Trial-Stimulus Mapping**: trial1→click000.wav, trial2→click001.wav, ..., trial5→click004.wav

## Installation

### Option 1: Using Conda (Recommended)

1. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate abr_analysis
```

### Option 2: Using pip

1. Create a virtual environment (optional but recommended):
```bash
python -m venv abr_env
source abr_env/bin/activate  # On Windows: abr_env\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Option 3: Manual Installation

1. Ensure all required Python packages are installed:
```bash
pip install numpy scipy matplotlib mne expyfun h5py pyglet PyOpenGL
```

2. For expyfun installation (if not available via pip):
```bash
pip install git+https://github.com/labsn/expyfun
```

3. Place your EEG data and click stimuli in the appropriate directories.

## Usage

### Single Subject Script (Music Preference Project)

#### Basic Usage
```bash
python derive_click_ABR_single_subject.py pilot_2
```

#### For All Subjects
```bash
for subject in pilot_1 pilot_2 pilot_3 pilot_4 pilot_5; do
    python derive_click_ABR_single_subject.py $subject
done
```

### Original Script Command Line Interface

#### Basic Usage
```bash
python derive_click_ABR.py /path/to/eeg_file.vhdr ./output_dir
```

#### With Custom Parameters
```bash
python derive_click_ABR.py /path/to/eeg_file.vhdr ./output_dir \
    --eeg_fs 25000 --eeg_f_hp 2.0 --t_click 30 --plot
```

#### With Custom Response Time Range
```bash
python derive_click_ABR.py /path/to/eeg_file.vhdr ./output_dir \
    --t_start -0.1 --t_stop 0.5 --plot
```

#### With Custom Click Directory
```bash
python derive_click_ABR.py /path/to/eeg_file.vhdr ./output_dir \
    --click_dir /path/to/click_stim --subject_id my_subject
```

### Python Module Usage

#### Basic Example
```python
from derive_click_ABR import derive_abr

abr_response, lags = derive_abr(
    eeg_file="/path/to/eeg_file.vhdr",
    output_dir="./abr_results",
    click_dir="../../click_stim",
    subject_id="subject001"
)
```

#### Custom Parameters Example
```python
abr_response, lags = derive_abr(
    eeg_file="/path/to/eeg_file.vhdr",
    output_dir="./abr_results",
    click_dir="../../click_stim",
    eeg_fs=25000,  # Higher sampling frequency
    eeg_f_hp=2.0,  # Higher high-pass cutoff
    t_click=30,    # Shorter trial length
    click_rate=50,  # Higher click rate
    n_epoch_click=10,  # More epochs
    plot_results=True,  # Generate plots
    subject_id="custom_subject",
    t_start=-100e-3,  # Custom start time (-100 ms)
    t_stop=500e-3     # Custom stop time (500 ms)
)
```

## Parameters

### Required Parameters
- `eeg_file`: Path to EEG .vhdr file
- `output_dir`: Output directory for results

### Single Subject Script Parameters
- `subject_id`: Subject identifier (e.g., pilot_2)
- Fixed parameters optimized for music preference project:
  - `eeg_fs`: 25000 Hz
  - `eeg_f_hp`: 1.0 Hz
  - `t_click`: 60 seconds
  - `click_rate`: 40 Hz
  - `t_start`: -200ms, `t_stop`: +600ms

### Original Script Optional Parameters
- `click_dir`: Directory containing click WAV files (defaults to `click_stim/`)
- `eeg_fs`: EEG sampling frequency (default: 10000 Hz)
- `eeg_f_hp`: High-pass cutoff frequency (default: 1.0 Hz)
- `t_click`: Click trial length in seconds (default: 60)
- `click_rate`: Click rate in Hz (default: 40)
- `stim_fs`: Stimulus sampling frequency (default: 48000 Hz)
- `n_epoch_click`: Number of click epochs to process (default: 5)
- `plot_results`: Whether to generate plots (default: False)
- `subject_id`: Subject identifier (default: "subject")
- `t_start`: Start time for ABR response in seconds (default: -200e-3)
- `t_stop`: Stop time for ABR response in seconds (default: 600e-3)

## Output Files

### Single Subject Script Output
- `{subject_id}_click_ABR.hdf5`: HDF5 file with ABR results (ExpyFun format)
- `{subject_id}_click_ABR.png`: Comprehensive plots (6-panel figure)

### Original Script Output
- `{subject_id}_abr_response.npy`: ABR response array
- `{subject_id}_lags.npy`: Time lags array
- `{subject_id}_abr_results.h5`: HDF5 file with data and metadata
- `{subject_id}_abr_results.txt`: Summary statistics
- `{subject_id}_abr_plot.png`: ABR response plot (if `plot_results=True`)

### HDF5 File Structure

#### Single Subject Script (ExpyFun format)
- `abr_response`: Raw ABR response array
- `abr_response_filtered`: Bandpass filtered ABR (1-1000Hz)
- `lags`: Time lags array (-200 to +600ms)
- `cc_trials`: Individual trial cross-correlations
- `x_in`: Click stimulus pulse trains
- `n_trials`: Number of trials processed
- `click_rate`, `t_click`, `eeg_fs`: Analysis parameters

#### Original Script (h5py format)
- **Datasets**:
  - `abr_response`: ABR response array
  - `lags`: Time lags array
- **Attributes**:
  - `subject_id`: Subject identifier
  - `response_length`: Number of samples
  - `time_range_ms`: Time range in milliseconds
  - `peak_amplitude_uv`: Peak amplitude in microvolts
  - `rms_amplitude_uv`: RMS amplitude in microvolts
  - `sampling_frequency_hz`: Sampling frequency

## Processing Pipeline

1. **EEG Data Loading**: Load BrainVision (.vhdr) files and extract events
2. **Channel Selection**: Pick ABR channels and create referenced channels
3. **Preprocessing**: Apply high-pass filtering and notch filtering
4. **Epoching**: Extract epochs around click events
5. **Stimulus Processing**: Load click WAV files and convert to pulse trains
6. **Cross-correlation**: Use FFT-based cross-correlation to derive ABR response
7. **Response Concatenation**: Create the final ABR response with proper time alignment
8. **Data Storage**: Save results in multiple formats (numpy arrays, HDF5, text summary)
9. **Plot Generation**: Generate and save ABR response plots (optional)

## File Structure

```
music_preference/
├── code/
│   └── analysis/
│       └── derive_click_ABR/          # ABR derivation module
│           ├── derive_click_ABR.py    # Original ABR script (BrainVision)
│           ├── derive_click_ABR_single_subject.py  # Music preference project script (.fif)
│           ├── README_ABR.md          # This documentation
│           ├── environment.yml        # Conda environment setup
│           └── requirements.txt       # pip requirements
├── click_stim/                        # Click stimulus files
│   ├── click000.wav
│   ├── click001.wav
│   ├── click002.wav
│   ├── click003.wav
│   └── click004.wav
├── data/                              # EEG data and results
│   ├── pilot_1/
│   │   ├── pilot_1_click_trial1.fif
│   │   ├── pilot_1_click_trial2.fif
│   │   └── ...
│   ├── pilot_2/
│   │   ├── pilot_2_click_trial1.fif
│   │   ├── pilot_2_click_trial2.fif
│   │   └── ...
│   └── ...
└── output/                            # Analysis results
    ├── pilot_1_click_ABR.hdf5
    ├── pilot_1_click_ABR.png
    └── ...
```

## Error Handling

The module includes comprehensive error handling for:
- **File Not Found**: Missing EEG or click stimulus files
- **Parameter Validation**: Invalid parameter values
- **Processing Errors**: Issues during ABR derivation
- **Output Directory**: Automatic creation of output directories

## Logging

The module provides detailed logging output including:
- Processing progress
- Parameter validation
- File operations
- Error messages

## Examples

See `example_abr_usage.py` for comprehensive usage examples including:
- Basic usage
- Custom parameters
- Batch processing
- Command-line interface examples

### HDF5 Data Access Example
```python
import h5py

# Load ABR results from HDF5 file
with h5py.File('subject001_abr_results.h5', 'r') as f:
    abr_response = f['abr_response'][:]
    lags = f['lags'][:]
    subject_id = f.attrs['subject_id']
    peak_amplitude = f.attrs['peak_amplitude_uv']
    rms_amplitude = f.attrs['rms_amplitude_uv']
    
print(f"Subject: {subject_id}")
print(f"Peak amplitude: {peak_amplitude:.3f} μV")
print(f"RMS amplitude: {rms_amplitude:.3f} μV")
```

## Troubleshooting

### Common Issues

1. **File Not Found Errors**
   - Ensure EEG file path is correct
   - Check that click stimulus files exist in the specified directory
   - Verify file permissions

2. **Parameter Validation Errors**
   - Ensure all frequency parameters are positive
   - Check that high-pass cutoff is below Nyquist frequency
   - Verify trial length and click rate are positive

3. **Memory Issues**
   - Reduce `n_epoch_click` for large datasets
   - Use shorter `t_click` values
   - Process data in smaller batches

4. **Dependency Issues**
   - Install expyfun from GitHub: `pip install git+https://github.com/labsn/expyfun`
   - Install pyglet with version constraint: `pip install "pyglet<1.6"`
   - Install PyOpenGL: `pip install PyOpenGL`

5. **HDF5 File Access**
   - Ensure h5py is installed: `pip install h5py`
   - Use appropriate file permissions for HDF5 files
   - Check HDF5 file integrity with h5py validation tools

### Performance Tips

- Use higher `eeg_fs` for better temporal resolution
- Increase `n_epoch_click` for better signal-to-noise ratio
- Adjust `eeg_f_hp` based on your specific requirements
- Use `plot_results=True` for visual verification
- HDF5 files provide efficient storage for large datasets
- Use HDF5 format for batch processing and data sharing

## Citation

If you use this module in your research, please cite the original work and include a reference to this implementation.

## License

This module is based on the original notebook by tshan@urmc-sh.rochester.edu and is provided for research purposes. 