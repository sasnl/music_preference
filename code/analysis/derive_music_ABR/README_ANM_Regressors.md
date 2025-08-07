# ANM Regressor Generation Script

This script generates Auditory Nerve Model (ANM) regressors for music files using the exact cochlea package implementation.

## Overview

The script processes music files from the `music_stim/preprocesed/` directory and generates ANM regressors for both positive and negative polarities. It uses the Zilany2014 auditory nerve fiber model to create realistic auditory nerve responses.

## Features

- **Exact Cochlea Implementation**: Uses the original cochlea package with Zilany2014 model
- **High Sampling Rate**: Configurable EEG sampling frequency (default: 25 kHz)
- **Batch Processing**: Processes all music files automatically
- **HDF5 Output**: Saves results in HDF5 format with metadata
- **Progress Tracking**: Comprehensive logging and progress reporting

## Installation

### Prerequisites

The script requires the exact cochlea package, which has specific dependencies. Follow these steps carefully:

### Step 1: Install Compatible Dependencies

```bash
# Install older Cython (required for cochlea compilation)
pip install "Cython<3.0"

# Install compatible NumPy (required for cochlea compilation)
pip install "numpy<2.0"
```

### Step 2: Install Cochlea Package

```bash
# Install cochlea from GitHub repository
pip install git+https://github.com/mrkrd/cochlea.git
```

### Step 3: Verify Other Dependencies

```bash
# Install other required packages if not already available
pip install mne scipy joblib expyfun h5py
```

## Troubleshooting Installation

If you encounter compilation errors during cochlea installation:

1. **Cython Issues**: Ensure you have Cython < 3.0
   ```bash
   pip uninstall Cython
   pip install "Cython<3.0"
   ```

2. **NumPy Compatibility**: Use NumPy < 2.0
   ```bash
   pip uninstall numpy
   pip install "numpy<2.0"
   ```

3. **C Compiler Issues**: On macOS, ensure Xcode command line tools are installed:
   ```bash
   xcode-select --install
   ```

4. **Alternative Installation**: If direct installation fails, try:
   ```bash
   # Clone and install manually
   git clone https://github.com/mrkrd/cochlea.git
   cd cochlea
   python setup.py install
   ```

## Usage

### Basic Usage

```bash
cd code/analysis
python generate_music_anm_regressors.py
```

### Output

The script generates:
- `data/music_anm_regressors.hdf5`: Main output file containing:
  - `x_in_pos`: Positive polarity ANM regressors
  - `x_in_neg`: Negative polarity ANM regressors
  - `fs`: Sampling frequency (25000 Hz)
  - `file_info`: Metadata for each processed file
  - `processing_params`: Processing parameters

### Output Structure

```python
import h5py

# Load results
with h5py.File('data/music_anm_regressors.hdf5', 'r') as f:
    x_in_pos = f['x_in_pos'][:]  # Shape: (n_files, max_length)
    x_in_neg = f['x_in_neg'][:]  # Shape: (n_files, max_length)
    fs = f['fs'][()]             # Sampling frequency: 25000 Hz
    file_info = f['file_info']   # File metadata
```

## Parameters

- **EEG Sampling Rate**: 25,000 Hz (as requested)
- **Stimulus Presentation Level**: 65 dB SPL
- **Center Frequency Range**: 125 Hz to 16 kHz
- **Frequency Resolution**: 1/6 octave steps
- **Species**: Human auditory model

## Input File Structure

The script expects preprocessed music files in this structure:
```
music_stim/preprocesed/
├── 1/
│   ├── 1-1_proc.wav
│   ├── 1-2_proc.wav
│   └── 1-3_proc.wav
├── 2/
│   ├── 2-1_proc.wav
│   ├── 2-2_proc.wav
│   └── 2-3_proc.wav
...
```

## Algorithm Details

1. **Audio Loading**: Loads preprocessed WAV files
2. **Upsampling**: Resamples to 100 kHz for cochlea model
3. **Level Scaling**: Converts to Pascal units based on presentation level
4. **Cochlea Processing**: Applies Zilany2014 auditory nerve fiber model
5. **Frequency Processing**: Processes multiple center frequencies (125 Hz - 16 kHz)
6. **Downsampling**: Resamples to target EEG sampling rate (25 kHz)
7. **ANM Generation**: Sums across frequency channels with scaling factor
8. **Polarity Processing**: Generates both positive and negative polarity responses

## Technical Notes

- Uses parallel processing for faster computation
- Handles variable-length audio files with zero-padding
- Implements proper temporal shifts and scaling factors
- Compatible with existing ABR analysis pipelines

## Dependencies

- `numpy`: Numerical computing
- `scipy`: Signal processing
- `mne`: Neurophysiological data processing
- `joblib`: Parallel processing
- `cochlea`: Auditory nerve fiber models (critical dependency)
- `expyfun`: Audio I/O and HDF5 utilities
- `h5py`: HDF5 file handling

## References

- Zilany, M. S., Bruce, I. C., & Carney, L. H. (2014). Updated parameters and expanded simulation options for a model of the auditory periphery. The Journal of the Acoustical Society of America, 135(1), 283-286.
- Rudnicki, M., Schoppe, O., Isik, M., Völk, F., & Hemmert, W. (2015). Modeling auditory coding: from sound to spikes. Cell and tissue research, 361(1), 159-175.