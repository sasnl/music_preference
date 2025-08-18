# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a music preference study codebase analyzing how neural responses to music correlate with subjective preference ratings. The repository implements EEG analysis for both Auditory Brainstem Response (ABR) and cortical processing of music stimuli.

### Core Analysis Types
1. **Click ABR Analysis**: Cross-correlation analysis for ABR derivation from click stimuli
2. **Music ABR Analysis**: Deconvolution-based continuous ABR analysis using ANM regressors
3. **Music Preference ABR Analysis**: Comparison of ABR responses between preferred and non-preferred music
4. **Cortical TRF Analysis**: Temporal Response Function modeling of cortical responses
5. **Inter-Subject Correlation (ISC)**: Measuring shared neural responses across participants

## Environment Setup

### Main EEG Analysis Environment
```bash
pip install -r requirements_eeg.txt
```
Core dependencies: `mne>=1.5.0`, `numpy>=1.21.0`, `matplotlib>=3.5.0`, `pandas>=1.3.0`, `scipy>=1.7.0`, `plotly>=5.0.0`

### Music Preprocessing Environment
```bash
conda env create -f code/stimulus_presentation/env.yml
conda activate music_preproc
```

### ABR Analysis Environment
```bash
cd code/analysis/derive_click_ABR
conda env create -f environment.yml
conda activate abr_analysis
```

## Common Commands

### Music Preprocessing
```bash
python code/stimulus_presentation/music_batch_preproc.py --input_dir music_stim/original --output_dir music_stim/preprocesed --no_trim
```

### ABR Analysis
**Basic click ABR derivation:**
```bash
cd code/analysis/derive_click_ABR
python derive_click_ABR.py /path/to/eeg_file.vhdr ./output_dir
```

**With custom parameters:**
```bash
python derive_click_ABR.py /path/to/eeg_file.vhdr ./output_dir \
    --eeg_fs 25000 --eeg_f_hp 2.0 --t_click 30 --plot
```

### Music ANM Regressor Generation
**Local processing:**
```bash
python code/analysis/derive_music_ABR/generate_music_anm_regressors.py
```

**Sherlock cluster processing:**
```bash
cd code/analysis/derive_music_ABR/sherlock_setup
sbatch sherlock_job.slurm
```

### Music Preference ABR Analysis
**Single subject analysis:**
```bash
python code/analysis/derive_music_ABR/derive_music_preference_ABR.py pilot_2
```

**For all subjects:**
```bash
for subject in pilot_1 pilot_2 pilot_3 pilot_4 pilot_5; do
    python code/analysis/derive_music_ABR/derive_music_preference_ABR.py $subject
done
```

### EEG Visualization
```bash
python eeg_visualization.py
python interactive_eeg_visualization.py
```

## Key Architecture Components

### Data Processing Pipeline
1. **Music Stimuli**: Original MP3 → Preprocessed WAV (mono, normalized, 48kHz)
2. **EEG Recording**: BrainVision format (.vhdr/.vmrk/.eeg files) at 10kHz/25kHz
3. **ANM Processing**: Auditory Nerve Model regressors for deconvolution analysis
4. **Output Storage**: HDF5 format with rich metadata

### Code Organization
- `code/stimulus_presentation/`: Experiment presentation and music preprocessing
- `code/analysis/derive_click_ABR/`: Modular ABR analysis with cross-correlation
- `code/analysis/derive_music_ABR/`: Continuous music ABR analysis using deconvolution
- `music_stim/`: Original, preprocessed, and ANM regressor files
- `data/`: EEG recordings and analysis outputs
- `click_stim/`: Click stimulus files (click000.wav, click001.wav, etc.)

### Key Data Formats
- **EEG Data**: BrainVision (.vhdr) with ABR channels (Plus_R, Minus_R, Plus_L, Minus_L)
- **Music Files**: 48kHz WAV files after preprocessing
- **Analysis Output**: HDF5 files with arrays and comprehensive metadata
- **ANM Regressors**: HDF5 files containing auditory nerve model responses

### Sherlock Cluster Integration
The repository includes complete Sherlock cluster setup for large-scale processing:
- Environment management with conda
- SLURM job scripts with memory optimization
- Automated file transfer and result collection
- 128GB memory allocation for ANM processing

### Analysis Parameters
**ABR Analysis:**
- EEG sampling: 10kHz/25kHz
- High-pass filter: 1-2Hz
- Response window: -200ms to +600ms
- Click rate: 40-50Hz

**Music Analysis:**
- Stimulus sampling: 48kHz
- Response window: 0-600ms for early responses
- Cross-validation: 80% train, 20% test
- TRF modeling with regularization

## File Naming Conventions
- Music files: `{participant}-{song}_proc.wav`
- EEG files: `{subject_id}.vhdr` or `{subject_id}-trial{N}_{song_id}_proc_originalptp-{N}.fif`
- Output files: `{subject_id}_abr_results.h5` or `{subject_id}_music_preference_ABR.hdf5`
- ANM files: `single_{song}_proc_anm.hdf5`

### Music Preference Study Specifics
- **Subject naming**: `pilot_1`, `pilot_2`, `pilot_3`, `pilot_4`, `pilot_5`
- **Song preference**: Pilot N prefers songs N-1, N-2, N-3 (e.g., pilot_2 prefers 2-1, 2-2, 2-3)
- **EEG trial files**: `{subject}-trial{N}_{song_id}_proc_originalptp-{N}.fif`
- **Output structure**: HDF5 files with `w_preferred`, `w_nonpreferred`, `abr_preferred`, `abr_nonpreferred`, `lags`

## Important Notes
- Always use absolute paths for file operations
- ANM processing requires the `cochlea` package (may need special installation)
- HDF5 files contain both data arrays and comprehensive metadata
- Behavioral data is collected via GUI during stimulus presentation
- Latin square randomization ensures balanced stimulus presentation order
- For music preference analysis, each song has exactly one corresponding EEG trial file
- ABR time window: -200ms to +600ms with 2.75ms shift for ANM regressors
- Bandpass filtering: 1-1000Hz for final ABR results

# Project Guidelines for Claude
Do not include attribution to Claude in the commit message or co-author attribution. Ever.