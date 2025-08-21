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

### Project Folder Structure
```
music_preference/
├── CLAUDE.md                              # Project guidance for Claude Code
├── README.md                              # Main project documentation
├── MusicianshipQuestionnaire.md           # Musicianship assessment questionnaire
├── reorganize_ratings_csv_to_json.py      # Behavioral data reorganization script
│
├── click_stim/                            # Click stimulus files
│   ├── click000.wav                       # Click train files for ABR analysis
│   ├── click001.wav
│   ├── click002.wav
│   ├── click003.wav
│   └── click004.wav
│
├── code/                                  # Analysis and presentation code
│   ├── analysis/                          # EEG and behavioral analysis scripts
│   │   ├── behavioral_data/               # Behavioral data processing
│   │   │   ├── behavioral_data_extraction.py     # Extract behavioral ratings
│   │   │   └── reorganize_ratings_csv_to_json.py # CSV to JSON converter
│   │   │
│   │   ├── derive_click_ABR/              # Click ABR analysis pipeline
│   │   │   ├── derive_click_ABR_single_subject.py # Single-subject click ABR
│   │   │   ├── environment.yml            # Conda environment specification
│   │   │   ├── requirements.txt           # Python package requirements
│   │   │   └── README_ABR.md             # ABR analysis documentation
│   │   │
│   │   ├── derive_music_ABR/              # Music ABR analysis pipeline
│   │   │   ├── derive_music_preference_ABR.py     # Preference-based ABR analysis
│   │   │   ├── ic_cn2018.py              # Cochlear nucleus model implementation
│   │   │   ├── README_ANM_Regressors.md  # ANM regressor documentation
│   │   │   └── sherlock_setup/           # Stanford Sherlock cluster setup
│   │   │       ├── QUICK_START_SHERLOCK.md       # Sherlock usage guide
│   │   │       ├── environment.yml       # Cluster environment specification
│   │   │       ├── generate_music_anm_regressors_sherlock.py # Cluster ANM script
│   │   │       ├── sherlock_job.slurm    # SLURM job submission script
│   │   │       ├── test_sherlock_job.slurm       # Test job script
│   │   │       ├── test_single_file.py   # Single file test script
│   │   │       ├── setup_conda.sh        # Conda setup script
│   │   │       ├── upload_to_sherlock.sh # Data upload script
│   │   │       └── download_from_sherlock.sh     # Results download script
│   │   │
│   │   ├── eeg_preprocessing/             # EEG preprocessing pipelines
│   │   │   ├── eeg_preprocessing_cortical.py     # Cortical EEG preprocessing
│   │   │   ├── eeg_ica_artifact_removal.py       # ICA artifact removal
│   │   │   ├── batch_ica_artifact_removal.py     # Batch ICA processing
│   │   │   └── split_eeg.ipynb           # EEG trial splitting notebook
│   │   │
│   │   └── feature_extraction/           # Audio feature extraction
│   │       └── extract_music_features.py # Comprehensive music feature extraction
│   │
│   └── stimulus_presentation/             # Experiment presentation code
│       ├── music_preference_presentation.py      # Main experiment script
│       ├── music_preference_presentation_improved.py # Enhanced version
│       ├── music_preference_presentation_preload_audio.py # Preloaded audio version
│       ├── music_presentation_with_randomization.py # Randomized presentation
│       ├── generate_music_orders.py      # Presentation order generation
│       ├── music_batch_preproc.py        # Batch audio preprocessing
│       ├── music_preference.yaml         # Experiment configuration
│       ├── env.yml                       # Presentation environment
│       ├── music_presentation_orders.csv # Generated presentation orders
│       └── music_presentation_report.txt # Presentation summary report
│
├── data/                                  # EEG recordings and processed data
│   ├── raw/                              # Raw EEG recordings
│   │   └── pilot_1/                      # Subject-specific raw data
│   │       ├── pilot_1.vhdr             # BrainVision header file
│   │       ├── pilot_1.vmrk             # BrainVision marker file
│   │       ├── pilot_1.eeg              # BrainVision data file
│   │       └── *.xlsx, *.log, *.tab     # Behavioral and log files
│   │
│   ├── preprocessed/                     # Preprocessed EEG data
│   │   ├── pilot_1/                     # Subject-specific preprocessed data
│   │   │   ├── pilot_1_cortical_preprocessed.{fif,vhdr,vmrk,eeg} # Full preprocessing
│   │   │   ├── pilot_1_cortical_preprocessing.png # Preprocessing visualization
│   │   │   └── preprocessed_trials/     # Individual trial files
│   │   │       ├── pilot_1-trial{N}_{song_id}_proc_originalptp-{N}_cortical_preproc.{fif,vhdr,vmrk,eeg}
│   │   │       ├── pilot_1_click_trial{N}_cortical_preproc.{fif,vhdr,vmrk,eeg}
│   │   │       └── pilot_1_trial_metadata.csv
│   │   ├── pilot_2/ ... pilot_5/        # Additional subjects
│   │
│   ├── ica_cleaned/                      # ICA artifact-removed data
│   │   ├── pilot_1/                     # Subject-specific cleaned data
│   │   │   ├── pilot_1-trial{N}_{song_id}_proc_originalptp-{N}_cortical_preproc_ica_cleaned.fif
│   │   │   ├── pilot_1_click_trial{N}_cortical_preproc_ica_cleaned.fif
│   │   │   ├── pilot_1_batch_ica.fif    # ICA decomposition
│   │   │   └── pilot_1_batch_ica_summary.txt # ICA summary
│   │   ├── pilot_2/ ... pilot_5/        # Additional subjects
│   │
│   ├── Organized Behavioral Folder - Ratings.csv # Original behavioral ratings
│   └── reorganized_ratings.json         # Restructured behavioral data
│
├── music_stim/                           # Music stimuli and features
│   ├── original/                         # Original MP3 files
│   │   ├── 1/                           # Artist 1 songs
│   │   │   ├── 1-1.mp3                  # Song 1-1
│   │   │   ├── 1-2.mp3                  # Song 1-2
│   │   │   └── 1-3.mp3                  # Song 1-3
│   │   ├── 2/ ... 5/                    # Artists 2-5 songs
│   │
│   ├── preprocesed/                      # Preprocessed WAV files
│   │   ├── 1/                           # Artist 1 processed songs
│   │   │   ├── 1-1_proc.wav             # Normalized, mono, 48kHz
│   │   │   ├── 1-2_proc.wav
│   │   │   └── 1-3_proc.wav
│   │   ├── 2/ ... 5/                    # Artists 2-5 processed songs
│   │   └── process_log.txt              # Preprocessing log
│   │
│   ├── music_anm/                        # Auditory Nerve Model regressors
│   │   ├── single_1-1_proc_anm.hdf5     # ANM regressors for deconvolution
│   │   ├── single_1-2_proc_anm.hdf5
│   │   └── ... (all 15 songs)
│   │
│   └── music_features/                   # Extracted audio features
│       ├── 1-1_proc_features.csv        # Feature CSV files
│       ├── 1-1_proc_features.npz        # Feature NumPy arrays
│       ├── 1-1_proc_features.png        # Feature visualizations
│       └── ... (all 15 songs × 3 formats)
│
└── output/                               # Analysis results and visualizations
    ├── pilot_2_click_ABR.{hdf5,png}     # Click ABR results
    ├── pilot_2_music_preference_ABR.{hdf5,png} # Music preference ABR results
    ├── single_1-1_proc_anm_plot.png     # ANM regressor visualizations
    ├── hdf5_summary.csv                 # Analysis output summary
    ├── interactive_*.html               # Interactive visualizations
    ├── ica_*.png                        # ICA analysis plots
    └── eeg_*.png                        # EEG visualization plots
```

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

### Behavioral Data Organization
The behavioral ratings are organized in a hierarchical JSON structure for easy access:

**Original format**: CSV with rows (songs) × columns (participant-question combinations)
**New format**: `reorganized_ratings.json` with structure:
```json
{
  "preference": {
    "pilot_1": {"1-1": 9, "1-2": 8, ...},
    "pilot_2": {"1-1": 6, "1-2": 7, ...},
    ...
  },
  "pleasantness": { ... },
  "arousal": { ... },
  "chills": { ... }
}
```

**Question mapping**:
- `"preference"` (Q1): Subjective liking ratings (1-9 scale)
- `"pleasantness"` (Q2): Valence dimension ratings (1-9 scale)  
- `"arousal"` (Q3): Activation/energy dimension ratings (1-9 scale)
- `"chills"` (Q4): Aesthetic chills/frisson intensity (1-9 scale)

**Access patterns**:
```python
import json
data = json.load(open('data/reorganized_ratings.json'))
rating = data["preference"]["pilot_2"]["2-1"]  # Get pilot_2's preference for song 2-1
```

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