# Music Preference Study

## Participants Onboarding

### [Musicianship Questionnair](https://github.com/sasnl/music_preference/blob/main/MusicianshipQuestionnaire.md)

### Selection of [Favorite Songs](https://docs.google.com/spreadsheets/d/1YDDWKmQ6O3HpwoQeA3kcLaOXuhWvbGlxDFgny0Mv1zk/edit?gid=0#gid=0)

total length of music = 68:02

### Music Stimuli Preprocessing
- **Music Stimuli Preprocessing Pipeline:**
  1. **Convert MP3 to WAV**
  2. **Convert Stereo to Mono**
  3. **Apply Low-Pass Envelope Division (Flatten)**
  4. **Normalize RMS (window size: 0.01)**
  5. **Resample to 48 kHz**
- **Script**: [`/code/stimulus_presentation/music_batch_preproc.py`](https://github.com/sasnl/music_preference/blob/main/code/stimulus_presentation/music_batch_preproc.py)

**To run the script**
1. install environment according to `/code/stimulus_presentation/env.yml`
```
conda env create -f env.yml
conda activate music_preproc
```
2. run this line in terminal:
```python
python code/stimulus_presentation/music_batch_preproc.py --input_dir music_stim/original --output_dir music_stim/preprocesed --no_trim
```
## Experiment Procedure
### 1. 5-Minute Click Trains: [`/click_stim`](https://github.com/sasnl/music_preference/tree/main/click_stim)
### 2. Latin Square Randomized Song Presentation
- code to generate randomized song order: `/code/stimulus_presentation/generate_music_orders.py`. Generated order file: `/code/stimulus_presentation/music_presentation_orders.csv`
- Participants will passively listening to the songs, while EEG recording with both ABR+Cortical system
- EEG recording at 10k Hz / 25k Hz
- Stimlus presentation at 48k Hz
run the script on stimlus computer: [`/code/stimulus_presentation/music_preference_presentation.py`](https://github.com/sasnl/music_preference/blob/main/code/stimulus_presentation/music_preference_presentation.py)
### 3. Behavioral Questions After Each Song
- in stimlus presentation script, questions pop up when a song ends
#### Preference for the Song
 How much did you like or enjoy the song overall?
 (1 = Not at all, 9 = Very much)
#### Pleasantness
 How pleasant or unpleasant did you find the song?
 (1 = Extremely unpleasant, 9 = Extremely pleasant)
#### Valence/Arousal
 How emotionally intense or stimulating was the song for you?
 (1 = Not intense or stimulating at all, 9 = Extremely intense or stimulating)
#### Musical Chills
 To what extent did you feel chills, goosebumps, or a strong emotional reaction while listening to the song?
 (1 = Not at all, 9 = Very strongly)

# Analysis
## Subcortical Responses

### **Click ABR**
Script: [`code/analysis/derive_click_ABR/derive_click_ABR_single_subject.py`](code/analysis/derive_click_ABR/derive_click_ABR_single_subject.py)

**Pipeline Steps:**
1. **Click stimulus processing**: Load click stimuli (click000.wav, click001.wav, etc.) and create pulse trains
   - Find click onset times when stimulus amplitude transitions from 0 to 1
   - Convert to EEG sample indices and create binary pulse train arrays
   - Handle multiple click files per subject (typically 5 trials × 60 seconds each)
2. **EEG preprocessing**: Load .fif files with comprehensive filtering pipeline
   - Create differential ABR channels: Plus_R - Minus_R, Plus_L - Minus_L
   - Apply 1 Hz high-pass filter for baseline removal
   - Apply notch filters at 60, 180, 300, 420 Hz (power line harmonics)
   - Average across left/right channels for single ABR trace per trial
3. **Cross-correlation analysis**: FFT-based frequency domain cross-correlation
   - Compute cross-correlation: `cc = ifft(EEG_fft × conj(Click_fft))`
   - Average cross-correlation across all click trials
   - Normalize by click rate (40 Hz) and trial duration (60 seconds)
4. **ABR extraction**: Extract -200ms to +600ms response window
   - Apply final bandpass filter (1-1000 Hz) for clean ABR waveform
   - Generate comprehensive visualization with 6-panel plots
   - Perform automatic peak detection in Wave I-V region (1-8 ms)

**Key Features:**
- **Robust preprocessing**: Comprehensive filtering pipeline with notch filtering
- **Multi-trial averaging**: Improved SNR through trial-wise cross-correlation averaging
- **Automatic peak detection**: Wave I-V analysis with latency and amplitude extraction
- **Comprehensive visualization**: Individual trials, averaged responses, zoomed regions
- **Summary statistics**: Peak latency, amplitude, RMS values, and processing metadata

**Usage:**
```bash
# Single subject analysis
python code/analysis/derive_click_ABR/derive_click_ABR_single_subject.py pilot_2

# Expected files:
# - Click stimuli: click_stim/click000.wav, click001.wav, etc.
# - EEG data: data/pilot_2/pilot_2_click_trial*.fif
```

**Requirements:**
- Click stimulus files: `click_stim/click{000-004}.wav`
- EEG trial files: `data/{subject}/{subject}_click_trial*.fif`
- ABR channels: Plus_R, Minus_R, Plus_L, Minus_L

**Output:**
- **HDF5 file**: `output/{subject}_click_ABR.hdf5` containing:
  - `abr_response`: Raw cross-correlation ABR
  - `abr_response_filtered`: Bandpass filtered ABR (1-1000 Hz)
  - `lags`: Time axis (-200 to +600 ms)
  - `cc_trials`: Individual trial cross-correlations
  - Processing metadata and parameters
- **Visualization**: `{subject}_click_ABR.png` with 6-panel analysis plots
- **Statistics**: Peak detection, latency analysis, and amplitude measures

#### **Modular Click ABR Implementation**
- **Modular Python implementation**: [`code/analysis/derive_click_ABR/derive_click_ABR.py`](https://github.com/sasnl/music_preference/blob/main/code/analysis/derive_click_ABR/derive_click_ABR.py)
- **Key Features**:
  - Configurable parameters (EEG sampling frequency, filtering, time ranges)
  - Command-line interface and Python module usage
  - HDF5 data storage with rich metadata
  - Optional plot generation
  - Batch processing support
  - Comprehensive error handling and logging
- **Input**: BrainVision (.vhdr) files with ABR channels (Plus_R, Minus_R, Plus_L, Minus_L)
- **Output**: ABR response arrays, time lags, HDF5 files, plots, and summary statistics
- **Documentation**: See [`code/analysis/derive_click_ABR/README_ABR.md`](https://github.com/sasnl/music_preference/blob/main/code/analysis/derive_click_ABR/README_ABR.md) for detailed usage

 ### Continuous Music ABR Analysis

#### **ANM Regressor Generation**
Script: [`code/analysis/derive_music_ABR/generate_music_anm_regressors.py`](code/analysis/derive_music_ABR/generate_music_anm_regressors.py)

**Sherlock Cluster Version**: [`code/analysis/derive_music_ABR/sherlock_setup/generate_music_anm_regressors_sherlock.py`](code/analysis/derive_music_ABR/sherlock_setup/generate_music_anm_regressors_sherlock.py)

**Pipeline Steps:**
1. **Audio Loading**: Process all preprocessed music files (`*_proc.wav`)
2. **ANM Generation**: Use Zilany2014 auditory nerve model via cochlea package
3. **Dual Polarity**: Generate both positive and negative polarity ANM responses
4. **Frequency Mapping**: Multiple characteristic frequencies (CF) from 125 Hz to 16 kHz  
5. **Resampling**: Downsample from 100 kHz model rate to 25 kHz EEG rate
6. **HDF5 Storage**: Save individual ANM files per song (`single_{song}_proc_anm.hdf5`)

**Key Features:**
- **High-performance computing**: Optimized for Stanford Sherlock cluster with 32 cores, 128GB RAM
- **Memory management**: Intelligent memory monitoring and garbage collection
- **Parallel processing**: Joblib parallelization across characteristic frequencies
- **Robust error handling**: Comprehensive logging and checkpoint recovery
- **Batch processing**: Processes all 15 music files automatically

**Local Usage:**
```bash
# Basic local processing
python code/analysis/derive_music_ABR/generate_music_anm_regressors.py

# Check dependencies
python -c "import cochlea, numpy, scipy, mne; print('Dependencies OK')"
```

**Sherlock Cluster Usage:**
```bash
# Submit job to Sherlock
cd code/analysis/derive_music_ABR/sherlock_setup
sbatch sherlock_job.slurm

# Monitor job status
squeue -u $USER

# Check outputs
ls -la /scratch/users/$USER/music_preference/data/
```

**Dependencies:**
- `cochlea` package: `pip install git+https://github.com/mrkrd/cochlea.git`
- `ic_cn2018.py`: Auditory nerve model implementation
- Standard packages: `numpy`, `scipy`, `mne`, `joblib`, `psutil`

**Output Structure:**
- **Individual files**: `music_stim/music_anm/single_{song}_proc_anm.hdf5`
- **Contents per file**: 
  - `key_x_in_pos`: Positive polarity ANM response
  - `key_x_in_neg`: Negative polarity ANM response  
  - `key_fs`: Sampling rate (25 kHz)
  - Metadata: Processing parameters and file information

#### **Music ABR Derivation**
Script: [`code/analysis/derive_music_ABR/derive_music_preference_ABR.py`](code/analysis/derive_music_ABR/derive_music_preference_ABR.py)

**Pipeline Steps:**
1. **Preference categorization**: Auto-identify preferred songs based on subject naming convention
   - Subject `pilot_N` prefers songs `N-1`, `N-2`, `N-3` (e.g., pilot_2 prefers 2-1, 2-2, 2-3)
   - Remaining 12 songs classified as non-preferred
2. **EEG preprocessing**: Load ABR channels (Plus_R, Minus_R, Plus_L, Minus_L)
   - Create differential channels (Plus - Minus) for left and right ears
   - Apply 1 Hz high-pass filter and 60/180/300/420 Hz notch filters
   - Average across left/right channels for single ABR trace
3. **ANM regressor matching**: Load corresponding ANM regressors for each song
   - Use both positive and negative polarity ANM responses
   - Match regressor length to EEG trial duration
4. **Deconvolution analysis**: FFT-based temporal response function (TRF) estimation
   - Compute TRF using: `H(ω) = conj(X(ω)) * Y(ω) / |X(ω)|²`
   - Average positive and negative polarity responses
   - Extract -200 to +600 ms ABR window with 2.75 ms ANM shift compensation
5. **Preference comparison**: Average ABR across preferred vs non-preferred trials
   - Final bandpass filter: 1-1000 Hz for clean ABR waveforms
   - Statistical comparison of preference-based responses

**Key Features:**
- **Automated processing**: Single command per subject with preference auto-detection
- **Robust preprocessing**: Comprehensive filtering and artifact removal
- **Dual-polarity deconvolution**: Improved SNR using both ANM polarities
- **Time window optimization**: 0-600 ms window captures early brainstem responses
- **Comparative analysis**: Direct preferred vs non-preferred ABR comparison

**Usage:**
```bash
# Single subject analysis
python code/analysis/derive_music_ABR/derive_music_preference_ABR.py pilot_2

# Batch processing for all subjects  
for subject in pilot_1 pilot_2 pilot_3 pilot_4 pilot_5; do
    python code/analysis/derive_music_ABR/derive_music_preference_ABR.py $subject
done
```

**Requirements:**
- Preprocessed EEG trial files: `data/{subject}/{subject}-trial*_proc_originalptp*.fif`
- ANM regressors: `music_stim/music_anm/single_{song}_proc_anm.hdf5`
- ABR channels: Plus_R, Minus_R, Plus_L, Minus_L

**Output:**
- **HDF5 file**: `output/{subject}_music_preference_ABR.hdf5` containing:
  - `w_preferred`, `w_nonpreferred`: Raw TRF responses
  - `abr_preferred`, `abr_nonpreferred`: Filtered ABR waveforms (1-1000 Hz)
  - `lags`: Time axis (-200 to +600 ms)
  - Metadata: Song lists, trial counts, processing parameters
- **Visualization**: `{subject}_music_preference_ABR.png` with full and zoomed ABR comparison plots

## Cortical Responses
### EEG preprocessing

The cortical EEG preprocessing pipeline consists of two main steps: basic preprocessing and artifact removal using Independent Component Analysis (ICA).

#### **Step 1: Basic EEG Preprocessing**
Script: [`code/analysis/eeg_preprocessing/eeg_preprocessing_cortical.py`](code/analysis/eeg_preprocessing/eeg_preprocessing_cortical.py)

**Pipeline Steps:**
1. **Load EEG data**: BrainVision format (.vhdr/.vmrk/.eeg files)
2. **Channel selection**: Pick cortical channels (exclude ABR channels: Plus_R, Minus_R, Plus_L, Minus_L, Audio)
3. **Reference restoration**: Add back Cz reference channel (set to zero since all channels were referenced to Cz during recording)
4. **Bandpass filtering**: 0.5-30 Hz, zero-phase bidirectional FIR filter
5. **Re-referencing**: Mean of TP9+TP10 (mastoid reference)
6. **Downsampling**: 128 Hz target sampling rate

**Usage:**
```bash
# Basic usage
python code/analysis/eeg_preprocessing/eeg_preprocessing_cortical.py pilot_1

# Custom filter settings
python code/analysis/eeg_preprocessing/eeg_preprocessing_cortical.py pilot_1 --l_freq 0.5 --h_freq 30.0

# Custom sampling rate and reference channels
python code/analysis/eeg_preprocessing/eeg_preprocessing_cortical.py pilot_1 --target_sfreq 128 --ref_channels Cz
```

**Output**: Preprocessed .fif files in `data/preprocessed/{subject_id}/`

#### **Step 2: ICA Artifact Removal**

##### **Single File ICA**: [`code/analysis/eeg_preprocessing/eeg_ica_artifact_removal.py`](code/analysis/eeg_preprocessing/eeg_ica_artifact_removal.py)

**Pipeline Steps:**
1. **Load preprocessed data**: From Step 1 output
2. **ICA preparation**: Apply 1 Hz high-pass filter for ICA stability
3. **ICA fitting**: FastICA or Infomax algorithm with 25 components (default)
4. **Interactive component selection**: Real-time click-to-select interface with live time series preview
5. **Artifact removal**: Remove selected components from original (unfiltered) data
6. **Save cleaned data**: ICA-cleaned .fif files

**Interactive Features:**
- Click components to select/deselect for removal
- Live time series preview of selected components
- Keyboard shortcuts: 'h' (help), 'r' (reset), 'q' (quit)
- Automatic artifact detection fallback

**Usage:**
```bash
# Interactive ICA with real-time selection
python code/analysis/eeg_preprocessing/eeg_ica_artifact_removal.py pilot_1_cortical_preprocessed.fif

# Custom ICA parameters
python code/analysis/eeg_preprocessing/eeg_ica_artifact_removal.py pilot_1_cortical_preprocessed.fif --n_components 25 --method fastica

# Automatic detection only (no interaction)
python code/analysis/eeg_preprocessing/eeg_ica_artifact_removal.py pilot_1_cortical_preprocessed.fif --no_interactive
```

##### **Batch ICA**: [`code/analysis/eeg_preprocessing/batch_ica_artifact_removal.py`](code/analysis/eeg_preprocessing/batch_ica_artifact_removal.py)

For consistent artifact removal across multiple trial files from the same subject:

**Pipeline Steps:**
1. **Fit ICA on continuous data**: Use full-length cortical recording for robust ICA decomposition
2. **Interactive component selection**: Select artifact components once
3. **Apply to all trials**: Automatically apply same ICA removal to all trial files
4. **Batch processing**: Process multiple trial files with consistent artifact removal

**Usage:**
```bash
# Batch ICA processing
python code/analysis/eeg_preprocessing/batch_ica_artifact_removal.py \
    data/preprocessed/pilot_1/pilot_1_cortical_preprocessed.fif \
    data/preprocessed/pilot_1/preprocessed_trials/

# Custom output suffix
python code/analysis/eeg_preprocessing/batch_ica_artifact_removal.py \
    data/preprocessed/pilot_1/pilot_1_cortical_preprocessed.fif \
    data/preprocessed/pilot_1/preprocessed_trials/ \
    --output_suffix "_ica_cleaned"
```

**Output**: ICA-cleaned trial files in `output_batch_ica/` with consistent artifact removal patterns

#### **Preprocessing Summary**
- **Input**: Raw BrainVision EEG files (25 kHz sampling)
- **Output**: Clean, preprocessed .fif files at 128 Hz
- **Key Features**: Cortical-specific channel selection, mastoid re-referencing, interactive ICA artifact removal
- **Applications**: Ready for TRF analysis, ISC analysis, and cortical response modeling

 ### Music Cortical TRF analysis
 - Derive music TRF using ANM regressor as in Shan et al. (2024)
 - Use conventional TRF methods to model the neural response:
  - Train/test split: 80% training, 20% testing.
  - Perform cross-validation.
  - Outcome: Average R² across subjects and models.
- Preprocessing and setup:
  - Normalize data.
  - Determine stimulus position.
  - Extract TRF weights and P2 component.
- Link neural data with behavioral responses.
  - Compile and align TRF results with survey/questionnaire data.
- Analyze topography to visualize spatial differences in TRF components (e.g., P2 differences across preference).

 ### Music ISC analysis
 - Step 1: Perform RCA (Reliable Components Analysis) to obtain spatial filters.
  - Identify the most reliable component (typically RC1) across all music conditions.
  - Check for differences in RCA components between preferred and non-preferred songs.
- Step 2: Apply the RCA weights to the EEG data to get spatially filtered signals.
- Step 3: Compute Inter-Subject Correlation (ISC) using the filtered signals.
  - Run ISC on the RCA-derived component (e.g., RC1) to measure shared neural responses across participants.
- Generate final ISC results as a matrix heat map for visualization.
- Additional Notes:
  - Perform ISC for each song individually.
  - Optionally, concatenate multiple songs (e.g., Songs 1, 2, and 3) per person to increase data length.
  - Concatenation can be done for short excerpts or full-length tracks—segment length is flexible.

 ### Correlate behavioral responses
- Compare ISC EEG amplitude to participants’ preference ratings:
- hypothesis: Higher P2 amplitude for more preferred songs; P1 is expected to remain similar.
- Instead of waveform amplitude correlations, use TRF model prediction accuracy:
  - Train on 8 minutes, test on 2 minutes.
  - Use R² value as the prediction accuracy metric (R will be higher, but R² is more informative).


