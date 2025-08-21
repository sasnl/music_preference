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
 ### [Click ABR Analysis](https://github.com/sasnl/music_preference/tree/main/code/analysis/derive_click_ABR)
 - **Cross-correlation analysis** for ABR derivation from click stimuli
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
 - Derive ANM regressor: ``
 - Use a 0–600 ms window to capture early responses.
 - Derive music ABR using **deconvolution** as in [Shan et al. (2024)](https://www.nature.com/articles/s41598-023-50438-0)
 - Future analyses can explore acoustic feature differences across stimuli.

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


