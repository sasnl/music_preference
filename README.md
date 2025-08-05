# Music Preference Study

## Table of Contents
- [Participants Onboarding](#participants-onboarding)
  - [Musicianship Questionnaire](#musicianship-questionnaire)
  - [Selection of Favorite Songs](#selection-of-favorite-songs)
  - [Music Stimuli Preprocessing](#music-stimuli-preprocessing)
- [Experiment Procedure](#experiment-procedure)
  - [5-Minute Click Trains](#5-minute-click-trains)
  - [Latin Square Randomized Song Presentation](#latin-square-randomized-song-presentation)
  - [Behavioral Questions After Each Song](#behavioral-questions-after-each-song)
    - [Preference for the Song](#preference-for-the-song)
    - [Pleasantness](#pleasantness)
    - [Valence/Arousal](#valencearousal)
    - [Musical Chills](#musical-chills)
- [Analysis](#analysis)
  - [Click ABR Analysis](#click-abr-analysis)
  - [Continuous Music ABR Analysis](#continuous-music-abr-analysis)

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
### 5-Minute Click Trains: [`/click_stim`](https://github.com/sasnl/music_preference/tree/main/click_stim)
### Latin Square Randomized Song Presentation
- code to generate randomized song order: `/code/stimulus_presentation/generate_music_orders.py`. Generated order file: `/code/stimulus_presentation/music_presentation_orders.csv`
- Participants will passively listening to the songs, while EEG recording with both ABR+Cortical system
- EEG recording at 10k Hz / 25k Hz
- Stimlus presentation at 48k Hz
run the script on stimlus computer: [`/code/stimulus_presentation/music_preference_presentation.py`](https://github.com/sasnl/music_preference/blob/main/code/stimulus_presentation/music_preference_presentation.py)
### Behavioral Questions After Each Song
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
 ## [Click ABR Analysis](https://github.com/sasnl/music_preference/tree/main/code/analysis/derive_click_ABR)
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

 ## Continuous Music ABR Analysis
 - Derive ANM regressor: ``
 - Derive music ABR using **deconvolution** as in Shan et al. (2024)
 - Derive music TRF using ANM regressor as in Shan et al. (2024)
 - Derive ISC among participants for each song
 - Extract behavioral question responses and make heatmap for each song
 - correlate neuro metrics with behavioral responses



