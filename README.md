# Music Preference Study

## Table of Contents
- [Participants onboarding](#participants-onboarding)
  - [Musicianship Questionnaire](#musicianship-questionnaire)
  - [Everyone choose 3 of their favorite songs](#everyone-choose-3-of-their-favorite-songs)
  - [Music stimuli preprocessing](#music-stimuli-preprocessing)
- [Experiment procedure](#experiment-procedure)
  - [5-min click trains](#5-min-click-trains)
  - [Latin randomized chosen songs](#latin-randomized-chosen-songs)
  - [Behavioral questions following each song](#behavioral-questions-following-each-song)
    - [Pleasantness](#pleasantness)
    - [Valence/Arousal](#valencearousal)
    - [Musical Chills](#musical-chills)
    - [Preference for the Song](#preference-for-the-song)

## Participants onboarding

### [Musicianship Questionnair]()

### Everyone choose 3 of their favorite songs
**[Song list](https://docs.google.com/spreadsheets/d/1YDDWKmQ6O3HpwoQeA3kcLaOXuhWvbGlxDFgny0Mv1zk/edit?gid=0#gid=0)**

- total length of music = 68:02

### Music stimli preprocessing
- Pipeline
    - mp3-2-wav -->
    stereo-2-mono -->
    low-passEnveDiv(flatten) --> rms0.01(normalize) 
    --> Resample-48kHz
- [`music_batch_preproc.py`](https://github.com/sasnl/music_preference/blob/main/code/stimulus_presentation/music_batch_preproc.py)

**To run the script**
1. install environment according to `env.yml`
```
conda env create -f env.yml
conda activate music_preproc
```
2. run this line in terminal:
```python
python code/stimulus_presentation/music_batch_preproc.py --input_dir music_stim/original --output_dir music_stim/preprocesed --no_trim
```
## Experiment procedure
### 5-min click trains: [`/click_stim`](https://github.com/sasnl/music_preference/tree/main/click_stim)
### Latin square randomized chosen songs
- code to generate randomized song order: `generate_music_orders.py`. Generated order file: `music_presentation_orders.csv`
- Participants will passively listening to the songs, while EEG recording with both ABR+Cortical system
- EEG recording at 10k Hz / 25k Hz
- Stimlus presentation at 48k Hz
run the script on stimlus computer: [`music_preference_presentation.py`](https://github.com/sasnl/music_preference/blob/main/code/stimulus_presentation/music_preference_presentation.py)
### Behavioral questions following each song
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
 ## Click ABR analysis
 - Cross-correlation
 - ``
 ## Continous music ABR analysis
 - Derive ANM regressor: ``
 - Derive music ABR using **deconvolution** as in Shan et al. (2024)
 - Derive music TRF using ANM regressor as in Shan et al. (2024)
 - Derive ISC among participants for each song
 - Extract behavioral question responses and make heatmap for each song
 - correlate neuro metrics with behavioral responses



