#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Music Preference Experiment Presentation Script (Improved Version)

This script presents music stimuli to participants and collects behavioral responses.
It uses the randomized music order from CSV file and presents questions after each song.
Improved with background audio preloading and loading screens to eliminate UI freezing.

@author: Tong, Ariya
"""

# %%
import os
# Set environment variable before importing sounddevice. Value is not important.
os.environ["SD_ENABLE_ASIO"] = "1"

import sounddevice as sd
sd.query_hostapis()
sd.query_devices()

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from expyfun import ExperimentController, decimals_to_binary
from expyfun.stimuli import window_edges
from expyfun.visual import ProgressBar
from expyfun.io import read_wav
import os
import random
from expyfun.visual import Circle
from datetime import datetime
import threading
import time

#%% Set parameters
n_epoch_total = 15
do_click = True # whether to play click sound before music

# hardware
fs = 48000
n_channel = 2
rms = 0.01
stim_db = 65

pause_dur = 1  # pause duration between each epoch

# %% Load music and click stimuli
# Use relative paths instead of hardcoded Windows paths
exp_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/"
file_path = exp_path+"music_stim/preprocesed/"  # Corrected path to preprocessed files
click_path = exp_path+"click_stim/"

# %% Input participants ID
participant_id = input("Enter participant ID (1-5): ")

#%% click setting: TRUE for click session, FALSE for no click session
click_input = input("Do click session? (y/n)").strip().lower()
click = True if click_input == 'y' else False
# click epochs setting
n_epoch_click = 5 # number of click epochs
click_dur = 60 # duration of each click epoch

# Load click train
print("Loading click stimuli...")
click_data = []
for c in range(n_epoch_click):
   temp, fs = read_wav(click_path + 'click{0:03d}'.format(c) + '.wav')
   click_data += [temp[0]]

# Load music stimuli in order of music_presentation_orders.csv and participant ID
# Load and validate CSV data
csv_path = exp_path+"code/stimulus_presentation/music_presentation_orders.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

orders_df = pd.read_csv(csv_path)
stim_df = orders_df[orders_df['Participant_ID'] == int(participant_id)]

if stim_df.empty:
    raise ValueError(f"No data found for participant ID: {participant_id}")

print(f"Found {len(stim_df)} songs for participant {participant_id}")

# Extract stimulus information
stim_name_list = stim_df['Song_File'].tolist()
stim_dur_list = stim_df['Duration_Seconds'].tolist()

# Build file paths using original participant information
stim_path_list = []
for _, row in stim_df.iterrows():
    original_participant = row['Original_Participant']
    song_file = row['Song_File']
    # Construct path: music_stim/preprocesed/{original_participant}/{song_file}
    file_path_full = os.path.join(file_path, str(original_participant), song_file)
    stim_path_list.append(file_path_full)
    
    # Verify file exists
    if not os.path.exists(file_path_full):
        print(f"Warning: File not found: {file_path_full}")

# Preload all audio files to avoid loading delays during trials
print("Preloading audio files...")
audio_data = []
for i, song_path in enumerate(stim_path_list):
    print(f"Loading song {i+1}/{len(stim_path_list)}: {os.path.basename(song_path)}")
    try:
        temp, fs = read_wav(song_path)
        audio_data.append(temp)
        print(f"  ✓ Loaded successfully ({len(temp)/fs:.1f}s)")
    except Exception as e:
        print(f"  ✗ Error loading {song_path}: {e}")
        # Create silent audio as fallback
        fallback_duration = stim_df.iloc[i]['Duration_Seconds']
        fallback_samples = int(fallback_duration * fs)
        audio_data.append(np.zeros((fallback_samples, 1)))
        print(f"  ✓ Created fallback silent audio ({fallback_duration}s)")

print("Audio preloading completed!")

# %%
click_instruction = ('Hi, thank you for participating this study!'
                     '\n You will first passively listen to sound for 5 min, which sounds like clicks.'
                     '\n Then you will hear several music.'
                     '\n Press the SPACE bar to start the experiment.')
music_instruction = ('Now you will hear the music.'
                    '\n After each music, you will be asked questions such as "how do you like the music?"' 
                    '\n Please use your mouse to choose from a scale of 1 to 9 accordingly.'
                    '\n Press the SPACE bar to continue.')
break_instruction = ('You can take a break after each song but not during the song.'
                    '\n Press the SPACE bar to continue.')

# %% Experiment

ec_args = dict(exp_name='Music Preference', window_size=[2560, 1440],
               session='00', full_screen=True, n_channels=n_channel,
               version='dev', stim_fs=fs, stim_db=65,
               force_quit=['end'])
trial_start_time = -np.inf

# %% Experiment start
n_bits_epoch = int(np.ceil(np.log2(n_epoch_total)))
n_bits_type = int(np.ceil(np.log2(2)))

# Pre-create UI elements for questions to avoid recreation
def create_question_ui(ec):
    """Create and cache UI elements for questions"""
    ui_elements = {}
    
    # Define positions for left/right labels and circles
    left_label_x = 0.35
    right_label_x = 1.2
    circles_y = -0.3
    number_y = -0.25
    n_circles = 9
    circle_spacing = 0.1
    circle_radius = (0.02, 0.03)
    number_circle_factor = 0.3
    
    # Pre-create circles for all questions
    circles = []
    for i in range(n_circles):
        circle = Circle(ec, radius=circle_radius, 
                       pos=(number_circle_factor+((-0.7 + (i*circle_spacing))), circles_y), 
                       units='norm', fill_color=None, line_color='white', line_width=3)
        circles.append(circle)
    
    ui_elements['circles'] = circles
    ui_elements['circle_spacing'] = circle_spacing
    ui_elements['number_circle_factor'] = number_circle_factor
    ui_elements['number_y'] = number_y
    ui_elements['left_label_x'] = left_label_x
    ui_elements['right_label_x'] = right_label_x
    
    return ui_elements

def show_loading_screen(ec, message="Loading next song...", progress=None):
    """Show a loading screen with optional progress"""
    ec.screen_text(message, pos=[0, 0.2], units='norm', color='w')
    ec.screen_text("Please wait...", pos=[0, -0.2], units='norm', color='w')
    
    if progress is not None:
        # Create a simple progress bar
        bar_width = 0.8
        bar_height = 0.05
        progress_width = bar_width * progress
        
        # Draw background bar
        ec.screen_text("█" * int(bar_width * 50), pos=[0, -0.1], units='norm', color='gray')
        # Draw progress bar
        ec.screen_text("█" * int(progress_width * 50), pos=[0, -0.1], units='norm', color='white')
    
    ec.flip()

def background_load_buffer(ec, audio_data, trial_id, callback=None):
    """Load audio buffer in background thread"""
    def load_thread():
        try:
            print(f"Background loading buffer for {trial_id}...")
            ec.load_buffer(audio_data)
            print(f"✓ Background loading completed for {trial_id}")
            # If a callback function is provided, call it to indicate successful completion.
            # The first argument True signals success, and the second argument None means there was no error.
            if callback:
                callback(True, None)
        except Exception as e:
            print(f"✗ Background loading failed for {trial_id}: {e}")
            if callback:
                callback(False, str(e))
    
    thread = threading.Thread(target=load_thread)
    thread.daemon = True
    thread.start()
    return thread

with ExperimentController(**ec_args) as ec:
    # Pre-create UI elements
    print("Initializing UI elements...")
    ui_elements = create_question_ui(ec)
    print("UI initialization completed!")
    
    # if click is True, play click sound
    if click:
        ec.screen_prompt(click_instruction, live_keys=['space']) # show click instruction
        pb_click = ProgressBar(ec, [0, -.2, 1.5, .2], colors=('xkcd:teal blue', 'w')) # progress bar
        for cn in range(n_epoch_click):
            ec.screen_text('Click trial number ' + str(cn+1) + ' out of ' + str(int(n_epoch_click)) + ' trials.')
            pb_click.draw()

            trial_id = "click train " + str(cn) # click session trial number 
            ec.load_buffer(click_data[cn]) # load click sound to sound card buffer

            ec.identify_trial(ec_id=trial_id, ttl_id=[]) # identify trial to record in log file

            ec.wait_until(trial_start_time + click_dur) # wait until the start of the trial
            trial_start_time = ec.start_stimulus() # start the sound stimulus playback
            ec.wait_secs(0.1) # wait for 0.1 sec
            ec.stamp_triggers([(b + 1) * 4 for b in decimals_to_binary([0, cn], [n_bits_type, n_bits_epoch])]) # stamp triggers of the specific trial number 

            # wait for the duration of the trial
            while ec.current_time < trial_start_time + click_dur:
                ec.check_force_quit()
                ec.wait_secs(0.1)

            ec.trial_ok() # mark the trial as ok
            ec.stop() # stop the sound stimulus playback

            pb_click.update_bar((cn + 1) / n_epoch_click * 100) # update the progress bar
            pb_click.draw() # draw the updated progress bar
    

    # %% Present music trials
    trial_start_time = -np.inf
    pb = ProgressBar(ec, [0, -.2, 1.5, .2], colors=('xkcd:teal blue', 'w'))

    ec.screen_prompt(music_instruction, live_keys=['space'])
    ec.screen_prompt(break_instruction, live_keys=['space'])
    
    # Preload first song buffer
    print("Preloading first song buffer...")
    show_loading_screen(ec, "Preparing first song...")
    ec.load_buffer(audio_data[0])
    print("✓ First song buffer loaded")
    
    for ei in range(n_epoch_total):
        
        # Load trial parameters
        trial_id = os.path.basename(stim_df.iloc[ei]['Song_File'])
        trial_dur = stim_df.iloc[ei]['Duration_Seconds']
        
        # Use preloaded audio data instead of loading from disk
        temp = audio_data[ei]
        
        # Draw progress bar
        ec.screen_text('Trial number ' + str(ei+1) + ' out of ' + str(int(n_epoch_total)) + ' trials.')
        pb.draw()

        # Define trial (buffer already loaded for first song)
        if ei > 0:
            # For subsequent songs, buffer was preloaded during questions
            pass
        ec.identify_trial(ec_id=trial_id, ttl_id=[])
        ec.wait_until(trial_start_time + trial_dur + pause_dur)
        # Start stimulus
        trial_start_time = ec.start_stimulus()
        ec.wait_secs(0.1)
        # Trigger
        ec.stamp_triggers([(b + 1) * 4 for b in decimals_to_binary([1, ei], [n_bits_type, n_bits_epoch])])
        while ec.current_time < trial_start_time + trial_dur:
            ec.check_force_quit()
            ec.wait_secs(0.1)
        ec.stop()
        
        
        ec.write_data_line("Trial data", {"trial_num": ei,
                                          "trial_id": trial_id,
                                          "song_file": stim_df.iloc[ei]['Song_File'],
                                          "duration": trial_dur})
        
        # Preload next song buffer in background during questions
        next_song_loaded = [False]  # Use list to allow modification in callback
        loading_thread = None
        if ei < n_epoch_total - 1:  # Not the last song
            next_audio = audio_data[ei + 1]
            next_trial_id = os.path.basename(stim_df.iloc[ei + 1]['Song_File'])
            
            def on_buffer_loaded(success, error):
                next_song_loaded[0] = success
                if not success:
                    print(f"Warning: Failed to preload next song: {error}")
            
            # Start background loading
            loading_thread = background_load_buffer(ec, next_audio, next_trial_id, on_buffer_loaded)
        
        # Ask preference and other questions (all 1-9 scale)
        ec.toggle_cursor(False)
        ec.wait_secs(0.1)

        questions = [
            {
                "prompt": "How much did you like or enjoy the song overall?",
                "left": "1 = Not at all",
                "right": "9 = Very much",
                "varname": "preference"
            },
            {
                "prompt": "How pleasant or unpleasant did you find the song?",
                "left": "1 = Extremely unpleasant",
                "right": "9 = Extremely pleasant",
                "varname": "pleasantness"
            },
            {
                "prompt": "How emotionally intense or stimulating was the song for you?",
                "left": "1 = Not intense or stimulating at all",
                "right": "9 = Extremely intense or stimulating",
                "varname": "valence_arousal"
            },
            {
                "prompt": "To what extent did you feel chills, goosebumps, \n or a strong emotional reaction while listening to the song?",
                "left": "1 = Not at all",
                "right": "9 = Very strongly",
                "varname": "chills"
            }
        ]

        responses = {}
        
        for q in questions:
            # Display the question prompt at the center of the screen
            ec.screen_text(q["prompt"], pos=[0.5, 0], units='norm', color='w')

            # Place "left" and "right" labels
            ec.screen_text(q["left"], pos=[ui_elements['left_label_x'], -0.5], units='norm', color='w')
            ec.screen_text(q["right"], pos=[ui_elements['right_label_x'], -0.5], units='norm', color='w')

            # Draw numbers above circles
            for i in range(9):
                ec.screen_text(str(i + 1), 
                              pos=[ui_elements['number_circle_factor']+(0.09+i*ui_elements['circle_spacing']), 
                                   ui_elements['number_y']], units='norm', color='w')
            
            # Draw all circles
            for c in ui_elements['circles']:
                c.draw()
            ec.flip()
            
            # Wait for click
            click, ind = ec.wait_for_click_on(ui_elements['circles'], max_wait=np.inf)

            # Show feedback: highlight the selected circle
            after_circles = []
            for i in range(9):
                if i == ind:
                    after_circles.append(
                        Circle(ec, radius=(0.02, 0.03), pos=(ui_elements['number_circle_factor']+((-0.7 + (i*ui_elements['circle_spacing']))), ui_elements['number_y']-0.05), units='norm',
                               fill_color='white', line_color='white', line_width=3)
                    )
                else:
                    after_circles.append(
                        Circle(ec, radius=(0.02, 0.03), pos=(ui_elements['number_circle_factor']+((-0.7 + (i*ui_elements['circle_spacing']))), ui_elements['number_y']-0.05), units='norm',
                               fill_color=None, line_color='white', line_width=3)
                    )
                ec.screen_text(str(i + 1), pos=[ui_elements['number_circle_factor']+(0.09+i*ui_elements['circle_spacing']), ui_elements['number_y']], units='norm', color='w')
            for c in after_circles:
                c.draw()
            ec.flip()
            
            # Show feedback for a moment before moving to next question
            ec.wait_secs(0.2)
            
            responses[q["varname"]] = ind + 1  # 1-based
           
        # save data
        ec.write_data_line("responses", responses)
        
        # Wait for background loading to complete or show loading screen
        if ei < n_epoch_total - 1:
            if loading_thread and loading_thread.is_alive():
                print("Background loading still in progress, waiting...")
                show_loading_screen(ec, "Loading next song...")
                # Wait for the thread to complete
                loading_thread.join()
                print("Background loading completed")
            
            # If loading failed, load it now
            if not next_song_loaded[0]:
                print("Loading next song buffer (fallback)...")
                show_loading_screen(ec, "Loading next song...")
                ec.load_buffer(audio_data[ei + 1])
        
        ec.trial_ok()
        ec.stop()
        # Update progress bar
        pb.update_bar((ei + 1) / n_epoch_total * 100)
        pb.draw()
        ec.wait_secs(0.5)
    # End exp
    ec.screen_prompt('ya did it!')
True 