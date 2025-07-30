#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul 29 2025

@author: Tong, Ariya
"""

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

#%% Set parameters
n_epoch_total = 15
do_click = True # whether to play click sound before music

# hardware
fs = 48000
n_channel = 2
rms = 0.01
stim_db = 65 #????

pause_dur = 1  # pause duration between each epoch

# %% Load music and click stimuli
exp_path = ""
file_path = exp_path+"music_stim/"
click_path = exp_path+"click_stim/"

#%% click setting: TRUE for click session, FALSE for no click session
click = False 
# click epochs setting
n_epoch_click = 5 # number of click epochs
click_dur = 60 # duration of each click epoch

# Load click train
click_data = []
for c in range(n_epoch_click):
   temp, fs = read_wav(click_path + 'click{0:03d}'.format(c) + '.wav')
   click_data += [temp[0]]

# Load music stimuli in order of music_presentation_orders.csv and participant ID
participant_id = input("Enter participant ID: ")
orders_df = pd.read_csv(exp_path+"stimulus_presentation/music_presentation_orders.csv")
stim_df = orders_df[orders_df['participant_id'] == participant_id]
stim_name_list = stim_df['Song_File'].tolist()
stim_path_list = [os.path.join(file_path, stim_name) for stim_name in stim_name_list]
stim_dur_list = stim_df['Duration_Seconds'].tolist()

# %%
click_instruction = ('Hi, thank you for participating this study!'
                     '\n You will first passively listen to sound for 5 min, which sounds like clicks.'
                     '\n Then you will hear several music.'
                     '\n Press the SPACE bar to start the experiment.')
music_instruction = ('Now you will hear the music.'
                    '\n After each music, you will be asked questions such as "how do you like the music?"' 
                    '\n Please use your mouse to choose from a scale of 1 to 9 accordingly.')
break_instruction = ('You can take a break after each song but not during the song.'
                    '\n Press the SPACE bar to continue.')

# %% Experiment

ec_args = dict(exp_name='Music Preference',
               session='00', full_screen=True, n_channels=n_channel,
               version='dev', enable_video=True, stim_fs=fs, stim_db=65,
               force_quit=['end'])
trial_start_time = -np.inf

# %% Experiment start
n_bits_epoch = int(np.ceil(np.log2(n_epoch_total)))
n_bits_type = int(np.ceil(np.log2(2)))

with ExperimentController(**ec_args) as ec:
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
    ec.screen_prompt(music_instruction, live_keys=['space'])
    ec.screen_prompt(break_instruction, live_keys=['space'])

    pb = ProgressBar(ec, [0, -.2, 1.5, .2], colors=('xkcd:teal blue', 'w'))
    for ei in range(n_epoch_total):
        # Draw progress bar
        ec.screen_text('Trial number ' + str(ei+1) + ' out of ' + str(int(n_epoch_total)) + ' trials.')
        pb.draw()
        # Load trial parameters
        trial_id = os.path.basename(stim_df.loc[ei, 'Song_File'])
        temp, fs = read_wav(stim_df.loc[ei, 'Song_file'])
        trial_dur = stim_df.loc[ei, 'duration']

        # Load buffer, define trial
        ec.load_buffer(temp)
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
                "prompt": "To what extent did you feel chills, goosebumps, or a strong emotional reaction while listening to the song?",
                "left": "1 = Not at all",
                "right": "9 = Very strongly",
                "varname": "chills"
            }
        ]

        responses = {}

        for q in questions:
            ec.screen_text(q["prompt"], pos=[0, 0.2], color='w')
            ec.screen_text(q["left"], pos=[-0.2, 0], color='w')
            ec.screen_text(q["right"], pos=[1.3, 0], color='w')
            init_circles = []
            for i in range(9):
                init_circles += [Circle(ec, radius=(0.02, 0.03), pos=((-1+(i+1)*0.22), -0.25), units='norm',
                                        fill_color=None, line_color='white', line_width=5)]
                ec.screen_text(str(i+1), pos=[(i)*0.22+0.04, -0.15], units='norm', color='w')
            for c in init_circles:
                c.draw()
            ec.flip()
            click, ind = ec.wait_for_click_on(init_circles, max_wait=np.inf)
            after_circles = []
            for i in range(9):
                if i == ind:
                    after_circles += [Circle(ec, radius=(0.02, 0.03), pos=((-1+(i+1)*0.22), -0.25), units='norm',
                                             fill_color='white', line_color='white', line_width=5)]
                else:
                    after_circles += [Circle(ec, radius=(0.02, 0.03), pos=((-1+(i+1)*0.22), -0.25), units='norm',
                                            fill_color=None, line_color='white', line_width=5)]
                ec.screen_text(str(i+1), pos=[(i)*0.22+0.04, -0.15], units='norm', color='w')
            for c in after_circles:
                c.draw()
            ec.flip()
            responses[q["varname"]] = ind + 1  # 1-based
        # Save data
        # Save trial_num, trial_id, and all responses
        trial_data = {
            "trial_num": ei,
            "trial_id": trial_id,
        }
        trial_data.update(responses)
        ec.write_data_line("Trial data", trial_data)
        ec.trial_ok()
        # Update progress bar
        pb.update_bar((ei + 1) / n_epoch_total * 100)
        pb.draw()
    # End exp
    ec.screen_prompt('ya did it!')
True
