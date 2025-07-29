#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 11:36:00 2025

@author: Tong, Ariya
"""
# Import necessary libraries (i.e. packages)
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from expyfun import ExperimentController, decimals_to_binary
from expyfun.stimuli import window_edges
from expyfun.visual import ProgressBar
from expyfun.io import read_wav
from expyfun.visual import Rectangle

# %% PARAMETERS

# hardware
fs = 48000 # sampling rate !!!
n_channel = 2 # number of channels: left and right
rms = 0.01 # RMS of the sound
stim_db = 65 # sound level in dB


#%% EXPERIMENT paths
# path to the folder that contains the experiment files
exp_path = 'C:/Users/Maddox/Code/Experiments/music_abr_diverse_beh/' 
# path to the folder that contains the music and speech files
file_path = exp_path + "present_files/" # path to the folder that contains the stimuli files

#%% click setting: TRUE for click session, FALSE for no click session
click = True 
# click epochs setting
n_epoch_click = 5 # number of click epochs
click_dur = 60 # duration of each click epoch

# Load click train
click_data = []
for c in range(n_epoch_click):
   temp, fs = read_wav(file_path + "click/" + 'click{0:03d}'.format(c) + '.wav')
   click_data += [temp[0]]

# instructions script
click_instruction = ('You will passively listen to sound for 5 min, which sounds like clicks.'
                     '\n Click your mouse to start the experiment.')
#%% general music /speech epochs
n_type_music = 6  # number of music types
n_type_speech = 6  # number of speech types
n_epoch = 8*60/12  # number of epoch in each type
n_epoch_total = (n_type_music + n_type_speech) * n_epoch # total number of epochs
pause_dur = 0.5  # pause duration between each epoch
epoch_dur = 10 # in sec

# music/speech session block setting
n_epoch_1_block = [5, 6, 7] # number of epochs in each block
n_block = int(n_epoch_total/np.average(n_epoch_1_block)) # number of blocks

#%% ###### RANDOM SET BLOCK ORDER ########## #
block_list = n_epoch_1_block*(int(n_block/3)) # list of number of epochs in each block
random.seed(0) # set random seed for reproducibility of experiment order
random.shuffle(block_list) # shuffle the order of blocks

question_trial = [] # list of trials that will have question
for i in range(len(block_list)):
    question_trial += [sum(block_list[0:i+1]) - 1]
# %% ###### RANDOM SET TRIAL CONTENT ########## #
music_path = file_path + "music/" # path to the folder that contains the music files
speech_path = file_path + "speech/" # path to the folder that contains the speech files

# Get the list of music and speech files
music_types = ["acoustic", "classical", "hiphop", "jazz", "metal", "pop"]
music_file_name =  []
music_file_type = []
for t in music_types:
    type_list = [f for f in os.listdir(file_path + t) if os.path.isfile(os.path.join(file_path+t, f))]
    music_file_name += [type_list]
    music_file_type += [t for i in range(len(type_list))]

speech_types = ["chn_aud", "eng_aud", "interview", "lecture", "news", "talk"]
speech_file_name =  []
speech_file_type = []
for t in speech_types:
    type_list = [f for f in os.listdir(file_path + t) if os.path.isfile(os.path.join(file_path+t, f))]
    speech_file_name += [type_list]
    speech_file_type += [t for i in range(len(type_list))]

# Get the number of epochs in each type
type_all = ["click"] + music_types + speech_types
type_all_list = music_file_name + speech_file_name
# Shuffle the order of each type
for i in range(12):
    random.shuffle(type_all_list[i])
# Random sampling 
file_all_list = [] # list of all files
for i in range(int(n_epoch)):
    order = random.sample(range(12),12)
    for k in range(len(order)):
        file_all_list += [type_all_list[order[k]][i]]

sound_instructions = ('Now you will listen to music and speech. '
                      '\n There will be a question asking you to compare the number of pieces of music and speech.'
                      '\n Please listen to the sound, count how many music and speech in each block '
                      'and answer questions appear at the end of each block. '
                      '\n Click to start the next session.')
# %% CALL expyfun experiment controller (EC)
# Trigger bit setting
n_bits_epoch = int(np.ceil(np.log2(n_epoch_total / (n_type_music + n_type_speech))))
n_bits_type = int(np.ceil(np.log2(n_type_music + n_type_speech + 1)))  # +1 meaning the click

trial_start_time = -np.inf # time of the start of the trial

with ExperimentController('music_diverse_abr', verbose=True, screen_num=0,
                          window_size=[1920, 1080], full_screen=False,
                          stim_db=stim_db, stim_fs=fs,
                          session='study', version='dev',
                          check_rms='wholefile', n_channels=n_channel,
                          force_quit=['end']) as ec:

    ec.write_data_line('n_bits_epoch', n_bits_epoch)
    ec.write_data_line('n_bits_type', n_bits_type)
# %% CLICK SESSION
    if click:
        ec.screen_prompt(click_instruction, click=True) # show click instruction
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

    # %% PRESENT MUSIC AND SPEECH TRIALS
    start_time = ec.current_time # get the current time
    pb = ProgressBar(ec, [0, -.2, 1.5, .2], colors=('xkcd:teal blue', 'w'))

    block_number = 0 # block number initialization
    music_number = 0 # music trial number initialization
    speech_number = 0 # speech trial number initialization

    ec.screen_prompt('You have finished the first session.'
                     '\n Click to continue.', click=True) # show instruction to continue
    ec.screen_prompt(sound_instructions, click=True) # show sound instruction

    for n in range(int(n_epoch_total)):

        current_type = file_all_list[n][0:-7] # get the current type of the trial
        if current_type in music_types:
            current_cat = "music"
            music_number += 1 # get the current music trial number
        else:
            current_cat = 'speech'
            speech_number += 1 # get the current speech trial number
        current_piece = int(file_all_list[n][-7:-4]) # get the current piece number of the trial

        # load the sound file
        wave_temp, fs = read_wav(file_path + current_cat + "_10s/" + current_type + '/' + file_all_list[n])
        if current_piece%2 == 0:
            sig = window_edges(wave_temp, fs, dur=0.03) # window the sound file
        else: 
            wave_temp = -wave_temp # invert the sound file if the piece number is odd, polarity issue
            sig = window_edges(wave_temp, fs, dur=0.03) # window the sound file

        type_number = type_all.index(current_type) # get the current type number

        trial_id = "trial number: " + str(n) + ", block number: " + str(block_number) + ", type: " + current_type + ', piece: {}'.format(file_all_list[n][-7:-4]) # get the trial identity to be recorded in log file

        ec.screen_text('Trial number ' + str(n+1) + ' out of ' + str(int(n_epoch_total)) + ' trials.') # show the trial number
        pb.draw() # draw the progress bar
        ec.load_buffer(sig) # load the sound file to sound card buffer

        ec.identify_trial(ec_id=trial_id, ttl_id=[]) # identify the trial to be recorded in log file

        ec.wait_until(trial_start_time + epoch_dur + pause_dur) # wait until the start of the trial
        trial_start_time = ec.start_stimulus() # start the sound stimulus playback
        ec.wait_secs(0.1) # wait for 0.1 sec
        ec.stamp_triggers([(b + 1) * 4 for b in decimals_to_binary([type_number, int(file_all_list[n][-7:-4])], [n_bits_type, n_bits_epoch])]) # stamp triggers of the specific trial number

        # wait for the duration of the trial
        while ec.current_time < trial_start_time + epoch_dur:
            ec.check_force_quit()
            ec.wait_secs(0.1)

        # if the trial is the end of the block, show the question
        if n in question_trial:
            ec.toggle_cursor(False) # hide the cursor at the beginning of the question
            ec.wait_secs(0.1)
            # show the question
            ec.screen_text("Did you hear more trials of music or speech?" + "\n Please click on the corresponding box.", pos=[0, 0.2], color='w')

            # Draw button
            more_music = Rectangle(ec, (-0.5, -0.2, 0.3, 0.3), units='norm', fill_color=None, line_color='b')
            ec.screen_text("More music", pos=[0.22,-0.23],units='norm', color='b')
            equal = Rectangle(ec, (0, -0.2, 0.3, 0.3), units='norm', fill_color=None, line_color='g')
            ec.screen_text("Equal", pos=[0.76, -0.23], units='norm', color='g')
            more_speech = Rectangle(ec, (0.5, -0.2, 0.3, 0.3), units='norm', fill_color=None, line_color='y')
            ec.screen_text("More speech", pos=[1.215,-0.23], units='norm', color='y')

            objects = [more_music, equal, more_speech]
            for o in objects:
                o.draw()
            ec.flip() # flip the screen to show the question and the buttons
            click, ind = ec.wait_for_click_on(objects, max_wait=np.inf) # wait for the user to click on the button

            # get the response from the user
            if ind==0:
                response = "music"
            elif ind==1:
                response = "equal"
            else:
                response = "speech"
            ec.write_data_line("response", response) # write the response to the log file

            if music_number > speech_number:
                answer = "music"
            elif music_number == speech_number:
                answer = "equal"
            else:
                answer = "speech"

            ec.write_data_line("answer", answer) # write the answer to the log file
            ec.wait_secs(1)

            # Reset music and speech number
            music_number = 0
            speech_number = 0
            
            # Show the feedback
            if response == answer:
                response_instruction = "Correct. "
            else:
                response_instruction = "Incorrect. "
            
            ec.screen_prompt(response_instruction + "\n Click to start the next block.", click=True) # show the feedback and wait for the user to click to continue

        # write the trial information to the log file
        ec.write_data_line("Trial Params", dict(trial_num=n,
                                                block_number=block_number,
                                                current_type=current_type,
                                                piece_num=file_all_list[n][-7:-4]))
        ec.trial_ok() # mark the trial as ok
        ec.stop() # stop the sound stimulus playback

        pb.update_bar((n + 1) / n_epoch_total * 100) # update the progress bar
        pb.draw() # draw the updated progress bar

    ec.screen_prompt('ya did it!') # show the end of the experiment message

