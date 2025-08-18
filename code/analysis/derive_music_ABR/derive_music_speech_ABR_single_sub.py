#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 14:36:02 2021

@author: tong
"""
# %%
import numpy as np
import pandas as pd
import scipy.signal as signal
from numpy.fft import fft, ifft
from expyfun.io import write_hdf5, read_hdf5
import mne
import matplotlib.pyplot as plt

"""
This script is used for deriving ABR using deconvolution with different regressors.
The regressors (half-wave rectified stimulus waveform, IHC, and ANM) were pre-generated.
(refer to rectified_regressor_gen.py and IHC_ANM_regressor_gen.py)
"""
# %% Define Filtering Functions

def butter_highpass(cutoff, fs, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=1):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=1):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


# %% Parameters
# Anlysis
is_click = False # if derive click ABR
is_ABR = True # if derive only ABR
Bayesian = True # Bayesian averaging
# Stim param
stim_fs = 48000 # stimulus sampling frequency
# t_click = 60 # click trial length
t_mus = 12 # music or speech trial length
# EEG param
eeg_n_channel = 2 # total channel of ABR
eeg_fs = 10000 # eeg sampling frequency
eeg_f_hp = 1 # high pass cutoff
# %% File paths
audio_file_root = '/Users/tongshan/Documents/ABR/present_files' # Present files waveforms root path
regressor_root = audio_file_root+'/ANM_new/' # Regressor files root pathfiles waveforms root path

# %% Loading and filtering EEG data

eeg_vhdr = '/Users/tongshan/Documents/ABR/data/pilot_10_20250716-selected/pilot_10_full_speech_mus.vhdr'
eeg_vhdr = '/Users/tongshan/Documents/ABR/data/old_lab/subject018/music_diverse_beh_018.vhdr'
eeg_raw = mne.io.read_raw_brainvision(eeg_vhdr, preload=True)

# %% redo abr reference channels
eeg_raw.pick_channels(['Plus_R','Minus_R','Plus_L','Minus_L'])
data_R = eeg_raw.get_data(picks=eeg_raw.ch_names[0]) - eeg_raw.get_data(picks=eeg_raw.ch_names[1])
data_L = eeg_raw.get_data(picks=eeg_raw.ch_names[2]) - eeg_raw.get_data(picks=eeg_raw.ch_names[3])
data = np.vstack((data_R, data_L))
data /= 100
info = mne.create_info(ch_names=["EP1","EP2"], sfreq=eeg_raw.info['sfreq'], ch_types='eeg')
eeg_raw_ref = mne.io.RawArray(data, info)

# %%
eeg_raw.pick_channels(["EP1","EP2"])
eeg_raw_ref = eeg_raw

# %% Read Events, correct for tube delay
events, event_dict = mne.events_from_annotations(eeg_raw)

# %% EEG Preprocessing
print('Filtering raw EEG data...')
# High-pass filter
eeg_raw_ref._data = butter_highpass_filter(eeg_raw_ref._data, eeg_f_hp, eeg_fs)
# Notch filter
notch_freq = np.arange(60, 540, 120)
notch_width = 5
for nf in notch_freq:
    bn, an = signal.iirnotch(nf / (eeg_fs / 2.), float(nf) / notch_width)
    eeg_raw_ref._data = signal.lfilter(bn, an, eeg_raw_ref._data)

# %% Epoch params
# general experiment
n_type_music = 6  # number of music types
n_type_speech = 6  # number of speech types
n_epoch = 40  # number of epoch in each type


n_epoch_total = (n_type_music + n_type_speech) * n_epoch
events_file_name = '/Users/tongshan/Documents/ABR/data/sub-18_task-MusicvsSpeech_events.tsv'
start_trial = 10
events_df = pd.read_csv(events_file_name, sep='\t')

file_all_list = []
for ti in np.arange(start_trial, len(events_df)):
    type_name = events_df['trial_type'][ti]
    piece =  events_df['number_trial'][ti]
    file_all_list += [type_name + f'{piece:03}']

# %% Epoching
print('Epoching EEG data...')
epochs = mne.Epochs(eeg_raw_ref, events, event_id=1, tmin=0,
                    tmax=(t_mus - 1/stim_fs + 1),
                    baseline=None, preload=True)
epoch = epochs.get_data()
#epoch = epoch[0:480,:]
epoch = epoch[10:490,:]

# %% Epoch indexing
music_types = ["acoustic", "classical", "hiphop", "jazz", "metal", "pop"]
speech_types = ["chn_aud", "eng_aud", "interview", "lecture", "news", "talk"]
types = music_types + speech_types
eeg_epi = dict(acoustic=np.zeros(n_epoch),
                classical=np.zeros(n_epoch),
                hiphop=np.zeros(n_epoch),
                jazz=np.zeros(n_epoch),
                metal=np.zeros(n_epoch),
                pop=np.zeros(n_epoch),
                chn_aud=np.zeros(n_epoch),
                eng_aud=np.zeros(n_epoch),
                interview=np.zeros(n_epoch),
                lecture=np.zeros(n_epoch),
                news=np.zeros(n_epoch),
                talk=np.zeros(n_epoch))
# Get epoch number for every types
for epi in range(len(file_all_list)):
    stim_type = file_all_list[epi][0:-3]
    stim_ind = int(file_all_list[epi][-3:])
    eeg_epi[stim_type][stim_ind] = epi
# %% Analysis 
# Regressor
regressor_list = ['ANM'] # half-wave rectified stimulus, IHC and ANM regressors
for regressor in regressor_list:
    # For music response
    len_eeg = int(t_mus*eeg_fs)
    data = read_hdf5(regressor_root + '/music_x_in.hdf5')
    t_start = -0.2
    t_stop = 0.6
    lags = np.arange(start=t_start*1000, stop=t_stop*1000, step=1e3/eeg_fs)
    
    w_music = dict(acoustic=np.zeros(len_eeg),
                    classical=np.zeros(len_eeg),
                    hiphop=np.zeros(len_eeg),
                    jazz=np.zeros(len_eeg),
                    metal=np.zeros(len_eeg),
                    pop=np.zeros(len_eeg))
    
    abr_music = dict(acoustic=np.zeros(8000),
                    classical=np.zeros(8000),
                    hiphop=np.zeros(8000),
                    jazz=np.zeros(8000),
                    metal=np.zeros(8000),
                    pop=np.zeros(8000))
    
    for ti in music_types:
        print(ti)
        n_epoch = 40
        # Load x_in
        x_in_pos = data['x_in_music_pos'][ti]
        x_in_neg = data['x_in_music_neg'][ti]
        # Load x_out
        x_out = np.zeros((n_epoch, eeg_n_channel, len_eeg))
    
        for ei in range(n_epoch):
            eeg_temp = epoch[int(eeg_epi[ti][ei]), :, :]
            x_out[ei, :, :] = eeg_temp[:, 0:len_eeg]
        x_out = np.mean(x_out, axis=1)
            
        # x_in fft
        x_in_pos_fft = fft(x_in_pos)
        x_in_neg_fft = fft(x_in_neg)
        # x_out fft
        x_out_fft = fft(x_out)


        if Bayesian:
            ivar = 1 / np.var(x_out, axis=1)
            weight = ivar/np.nansum(ivar)
        else:
            weight = np.ones(n_epoch)  # uniform weight       

        # TRF
        denom_pos = np.mean(x_in_pos_fft * np.conj(x_in_pos_fft), axis=0)
        denom_neg = np.mean(x_in_neg_fft * np.conj(x_in_neg_fft), axis=0)
        w_pos = []
        w_neg = []
        for ei in range(n_epoch):
            w_i_pos = (weight[ei] * np.conj(x_in_pos_fft[ei, :]) *
                        x_out_fft[ei, :]) / denom_pos
            w_i_neg = (weight[ei] * np.conj(x_in_neg_fft[ei, :]) *
                        x_out_fft[ei, :]) / denom_neg
            w_pos += [w_i_pos]
            w_neg += [w_i_neg]
        w_music[ti] = (ifft(np.array(w_pos).sum(0)).real +
                        ifft(np.array(w_neg).sum(0)).real) / 2
        abr_music[ti] = np.concatenate((w_music[ti][int(t_start*eeg_fs):],
                                        w_music[ti][0:int(t_stop*eeg_fs)]))
        # shift ABR for IHC and ANM regressor
        abr_music[ti] = np.roll(abr_music[ti], int(2.75*eeg_fs/1000))
    
    # For speech response
    data = read_hdf5(regressor_root + '/speech_x_in.hdf5')

    t_start = -0.2
    t_stop = 0.6
    lags = np.arange(start=t_start*1000, stop=t_stop*1000, step=1e3/eeg_fs)
    
    w_speech = dict(chn_aud=np.zeros(len_eeg),
                    eng_aud=np.zeros(len_eeg),
                    interview=np.zeros(len_eeg),
                    lecture=np.zeros(len_eeg),
                    news=np.zeros(len_eeg),
                    talk=np.zeros(len_eeg))
    
    abr_speech = dict(chn_aud=np.zeros(8000),
                        eng_aud=np.zeros(8000),
                        interview=np.zeros(8000),
                        lecture=np.zeros(8000),
                        news=np.zeros(8000),
                        talk=np.zeros(8000))
    
    for ti in speech_types:
        print(ti)
        n_epoch = 40
        # Load x_in
        x_in_pos = data['x_in_speech_pos'][ti]
        x_in_neg = data['x_in_speech_neg'][ti]
        # Load x_out
        x_out = np.zeros((n_epoch, eeg_n_channel, len_eeg))
        for ei in range(n_epoch):
            eeg_temp = epoch[int(eeg_epi[ti][ei]), :, :]
            x_out[ei, :, :] = eeg_temp[:, 0:len_eeg]
        x_out = np.mean(x_out, axis=1)
        
        # x_in fft
        x_in_pos_fft = fft(x_in_pos)
        x_in_neg_fft = fft(x_in_neg)
        # x_out fft
        x_out_fft = fft(x_out)
        if Bayesian:
            ivar = 1 / np.var(x_out, axis=1)
            weight = ivar/np.nansum(ivar)
        else:
            weight = np.ones(n_epoch)  # uniform weight  
        # TRF
        denom_pos = np.mean(x_in_pos_fft * np.conj(x_in_pos_fft), axis=0)
        denom_neg = np.mean(x_in_neg_fft * np.conj(x_in_neg_fft), axis=0)
        w_pos = []
        w_neg = []
        for ei in range(n_epoch):
            w_i_pos = (weight[ei] * np.conj(x_in_pos_fft[ei, :]) *
                        x_out_fft[ei, :]) / denom_pos
            w_i_neg = (weight[ei] * np.conj(x_in_neg_fft[ei, :]) *
                        x_out_fft[ei, :]) / denom_neg
            w_pos += [w_i_pos]
            w_neg += [w_i_neg]
        w_speech[ti] = (ifft(np.array(w_pos).sum(0)).real +
                        ifft(np.array(w_neg).sum(0)).real) / 2
        abr_speech[ti] = np.concatenate((w_speech[ti][int(t_start*eeg_fs):],
                                        w_speech[ti][0:int(t_stop*eeg_fs)]))
        # shift ABR for IHC and ANM regressor
        abr_speech[ti] = np.roll(abr_speech[ti], int(2.75*eeg_fs/1000))
    
# %% bandpassing
abr_music_bp = dict(acoustic=np.zeros(8000),
                    classical=np.zeros(8000),
                    hiphop=np.zeros(8000),
                    jazz=np.zeros(8000),
                    metal=np.zeros(8000),
                    pop=np.zeros(8000))
abr_music_ave = np.zeros(8000,)
for ti in music_types:
    abr_music_bp[ti] = butter_bandpass_filter(abr_music[ti], 1, 1000, eeg_fs, order=1)
    abr_music_ave += abr_music_bp[ti]
abr_music_ave = abr_music_ave / len(music_types)

abr_speech_bp = dict(chn_aud=np.zeros(8000),
                    eng_aud=np.zeros(8000),
                    interview=np.zeros(8000),
                    lecture=np.zeros(8000),
                    news=np.zeros(8000),
                    talk=np.zeros(8000))
abr_speech_ave = np.zeros(8000,)
for ti in speech_types:
    abr_speech_bp[ti] = butter_bandpass_filter(abr_speech[ti], 1, 1000, eeg_fs, order=1)
    abr_speech_ave += abr_speech_bp[ti]
abr_speech_ave = abr_speech_ave / len(music_types)
        
        # write_hdf5('/' + subject + '_abr_response_' + regressor + '.hdf5',
        #           dict(w_music=w_music, abr_music=abr_music,
        #                 w_speech=w_speech, abr_speech=abr_speech,
        #                 abr_music_ave=abr_music_ave, abr_speech_ave=abr_speech_ave,
        #                 lags=lags), overwrite=True)

# %% Plotting
plt.figure(figsize=(10, 5))
plt.plot(lags, abr_music_ave, label='Music ABR Average')
plt.plot(lags, abr_speech_ave, label='Speech ABR Average')
plt.xlabel('time (ms)')
plt.ylabel('Amplitude')
plt.title('Average ABR for Music and Speech')
plt.xlim(-200, 600)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(lags, abr_music_ave, label='Music ABR Average')
plt.plot(lags, abr_speech_ave, label='Speech ABR Average')
plt.xlabel('time (ms)')
plt.ylabel('Amplitude')
plt.title('Average ABR for Music and Speech')
plt.xlim(-20, 30)
plt.legend()
plt.tight_layout()
plt.show()

# %%
