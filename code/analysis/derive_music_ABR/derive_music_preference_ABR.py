#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Music Preference ABR Analysis Script

Derives ABR from preferred vs non-preferred music stimuli using deconvolution 
with ANM regressors. Based on the continuous stimulus ABR approach.

For a single subject:
- Identifies preferred songs based on naming convention (e.g., pilot 1 prefers 1-1, 1-2, 1-3)
- Derives ABR for preferred vs non-preferred music trials
- Saves results as HDF5 files

Usage: python derive_music_preference_ABR.py <subject_id>
Example: python derive_music_preference_ABR.py pilot_2

"""

import numpy as np
import scipy.signal as signal
from numpy.fft import fft, ifft
from expyfun.io import write_hdf5, read_hdf5
import mne
import matplotlib.pyplot as plt
import os
import glob
import sys

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
# Analysis
Bayesian = True  # Bayesian averaging
# Stim param
stim_fs = 25000  # stimulus sampling frequency from ANM regressor
# EEG param
eeg_n_channel = 2  # total channel of ABR (Plus_R-Minus_R, Plus_L-Minus_L)
eeg_fs = 25000  # eeg sampling frequency
eeg_f_hp = 1  # high pass cutoff

# ABR analysis window
t_start = -0.2
t_stop = 0.6
lags = np.arange(start=t_start*1000, stop=t_stop*1000, step=1e3/eeg_fs)

# %% File paths
data_root = './data/'
regressor_root = './music_stim/music_anm/'
output_root = './output/'

# Create output directory if it doesn't exist
os.makedirs(output_root, exist_ok=True)

# %% Subject configuration
# Define preferred songs for each subject (based on naming convention)
all_songs = ['1-1', '1-2', '1-3', '2-1', '2-2', '2-3', '3-1', '3-2', '3-3', 
             '4-1', '4-2', '4-3', '5-1', '5-2', '5-3']

def get_preferred_songs(subject_id):
    """Get preferred songs for a subject based on naming convention"""
    # Extract pilot number from subject_id (e.g., 'pilot_2' -> '2')
    pilot_num = subject_id.split('_')[1]
    preferred = [f"{pilot_num}-1", f"{pilot_num}-2", f"{pilot_num}-3"]
    nonpreferred = [song for song in all_songs if song not in preferred]
    return preferred, nonpreferred

# %% Get subject from command line argument
if len(sys.argv) != 2:
    print("Usage: python derive_music_preference_ABR.py <subject_id>")
    print("Example: python derive_music_preference_ABR.py pilot_2")
    sys.exit(1)

subject = sys.argv[1]
print(f"\n=== Processing {subject} ===")
# Check if subject data exists
subject_data_dir = os.path.join(data_root, subject)
if not os.path.exists(subject_data_dir):
    print(f"Error: Data directory for {subject} not found at {subject_data_dir}")
    sys.exit(1)
    
# Get preferred and non-preferred songs for this subject
preferred_songs, nonpreferred_songs = get_preferred_songs(subject)
print(f"Preferred songs: {preferred_songs}")
print(f"Non-preferred songs: {nonpreferred_songs}")

# Find all EEG files for this subject
eeg_files = glob.glob(os.path.join(subject_data_dir, f"{subject}-trial*_proc_originalptp*.fif"))

if not eeg_files:
    print(f"Error: No EEG files found for {subject} in {subject_data_dir}")
    sys.exit(1)
    
# Initialize storage for preferred and non-preferred trials
preferred_trials = {}
nonpreferred_trials = {}

# Parse EEG files and categorize by song
for eeg_file in eeg_files:
    filename = os.path.basename(eeg_file)
    # Extract song ID from filename (e.g., "pilot_2-trial1_2-2_proc_originalptp-2.fif" -> "2-2")
    parts = filename.split('_')
    if len(parts) >= 3:
        song_id = parts[2]  # This should be like "2-2"
        
        if song_id in preferred_songs:
            preferred_trials[song_id] = eeg_file  # Store single file per song
        elif song_id in nonpreferred_songs:
            nonpreferred_trials[song_id] = eeg_file  # Store single file per song

print(f"Found preferred trials: {list(preferred_trials.keys())}")
print(f"Found non-preferred trials: {list(nonpreferred_trials.keys())}")

# Initialize arrays for storing ABR results
abr_length = int((t_stop - t_start) * eeg_fs)

w_preferred_all = []
w_nonpreferred_all = []
    
# Process preferred songs
print("\nProcessing preferred songs...")
for song_id in preferred_trials:
    print(f"Processing preferred song: {song_id}")
    
    # Check if ANM regressor exists
    regressor_file = os.path.join(regressor_root, f"single_{song_id}_proc_anm.hdf5")
    if not os.path.exists(regressor_file):
        print(f"Warning: ANM regressor not found for {song_id}. Skipping...")
        continue
    
    # Load ANM regressor
    regressor_data = read_hdf5(regressor_file)
    x_in_pos = regressor_data['x_in_pos'][0]  # Remove singleton dimension
    x_in_neg = regressor_data['x_in_neg'][0]  # Remove singleton dimension
    
    # Calculate song length from regressor length
    len_eeg = len(x_in_pos)  # Use the actual regressor length
    print(f"  Song duration: {len_eeg / eeg_fs:.2f} seconds")
    
    # Process the single trial for this song
    eeg_file = preferred_trials[song_id]
    print(f"  Processing trial: {os.path.basename(eeg_file)}")
    
    # Load and preprocess EEG data
    eeg_raw = mne.io.read_raw_fif(eeg_file, preload=True, verbose=False)
    
    # Create ABR channels (Plus - Minus for R and L)
    eeg_raw.pick_channels(['Plus_R', 'Minus_R', 'Plus_L', 'Minus_L'])
    data_R = eeg_raw.get_data(picks=['Plus_R'])[0] - eeg_raw.get_data(picks=['Minus_R'])[0]
    data_L = eeg_raw.get_data(picks=['Plus_L'])[0] - eeg_raw.get_data(picks=['Minus_L'])[0]
    data = np.vstack((data_R, data_L))
    data /= 100  # Scale factor
    
    # Apply high-pass filter
    data = butter_highpass_filter(data, eeg_f_hp, eeg_fs)
    
    # Apply notch filter
    notch_freq = np.arange(60, 540, 120)
    notch_width = 5
    for nf in notch_freq:
        bn, an = signal.iirnotch(nf / (eeg_fs / 2.), float(nf) / notch_width)
        data = signal.lfilter(bn, an, data)
    
    # Extract EEG data for analysis (trim to match regressor length)
    x_out = data[:, :len_eeg]
    x_out = np.mean(x_out, axis=0)  # Average across L/R channels
    
    # Use full regressor length (no trimming needed)
    x_in_pos_trimmed = x_in_pos
    x_in_neg_trimmed = x_in_neg
    
    # FFT
    x_in_pos_fft = fft(x_in_pos_trimmed)
    x_in_neg_fft = fft(x_in_neg_trimmed)
    x_out_fft = fft(x_out)
    
    # Deconvolution (TRF estimation)
    denom_pos = x_in_pos_fft * np.conj(x_in_pos_fft)
    denom_neg = x_in_neg_fft * np.conj(x_in_neg_fft)
    
    w_pos = (np.conj(x_in_pos_fft) * x_out_fft) / denom_pos
    w_neg = (np.conj(x_in_neg_fft) * x_out_fft) / denom_neg
    
    # Average positive and negative regressors
    w_trial = (ifft(w_pos).real + ifft(w_neg).real) / 2
    
    # Extract ABR window and apply time shift for ANM
    abr_trial = np.concatenate((w_trial[int(t_start*eeg_fs):],
                              w_trial[0:int(t_stop*eeg_fs)]))
    abr_trial = np.roll(abr_trial, int(2.75*eeg_fs/1000))  # 2.75ms shift for ANM
    
    w_preferred_all.append(abr_trial)
    
# Process non-preferred songs
print("\nProcessing non-preferred songs...")
for song_id in nonpreferred_trials:
    print(f"Processing non-preferred song: {song_id}")
    
    # Check if ANM regressor exists
    regressor_file = os.path.join(regressor_root, f"single_{song_id}_proc_anm.hdf5")
    if not os.path.exists(regressor_file):
        print(f"Warning: ANM regressor not found for {song_id}. Skipping...")
        continue
    
    # Load ANM regressor
    regressor_data = read_hdf5(regressor_file)
    x_in_pos = regressor_data['x_in_pos'][0]  # Remove singleton dimension
    x_in_neg = regressor_data['x_in_neg'][0]  # Remove singleton dimension
    
    # Calculate song length from regressor length
    len_eeg = len(x_in_pos)  # Use the actual regressor length
    print(f"  Song duration: {len_eeg / eeg_fs:.2f} seconds")
    
    # Process the single trial for this song
    eeg_file = nonpreferred_trials[song_id]
    print(f"  Processing trial: {os.path.basename(eeg_file)}")
    
    # Load and preprocess EEG data
    eeg_raw = mne.io.read_raw_fif(eeg_file, preload=True, verbose=False)
    
    # Create ABR channels (Plus - Minus for R and L)
    eeg_raw.pick_channels(['Plus_R', 'Minus_R', 'Plus_L', 'Minus_L'])
    data_R = eeg_raw.get_data(picks=['Plus_R'])[0] - eeg_raw.get_data(picks=['Minus_R'])[0]
    data_L = eeg_raw.get_data(picks=['Plus_L'])[0] - eeg_raw.get_data(picks=['Minus_L'])[0]
    data = np.vstack((data_R, data_L))
    data /= 100  # Scale factor
    
    # Apply high-pass filter
    data = butter_highpass_filter(data, eeg_f_hp, eeg_fs)
    
    # Apply notch filter
    notch_freq = np.arange(60, 540, 120)
    notch_width = 5
    for nf in notch_freq:
        bn, an = signal.iirnotch(nf / (eeg_fs / 2.), float(nf) / notch_width)
        data = signal.lfilter(bn, an, data)
    
    # Extract EEG data for analysis (trim to match regressor length)
    x_out = data[:, :len_eeg]
    x_out = np.mean(x_out, axis=0)  # Average across L/R channels
    
    # Use full regressor length (no trimming needed)
    x_in_pos_trimmed = x_in_pos
    x_in_neg_trimmed = x_in_neg
    
    # FFT
    x_in_pos_fft = fft(x_in_pos_trimmed)
    x_in_neg_fft = fft(x_in_neg_trimmed)
    x_out_fft = fft(x_out)
    
    # Deconvolution (TRF estimation)
    denom_pos = x_in_pos_fft * np.conj(x_in_pos_fft)
    denom_neg = x_in_neg_fft * np.conj(x_in_neg_fft)
    
    w_pos = (np.conj(x_in_pos_fft) * x_out_fft) / denom_pos
    w_neg = (np.conj(x_in_neg_fft) * x_out_fft) / denom_neg
    
    # Average positive and negative regressors
    w_trial = (ifft(w_pos).real + ifft(w_neg).real) / 2
    
    # Extract ABR window and apply time shift for ANM
    abr_trial = np.concatenate((w_trial[int(t_start*eeg_fs):],
                              w_trial[0:int(t_stop*eeg_fs)]))
    abr_trial = np.roll(abr_trial, int(2.75*eeg_fs/1000))  # 2.75ms shift for ANM
    
    w_nonpreferred_all.append(abr_trial)
    
# Average across trials
if w_preferred_all:
    w_preferred = np.mean(w_preferred_all, axis=0)
    abr_preferred = butter_bandpass_filter(w_preferred, 1, 1000, eeg_fs, order=1)
else:
    w_preferred = np.zeros(abr_length)
    abr_preferred = np.zeros(abr_length)
    print(f"Warning: No preferred trials processed for {subject}")

if w_nonpreferred_all:
    w_nonpreferred = np.mean(w_nonpreferred_all, axis=0)
    abr_nonpreferred = butter_bandpass_filter(w_nonpreferred, 1, 1000, eeg_fs, order=1)
else:
    w_nonpreferred = np.zeros(abr_length)
    abr_nonpreferred = np.zeros(abr_length)
    print(f"Warning: No non-preferred trials processed for {subject}")

# Save results
output_file = os.path.join(output_root, f'{subject}_music_preference_ABR.hdf5')
write_hdf5(output_file, 
           dict(w_preferred=w_preferred, 
                w_nonpreferred=w_nonpreferred,
                abr_preferred=abr_preferred, 
                abr_nonpreferred=abr_nonpreferred,
                lags=lags,
                preferred_songs=preferred_songs,
                nonpreferred_songs=nonpreferred_songs,
                n_preferred_trials=len(w_preferred_all),
                n_nonpreferred_trials=len(w_nonpreferred_all)), 
           overwrite=True)

print(f"Results saved to: {output_file}")
print(f"Processed {len(w_preferred_all)} preferred trials and {len(w_nonpreferred_all)} non-preferred trials")

# Plot results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(lags, abr_preferred, label='Preferred Music ABR', color='blue')
plt.plot(lags, abr_nonpreferred, label='Non-Preferred Music ABR', color='red')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.title(f'{subject}: Preferred vs Non-Preferred Music ABR')
plt.xlim(-200, 600)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(lags, abr_preferred, label='Preferred Music ABR', color='blue')
plt.plot(lags, abr_nonpreferred, label='Non-Preferred Music ABR', color='red')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.title(f'{subject}: ABR Early Response (Zoomed)')
plt.xlim(-20, 30)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plot_file = os.path.join(output_root, f'{subject}_music_preference_ABR.png')
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
plt.show()

print(f"Plot saved to: {plot_file}")
print("\n=== Analysis Complete ===")