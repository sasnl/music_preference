#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single Subject Click ABR Analysis Script

Derives ABR from click stimuli using cross-correlation analysis in frequency domain.
Adapted for the music preference project structure with .fif files.

Usage: python derive_click_ABR_single_subject.py <subject_id>
Example: python derive_click_ABR_single_subject.py pilot_2
"""

import numpy as np
import scipy.signal as signal
from numpy.fft import fft, ifft
from expyfun.io import write_hdf5, read_wav
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
# EEG param
eeg_n_channel = 2  # total channel of ABR (Plus_R-Minus_R, Plus_L-Minus_L)
eeg_fs = 25000  # eeg sampling frequency (updated to 25kHz)
eeg_f_hp = 1  # high pass cutoff

# Click stimulus parameters
t_click = 60  # click trial length in seconds
click_rate = 40  # click rate in Hz
stim_fs = 48000  # stimulus sampling frequency

# ABR analysis window
t_start = -0.2  # -200ms
t_stop = 0.6   # +600ms
lags = np.arange(start=t_start*1000, stop=t_stop*1000, step=1e3/eeg_fs)

# %% File paths
data_root = './data/'
click_dir = './click_stim/'
output_root = './output/'

# Create output directory if it doesn't exist
os.makedirs(output_root, exist_ok=True)

# %% Get subject from command line argument
if len(sys.argv) != 2:
    print("Usage: python derive_click_ABR_single_subject.py <subject_id>")
    print("Example: python derive_click_ABR_single_subject.py pilot_2")
    sys.exit(1)

subject = sys.argv[1]
print(f"\n=== Processing Click ABR for {subject} ===")

# Check if subject data exists
subject_data_dir = os.path.join(data_root, subject)
if not os.path.exists(subject_data_dir):
    print(f"Error: Data directory for {subject} not found at {subject_data_dir}")
    sys.exit(1)

# Find all click EEG files for this subject
click_files = glob.glob(os.path.join(subject_data_dir, f"{subject}_click_trial*.fif"))
click_files.sort()  # Ensure proper order

if not click_files:
    print(f"Error: No click EEG files found for {subject} in {subject_data_dir}")
    sys.exit(1)

print(f"Found {len(click_files)} click trial files")
for cf in click_files:
    print(f"  {os.path.basename(cf)}")

# %% Load click stimuli and create pulse trains
print("\nLoading click stimuli...")
n_trials = len(click_files)
len_eeg = int(t_click * eeg_fs)

# Initialize pulse train array
x_in = np.zeros((n_trials, len_eeg), dtype=float)

for trial_idx in range(n_trials):
    # Click stimulus files: click000.wav, click001.wav, ..., click004.wav
    click_file = os.path.join(click_dir, f'click{trial_idx:03d}.wav')
    
    if not os.path.exists(click_file):
        print(f"Warning: Click file not found: {click_file}. Skipping trial {trial_idx+1}")
        continue
    
    print(f"  Loading click stimulus: {os.path.basename(click_file)}")
    
    # Load click stimulus
    stim, fs_stim = read_wav(click_file)
    stim_abs = np.abs(stim)
    
    # Find click times (when stimulus goes from 0 to 1)
    click_times = []
    for s in stim_abs:
        click_indices = np.where(np.diff(s) > 0)[0] + 1
        click_times.append(click_indices / float(fs_stim))
    
    # Convert click times to EEG sample indices
    for ct in click_times:
        click_inds = (ct * eeg_fs).astype(int)
        # Only include clicks within the trial duration
        valid_clicks = click_inds[click_inds < len_eeg]
        x_in[trial_idx, valid_clicks] = 1

print(f"Loaded {n_trials} click stimuli")

# %% Process EEG data
print("\nProcessing EEG data...")
x_out = np.zeros((n_trials, len_eeg), dtype=float)

for trial_idx, eeg_file in enumerate(click_files):
    print(f"  Processing trial {trial_idx+1}: {os.path.basename(eeg_file)}")
    
    # Load and preprocess EEG data
    eeg_raw = mne.io.read_raw_fif(eeg_file, preload=True, verbose=False)
    
    # Create ABR channels (Plus - Minus for R and L)
    eeg_raw.pick(['Plus_R', 'Minus_R', 'Plus_L', 'Minus_L'])
    data_R = eeg_raw.get_data(picks=['Plus_R'])[0] - eeg_raw.get_data(picks=['Minus_R'])[0]
    data_L = eeg_raw.get_data(picks=['Plus_L'])[0] - eeg_raw.get_data(picks=['Minus_L'])[0]
    data = np.vstack((data_R, data_L))
    data /= 100  # Scale factor
    
    # Apply high-pass filter
    data = butter_highpass_filter(data, eeg_f_hp, eeg_fs)
    
    # Apply notch filter
    notch_freq = np.arange(60, 540, 180)
    notch_width = 5
    for nf in notch_freq:
        bn, an = signal.iirnotch(nf / (eeg_fs / 2.), float(nf) / notch_width)
        data = signal.lfilter(bn, an, data)
    
    # Extract EEG data for analysis (full trial length)
    eeg_data = data[:, :len_eeg]
    # Average across L/R channels
    x_out[trial_idx, :] = np.mean(eeg_data, axis=0)

print("EEG data preprocessing completed")

# %% Derive ABR using cross-correlation in frequency domain
print("\nDeriving ABR through cross-correlation...")

# FFT
x_in_fft = fft(x_in, axis=-1)
x_out_fft = fft(x_out, axis=-1)

# Cross-correlation in frequency domain for each trial
cc_trials = []
for trial_idx in range(n_trials):
    cc_trial = np.real(ifft(x_out_fft[trial_idx] * np.conj(x_in_fft[trial_idx])))
    cc_trials.append(cc_trial)

# Average across trials
cc_trials = np.array(cc_trials)
abr = np.mean(cc_trials, axis=0)

# Normalize by click rate and trial length
abr /= (click_rate * t_click)

# Extract ABR response window [-200ms, +600ms]
abr_response = np.concatenate((abr[int(t_start*eeg_fs):],
                              abr[0:int(t_stop*eeg_fs)]))

# Apply bandpass filter for final ABR
abr_response_filtered = butter_bandpass_filter(abr_response, 1, 1000, eeg_fs, order=1)

print("ABR derivation completed")

# %% Save results
output_file = os.path.join(output_root, f'{subject}_click_ABR.hdf5')
write_hdf5(output_file, 
           dict(abr_response=abr_response,
                abr_response_filtered=abr_response_filtered,
                lags=lags,
                cc_trials=cc_trials,
                x_in=x_in,
                n_trials=n_trials,
                click_rate=click_rate,
                t_click=t_click,
                eeg_fs=eeg_fs,
                t_start=t_start,
                t_stop=t_stop), 
           overwrite=True)

print(f"Results saved to: {output_file}")
print(f"Processed {n_trials} click trials")

# %% Generate plots
plt.figure(figsize=(15, 10))

# Plot 1: Individual trial cross-correlations
plt.subplot(2, 3, 1)
for trial_idx in range(n_trials):
    trial_abr = np.concatenate((cc_trials[trial_idx][int(t_start*eeg_fs):],
                               cc_trials[trial_idx][0:int(t_stop*eeg_fs)]))
    trial_abr /= (click_rate * t_click)
    plt.plot(lags, trial_abr, alpha=0.6, label=f'Trial {trial_idx+1}')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude')
plt.title(f'{subject}: Individual Trial ABRs')
plt.xlim(-200, 600)
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Average ABR (full window)
plt.subplot(2, 3, 2)
plt.plot(lags, abr_response, 'b-', linewidth=2, label='Raw ABR')
plt.plot(lags, abr_response_filtered, 'r-', linewidth=2, label='Filtered ABR (1-1000Hz)')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (μV)')
plt.title(f'{subject}: Average Click ABR')
plt.xlim(-200, 600)
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: ABR early response (zoomed)
plt.subplot(2, 3, 3)
plt.plot(lags, abr_response, 'b-', linewidth=2, label='Raw ABR')
plt.plot(lags, abr_response_filtered, 'r-', linewidth=2, label='Filtered ABR')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (μV)')
plt.title(f'{subject}: ABR Early Response (Zoomed)')
plt.xlim(-20, 60)
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Click stimulus example (first trial)
plt.subplot(2, 3, 4)
time_vec = np.arange(len_eeg) / eeg_fs
plt.plot(time_vec[:int(0.1*eeg_fs)], x_in[0, :int(0.1*eeg_fs)])  # First 100ms
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Click Stimulus (First 100ms, Trial 1)')
plt.grid(True, alpha=0.3)

# Plot 5: EEG response example (first trial)
plt.subplot(2, 3, 5)
plt.plot(time_vec[:int(0.1*eeg_fs)], x_out[0, :int(0.1*eeg_fs)])  # First 100ms
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (μV)')
plt.title('EEG Response (First 100ms, Trial 1)')
plt.grid(True, alpha=0.3)

# Plot 6: ABR peaks analysis
plt.subplot(2, 3, 6)
# Focus on Wave I-V region (0-10ms)
wave_mask = (lags >= 0) & (lags <= 10)
wave_region = abr_response_filtered[wave_mask]
wave_lags = lags[wave_mask]
plt.plot(wave_lags, wave_region, 'r-', linewidth=2)
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (μV)')
plt.title(f'{subject}: ABR Waves I-V Region')
plt.xlim(0, 10)
plt.grid(True, alpha=0.3)

# Mark potential peaks
from scipy.signal import find_peaks
peaks, _ = find_peaks(wave_region, height=np.std(wave_region), distance=int(0.5*eeg_fs/1000))  # At least 0.5ms apart
if len(peaks) > 0:
    plt.plot(wave_lags[peaks], wave_region[peaks], 'ro', markersize=8)
    for i, peak in enumerate(peaks):
        plt.annotate(f'{wave_lags[peak]:.1f}ms', 
                    (wave_lags[peak], wave_region[peak]), 
                    xytext=(5, 5), textcoords='offset points')

plt.tight_layout()
plot_file = os.path.join(output_root, f'{subject}_click_ABR.png')
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
plt.show()

print(f"Plot saved to: {plot_file}")

# %% Summary statistics
print(f"\n=== ABR Analysis Summary for {subject} ===")
print(f"Number of trials processed: {n_trials}")
print(f"Trial duration: {t_click} seconds")
print(f"Click rate: {click_rate} Hz")
print(f"Total clicks per trial: ~{int(click_rate * t_click)}")
print(f"EEG sampling rate: {eeg_fs} Hz")
print(f"ABR time window: {t_start*1000:.0f} to {t_stop*1000:.0f} ms")
print(f"Peak ABR amplitude: {np.max(np.abs(abr_response_filtered)):.3f} μV")
print(f"RMS ABR amplitude: {np.sqrt(np.mean(abr_response_filtered**2)):.3f} μV")

# Peak analysis in Wave I-V region
wave_mask = (lags >= 1) & (lags <= 8)  # Typical ABR wave region
wave_region = abr_response_filtered[wave_mask]
wave_lags = lags[wave_mask]

if len(wave_region) > 0:
    max_idx = np.argmax(np.abs(wave_region))
    peak_latency = wave_lags[max_idx]
    peak_amplitude = wave_region[max_idx]
    print(f"Dominant peak latency: {peak_latency:.2f} ms")
    print(f"Dominant peak amplitude: {peak_amplitude:.3f} μV")

print("\n=== Analysis Complete ===")