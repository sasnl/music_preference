#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABR (Auditory Brainstem Response) Derivation Module

This module provides functionality to derive ABR responses from click stimuli
using cross-correlation analysis in the frequency domain.

Author: Based on original notebook by tshan@urmc-sh.rochester.edu
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
import scipy.signal as signal
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
from expyfun.io import read_wav
import mne


def butter_highpass(cutoff: float, fs: float, order: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a high-pass Butterworth filter.
    
    Args:
        cutoff: Cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
        
    Returns:
        Tuple of (b, a) filter coefficients
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 1) -> np.ndarray:
    """
    Apply high-pass Butterworth filter to data.
    
    Args:
        data: Input data array
        cutoff: Cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
        
    Returns:
        Filtered data array
    """
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


def butter_lowpass(cutoff: float, fs: float, order: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a low-pass Butterworth filter.
    
    Args:
        cutoff: Cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
        
    Returns:
        Tuple of (b, a) filter coefficients
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 1) -> np.ndarray:
    """
    Apply low-pass Butterworth filter to data.
    
    Args:
        data: Input data array
        cutoff: Cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
        
    Returns:
        Filtered data array
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design a band-pass Butterworth filter.
    
    Args:
        lowcut: Lower cutoff frequency in Hz
        highcut: Upper cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
        
    Returns:
        Tuple of (b, a) filter coefficients
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 1) -> np.ndarray:
    """
    Apply band-pass Butterworth filter to data.
    
    Args:
        data: Input data array
        lowcut: Lower cutoff frequency in Hz
        highcut: Upper cutoff frequency in Hz
        fs: Sampling frequency in Hz
        order: Filter order
        
    Returns:
        Filtered data array
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


def validate_parameters(params: Dict[str, Any]) -> None:
    """
    Validate input parameters.
    
    Args:
        params: Dictionary of parameters to validate
        
    Raises:
        ValueError: If any parameter is invalid
    """
    if params['eeg_fs'] <= 0:
        raise ValueError("EEG sampling frequency must be positive")
    
    if params['eeg_f_hp'] <= 0 or params['eeg_f_hp'] >= params['eeg_fs'] / 2:
        raise ValueError("High-pass cutoff must be between 0 and Nyquist frequency")
    
    if params['t_click'] <= 0:
        raise ValueError("Click trial length must be positive")
    
    if params['click_rate'] <= 0:
        raise ValueError("Click rate must be positive")
    
    if params['stim_fs'] <= 0:
        raise ValueError("Stimulus sampling frequency must be positive")


def load_eeg_data(eeg_file: str) -> Tuple[mne.io.Raw, np.ndarray, Dict[str, int]]:
    """
    Load EEG data from BrainVision file.
    
    Args:
        eeg_file: Path to .vhdr file
        
    Returns:
        Tuple of (raw_eeg, events, event_dict)
    """
    logging.info(f"Loading EEG data from {eeg_file}")
    
    if not os.path.exists(eeg_file):
        raise FileNotFoundError(f"EEG file not found: {eeg_file}")
    
    eeg_raw = mne.io.read_raw_brainvision(eeg_file, preload=True)
    events, event_dict = mne.events_from_annotations(eeg_raw)
    
    logging.info(f"Loaded EEG data with {len(events)} events")
    return eeg_raw, events, event_dict


def create_abr_channels(eeg_raw: mne.io.Raw) -> mne.io.Raw:
    """
    Create referenced ABR channels from raw EEG data.
    
    Args:
        eeg_raw: Raw EEG data with ABR channels
        
    Returns:
        Referenced ABR data
    """
    logging.info("Creating referenced ABR channels")
    
    # Select ABR channels
    abr_channels = ['Plus_R', 'Minus_R', 'Plus_L', 'Minus_L']
    eeg_raw.pick_channels(abr_channels)
    
    # Create referenced channels
    # Right ear: Plus_R - Minus_R
    # Left ear: Plus_L - Minus_L
    data_R = eeg_raw.get_data(picks=eeg_raw.ch_names[0]) - eeg_raw.get_data(picks=eeg_raw.ch_names[1])
    data_L = eeg_raw.get_data(picks=eeg_raw.ch_names[2]) - eeg_raw.get_data(picks=eeg_raw.ch_names[3])
    data = np.vstack((data_R, data_L))
    data /= 100  # Scale data to microvolts
    
    # Create info for RawArray
    info = mne.create_info(ch_names=["EP1", "EP2"], sfreq=eeg_raw.info['sfreq'], ch_types='eeg')
    eeg_raw_ref = mne.io.RawArray(data, info)
    
    logging.info("Created referenced ABR channels")
    return eeg_raw_ref


def preprocess_eeg(eeg_raw: mne.io.Raw, eeg_fs: float, eeg_f_hp: float) -> mne.io.Raw:
    """
    Preprocess EEG data with filtering.
    
    Args:
        eeg_raw: Raw EEG data
        eeg_fs: EEG sampling frequency
        eeg_f_hp: High-pass cutoff frequency
        
    Returns:
        Preprocessed EEG data
    """
    logging.info("Preprocessing EEG data")
    
    # High-pass filter
    eeg_raw._data = butter_highpass_filter(eeg_raw._data, eeg_f_hp, eeg_fs)
    
    # Notch filter
    notch_freq = np.arange(60, 540, 180)
    notch_width = 5
    for nf in notch_freq:
        bn, an = signal.iirnotch(nf / (eeg_fs / 2.), float(nf) / notch_width)
        eeg_raw._data = signal.lfilter(bn, an, eeg_raw._data)
    
    logging.info("EEG preprocessing completed")
    return eeg_raw


def epoch_data(eeg_raw: mne.io.Raw, events: np.ndarray, t_click: float, stim_fs: float) -> np.ndarray:
    """
    Extract epochs around click events.
    
    Args:
        eeg_raw: Preprocessed EEG data
        events: Event array
        t_click: Click trial length in seconds
        stim_fs: Stimulus sampling frequency
        
    Returns:
        Epoch data array
    """
    logging.info("Epoching EEG data")
    
    epochs_click = mne.Epochs(eeg_raw, events, tmin=0,
                              tmax=(t_click - 1/stim_fs + 1),
                              event_id=1, baseline=None,
                              preload=True, proj=False)
    epoch_click = epochs_click.get_data()
    
    logging.info(f"Created {len(epoch_click)} epochs")
    return epoch_click


def load_click_stimuli(click_dir: str, n_epoch_click: int, t_click: float, eeg_fs: float) -> np.ndarray:
    """
    Load click WAV files and convert to pulse trains.
    
    Args:
        click_dir: Directory containing click WAV files
        n_epoch_click: Number of click epochs
        t_click: Click trial length in seconds
        eeg_fs: EEG sampling frequency
        
    Returns:
        Pulse train array
    """
    logging.info(f"Loading click stimuli from {click_dir}")
    
    x_in = np.zeros((n_epoch_click, int(t_click * eeg_fs)), dtype=float)
    
    for ei in range(n_epoch_click):
        click_file = os.path.join(click_dir, f'click{ei:03d}.wav')
        
        if not os.path.exists(click_file):
            raise FileNotFoundError(f"Click file not found: {click_file}")
        
        stim, fs_stim = read_wav(click_file)
        stim_abs = np.abs(stim)
        click_times = [(np.where(np.diff(s) > 0)[0] + 1) / float(fs_stim) for s in stim_abs]
        click_inds = [(ct * eeg_fs).astype(int) for ct in click_times]
        x_in[ei, click_inds] = 1  # Generate click train as x_in
    
    logging.info(f"Loaded {n_epoch_click} click stimuli")
    return x_in


def derive_abr_response(x_in: np.ndarray, epoch_click: np.ndarray, eeg_fs: float, 
                       t_click: float, click_rate: float, t_start: float = -200e-3, 
                       t_stop: float = 600e-3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Derive ABR response using cross-correlation in frequency domain.
    
    Args:
        x_in: Input pulse train array
        epoch_click: Epoch data array
        eeg_fs: EEG sampling frequency
        t_click: Click trial length
        click_rate: Click rate
        t_start: Start time for ABR response (seconds)
        t_stop: Stop time for ABR response (seconds)
        
    Returns:
        Tuple of (abr_response, lags)
    """
    logging.info("Deriving ABR through cross-correlation")
    
    # Get x_out (EEG response)
    len_eeg = int(eeg_fs * t_click)
    x_out = np.zeros((len(x_in), 2, len_eeg))
    
    for i in range(len(x_in)):
        x_out_i = epoch_click[i, :, 0:int(eeg_fs*t_click)]
        x_out[i, :, :] = mne.filter.resample(x_out_i, eeg_fs, eeg_fs)
    
    x_out = np.mean(x_out, axis=1)  # Average the two channels
    
    # FFT
    x_in_fft = fft(x_in, axis=-1)
    x_out_fft = fft(x_out, axis=-1)
    
    # Cross-correlation in frequency domain
    cc = np.real(ifft(x_out_fft * np.conj(x_in_fft)))
    abr = np.mean(cc, axis=0)  # Average across trials
    abr /= (click_rate * t_click)  # Real unit value
    
    # Concatenate ABR response as [-200, 600] ms lag range
    abr_response = np.concatenate((abr[int(t_start*eeg_fs):],
                                   abr[0:int(t_stop*eeg_fs)]))
    
    # Generate time vector
    lags = np.arange(start=t_start*1000, stop=t_stop*1000, step=1e3/eeg_fs)
    
    logging.info("ABR derivation completed")
    return abr_response, lags


def plot_abr_response(abr_response: np.ndarray, lags: np.ndarray, 
                     output_file: Optional[str] = None) -> None:
    """
    Plot ABR response.
    
    Args:
        abr_response: ABR response array
        lags: Time lags array
        output_file: Optional file path to save plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(lags, abr_response)
    plt.xlim([-20, 60])
    plt.xlabel('Lag (ms)')
    plt.ylabel('Amplitude (μV)')
    plt.title('ABR Response')
    plt.grid(True, alpha=0.3)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logging.info(f"ABR plot saved to {output_file}")
    
    plt.show()


def save_results(abr_response: np.ndarray, lags: np.ndarray, output_dir: str, 
                subject_id: str = "subject") -> None:
    """
    Save ABR results to files.
    
    Args:
        abr_response: ABR response array
        lags: Time lags array
        output_dir: Output directory
        subject_id: Subject identifier
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as numpy arrays
    np.save(os.path.join(output_dir, f"{subject_id}_abr_response.npy"), abr_response)
    np.save(os.path.join(output_dir, f"{subject_id}_lags.npy"), lags)
    
    # Save as text file for easy viewing
    results_file = os.path.join(output_dir, f"{subject_id}_abr_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"ABR Results for {subject_id}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Response length: {len(abr_response)} samples\n")
        f.write(f"Time range: {lags[0]:.1f} to {lags[-1]:.1f} ms\n")
        f.write(f"Peak amplitude: {np.max(np.abs(abr_response)):.3f} μV\n")
        f.write(f"RMS amplitude: {np.sqrt(np.mean(abr_response**2)):.3f} μV\n")
    
    logging.info(f"ABR results saved to {output_dir}")


def derive_abr(eeg_file: str, output_dir: str, click_dir: str = None,
               eeg_fs: float = 10000, eeg_f_hp: float = 1.0,
               t_click: float = 60, click_rate: float = 40, stim_fs: float = 48000,
               n_epoch_click: int = 5, plot_results: bool = False,
               subject_id: str = "subject", t_start: float = -200e-3, 
               t_stop: float = 600e-3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Main function to derive ABR from click stimuli.
    
    Args:
        eeg_file: Path to EEG .vhdr file
        output_dir: Output directory for results
        click_dir: Directory containing click WAV files (defaults to click_stim/)
        eeg_fs: EEG sampling frequency (default: 10000 Hz)
        eeg_f_hp: High-pass cutoff frequency (default: 1.0 Hz)
        t_click: Click trial length in seconds (default: 60)
        click_rate: Click rate in Hz (default: 40)
        stim_fs: Stimulus sampling frequency (default: 48000 Hz)
        n_epoch_click: Number of click epochs to process (default: 5)
        plot_results: Whether to generate plots (default: False)
        subject_id: Subject identifier (default: "subject")
        t_start: Start time for ABR response in seconds (default: -200e-3)
        t_stop: Stop time for ABR response in seconds (default: 600e-3)
        
    Returns:
        Tuple of (abr_response, lags)
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Validate parameters
    params = {
        'eeg_fs': eeg_fs,
        'eeg_f_hp': eeg_f_hp,
        't_click': t_click,
        'click_rate': click_rate,
        'stim_fs': stim_fs
    }
    validate_parameters(params)
    
    # Set default click directory if not provided
    if click_dir is None:
        click_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'click_stim')
    
    logging.info(f"Starting ABR derivation for {subject_id}")
    logging.info(f"EEG file: {eeg_file}")
    logging.info(f"Click directory: {click_dir}")
    logging.info(f"Output directory: {output_dir}")
    
    try:
        # Load EEG data
        eeg_raw, events, event_dict = load_eeg_data(eeg_file)
        
        # Create ABR channels
        eeg_raw_ref = create_abr_channels(eeg_raw)
        
        # Preprocess EEG
        eeg_raw_ref = preprocess_eeg(eeg_raw_ref, eeg_fs, eeg_f_hp)
        
        # Epoch data
        epoch_click = epoch_data(eeg_raw_ref, events, t_click, stim_fs)
        
        # Load click stimuli
        x_in = load_click_stimuli(click_dir, n_epoch_click, t_click, eeg_fs)
        
        # Derive ABR response
        abr_response, lags = derive_abr_response(x_in, epoch_click, eeg_fs, t_click, click_rate, t_start, t_stop)
        
        # Save results
        save_results(abr_response, lags, output_dir, subject_id)
        
        # Generate plot if requested
        if plot_results:
            plot_file = os.path.join(output_dir, f"{subject_id}_abr_plot.png")
            plot_abr_response(abr_response, lags, plot_file)
        
        logging.info(f"ABR derivation completed successfully for {subject_id}")
        return abr_response, lags
        
    except Exception as e:
        logging.error(f"Error during ABR derivation: {str(e)}")
        raise


def main():
    """Command-line interface for ABR derivation."""
    parser = argparse.ArgumentParser(description='Derive ABR from click stimuli')
    
    parser.add_argument('eeg_file', help='Path to EEG .vhdr file')
    parser.add_argument('output_dir', help='Output directory for results')
    parser.add_argument('--click_dir', help='Directory containing click WAV files')
    parser.add_argument('--eeg_fs', type=float, default=10000, help='EEG sampling frequency (default: 10000)')
    parser.add_argument('--eeg_f_hp', type=float, default=1.0, help='High-pass cutoff frequency (default: 1.0)')
    parser.add_argument('--t_click', type=float, default=60, help='Click trial length in seconds (default: 60)')
    parser.add_argument('--click_rate', type=float, default=40, help='Click rate in Hz (default: 40)')
    parser.add_argument('--stim_fs', type=float, default=48000, help='Stimulus sampling frequency (default: 48000)')
    parser.add_argument('--n_epoch_click', type=int, default=5, help='Number of click epochs (default: 5)')
    parser.add_argument('--subject_id', default='subject', help='Subject identifier (default: subject)')
    parser.add_argument('--t_start', type=float, default=-200e-3, help='Start time for ABR response in seconds (default: -0.2)')
    parser.add_argument('--t_stop', type=float, default=600e-3, help='Stop time for ABR response in seconds (default: 0.6)')
    parser.add_argument('--plot', action='store_true', help='Generate ABR plots')
    
    args = parser.parse_args()
    
    try:
        abr_response, lags = derive_abr(
            eeg_file=args.eeg_file,
            output_dir=args.output_dir,
            click_dir=args.click_dir,
            eeg_fs=args.eeg_fs,
            eeg_f_hp=args.eeg_f_hp,
            t_click=args.t_click,
            click_rate=args.click_rate,
            stim_fs=args.stim_fs,
            n_epoch_click=args.n_epoch_click,
            plot_results=args.plot,
            subject_id=args.subject_id,
            t_start=args.t_start,
            t_stop=args.t_stop
        )
        
        print(f"ABR derivation completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 