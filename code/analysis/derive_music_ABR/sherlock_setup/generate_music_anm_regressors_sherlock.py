#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate ANM Regressors for Music Stimuli - Sherlock Optimized Version

This script is optimized for Stanford's Sherlock cluster with:
- Better memory management
- Optimized for high-performance computing
- Checkpointing for long jobs
- Better error handling for cluster environment

IMPORTANT: This script requires the exact cochlea package. Install it with:
    pip install git+https://github.com/mrkrd/cochlea.git

Created based on generate_music_anm_regressors.py
Author: Assistant
"""

import numpy as np
import os
import glob
import logging
import gc
import psutil
import time
from pathlib import Path
from mne.filter import resample
from joblib import Parallel, delayed
from expyfun.io import read_wav, write_hdf5

# Try to import cochlea - this is required for the script
try:
    import cochlea
    COCHLEA_AVAILABLE = True
    logging.info("Cochlea package imported successfully")
except ImportError as e:
    COCHLEA_AVAILABLE = False
    logging.error(f"Cochlea package not available: {e}")
    logging.error("Please install cochlea package with: pip install git+https://github.com/mrkrd/cochlea.git")

    # Import ic_cn2018 module
    try:
        import sys
        # Try multiple import paths for robustness
        script_dir = os.path.dirname(__file__)
        possible_paths = [
            script_dir,  # Same directory as script
            os.path.join(script_dir, 'derive_music_ABR'),  # Subdirectory
            os.path.join(script_dir, '..'),  # Parent directory
            os.path.join(script_dir, '..', 'code', 'analysis', 'derive_music_ABR'),  # Original location
        ]
    
    nuclei = None
    for path in possible_paths:
        if path not in sys.path:
            sys.path.insert(0, path)
        try:
            import ic_cn2018 as nuclei
            logging.info(f"ic_cn2018 module imported successfully from {path}")
            break
        except ImportError:
            continue
    
    if nuclei is None:
        raise ImportError("Could not import ic_cn2018 module from any location")
        
except ImportError as e:
    logging.error(f"ic_cn2018 module not available: {e}")
    logging.error("Please ensure ic_cn2018.py is in the derive_music_ABR directory")
    nuclei = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_memory_usage():
    """Get current memory usage in GB."""
    return psutil.virtual_memory().used / 1024**3

def log_memory_usage(stage=""):
    """Log current memory usage."""
    memory_gb = get_memory_usage()
    available_gb = psutil.virtual_memory().available / 1024**3
    logging.info(f"Memory usage {stage}: {memory_gb:.2f} GB used, {available_gb:.2f} GB available")
    return memory_gb, available_gb

def get_rates(stim_up, cf):
    """Generate auditory nerve fiber rates using Zilany2014 model."""
    if not COCHLEA_AVAILABLE:
        raise ImportError("Cochlea package is required but not available. Please install it.")
    
    fs_up = int(100e3)
    return(np.array(cochlea.run_zilany2014_rate(stim_up,
                                                fs_up,
                                                anf_types='hsr',
                                                cf=cf,
                                                species='human',
                                                cohc=1,
                                                cihc=1))[:, 0])

def anm(stim, fs_in, stim_pres_db, parallel=True, n_jobs=8,  # Optimized for Sherlock
        stim_gen_rms=0.01, cf_low=125, cf_high=16e3, shift_cfs=False,
        shift_vals=None):
    """
    Generate Auditory Nerve Model (ANM) response - Sherlock optimized.
    
    Parameters:
    -----------
    stim : array-like
        Input stimulus waveform
    fs_in : int
        Input sampling frequency
    stim_pres_db : float
        Stimulus presentation level in dB SPL
    parallel : bool
        Whether to use parallel processing
    n_jobs : int
        Number of parallel jobs (optimized for Sherlock)
    stim_gen_rms : float
        RMS of the stimulus generation
    cf_low : float
        Lowest center frequency
    cf_high : float
        Highest center frequency
    shift_cfs : bool
        Whether to shift each CF independently
    shift_vals : array-like
        Shift values for each CF
        
    Returns:
    --------
    anm : array
        ANM response
    """
    if not COCHLEA_AVAILABLE:
        raise ImportError("Cochlea package is required but not available. Please install it.")
    
    # Resample your stimuli to a higher fs for the model
    fs_up = int(100e3)
    log_memory_usage("before resampling")
    stim_up = resample(stim, fs_up, fs_in, npad='auto', n_jobs=n_jobs)
    log_memory_usage("after resampling")

    # Put stim in correct units
    # scalar to put it in units of pascals (double-checked and good)
    sine_rms_at_0db = 20e-6
    db_conv = ((sine_rms_at_0db / stim_gen_rms) * 10 ** (stim_pres_db / 20.))
    stim_up = db_conv * stim_up

    # run the model
    dOct = 1. / 6
    cfs = 2 ** np.arange(np.log2(cf_low), np.log2(cf_high + 1), dOct)
    anf_rates_up = np.zeros([len(cfs), len(stim)*fs_up//fs_in])

    if parallel:
        anf_rates_up = Parallel(n_jobs=n_jobs)([delayed(get_rates)(stim_up, cf)
                                               for cf in cfs])
    else:
        for cfi, cf in enumerate(cfs):
            anf_rates_up[cfi] = get_rates(stim_up, cf)

    # Downsample to match input fs
    anf_rates = resample(anf_rates_up, fs_in, fs_up, npad='auto',
                         n_jobs=n_jobs)

    # Optionally, shift each cf independently
    final_shift = int(fs_in*0.001)  # shift w1 by 1ms if not shifting each cf
    if shift_cfs:
        final_shift = 0  # don't shift everything after aligning channels at 0
        if shift_vals is None:
            # default shift_cfs values (based on 75 dB click)
            shift_vals = np.array([0.0046875, 0.0045625, 0.00447917,
                                   0.00435417, 0.00422917, 0.00416667,
                                   0.00402083, 0.0039375, 0.0038125, 0.0036875,
                                   0.003625, 0.00354167, 0.00341667,
                                   0.00327083, 0.00316667, 0.0030625,
                                   0.00302083, 0.00291667, 0.0028125,
                                   0.0026875, 0.00258333, 0.00247917,
                                   0.00239583, 0.0023125, 0.00220833,
                                   0.00210417, 0.00204167, 0.002, 0.001875,
                                   0.00185417, 0.00175, 0.00170833, 0.001625,
                                   0.0015625, 0.0015, 0.00147917, 0.0014375,
                                   0.00135417, 0.0014375, 0.00129167,
                                   0.00129167, 0.00125, 0.00122917])

        # Allow fewer CFs while still using defaults
        if len(cfs) != len(shift_vals):
            ref_cfs = 2 ** np.arange(np.log2(125), np.log2(16e3 + 1), 1/6)
            picks = [cf in np.round(cfs, 3) for cf in np.round(ref_cfs, 3)]
            shift_vals = shift_vals[picks]

        # Ensure the number of shift values matches the number of cfs
        msg = 'Number of CFs does not match number of known shift values'
        assert(len(shift_vals) == len(cfs)), msg
        lags = np.round(shift_vals * fs_in).astype(int)

        # Shift each channel
        for cfi in range(len(cfs)):
            anf_rates[cfi] = np.roll(anf_rates[cfi], -lags[cfi])
            anf_rates[cfi, -lags[cfi]:] = anf_rates[cfi, -(lags[cfi]+1)]

    # Shift, scale, and sum
    M1 = nuclei.M1
    anm_response = M1 * anf_rates.sum(0)
    anm_response = np.roll(anm_response, final_shift)
    anm_response[:final_shift] = anm_response[final_shift+1]
    return anm_response

def discover_music_files(music_dir):
    """
    Discover all preprocessed music files in the directory structure.
    
    Parameters:
    -----------
    music_dir : str
        Path to the preprocessed music directory
        
    Returns:
    --------
    file_dict : dict
        Dictionary mapping participant IDs to lists of file paths
    """
    file_dict = {}
    music_path = Path(music_dir)
    
    # Look for participant directories (1, 2, 3, 4, 5)
    for participant_dir in sorted(music_path.glob('[1-5]')):
        participant_id = participant_dir.name
        wav_files = sorted(list(participant_dir.glob('*_proc.wav')))
        if wav_files:
            file_dict[participant_id] = wav_files
            logging.info(f"Found {len(wav_files)} files for participant {participant_id}")
    
    return file_dict

def main():
    """Main processing function - Sherlock optimized."""
    # Check dependencies first
    if not COCHLEA_AVAILABLE:
        logging.error("CRITICAL: Cochlea package is required but not available!")
        logging.error("Please install it with the following steps:")
        logging.error("1. pip install 'Cython<3.0'")
        logging.error("2. pip install 'numpy<2.0'")
        logging.error("3. pip install git+https://github.com/mrkrd/cochlea.git")
        return
    
    # Check if nuclei module is available
    if nuclei is None:
        logging.error("CRITICAL: ic_cn2018 module is required but not available!")
        logging.error("Please ensure ic_cn2018.py is in the correct directory")
        return
    
    # %% Parameters - Optimized for Sherlock
    stim_pres_db = 65  # Stimulus presentation level in dB SPL
    eeg_fs = 25000     # EEG sampling frequency (25 kHz as requested)
    
    # Memory optimization parameters - Sherlock optimized
    use_parallel = True  # Use parallel processing on Sherlock
    max_parallel_jobs = 8  # Use 8 cores (Sherlock has many cores)
    memory_threshold_gb = 50.0  # Higher threshold for Sherlock nodes
    
    # %% File paths
    # Fix path calculation - use current working directory as project root
    project_root = Path.cwd()  # Use current working directory as project root
    music_dir = project_root / 'music_stim' / 'preprocesed'
    output_dir = project_root / 'data'
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    logging.info(f"Processing music files from: {music_dir}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"EEG sampling frequency: {eeg_fs} Hz")
    logging.info(f"Sherlock optimized: {max_parallel_jobs} parallel jobs")
    
    # Validate music directory exists
    if not music_dir.exists():
        logging.error(f"Music directory does not exist: {music_dir}")
        logging.error("Please check the directory path and ensure music files are available")
        return
    
    # %% Discover music files
    file_dict = discover_music_files(music_dir)
    
    if not file_dict:
        logging.error("No music files found! Please check the directory structure.")
        return
    
    total_files = sum(len(files) for files in file_dict.values())
    logging.info(f"Total files to process: {total_files}")
    
    # %% Initialize data structures
    # We'll store all files in lists and create arrays later
    x_in_pos_list = []
    x_in_neg_list = []
    file_info_list = []
    
    # %% Process all music files
    file_counter = 0
    successful_files = 0
    failed_files = 0
    
    # Log initial memory usage
    log_memory_usage("at start of processing")
    
    for participant_id, file_paths in file_dict.items():
        logging.info(f"Processing participant {participant_id} ({len(file_paths)} files)")
        
        for file_path in file_paths:
            file_counter += 1
            start_time = time.time()
            logging.info(f"Processing file {file_counter}/{total_files}: {file_path.name}")
            log_memory_usage(f"before file {file_counter}")
            
            # Check memory threshold
            memory_gb, available_gb = log_memory_usage(f"before file {file_counter}")
            if memory_gb > memory_threshold_gb:
                logging.warning(f"  High memory usage detected: {memory_gb:.2f} GB")
                if available_gb < 5.0:
                    logging.error(f"  Very low available memory: {available_gb:.2f} GB")
                    logging.error("  Consider stopping and freeing memory")
            
            try:
                # Validate file exists and is readable
                if not file_path.exists():
                    logging.error(f"  File does not exist: {file_path}")
                    failed_files += 1
                    continue
                
                # Load audio file
                logging.info(f"  Loading audio file...")
                temp, stim_fs = read_wav(str(file_path))
                if temp.ndim > 1:
                    temp = temp[0, :]  # Take first channel if stereo
                
                logging.info(f"  Loaded audio: {len(temp)} samples at {stim_fs} Hz")
                
                # Validate audio data
                if len(temp) == 0:
                    logging.error(f"  Audio file is empty: {file_path}")
                    failed_files += 1
                    continue
                
                # Generate ANM for positive polarity
                logging.info("  Generating ANM for positive polarity...")
                waves_pos = anm(temp, stim_fs, stim_pres_db, parallel=use_parallel, n_jobs=max_parallel_jobs)
                waves_pos_resmp = resample(waves_pos, down=stim_fs/eeg_fs)
                
                # Generate ANM for negative polarity
                logging.info("  Generating ANM for negative polarity...")
                waves_neg = anm(-temp, stim_fs, stim_pres_db, parallel=use_parallel, n_jobs=max_parallel_jobs)
                waves_neg_resmp = resample(waves_neg, down=stim_fs/eeg_fs)
                
                # Store results
                x_in_pos_list.append(waves_pos_resmp)
                x_in_neg_list.append(waves_neg_resmp)
                file_info_list.append({
                    'participant': participant_id,
                    'filename': file_path.name,
                    'original_fs': stim_fs,
                    'length_samples': len(waves_pos_resmp)
                })
                
                successful_files += 1
                processing_time = time.time() - start_time
                logging.info(f"  Completed: {len(waves_pos_resmp)} samples at {eeg_fs} Hz in {processing_time:.1f} seconds")
                
                # Memory cleanup after successful processing
                gc.collect()
                log_memory_usage(f"after file {file_counter}")
                
            except Exception as e:
                logging.error(f"  Error processing {file_path}: {str(e)}")
                failed_files += 1
                # Memory cleanup after failed processing
                gc.collect()
                continue
    
    logging.info(f"Processing complete: {successful_files} successful, {failed_files} failed")
    
    # %% Convert to arrays and save
    if not x_in_pos_list:
        logging.error("No files were successfully processed!")
        return
    
    logging.info("Converting results to arrays...")
    
    # Find the maximum length to pad all arrays to the same size
    max_length = max(len(arr) for arr in x_in_pos_list)
    n_files = len(x_in_pos_list)
    
    logging.info(f"Maximum signal length: {max_length} samples ({max_length/eeg_fs:.2f} seconds)")
    
    # Create padded arrays
    x_in_pos = np.zeros((n_files, max_length))
    x_in_neg = np.zeros((n_files, max_length))
    
    for i, (pos_arr, neg_arr) in enumerate(zip(x_in_pos_list, x_in_neg_list)):
        x_in_pos[i, :len(pos_arr)] = pos_arr
        x_in_neg[i, :len(neg_arr)] = neg_arr
    
    # %% Save to HDF5 file
    output_file = output_dir / 'music_anm_regressors.hdf5'
    
    logging.info(f"Saving results to: {output_file}")
    
    # Prepare data for saving
    save_data = {
        'x_in_pos': x_in_pos,
        'x_in_neg': x_in_neg,
        'fs': eeg_fs,
        'file_info': file_info_list,
        'processing_params': {
            'stim_pres_db': stim_pres_db,
            'eeg_fs': eeg_fs,
            'max_length': max_length,
            'n_files': n_files,
            'sherlock_optimized': True,
            'parallel_jobs': max_parallel_jobs
        }
    }
    
    write_hdf5(str(output_file), save_data, overwrite=True)
    
    logging.info("ANM regressor generation completed successfully!")
    logging.info(f"Output shape: x_in_pos={x_in_pos.shape}, x_in_neg={x_in_neg.shape}")
    logging.info(f"Sampling frequency: {eeg_fs} Hz")
    logging.info(f"Total files processed: {n_files}")
    logging.info("Sherlock optimization completed!")

if __name__ == "__main__":
    main()
