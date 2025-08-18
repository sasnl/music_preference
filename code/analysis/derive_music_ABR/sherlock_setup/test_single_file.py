#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Single File ANM Processing - Sherlock Test Script

This script processes a single music file to test the ANM pipeline
before running the full batch processing on Sherlock.

Author: Assistant
"""

import numpy as np
import os
import sys
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def find_test_file():
    """Find a single test file to process."""
    project_root = Path.cwd()
    music_dir = project_root / 'music_stim' / 'preprocesed'
    
    # Look for any preprocessed music file
    for participant_dir in sorted(music_dir.glob('[1-5]')):
        wav_files = list(participant_dir.glob('*_proc.wav'))
        if wav_files:
            return wav_files[0]  # Return first file found
    
    return None

def test_imports():
    """Test if all required modules can be imported."""
    logging.info("Testing module imports...")
    
    # Test cochlea import
    try:
        import cochlea
        logging.info("✓ Cochlea package imported successfully")
        cochlea_available = True
    except ImportError as e:
        logging.error(f"✗ Cochlea package import failed: {e}")
        cochlea_available = False
    
    # Test ic_cn2018 import
    try:
        # Add parent directory to path to find ic_cn2018
        script_dir = Path(__file__).parent
        parent_dir = script_dir.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        
        import ic_cn2018
        logging.info("✓ ic_cn2018 module imported successfully")
        nuclei_available = True
    except ImportError as e:
        logging.error(f"✗ ic_cn2018 module import failed: {e}")
        nuclei_available = False
    
    # Test other required modules
    required_modules = ['numpy', 'mne', 'joblib', 'expyfun']
    for module_name in required_modules:
        try:
            __import__(module_name)
            logging.info(f"✓ {module_name} imported successfully")
        except ImportError as e:
            logging.error(f"✗ {module_name} import failed: {e}")
    
    return cochlea_available and nuclei_available

def test_single_file_processing():
    """Test processing of a single music file."""
    logging.info("Starting single file test...")
    
    # Test imports first
    if not test_imports():
        logging.error("Import test failed - cannot proceed with file processing")
        return False
    
    # Find a test file
    test_file = find_test_file()
    if test_file is None:
        logging.error("No test file found in music_stim/preprocesed/")
        return False
    
    logging.info(f"Testing with file: {test_file}")
    
    try:
        # Import required modules
        from expyfun.io import read_wav
        from mne.filter import resample
        import cochlea
        import ic_cn2018 as nuclei
        
        # Test parameters
        stim_pres_db = 65
        eeg_fs = 25000
        
        # Load test file
        logging.info("Loading audio file...")
        start_time = time.time()
        temp, stim_fs = read_wav(str(test_file))
        if temp.ndim > 1:
            temp = temp[0, :]  # Take first channel if stereo
        
        load_time = time.time() - start_time
        logging.info(f"Loaded {len(temp)} samples at {stim_fs} Hz in {load_time:.2f} seconds")
        
        # Truncate to first 5 seconds for quick test
        max_samples = int(5 * stim_fs)  # 5 seconds
        if len(temp) > max_samples:
            temp = temp[:max_samples]
            logging.info(f"Truncated to {len(temp)} samples ({len(temp)/stim_fs:.1f} seconds) for quick test")
        
        # Test ANM processing with simplified version
        logging.info("Testing ANM processing...")
        start_time = time.time()
        
        # Resample to model frequency
        fs_up = int(100e3)
        stim_up = resample(temp, fs_up, stim_fs, npad='auto', n_jobs=1)
        
        # Convert to pascals
        sine_rms_at_0db = 20e-6
        stim_gen_rms = 0.01
        db_conv = ((sine_rms_at_0db / stim_gen_rms) * 10 ** (stim_pres_db / 20.))
        stim_up = db_conv * stim_up
        
        # Test with a single CF for quick processing
        cf = 1000.0  # Single center frequency
        logging.info(f"Testing with single CF: {cf} Hz")
        
        anf_rates_up = np.array(cochlea.run_zilany2014_rate(stim_up,
                                                           fs_up,
                                                           anf_types='hsr',
                                                           cf=cf,
                                                           species='human',
                                                           cohc=1,
                                                           cihc=1))[:, 0]
        
        # Downsample
        anf_rates = resample(anf_rates_up, stim_fs, fs_up, npad='auto', n_jobs=1)
        
        # Apply nuclei processing
        M1 = nuclei.M1
        anm_response = M1 * anf_rates
        
        # Final resample to EEG frequency
        final_response = resample(anm_response, down=stim_fs/eeg_fs)
        
        processing_time = time.time() - start_time
        logging.info(f"ANM processing completed in {processing_time:.2f} seconds")
        logging.info(f"Output: {len(final_response)} samples at {eeg_fs} Hz")
        
        # Basic validation
        if len(final_response) == 0:
            logging.error("Output is empty!")
            return False
        
        if np.all(final_response == 0):
            logging.error("Output is all zeros!")
            return False
        
        logging.info(f"Output statistics: mean={np.mean(final_response):.2e}, std={np.std(final_response):.2e}")
        logging.info("✓ Single file test completed successfully!")
        
        return True
        
    except Exception as e:
        logging.error(f"Single file test failed: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False

def main():
    """Main test function."""
    logging.info("=" * 50)
    logging.info("SHERLOCK ANM TEST - Single File Processing")
    logging.info("=" * 50)
    
    # Test environment
    logging.info(f"Python version: {sys.version}")
    logging.info(f"Working directory: {os.getcwd()}")
    
    # Run tests
    success = test_single_file_processing()
    
    if success:
        logging.info("=" * 50)
        logging.info("✓ ALL TESTS PASSED - Ready for full processing!")
        logging.info("=" * 50)
        return 0
    else:
        logging.error("=" * 50)
        logging.error("✗ TESTS FAILED - Please fix issues before full processing")
        logging.error("=" * 50)
        return 1

if __name__ == "__main__":
    sys.exit(main())
