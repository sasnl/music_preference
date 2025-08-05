#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example usage of the ABR derivation module.

This script demonstrates how to use the derive_abr module with different
parameter configurations.
"""

import os
import sys
from pathlib import Path

# Add the current directory to the path so we can import the module
sys.path.append(os.path.dirname(__file__))

from derive_click_abr import derive_abr


def example_basic_usage():
    """Example of basic ABR derivation usage."""
    print("=== Basic ABR Derivation Example ===")
    
    # Example parameters (you'll need to adjust these paths)
    eeg_file = "/path/to/your/eeg_file.vhdr"  # Replace with actual path
    output_dir = "./abr_results"
    click_dir = "../../click_stim"  # Relative to the click_stim directory
    
    try:
        # Basic usage with default parameters
        abr_response, lags = derive_abr(
            eeg_file=eeg_file,
            output_dir=output_dir,
            click_dir=click_dir,
            subject_id="example_subject"
        )
        
        print(f"ABR derivation completed!")
        print(f"Response shape: {abr_response.shape}")
        print(f"Time range: {lags[0]:.1f} to {lags[-1]:.1f} ms")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please update the file paths in this example.")
    except Exception as e:
        print(f"Error: {e}")


def example_custom_parameters():
    """Example with custom parameters."""
    print("\n=== Custom Parameters Example ===")
    
    eeg_file = "/path/to/your/eeg_file.vhdr"  # Replace with actual path
    output_dir = "./abr_results_custom"
    click_dir = "../../click_stim"
    
    try:
        # Custom parameters
        abr_response, lags = derive_abr(
            eeg_file=eeg_file,
            output_dir=output_dir,
            click_dir=click_dir,
            eeg_fs=25000,  # Higher sampling frequency
            eeg_f_hp=2.0,  # Higher high-pass cutoff
            t_click=30,    # Shorter trial length
            click_rate=50,  # Higher click rate
            n_epoch_click=10,  # More epochs
            plot_results=True,  # Generate plots
            subject_id="custom_subject",
            t_start=-100e-3,  # Custom start time (-100 ms)
            t_stop=500e-3     # Custom stop time (500 ms)
        )
        
        print(f"Custom ABR derivation completed!")
        print(f"Response shape: {abr_response.shape}")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please update the file paths in this example.")
    except Exception as e:
        print(f"Error: {e}")


def example_batch_processing():
    """Example of batch processing multiple subjects."""
    print("\n=== Batch Processing Example ===")
    
    # Example subject list
    subjects = [
        {"id": "subject001", "eeg_file": "/path/to/subject001.vhdr"},
        {"id": "subject002", "eeg_file": "/path/to/subject002.vhdr"},
        # Add more subjects as needed
    ]
    
    output_base_dir = "./batch_abr_results"
    click_dir = "../../click_stim"
    
    for subject in subjects:
        try:
            print(f"Processing {subject['id']}...")
            
            # Create subject-specific output directory
            subject_output_dir = os.path.join(output_base_dir, subject['id'])
            
            abr_response, lags = derive_abr(
                eeg_file=subject['eeg_file'],
                output_dir=subject_output_dir,
                click_dir=click_dir,
                subject_id=subject['id'],
                plot_results=True
            )
            
            print(f"Completed {subject['id']}")
            
        except FileNotFoundError as e:
            print(f"File not found for {subject['id']}: {e}")
        except Exception as e:
            print(f"Error processing {subject['id']}: {e}")


def example_command_line_usage():
    """Example of how to use the command-line interface."""
    print("\n=== Command Line Usage Example ===")
    print("You can also use the module from the command line:")
    print()
    print("Basic usage:")
    print("python derive_click_abr.py /path/to/eeg_file.vhdr ./output_dir")
    print()
    print("With custom parameters:")
    print("python derive_click_abr.py /path/to/eeg_file.vhdr ./output_dir \\")
    print("    --eeg_fs 25000 --eeg_f_hp 2.0 --t_click 30 --plot")
    print()
    print("With custom response time range:")
    print("python derive_click_abr.py /path/to/eeg_file.vhdr ./output_dir \\")
    print("    --t_start -0.1 --t_stop 0.5 --plot")
    print()
    print("With custom click directory:")
    print("python derive_click_abr.py /path/to/eeg_file.vhdr ./output_dir \\")
    print("    --click_dir /path/to/click_stim --subject_id my_subject")


if __name__ == "__main__":
    print("ABR Derivation Module - Usage Examples")
    print("=" * 50)
    
    # Run examples
    example_basic_usage()
    example_custom_parameters()
    example_batch_processing()
    example_command_line_usage()
    
    print("\n" + "=" * 50)
    print("Note: Update the file paths in these examples to match your data.")
    print("The module will automatically create output directories and save results.") 