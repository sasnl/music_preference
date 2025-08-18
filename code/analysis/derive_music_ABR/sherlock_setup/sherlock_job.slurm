#!/bin/bash
#SBATCH --job-name=music_anm
#SBATCH --output=music_anm_%j.out
#SBATCH --error=music_anm_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$USER@stanford.edu

# Load required modules
echo "Loading Python module..."
module load python/3.9.0
if [ $? -ne 0 ]; then
    echo "Error: Failed to load Python module"
    exit 1
fi

# Initialize conda (Sherlock uses miniconda3)
echo "Initializing conda..."
CONDA_SCRIPT="/share/software/user/open/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_SCRIPT" ]; then
    source "$CONDA_SCRIPT"
    echo "✓ Conda initialized from $CONDA_SCRIPT"
else
    echo "Error: Conda script not found at $CONDA_SCRIPT"
    echo "Trying alternative paths..."
    # Try alternative conda paths
    for alt_path in "/usr/local/miniconda3/etc/profile.d/conda.sh" "$HOME/miniconda3/etc/profile.d/conda.sh"; do
        if [ -f "$alt_path" ]; then
            source "$alt_path"
            echo "✓ Conda initialized from $alt_path"
            break
        fi
    done
    
    # Verify conda is available
    if ! command -v conda &> /dev/null; then
        echo "Error: Conda not available after initialization"
        exit 1
    fi
fi

# Set working directory first
echo "Setting working directory..."
cd /scratch/users/$USER/music_preference
if [ $? -ne 0 ]; then
    echo "Error: Failed to change to working directory /scratch/users/$USER/music_preference"
    exit 1
fi
echo "✓ Working directory: $(pwd)"

# Verify environment file exists
ENV_FILE="code/analysis/derive_music_ABR/sherlock_setup/environment.yml"
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: Environment file not found: $ENV_FILE"
    echo "Available files in sherlock_setup:"
    ls -la code/analysis/derive_music_ABR/sherlock_setup/ || echo "Directory not found"
    exit 1
fi

# Activate environment
echo "Activating conda environment..."
conda activate music_anm_env
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate music_anm_env"
    echo "Available environments:"
    conda env list
    exit 1
fi
echo "✓ Environment activated: $(conda info --envs | grep '*')"

# Verify required packages
echo "Verifying required packages..."
python -c "import numpy, scipy, mne, joblib, psutil, h5py; print('✓ Basic packages available')" || {
    echo "Error: Basic packages not available"
    exit 1
}

# Check for cochlea (this might fail, but we'll continue)
python -c "import cochlea; print('✓ Cochlea package available')" || {
    echo "Warning: Cochlea package not available - this will cause the script to fail"
    echo "Please ensure cochlea is installed in the environment"
}

# Create output directory
echo "Creating output directory..."
mkdir -p data
if [ $? -ne 0 ]; then
    echo "Error: Failed to create data directory"
    exit 1
fi
echo "✓ Output directory created: $(pwd)/data"

# Verify music files exist
echo "Checking for music files..."
if [ ! -d "music_stim/preprocesed" ]; then
    echo "Error: Music directory not found: music_stim/preprocesed"
    echo "Available directories:"
    ls -la music_stim/ || echo "music_stim directory not found"
    exit 1
fi

MUSIC_COUNT=$(find music_stim/preprocesed -name "*_proc.wav" | wc -l)
echo "✓ Found $MUSIC_COUNT music files to process"

if [ $MUSIC_COUNT -eq 0 ]; then
    echo "Error: No music files found to process"
    exit 1
fi

# Run the ANM processing script
echo "Starting ANM processing at $(date)"
echo "=================================="
python code/analysis/derive_music_ABR/sherlock_setup/generate_music_anm_regressors_sherlock.py
SCRIPT_EXIT_CODE=$?

echo "=================================="
echo "ANM processing completed at $(date)"

if [ $SCRIPT_EXIT_CODE -eq 0 ]; then
    echo "✓ Processing completed successfully!"
    
    # Check if output file was created
    if [ -f "data/music_anm_regressors.hdf5" ]; then
        echo "✓ Output file created: data/music_anm_regressors.hdf5"
        
        # Get file size
        FILE_SIZE=$(ls -lh data/music_anm_regressors.hdf5 | awk '{print $5}')
        echo "✓ File size: $FILE_SIZE"
        
        # Copy results to home directory for easy access
        mkdir -p $HOME/music_preference_results/
        cp data/music_anm_regressors.hdf5 $HOME/music_preference_results/
        
        if [ $? -eq 0 ]; then
            echo "✓ Results copied to $HOME/music_preference_results/"
            echo "✓ Job completed successfully!"
        else
            echo "⚠ Warning: Failed to copy results to home directory"
            echo "Results are available at: $(pwd)/data/music_anm_regressors.hdf5"
        fi
    else
        echo "✗ Error: Expected output file not found: data/music_anm_regressors.hdf5"
        echo "Check the processing logs above for errors"
        exit 1
    fi
else
    echo "✗ Processing failed with exit code: $SCRIPT_EXIT_CODE"
    echo "Check the error logs for details"
    exit $SCRIPT_EXIT_CODE
fi
