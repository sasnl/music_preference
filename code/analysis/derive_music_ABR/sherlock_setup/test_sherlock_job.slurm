#!/bin/bash
#SBATCH --job-name=test_anm
#SBATCH --output=test_anm_%j.out
#SBATCH --error=test_anm_%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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

# Create output directory
echo "Creating output directory..."
mkdir -p data
if [ $? -ne 0 ]; then
    echo "Error: Failed to create data directory"
    exit 1
fi
echo "✓ Output directory created: $(pwd)/data"

# Verify test script exists
TEST_SCRIPT="code/analysis/derive_music_ABR/sherlock_setup/test_single_file.py"
if [ ! -f "$TEST_SCRIPT" ]; then
    echo "Error: Test script not found: $TEST_SCRIPT"
    echo "Available files in sherlock_setup:"
    ls -la code/analysis/derive_music_ABR/sherlock_setup/ || echo "Directory not found"
    exit 1
fi
echo "✓ Test script found: $TEST_SCRIPT"

# Run the test script
echo "Starting test ANM processing at $(date)"
echo "=================================="
python "$TEST_SCRIPT"
SCRIPT_EXIT_CODE=$?

echo "=================================="
echo "Test ANM processing completed at $(date)"

# Create test results directory
mkdir -p $HOME/music_preference_test_results/

if [ $SCRIPT_EXIT_CODE -eq 0 ]; then
    echo "✓ Test completed successfully!"
    
    # Copy any generated data files
    if [ -d "data" ] && [ "$(ls -A data 2>/dev/null)" ]; then
        cp data/* $HOME/music_preference_test_results/ 2>/dev/null
        echo "✓ Test data copied to $HOME/music_preference_test_results/"
    else
        echo "ℹ No data files generated (this is normal for the test)"
    fi
    
    # Copy log files
    cp test_anm_*.out $HOME/music_preference_test_results/ 2>/dev/null || echo "No output log to copy"
    cp test_anm_*.err $HOME/music_preference_test_results/ 2>/dev/null || echo "No error log to copy"
    
    echo "✓ Test job completed successfully!"
    echo "✓ Ready to run full processing with: sbatch sherlock_job.slurm"
else
    echo "✗ Test failed with exit code: $SCRIPT_EXIT_CODE"
    echo "Check the error logs for details"
    
    # Still copy logs for debugging
    cp test_anm_*.out $HOME/music_preference_test_results/ 2>/dev/null || echo "No output log to copy"
    cp test_anm_*.err $HOME/music_preference_test_results/ 2>/dev/null || echo "No error log to copy"
    
    exit $SCRIPT_EXIT_CODE
fi
