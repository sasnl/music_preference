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
module load python/3.9.0

# Initialize conda (Sherlock uses miniconda3)
source /share/software/user/open/miniconda3/etc/profile.d/conda.sh

# Set up environment
conda activate music_anm_env

# Copy environment file to working directory
cp code/analysis/derive_music_ABR/sherlock_setup/environment.yml .

# Set working directory
cd /scratch/users/$USER/music_preference

# Create output directory
mkdir -p data

# Run the test script with a single file
echo "Starting test ANM processing at $(date)"
python code/analysis/derive_music_ABR/sherlock_setup/test_single_file.py

echo "Test ANM processing completed at $(date)"

# Copy results to home directory for easy access
mkdir -p $HOME/music_preference_test_results/
cp data/* $HOME/music_preference_test_results/ 2>/dev/null || echo "No data files to copy"

echo "Test results copied to $HOME/music_preference_test_results/"
