#!/bin/bash
#SBATCH --job-name=music_anm
#SBATCH --output=music_anm_%j.out
#SBATCH --error=music_anm_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
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

# Run the ANM processing script
echo "Starting ANM processing at $(date)"
python code/analysis/derive_music_ABR/sherlock_setup/generate_music_anm_regressors_sherlock.py

echo "ANM processing completed at $(date)"

# Copy results to home directory for easy access
cp data/music_anm_regressors.hdf5 $HOME/music_preference_results/

echo "Results copied to $HOME/music_preference_results/"
