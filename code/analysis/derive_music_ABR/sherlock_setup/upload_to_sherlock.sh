#!/bin/bash
# Script to upload music preference project to Stanford Sherlock cluster

# Configuration
SHERLOCK_USER=$USER
SHERLOCK_HOST="login.sherlock.stanford.edu"
REMOTE_DIR="/scratch/users/$USER/music_preference"
LOCAL_DIR="."

echo "Uploading music preference project to Sherlock..."
echo "User: $SHERLOCK_USER"
echo "Host: $SHERLOCK_HOST"
echo "Remote directory: $REMOTE_DIR"

# Create remote directory structure
ssh $SHERLOCK_USER@$SHERLOCK_HOST "mkdir -p $REMOTE_DIR"

# Upload the entire project (excluding large files initially)
echo "Uploading code and configuration files..."
rsync -avz --exclude='*.wav' --exclude='*.mp3' --exclude='*.hdf5' --exclude='__pycache__' --exclude='.DS_Store' \
    $LOCAL_DIR/ $SHERLOCK_USER@$SHERLOCK_HOST:$REMOTE_DIR/

# Upload music files separately (these are large)
echo "Uploading music files..."
rsync -avz --progress --partial --timeout=300 --contimeout=300 music_stim/preprocesed/ $SHERLOCK_USER@$SHERLOCK_HOST:$REMOTE_DIR/music_stim/preprocesed/

# Upload Sherlock setup files
echo "Uploading Sherlock setup files..."
rsync -avz code/analysis/derive_music_ABR/sherlock_setup/ $SHERLOCK_USER@$SHERLOCK_HOST:$REMOTE_DIR/code/analysis/derive_music_ABR/sherlock_setup/

echo "Upload completed!"
echo ""
echo "Next steps:"
echo "1. SSH to Sherlock: ssh $SHERLOCK_USER@$SHERLOCK_HOST"
echo "2. Navigate to project: cd $REMOTE_DIR"
echo "3. Set up environment: ./code/analysis/derive_music_ABR/sherlock_setup/setup_conda.sh"
echo "4. Activate environment: conda activate music_anm_env"
echo "5. Submit job: sbatch code/analysis/derive_music_ABR/sherlock_setup/sherlock_job.slurm"
