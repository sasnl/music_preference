#!/bin/bash
# Script to download results from Stanford Sherlock cluster

# Configuration
SHERLOCK_USER=$USER
SHERLOCK_HOST="login.sherlock.stanford.edu"
REMOTE_DIR="$SCRATCH/music_preference"
LOCAL_DIR="."

echo "Downloading results from Sherlock..."
echo "User: $SHERLOCK_USER"
echo "Host: $SHERLOCK_HOST"
echo "Remote directory: $REMOTE_DIR"

# Create local results directory
mkdir -p $LOCAL_DIR/results

# Download results
echo "Downloading ANM regressors..."
rsync -avz --progress $SHERLOCK_USER@$SHERLOCK_HOST:$REMOTE_DIR/data/ $LOCAL_DIR/results/

# Download log files
echo "Downloading log files..."
rsync -avz --progress $SHERLOCK_USER@$SHERLOCK_HOST:$REMOTE_DIR/*.out $LOCAL_DIR/results/
rsync -avz --progress $SHERLOCK_USER@$SHERLOCK_HOST:$REMOTE_DIR/*.err $LOCAL_DIR/results/

echo "Download completed!"
echo "Results are in: $LOCAL_DIR/results/"
