# Quick Start: Sherlock ANM Processing

## Immediate Steps

### 1. Upload to Sherlock (5 minutes)
```bash
./code/analysis/derive_music_ABR/sherlock_setup/upload_to_sherlock.sh
```

### 2. Connect to Sherlock
```bash
ssh $USER@login.sherlock.stanford.edu
cd /scratch/users/$USER/music_preference
```

### 3. Set Up Environment (10 minutes)
```bash
source /share/software/user/open/miniconda3/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate music_anm_env
```

### 4. Test with Single File (1 hour)
```bash
sbatch code/analysis/derive_music_ABR/sherlock_setup/test_sherlock_job.slurm
squeue -u $USER  # Check status
tail -f test_anm_*.out  # Monitor progress
```

### 5. Run Full Processing (8-12 hours)
```bash
sbatch code/analysis/derive_music_ABR/sherlock_setup/sherlock_job.slurm
squeue -u $USER  # Check status
tail -f music_anm_*.out  # Monitor progress
```

### 6. Download Results
```bash
# From your local machine
./code/analysis/derive_music_ABR/sherlock_setup/download_from_sherlock.sh
```

## Key Commands

### Check Job Status
```bash
squeue -u $USER
```

### Monitor Progress
```bash
tail -f music_anm_*.out
```

### Cancel Job
```bash
scancel <job_id>
```

## Expected Timeline
- **Upload**: 5 minutes
- **Setup**: 10 minutes  
- **Test**: 1 hour
- **Full processing**: 8-12 hours
- **Download**: 5 minutes

## Troubleshooting
- If test fails: Check logs in `test_anm_*.err`
- If memory issues: Reduce `--cpus-per-task` in SLURM script
- If timeout: Increase `--time` in SLURM script

## Support Files
- `code/analysis/derive_music_ABR/sherlock_setup/sherlock_README.md` - Detailed instructions
- `code/analysis/derive_music_ABR/sherlock_setup/sherlock_job.slurm` - Main job script
- `code/analysis/derive_music_ABR/sherlock_setup/test_sherlock_job.slurm` - Test job script
- `code/analysis/derive_music_ABR/sherlock_setup/environment.yml` - Dependencies
