#!/bin/bash
# Robust script to set up conda environment on Sherlock

echo "Setting up conda environment on Sherlock..."

# Try different conda paths
CONDA_PATHS=(
    "/share/software/user/open/miniconda3/etc/profile.d/conda.sh"
    "/share/software/user/open/anaconda3/etc/profile.d/conda.sh"
    "/usr/local/miniconda3/etc/profile.d/conda.sh"
    "/usr/local/anaconda3/etc/profile.d/conda.sh"
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
)

CONDA_LOADED=false

for conda_path in "${CONDA_PATHS[@]}"; do
    if [ -f "$conda_path" ]; then
        echo "Found conda at: $conda_path"
        source "$conda_path"
        if command -v conda &> /dev/null; then
            echo "Conda loaded successfully from: $conda_path"
            CONDA_LOADED=true
            break
        fi
    fi
done

if [ "$CONDA_LOADED" = false ]; then
    echo "Error: Could not find conda in any of the expected locations"
    echo "Available conda installations:"
    find /share/software/user/open/ -name "conda.sh" 2>/dev/null || echo "No conda.sh found in /share/software/user/open/"
    find /usr/local/ -name "conda.sh" 2>/dev/null || echo "No conda.sh found in /usr/local/"
    echo ""
    echo "Please contact Stanford Research Computing for conda setup assistance"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "music_anm_env"; then
    echo "Environment music_anm_env already exists"
    echo "To activate it, run: conda activate music_anm_env"
else
    echo "Creating conda environment..."
    conda env create -f environment.yml
    
    if [ $? -eq 0 ]; then
        echo "Environment created successfully!"
        echo "To activate it, run: conda activate music_anm_env"
    else
        echo "Error creating environment. Please check the error messages above."
        exit 1
    fi
fi

echo ""
echo "Setup complete! Next steps:"
echo "1. Activate environment: conda activate music_anm_env"
echo "2. Test environment: python -c 'import cochlea; print(\"Cochlea available\")'"
echo "3. Submit job: sbatch code/analysis/derive_music_ABR/sherlock_setup/test_sherlock_job.slurm"
