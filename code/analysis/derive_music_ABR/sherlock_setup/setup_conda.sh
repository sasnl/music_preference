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
    echo "Checking if environment is functional..."
    
    # Test if environment can be activated and has required packages
    if conda activate music_anm_env 2>/dev/null && python -c "import numpy, scipy, mne" 2>/dev/null; then
        echo "✓ Environment is functional"
        echo "To activate it, run: conda activate music_anm_env"
    else
        echo "⚠ Environment exists but may have issues. Consider recreating it."
        echo "To recreate: conda env remove -n music_anm_env && conda env create -f code/analysis/derive_music_ABR/sherlock_setup/environment.yml"
    fi
else
    echo "Creating conda environment..."
    
    # Use full path to environment file for robustness
    ENV_FILE="code/analysis/derive_music_ABR/sherlock_setup/environment.yml"
    if [ -f "$ENV_FILE" ]; then
        conda env create -f "$ENV_FILE"
        
        if [ $? -eq 0 ]; then
            echo "✓ Environment created successfully!"
            echo "Testing environment..."
            
            # Test the environment
            if conda activate music_anm_env 2>/dev/null && python -c "import numpy, scipy, mne; print('Basic packages OK')" 2>/dev/null; then
                echo "✓ Environment test passed"
                echo "To activate it, run: conda activate music_anm_env"
            else
                echo "⚠ Environment created but basic test failed"
                echo "You may need to install additional packages manually"
            fi
        else
            echo "✗ Error creating environment. Please check the error messages above."
            echo "Common solutions:"
            echo "1. Check internet connectivity"
            echo "2. Try: conda clean --all"
            echo "3. Update conda: conda update conda"
            exit 1
        fi
    else
        echo "✗ Environment file not found: $ENV_FILE"
        echo "Please ensure you're in the project root directory"
        exit 1
    fi
fi

echo ""
echo "Setup complete! Next steps:"
echo "1. Activate environment: conda activate music_anm_env"
echo "2. Test environment: python -c 'import cochlea; print(\"Cochlea available\")'"
echo "3. Submit job: sbatch code/analysis/derive_music_ABR/sherlock_setup/test_sherlock_job.slurm"
