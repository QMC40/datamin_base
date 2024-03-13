#!/bin/bash

# Note that if conda does not properly activate from the script simply run the steps manually in the terminal

# 1. Install mamba

# Check if mamba is installed
if ! [ -x "$(command -v mamba)" ]; then
    echo "Installing mamba"
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh
    rm Miniforge3-$(uname)-$(uname -m).sh
else
    echo "Mamba is already installed"
fi

# 2. Create environment
mamba env create --file environment.yml

eval "$(conda shell.bash hook)"
conda activate datamin-env
echo "Finished creating datamin environment - running in $PATH"

# 3. Install sktree
echo "Installing sktree"
cd sktree
# If you are in datamin-env you can remove the conda run -n datamin-env
conda run -n datamin-env ./build.sh   
cd ..
echo "Setup finished"