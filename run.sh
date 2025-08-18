#!/bin/bash

#SBATCH --job-name=evoaug22_deepstarr
#SBATCH --output=slurm_outputs/evoaug2_deepstarr_%A_%a.out
#SBATCH --error=slurm_outputs/evoaug2_deepstarr_%A_%a.err
#SBATCH --time=1:00:00
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64GB
#SBATCH --qos=fast

source ~/.bashrc
mamba activate d3-old

cd ~/evoaug

echo "Starting EvoAug2 DeepSTARR Two-Stage Training: $(date -u +%H:%M:%S)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working Directory: $(pwd)"

# Check if data file exists
if [ ! -f "deepstarr-data.h5" ]; then
    echo "ERROR: deepstarr-data.h5 not found!"
    echo "Downloading data from Zenodo..."
    wget https://zenodo.org/record/7265991/files/DeepSTARR_data.h5 -O deepstarr-data.h5
fi

echo "Running EvoAug2 DeepSTARR training with two-stage approach..."
echo "Stage 1: Training with augmentations"
echo "Stage 2: Fine-tuning on original data"
echo "Control: Training on original data only"

python -u evoaug2_deepstarr_training.py

echo "EvoAug2 DeepSTARR training completed: $(date -u +%H:%M:%S)"
echo "Check output files for training results and saved models"