#!/bin/bash
#SBATCH --job-name=mnist_encoder
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=mnist_encoder_%j.out
#SBATCH --error=mnist_encoder_%j.err

source /home/groups/mukerji/minghuix/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/users/minghuix/conda_envs/drift

cd /home/groups/mukerji/minghuix/Driftmodel
python run_mnist_encoder.py --save_dir ./mnist_encoder_outputs
