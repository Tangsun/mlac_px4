#!/bin/bash
#SBATCH --job-name=train_3_on_gpu
#SBATCH -o %j.log
#SBATCH --partition=cpu-gpu-rtx8000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G

# Initialize the module command first source
source /mnt/home/tangsun/miniconda3/etc/profile.d/conda.sh
conda activate coml_env

# Run scripts
python train_z_up_kR.py "$@"
