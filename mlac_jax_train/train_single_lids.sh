#!/bin/bash
#SBATCH --job-name=train_3_on_gpu
#SBATCH -o %j.log
#SBATCH --partition=cpu
#SBATCH --nodes=1

# Initialize the module command first source
source /mnt/home/tangsun/miniconda3/etc/profile.d/conda.sh
conda activate coml_env

# Run scripts
python train_z_up_kR.py "$@"
