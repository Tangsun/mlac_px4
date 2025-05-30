#!/bin/bash

source /home/sunbochen/mlac_px4/mlac_env/bin/activate

python train_z_up_kR.py --seed 0 --M 50 --pnorm_init 2.0 --p_freq 2000 --meta_epochs 1000 --reg_P 1.0 --reg_k_R 0.001 --k_R_scale 1 --k_R_z 1.26 --output_dir "reg_P_1_reg_k_R_1e-3_k_R_scale_1_k_R_z_1.26_z_training"