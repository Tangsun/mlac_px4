#!/bin/bash
#SBATCH --job-name=submission

for seed in 0
do
    for M in 50
    do
        for p in 2.0
        do
            for p_freq in 2000
            do
                for reg_P in 1.0
                    do
                    for reg_k_R in 0.001
                    do
                        for k_R_scale in 1
                        do
                            for k_R_z in 1.26
                            do
                                echo "seed = $seed, M = $M, pnorm_init = $p, p_freq = $p_freq, meta_epochs = 1000, reg_P = $reg_P reg_k_R = $reg_k_R k_R_scale = $k_R_scale k_R_z = $k_R_z output_dir = reg_P_${reg_P}_reg_k_R_${reg_k_R}_k_R_scale_${k_R_scale}_k_R_z_${k_R_z}_z_training"
                                sbatch train_single_lids.sh $seed $M --pnorm_init $p --p_freq $p_freq --meta_epochs 1000 --reg_P $reg_P --reg_k_R $reg_k_R --k_R_scale $k_R_scale --k_R_z $k_R_z --output_dir "reg_P_${reg_P}_reg_k_R_${reg_k_R}_k_R_scale_${k_R_scale}_k_R_z_${k_R_z}_z_training"
                            done
                        done
                    done
                done
            done
        done
    done
done