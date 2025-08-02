#!/bin/bash

source /Users/kaiyun/OneDrive/Documents/32.Azizan_Lab/MLAC/mlac_px4/mlac_env/bin/activate

# Define the arrays of values to loop through
# Corrected bash array syntax
reg_L_vals=(0.02 0.01 0.005)
reg_K_vals=(0.01 0.005 0.0025)
lr_vals=(0.005 0.002 0.001)

# Common parameters (fixed for all runs in this script)
SEED=0
M_TRAJ=50
PNORM_INIT=2.0
P_FREQ=2000
META_EPOCHS=1000 # Keeping this low as per your original script for testing
REG_P=1.0
K_R_XY=1.6
K_R_Z=0.4
BASE_OUTPUT_DIR="train_fixed_kR_kOmega_2norm"

# Check if the lengths of reg_L_vals and reg_K_vals are the same
if [ ${#reg_L_vals[@]} -ne ${#reg_K_vals[@]} ]; then
    echo "Error: reg_L_vals and reg_K_vals arrays must have the same number of elements."
    exit 1
fi

# Loop over reg_L_vals and reg_K_vals as pairs
for (( i=0; i<${#reg_L_vals[@]}; i++ )); do
    current_reg_L=${reg_L_vals[$i]}
    current_reg_K=${reg_K_vals[$i]}

    # Loop over lr_vals (learning rates)
    for current_lr in "${lr_vals[@]}"; do
        echo "------------------------------------------------------------------------"
        echo "Starting training with: reg_Lambda=$current_reg_L, reg_K=$current_reg_K, learning_rate=$current_lr"
        echo "------------------------------------------------------------------------"

        # Construct a unique output directory for each combination of parameters
        # Appending parameters to the base output directory name for clarity
        output_dir_name="${BASE_OUTPUT_DIR}_regL_${current_reg_L}_regK_${current_reg_K}_lr_${current_lr}"

        # Run the Python training script
        python train_kR_fixed.py \
            --seed $SEED \
            --M $M_TRAJ \
            --pnorm_init $PNORM_INIT \
            --p_freq $P_FREQ \
            --meta_epochs $META_EPOCHS \
            --reg_P $REG_P \
            --reg_Lambda $current_reg_L \
            --reg_K $current_reg_K \
            --learning_rate $current_lr \
            --output_dir "$output_dir_name"

        echo "------------------------------------------------------------------------"
        echo "Finished training for: reg_Lambda=$current_reg_L, reg_K=$current_reg_K, learning_rate=$current_lr"
        echo "Output saved to: $output_dir_name"
        echo "------------------------------------------------------------------------"
        echo # Add an empty line for better readability between runs
    done
done

P_FREQ=20
BASE_OUTPUT_DIR="train_fixed_kR_kOmega_pnorm_freq20"

# Loop over reg_L_vals and reg_K_vals as pairs
for (( i=0; i<${#reg_L_vals[@]}; i++ )); do
    current_reg_L=${reg_L_vals[$i]}
    current_reg_K=${reg_K_vals[$i]}

    # Loop over lr_vals (learning rates)
    for current_lr in "${lr_vals[@]}"; do
        echo "------------------------------------------------------------------------"
        echo "Starting training with: reg_Lambda=$current_reg_L, reg_K=$current_reg_K, learning_rate=$current_lr, p_freq=$P_FREQ"
        echo "------------------------------------------------------------------------"

        # Construct a unique output directory for each combination of parameters
        # Appending parameters to the base output directory name for clarity
        output_dir_name="${BASE_OUTPUT_DIR}_regL_${current_reg_L}_regK_${current_reg_K}_lr_${current_lr}"

        # Run the Python training script
        python train_kR_fixed.py \
            --seed $SEED \
            --M $M_TRAJ \
            --pnorm_init $PNORM_INIT \
            --p_freq $P_FREQ \
            --meta_epochs $META_EPOCHS \
            --reg_P $REG_P \
            --reg_Lambda $current_reg_L \
            --reg_K $current_reg_K \
            --learning_rate $current_lr \
            --output_dir "$output_dir_name"

        echo "------------------------------------------------------------------------"
        echo "Finished training for: reg_Lambda=$current_reg_L, reg_K=$current_reg_K, learning_rate=$current_lr"
        echo "Output saved to: $output_dir_name"
        echo "------------------------------------------------------------------------"
        echo # Add an empty line for better readability between runs
    done
done

echo "All training runs completed."