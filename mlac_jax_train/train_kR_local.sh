#!/bin/bash

source /Users/kaiyun/OneDrive/Documents/32.Azizan_Lab/MLAC/mlac_px4/mlac_env/bin/activate

# Define the arrays of values for parameters
reg_Lambda_values=(0.008 0.012 0.016 0.02) # 4 values
reg_K_values=(0.004 0.006 0.008 0.01)     # 4 values
pfreq_values=(10 20)                       # 2 values
pnorm_init_values=(2.0 3.0)                # 2 values

# Check if the arrays for pairing have the same number of elements
if [ ${#reg_Lambda_values[@]} -ne ${#reg_K_values[@]} ]; then
  echo "Error: reg_Lambda_values and reg_K_values arrays must have the same number of elements for pairing."
  exit 1
fi

# Common parameters (those not being looped over for every single experiment variation)
SEED=0
M_TRAJ=50
META_EPOCHS=1000 # Keeping this low as per your original script for testing
REG_P=1.0
REG_K_R=0.0001
Z_WEIGHT=1.5

# Base output directory
BASE_OUTPUT_DIR="pnorm_var_reg_L_K_pfreq_pnorm_16runs" # Adjusted for clarity

# Experiment counter
experiment_count=0

# Loop through the indices of the paired arrays
for i in "${!reg_Lambda_values[@]}" # This gives the indices: 0, 1, 2, ... (4 iterations)
do
  lambda_val="${reg_Lambda_values[$i]}"
  k_val="${reg_K_values[$i]}"

  # Loop through pfreq_values
  for p_freq_current in "${pfreq_values[@]}"; do # (2 iterations)
    # Loop through pnorm_init_values
    for pnorm_init_current in "${pnorm_init_values[@]}"; do # (2 iterations)
      experiment_count=$((experiment_count + 1))

      # Create a unique output directory for each combination of parameters
      # This ensures that each of the 16 runs has its own output folder
      OUTPUT_DIR="${BASE_OUTPUT_DIR}/run_${experiment_count}_lambda_${lambda_val}_k_${k_val}_pfreq_${p_freq_current}_pnorm_${pnorm_init_current}"
      mkdir -p "$OUTPUT_DIR" # Create the directory if it doesn't exist

      echo "----------------------------------------------------------------------"
      echo "Running Experiment #$experiment_count"
      echo "Parameters: reg_Lambda=$lambda_val, reg_K=$k_val, P_FREQ=$p_freq_current, PNORM_INIT=$pnorm_init_current"
      echo "Outputting to: $OUTPUT_DIR"
      echo "----------------------------------------------------------------------"

      # Execute the python script with the current combination of parameters
      python train_kR.py \
        --seed $SEED \
        --M $M_TRAJ \
        --pnorm_init "$pnorm_init_current" \
        --p_freq "$p_freq_current" \
        --meta_epochs $META_EPOCHS \
        --reg_P $REG_P \
        --reg_Lambda "$lambda_val" \
        --reg_K "$k_val" \
        --reg_k_R $REG_K_R \
        --z_weight $Z_WEIGHT \
        --output_dir "$OUTPUT_DIR"

      echo "Finished Experiment #$experiment_count: reg_Lambda=$lambda_val, reg_K=$k_val, P_FREQ=$p_freq_current, PNORM_INIT=$pnorm_init_current"
      echo "######################################################################"
      echo ""

    done # End of pnorm_init_values loop
  done # End of pfreq_values loop
done # End of paired reg_Lambda_values/reg_K_values loop

echo "All $experiment_count experiments finished."
