#!/bin/bash

source /home/sunbochen/mlac_px4/mlac_env/bin/activate

# Define the arrays of values for k_R_xy and k_R_z
# For k_R_xy between 1.0 and 2.5 (4 values)
# For k_R_z between 1.0 and 2.0 (4 values)

k_R_xy_values=(1.0 1.5 2.0 2.5)
k_R_z_values=(1.0 1.33 1.66 2.0) # Example: 3 intervals, 4 values

# Common parameters
SEED=0
M_TRAJ=50
PNORM_INIT=2.0
P_FREQ=2000
META_EPOCHS=1000 # Keeping this low as per your original script for testing
REG_P=1.0

# Counter for experiments
experiment_count=0

# Nested loops to iterate through all combinations
for kr_xy in "${k_R_xy_values[@]}"; do
  for kr_z in "${k_R_z_values[@]}"; do
    experiment_count=$((experiment_count + 1))
    echo "Running experiment $experiment_count: k_R_xy = $kr_xy, k_R_z = $kr_z"

    OUTPUT_DIR_NAME="exp_${experiment_count}_kRxy_${kr_xy}_kRz_${kr_z}_pnorm_${PNORM_INIT}"

    python train_z_up_kR.py \
      --seed $SEED \
      --M $M_TRAJ \
      --pnorm_init $PNORM_INIT \
      --p_freq $P_FREQ \
      --meta_epochs $META_EPOCHS \
      --reg_P $REG_P \
      --k_R_xy $kr_xy \
      --k_R_z $kr_z \
      --output_dir "$OUTPUT_DIR_NAME"
      # Removed --k_R_scale assuming you'll absorb it into the values directly
      # or add it back if your train_z_up_kR.py still uses it and you want a global scale

    echo "Finished experiment $experiment_count: $OUTPUT_DIR_NAME"
    echo "----------------------------------------------------"
  done
done

echo "All 16 experiments complete."