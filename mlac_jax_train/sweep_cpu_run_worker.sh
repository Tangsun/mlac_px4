#!/bin/bash

# This script is called by sbatch. 
# It expects: WORKER_OFFSET, TOTAL_WORKERS, BATCH_TIMESTAMP

# Load Environment
source /etc/profile
module unload anaconda 2>/dev/null
module load anaconda/Python-ML-2024b

# Calculate Global Worker ID
# (e.g., if P8 has 8 workers, the first Volta worker is ID 8)
MY_WORKER_ID=$(( SLURM_ARRAY_TASK_ID + WORKER_OFFSET ))

# Get Total Number of Experiments from Python
TOTAL_EXPS=$(python3 sweep_gen_params.py)

# --- NEW: Fetch Experiment Name from Python ---
EXP_NAME=$(python3 sweep_gen_params.py name)

# Calculate "Chunk" size (Experiments per worker)
# Ceiling division: (A + B - 1) / B
CHUNK_SIZE=$(( (TOTAL_EXPS + TOTAL_WORKERS - 1) / TOTAL_WORKERS ))

START_ID=$(( MY_WORKER_ID * CHUNK_SIZE ))
END_ID=$(( START_ID + CHUNK_SIZE - 1 ))

echo "------------------------------------------------"
echo "Worker: $MY_WORKER_ID (Slurm ID: $SLURM_ARRAY_TASK_ID)"
echo "Node: $(hostname)"
echo "Processing Indices: $START_ID to $END_ID"
echo "Timestamp: $BATCH_TIMESTAMP"
echo "------------------------------------------------"

# Force JAX to use CPU only (Crucial for the Volta node!)
export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES="" 
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$SLURM_CPUS_PER_TASK"

export WANDB_MODE=offline  # Disable online logging for large sweeps

# --- Loop through assigned experiments ---
for (( exp_id=START_ID; exp_id<=END_ID; exp_id++ ))
do
    if [ "$exp_id" -ge "$TOTAL_EXPS" ]; then
        break
    fi

    # Fetch Parameters
    PARAMS=$(python3 sweep_gen_params.py $exp_id)
    read -r R_K R_L R_K_R SEED <<< "$PARAMS"

    # Create specific Run ID
    RUN_ID="seed_${SEED}_regK_${R_K}_regL_${R_L}_regKR_${R_K_R}"

    # Define Paths (with "noPnorm" and all params in name)
    DIR_NAME="${BATCH_TIMESTAMP}_${EXP_NAME}/${RUN_ID}"

    LOG_DIR="train_log/${DIR_NAME}"
    RESULT_DIR="${DIR_NAME}"
    mkdir -p $LOG_DIR

    echo "Running Exp $exp_id: $RUN_ID"

    # --- RUN TRAINING ---
    # Note: Redirecting python stdout to a specific log file for cleanliness
    python train_bodyrate.py \
        --seed $SEED \
        --M 50 \
        --pnorm_init 2.0 \
        --p_freq 2000 \
        --meta_epochs 1000 \
        --reg_K $R_K \
        --reg_Lambda $R_L \
        --reg_k_R $R_K_R \
        --output_dir $RESULT_DIR \
        --exp_name "${BATCH_TIMESTAMP}_${EXP_NAME}_${RUN_ID}" \
        --use_x64 > "${LOG_DIR}/exp_${exp_id}_seed_${SEED}.log" 2>&1

done

echo "Worker $MY_WORKER_ID Finished."