#!/bin/bash
#SBATCH --partition=xeon-p8       # CPU partition
#SBATCH --nodes=1                 # 1 Node per task
#SBATCH --cpus-per-task=48        # Use ALL 48 cores on the node (Max resource per run)
#SBATCH --mem=350G                # Request mostly full RAM (Safe upper limit for p8 nodes)
#SBATCH --time=12:00:00           # 12 Hour limit
#SBATCH --exclusive               # Request exclusive access to the node (no other users)

# --- Parameter Sweep Definition ---
# Define the values for reg_k_R you want to test.
# Space-separated list.
REG_KR_VALUES=(0.001 0.01 0.1 1.0)
REG_K=(0.0 0.01 0.02 0.05)
REG_LAMBDA=(0.0 0.01 0.02 0.05)
P_NORM_FREQ=2000

# Get the array length
NUM_VALS=${#REG_KR_VALUES[@]}

# Calculate which index to use. 
# Note: You must submit this script with --array=0-$(($NUM_VALS-1))
CURRENT_REG_KR=${REG_KR_VALUES[$SLURM_ARRAY_TASK_ID]}

# --- Timestamp Setup ---
# We expect the TIMESTAMP variable to be passed in via --export.
# If not, generate one now (though this might differ by seconds between tasks if not careful).
if [ -z "$BATCH_TIMESTAMP" ]; then
    BATCH_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
fi

# --- Directory Setup ---
# Construct paths using the timestamp
LOG_DIR="train_log/${BATCH_TIMESTAMP}_sweep_noPnorm"
RESULT_DIR="train_results/${BATCH_TIMESTAMP}_sweep_noPnorm/reg_k_R_${CURRENT_REG_KR}"

mkdir -p $LOG_DIR
mkdir -p $RESULT_DIR

# Redirect Slurm output to the specific log file for this task
# (We can't use #SBATCH directives for dynamic paths, so we use exec)
exec > "${LOG_DIR}/task_${SLURM_ARRAY_TASK_ID}_reg_k_R_${CURRENT_REG_KR}.out" 2>&1

# --- Environment ---
source /etc/profile
module unload anaconda 2>/dev/null
module load anaconda/Python-ML-2024b

echo "==================================================="
echo "Starting CPU Job on Node: $(hostname)"
echo "Batch Timestamp: $BATCH_TIMESTAMP"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Parameter reg_k_R: $CURRENT_REG_KR"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Output Dir: $RESULT_DIR"
echo "==================================================="

# --- JAX CPU Configuration ---
# Since we requested the whole node (48 cores), tell JAX to use them.
# JAX usually detects this automatically, but explicit flags help.
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$SLURM_CPUS_PER_TASK"

# --- Run Training ---
python train_bodyrate.py \
    --seed 0 \
    --M 50 \
    --pnorm_init 2.0 \
    --p_freq $P_NORM_FREQ \
    --meta_epochs 1000 \
    --reg_k_R $CURRENT_REG_KR \
    --output_dir $RESULT_DIR \
    --use_x64

echo "==================================================="
echo "Job Finished"