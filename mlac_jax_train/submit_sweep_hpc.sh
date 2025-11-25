#!/bin/bash

# --- Slurm Resource Requests ---

# 1. Job Array: Request 8 tasks (indices 0-7).
#    This fills your 8-GPU quota exactly.
#SBATCH -a 0-7

# 2. Partition: Request the GPU-enabled partition
#SBATCH -p xeon-g6-volta

# 3. GPUs: Request 1 Volta GPU *per task*
#SBATCH --gres=gpu:volta:1

# 4. CPUs: Request 20 CPUs *per task*
#    (Each node has 40 cores and 2 GPUs, so 20 is the fair share per GPU)
#SBATCH -c 20

# 5. Output: Logs
#SBATCH -o train_log/bodyrate_%A_%a.log
#SBATCH -e train_log/bodyrate_%A_%a.err

# --- End Slurm ---

# Create log/output directories
mkdir -p train_log
mkdir -p train_results

# Initialize environment modules (matches your working script)
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
elif [ -f /usr/share/lmod/lmod/init/bash ]; then
    source /usr/share/lmod/lmod/init/bash
fi

# Load Anaconda environment
module unload anaconda 2>/dev/null || true
module load anaconda/Python-ML-2024b

echo "----------------------------------------------------"
echo "STARTING WORKER: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "CUDA Visible Devices: $CUDA_VISIBLE_DEVICES"
echo "----------------------------------------------------" 

# --- BUNDLING LOGIC ---
# We have 8 GPUs. To run 50 experiments total, each GPU needs to do ~7 runs.
# Worker 0 does seeds 0-6, Worker 1 does 7-13, etc.

RUNS_PER_WORKER=7
START_SEED=$(( SLURM_ARRAY_TASK_ID * RUNS_PER_WORKER ))
END_SEED=$(( START_SEED + RUNS_PER_WORKER - 1 ))
MAX_TOTAL_SEEDS=49 # Stop if we exceed seed 49 (total 50 runs)

# --- SEQUENTIAL LOOP ---
for (( seed=START_SEED; seed<=END_SEED; seed++ ))
do
    # Stop if we go past our desired total experiment count
    if [ "$seed" -gt "$MAX_TOTAL_SEEDS" ]; then
        break
    fi

    echo ">>> Running Seed: $seed"
    
    # Create unique output directory for this seed
    # (Assuming you want to organize by experiment ID then seed)
    OUTPUT_DIR="train_results/bodyrate_sweep_${SLURM_ARRAY_JOB_ID}/seed_${seed}"
    mkdir -p $OUTPUT_DIR

    # Run the Python Training Code
    python train_bodyrate.py \
        --seed $seed \
        --M 50 \
        --pnorm_init 2.0 \
        --meta_epochs 1000 \
        --output_dir $OUTPUT_DIR \
        --use_x64

    echo ">>> Finished Seed: $seed"
done

echo "🎉 Worker $SLURM_ARRAY_TASK_ID finished all assigned seeds."

