#!/bin/bash

# 1. Generate a single timestamp for the whole batch
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "Launching Hybrid Sweep. Timestamp: $TIMESTAMP"

# --- CONFIGURATION ---
# We have 12 workers total.
# P8: 8 Workers (Indices 0-7)
# Volta: 4 Workers (Indices 0-3, but logically 8-11)
TOTAL_WORKERS=12

# --- SUBMISSION 1: XEON-P8 ---
# 2 Nodes * 48 Cores = 96 Cores.
# 8 Workers * 12 CPUs = 96 Cores. (Perfect fit)
echo "Submitting P8 Array (Workers 0-7)..."
sbatch \
    --job-name="p8_sweep" \
    --partition=xeon-p8 \
    --array=0-7 \
    --cpus-per-task=12 \
    --mem=40G \
    --time=24:00:00 \
    --export=ALL,BATCH_TIMESTAMP=$TIMESTAMP,WORKER_OFFSET=0,TOTAL_WORKERS=$TOTAL_WORKERS \
    sweep_cpu_run_worker.sh

# --- SUBMISSION 2: XEON-G6-VOLTA ---
# 1 Node * 40 Cores = 40 Cores.
# 4 Workers * 10 CPUs = 40 Cores. (Perfect fit)
echo "Submitting Volta Array (Workers 8-11)..."
sbatch \
    --job-name="volta_cpu_sweep" \
    --partition=xeon-g6-volta \
    --array=0-3 \
    --cpus-per-task=10 \
    --mem=40G \
    --time=24:00:00 \
    --export=ALL,BATCH_TIMESTAMP=$TIMESTAMP,WORKER_OFFSET=8,TOTAL_WORKERS=$TOTAL_WORKERS \
    sweep_cpu_run_worker.sh

echo "Done. Check queue with 'squeue -u $USER'"