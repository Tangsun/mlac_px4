#!/bin/bash

# 1. Generate a single timestamp for the whole batch
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "Launching Massive Hybrid Sweep. Timestamp: $TIMESTAMP"

# --- CONFIGURATION ---
# New Limits:
# P8: 8 Nodes * 48 Cores = 384 Cores.
# Volta: 4 Nodes * 40 Cores = 160 Cores.
# Strategy: Run 4 workers per node on both partitions.
# P8 Workers: 8 * 4 = 32
# Volta Workers: 4 * 4 = 16
# Total Workers = 48
TOTAL_WORKERS=48

# --- SUBMISSION 1: XEON-P8 ---
# 32 Workers (Indices 0-31)
# Each gets 12 CPUs (32 * 12 = 384 Cores = 100% of 8 Nodes)
echo "Submitting P8 Array (Workers 0-31)..."
sbatch \
    --job-name="p8_sweep" \
    --partition=xeon-p8 \
    --array=0-31 \
    --cpus-per-task=12 \
    --mem=40G \
    --time=48:00:00 \
    --export=ALL,BATCH_TIMESTAMP=$TIMESTAMP,WORKER_OFFSET=0,TOTAL_WORKERS=$TOTAL_WORKERS \
    sweep_cpu_run_worker.sh

# --- SUBMISSION 2: XEON-G6-VOLTA ---
# 16 Workers (Indices 0-15)
# Each gets 10 CPUs (16 * 10 = 160 Cores = 100% of 4 Nodes)
# Offset starts at 32 so these logical IDs are 32-47
echo "Submitting Volta Array (Workers 32-47)..."
sbatch \
    --job-name="volta_cpu_sweep" \
    --partition=xeon-g6-volta \
    --array=0-15 \
    --cpus-per-task=10 \
    --mem=40G \
    --time=48:00:00 \
    --export=ALL,BATCH_TIMESTAMP=$TIMESTAMP,WORKER_OFFSET=32,TOTAL_WORKERS=$TOTAL_WORKERS \
    sweep_cpu_run_worker.sh

echo "Done. Check queue with 'squeue -u $USER'. You should see 48 jobs running/pending."