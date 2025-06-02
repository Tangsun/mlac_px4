#!/bin/bash
#SBATCH --job-name=gpu_param_sweep
#SBATCH -o %j.log                     # Standard output and error log (%j expands to job ID)
#SBATCH --nodes=1                     # We want to run on a single node
#SBATCH --ntasks=1                    # The main script is one task; it will manage sub-processes
#SBATCH --gres=gpu:6                  # Request all 6 GPUs on the node

# --- Node Specification (Choose ONE of the following options) ---
# Option 1: If 'cpu-gpu-rtx8000' is the EXACT hostname of the node:
#SBATCH --nodelist=cpu-gpu-rtx8000

# --- Resource Allocation for the main script and its children ---
# These CPUs are for the Python processes. If each needs ~4 CPUs: 6 GPUs * 4 CPUs/GPU = 24
SBATCH --cpus-per-task=24            # Adjust based on CPU needs of your Python scripts
# Host memory for all processes. Adjust based on combined RAM needs (not VRAM).
SBATCH --mem=128G                    # Example: 128GB of host RAM

echo "Job started on: $(hostname)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Node List: $SLURM_JOB_NODELIST"
echo "SLURM GPUs on Node (total on node): $SLURM_GPUS_ON_NODE"
echo "SLURM Job GPUs (allocated to this job): $SLURM_JOB_GPUS" # Expected: 0,1,2,3,4,5 or similar
echo "CUDA_VISIBLE_DEVICES (set by SLURM): $CUDA_VISIBLE_DEVICES"

# --- Environment Setup ---
# Adjust the path to your miniconda and environment name if necessary
echo "Initializing Conda environment..."
source /mnt/home/tangsun/miniconda3/etc/profile.d/conda.sh
conda activate mlac_env
echo "Conda environment activated."

# --- Parameter Definitions for the Sweep ---
# Define arrays for each parameter you want to sweep
reg_Lambda_values=(0.008 0.012 0.016 0.02)
reg_K_values=(0.004 0.006 0.008 0.01)
pfreq_values=(10 20)
pnorm_init_values=(1.5 2.5)
learning_rate_values=(0.01 0.005 0.002) # Added learning rate

# Check if paired arrays have the same number of elements
if [ ${#reg_Lambda_values[@]} -ne ${#reg_K_values[@]} ]; then
  echo "Error: reg_Lambda_values and reg_K_values arrays must have the same number of elements for pairing."
  exit 1
fi

# Common parameters for all Python script runs (if any)
SEED_BASE=0 # Base seed, can be varied per run if desired
M_TRAJ=50
META_EPOCHS=1000
REG_P=1.0
REG_K_R=0.0001
Z_WEIGHT=1.5

# --- Output Directory ---
# Create a base directory for all results from this sweep job
BASE_OUTPUT_DIR="sweep_results_job_${SLURM_JOB_ID}_paired_lk" # Clarified base dir name
mkdir -p "$BASE_OUTPUT_DIR"
echo "Base output directory: $BASE_OUTPUT_DIR"

# --- Generate All Parameter Combinations ---
declare -a all_parameter_sets # Array to hold argument strings for each run
experiment_counter=0

echo "Generating parameter sets..."
# Loop through indices for paired reg_Lambda and reg_K
for i in "${!reg_Lambda_values[@]}"; do
  lambda_val="${reg_Lambda_values[$i]}"
  k_val="${reg_K_values[$i]}"

  for p_freq in "${pfreq_values[@]}"; do
    for pnorm_init in "${pnorm_init_values[@]}"; do
      for lr_val in "${learning_rate_values[@]}"; do # Loop for learning_rate
        experiment_counter=$((experiment_counter + 1))
        current_seed=$((SEED_BASE + experiment_counter -1)) # Optional: vary seed per run

        # Define a unique output directory for this specific parameter combination
        output_dir="${BASE_OUTPUT_DIR}/run_${experiment_counter}_lambda_${lambda_val}_k_${k_val}_pfreq_${p_freq}_pnorm_${pnorm_init}_lr_${lr_val}"
        mkdir -p "$output_dir"

        # Construct the argument string for the Python script
        # Ensure your train_z_up_kR.py script accepts these arguments
        param_set="--seed $current_seed "
        param_set+="--M $M_TRAJ "
        param_set+="--pnorm_init $pnorm_init "
        param_set+="--p_freq $p_freq "
        param_set+="--learning_rate $lr_val " # Added learning rate argument
        param_set+="--meta_epochs $META_EPOCHS "
        param_set+="--reg_P $REG_P "
        param_set+="--reg_Lambda $lambda_val "
        param_set+="--reg_K $k_val "
        param_set+="--reg_k_R $REG_K_R "
        param_set+="--z_weight $Z_WEIGHT "
        param_set+="--output_dir $output_dir"
        # Add other parameters to the string as needed

        all_parameter_sets+=("$param_set")
      done
    done
  done
done
echo "Generated ${#all_parameter_sets[@]} unique parameter sets for experiments."

# --- GPU Management and Job Launching Logic ---
# Determine the number of GPUs allocated by SLURM for this job
num_gpus_allocated=$(echo "$SLURM_JOB_GPUS" | tr ',' ' ' | wc -w)

if [[ -z "$SLURM_JOB_GPUS" || "$num_gpus_allocated" -eq 0 ]]; then
    if [[ ! -z "$CUDA_VISIBLE_DEVICES" ]]; then
        num_gpus_allocated=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' ' ' | wc -w)
    else
        echo "Error: Could not determine number of allocated GPUs from SLURM environment variables."
        echo "Please ensure --gres=gpu:N is working correctly. Defaulting to 1 GPU, which might not be optimal."
        num_gpus_allocated=1
    fi
fi
# Ensure num_gpus_allocated is at least 1, even if parsing failed, to avoid issues with slot array
if ! [[ "$num_gpus_allocated" =~ ^[0-9]+$ ]] || [[ "$num_gpus_allocated" -lt 1 ]]; then
    echo "Warning: num_gpus_allocated was not a positive integer. Defaulting to 1."
    num_gpus_allocated=1
fi

echo "This script will manage up to $num_gpus_allocated concurrent Python processes on distinct GPUs."

max_concurrent_jobs=$num_gpus_allocated
declare -A running_pid_to_gpu_slot_map # Stores PID -> GPU_Slot_ID (0 to N-1)
declare -a active_job_pids             # Stores PIDs of currently running Python scripts
available_gpu_slots=()
for (( j=0; j<num_gpus_allocated; j++ )); do
    available_gpu_slots+=($j)
done

current_command_index=0
total_commands_to_run=${#all_parameter_sets[@]}

# Function to launch a single Python script on a specific GPU slot
launch_experiment_on_gpu() {
    local script_params="$1"
    local gpu_slot_id="$2"
    local exp_num="$3"

    echo "Experiment ${exp_num}/${total_commands_to_run}: Launching on GPU Slot ${gpu_slot_id}..."
    echo "  Params: ${script_params}"
    (
      export CUDA_VISIBLE_DEVICES=${gpu_slot_id} # Assign specific GPU to this sub-process
      # Ensure your python script is correctly named and located
      python train_z_up_kR.py ${script_params}
    ) &
    local job_pid=$!
    running_pid_to_gpu_slot_map[$job_pid]=${gpu_slot_id}
    active_job_pids+=(${job_pid})
    echo "  PID ${job_pid} started on GPU Slot ${gpu_slot_id} (CUDA_VISIBLE_DEVICES=${gpu_slot_id})."
}

# Main loop to manage launching experiments
while [ "$current_command_index" -lt "$total_commands_to_run" ] || [ "${#active_job_pids[@]}" -gt 0 ]; do
    while [ "$current_command_index" -lt "$total_commands_to_run" ] && [ "${#available_gpu_slots[@]}" -gt 0 ]; do
        params_for_current_run="${all_parameter_sets[$current_command_index]}"
        gpu_slot_to_use=${available_gpu_slots[0]}
        available_gpu_slots=("${available_gpu_slots[@]:1}")

        launch_experiment_on_gpu "$params_for_current_run" "$gpu_slot_to_use" "$((current_command_index + 1))"
        current_command_index=$((current_command_index + 1))
    done

    if [ "${#active_job_pids[@]}" -gt 0 ]; then
        wait -n -p finished_pid # Requires bash 4.3+
        exit_status=$?
        if [ -z "$finished_pid" ]; then # Fallback if -p not supported or failed
            # This fallback is more complex: iterate and check which PID is gone
            # For simplicity, assuming bash 4.3+ for now. If issues, this part needs refinement.
            echo "Warning: 'wait -n -p' might not be supported or failed to capture PID. Checking PIDs manually (not implemented in this simple fallback)."
            # A simple 'wait -n' would still wait, but we wouldn't know which PID finished directly
            # We would then need to iterate through active_job_pids and use 'kill -0 $pid' to see which one is gone
            # For now, we proceed hoping finished_pid was captured or the next check handles it.
             sleep 1 # Give a moment for process table to update
             # Crude check: find a PID that is no longer in running_pid_to_gpu_slot_map keys
             # This is not robust, 'wait -n -p' is preferred.
            local found_finished_pid=""
            for pid_in_map in "${!running_pid_to_gpu_slot_map[@]}"; do
                if ! kill -0 "$pid_in_map" 2>/dev/null; then # Check if process exists
                    found_finished_pid="$pid_in_map"
                    break
                fi
            done
            if [ -n "$found_finished_pid" ]; then
                finished_pid="$found_finished_pid"
                echo "Fallback: Detected finished PID $finished_pid"
            else
                echo "Error: Could not determine which PID finished. GPU slot might not be reclaimed correctly."
                # To prevent infinite loop in case of error, break after some time or retries (not implemented)
                # Or simply remove the first active_job_pid as a guess (risky)
                if [ "${#active_job_pids[@]}" -gt 0 ]; then
                    finished_pid="${active_job_pids[0]}" # Risky guess
                     echo "Warning: Assuming PID ${finished_pid} finished to attempt recovery."
                else
                    continue # No active PIDs, loop should eventually exit
                fi
            fi
        fi


        gpu_slot_freed=${running_pid_to_gpu_slot_map[$finished_pid]}
        if [ -n "$gpu_slot_freed" ]; then # Ensure gpu_slot_freed is not empty
             available_gpu_slots+=("$gpu_slot_freed")
             echo "Job PID $finished_pid (which was on GPU Slot $gpu_slot_freed) finished with status $exit_status."
        else
            echo "Warning: Could not determine GPU slot for finished PID $finished_pid. Slot not reclaimed."
        fi


        new_active_pids=()
        for pid_in_list in "${active_job_pids[@]}"; do
            if [ "$pid_in_list" -ne "$finished_pid" ]; then
                new_active_pids+=("$pid_in_list")
            fi
        done
        active_job_pids=("${new_active_pids[@]}")
        unset running_pid_to_gpu_slot_map[$finished_pid]
    fi

    if [ "$current_command_index" -ge "$total_commands_to_run" ] && [ "${#active_job_pids[@]}" -eq 0 ]; then
        break
    fi
done

echo "All $total_commands_to_run experiments have been processed."
echo "Job completed successfully."