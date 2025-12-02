import sys
import itertools

# --- Define Sweep Grid ---
REG_K = [0.001, 0.01, 0.1]
REG_LAMBDA = [0.001, 0.01, 0.1]
REG_K_R = [0.001, 0.01, 0.1]
SEEDS = [0, 2, 4] # 1 is bad, 2 and 3 never ran

EXPERIMENT_NAME = "sweep_noPnorm_b1ortho_kR_sched"

def get_params(job_id):
    # Create all combinations
    configs = list(itertools.product(REG_K, REG_LAMBDA, REG_K_R, SEEDS))

    if job_id < len(configs):
        return configs[job_id], len(configs)
    return None, len(configs)

if __name__ == "__main__":
    # If checking for experiment name
    if len(sys.argv) > 1 and sys.argv[1] == "name":
        print(EXPERIMENT_NAME)
        sys.exit(0)

    if len(sys.argv) < 2:
        # Just return total count if no ID provided
        print(len(list(itertools.product(REG_K, REG_LAMBDA, REG_K_R, SEEDS))))
        sys.exit(0)

    job_id = int(sys.argv[1])
    params, total = get_params(job_id)
    
    if params:
        # Output: reg_k reg_lambda reg_k_R seed
        print(f"{params[0]} {params[1]} {params[2]} {params[3]}")
    else:
        sys.exit(1)