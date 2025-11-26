import sys
import itertools

# --- Define Sweep Grid ---
# 4 * 4 * 5 = 80 total experiments
REG_K = [0.0, 0.01, 0.02, 0.05]
REG_LAMBDA = [0.0, 0.01, 0.02, 0.05]
SEEDS = [0, 4] # 1 is bad, 2 and 3 never ran

def get_params(job_id):
    # Create all combinations
    configs = list(itertools.product(REG_K, REG_LAMBDA, SEEDS))

    if job_id < len(configs):
        return configs[job_id], len(configs)
    return None, len(configs)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Just return total count if no ID provided
        print(len(list(itertools.product(REG_K, REG_LAMBDA, SEEDS))))
        sys.exit(0)

    job_id = int(sys.argv[1])
    params, total = get_params(job_id)
    
    if params:
        # Output: reg_k reg_lambda seed
        print(f"{params[0]} {params[1]} {params[2]}")
    else:
        sys.exit(1)