import os
import re
import glob

def parse_log_file(filepath):
    """
    Parses a single log file to find completion status, best step, and best valid loss.
    Returns:
        dict: {'status': 'complete', 'step': int, 'loss': float}
        OR
        dict: {'status': 'incomplete'}
    """
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception as e:
        return {'status': 'error', 'msg': str(e)}

    # 1. Check if training completed successfully
    # Looking for: "done (5183.53 s)! Best step index for meta params: 703"
    done_pattern = r"done .*! Best step index for meta params:\s*(\d+)"
    done_match = re.search(done_pattern, content)

    if not done_match:
        return {'status': 'incomplete'}

    best_step = int(done_match.group(1))

    # 2. Find the valid loss associated with that specific step
    # Logic: Look for the specific block "update best meta params at step X" 
    # and capture the "valid loss" immediately following it.
    
    # We use re.DOTALL so the dot (.) matches newlines, allowing us to search across multiple lines
    # Pattern explanation:
    #   update best meta params at step {best_step}  <- find the specific header
    #   .*?                                          <- match any text (non-greedy)
    #   valid loss:\s*([\d\.]+)                      <- capture the loss float
    
    loss_pattern = rf"update best meta params at step\s+{best_step}.*?valid loss:\s*([\d\.]+)"
    loss_match = re.search(loss_pattern, content, re.DOTALL)

    if loss_match:
        best_loss = float(loss_match.group(1))
    else:
        # Fallback: If regex fails (formatting oddity), find the global minimum valid loss
        # This acts as a safety net
        all_losses = re.findall(r"valid loss:\s*([\d\.]+)", content)
        if all_losses:
            best_loss = min([float(l) for l in all_losses])
        else:
            # Should ideally not happen if "done" was found
            return {'status': 'incomplete'} 

    return {
        'status': 'complete',
        'step': best_step,
        'loss': best_loss
    }

def generate_training_report(target_directory):
    """
    Scans directory, parses logs, ranks results, and writes a report file.
    """
    if not os.path.exists(target_directory):
        print(f"Error: Directory '{target_directory}' not found.")
        return

    completed_runs = []
    incomplete_runs = []

    print(f"Scanning directory: {target_directory} ...")

    # Walk through the directory to find all .log files
    for root, dirs, files in os.walk(target_directory):
        for file in files:
            if file.endswith(".log"):
                full_path = os.path.join(root, file)
                
                # Use the parent folder name as the "Training Name"
                # e.g., seed_0_regK_0.1_...
                training_name = os.path.basename(root)
                
                result = parse_log_file(full_path)

                if result['status'] == 'complete':
                    completed_runs.append({
                        'name': training_name,
                        'loss': result['loss'],
                        'step': result['step'],
                        'file': file # Keep track of which log file it was
                    })
                else:
                    incomplete_runs.append({
                        'name': training_name,
                        'file': file
                    })

    # --- Sort the Results ---
    # Sort by loss (ascending), then by step (ascending)
    completed_runs.sort(key=lambda x: (x['loss'], x['step']))

    # --- Generate Report Content ---
    lines = []
    lines.append("=" * 80)
    lines.append(f"TRAINING SUMMARY REPORT")
    lines.append(f"Directory: {target_directory}")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Total Logs Found: {len(completed_runs) + len(incomplete_runs)}")
    lines.append(f"Completed:        {len(completed_runs)}")
    lines.append(f"Incomplete:       {len(incomplete_runs)}")
    lines.append("")
    
    lines.append("-" * 80)
    lines.append("COMPLETED RUNS (Ranked by Lowest Valid Loss)")
    lines.append("-" * 80)
    # Table Header
    header = f"{'Rank':<6} | {'Valid Loss':<12} | {'Step':<6} | {'Training Name'}"
    lines.append(header)
    lines.append("-" * len(header))

    for idx, run in enumerate(completed_runs, 1):
        line = f"{idx:<6} | {run['loss']:<12.4f} | {run['step']:<6} | {run['name']}"
        lines.append(line)

    lines.append("")
    lines.append("-" * 80)
    lines.append("INCOMPLETE / FAILED TRAINING INSTANCES")
    lines.append("-" * 80)
    
    if not incomplete_runs:
        lines.append("None! All runs completed successfully.")
    else:
        for run in incomplete_runs:
            lines.append(f"- {run['name']}  (File: {run['file']})")

    # --- Write to File ---
    report_path = os.path.join(target_directory, "summary_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nDone! Report generated at:\n{report_path}")
    
    # Print top 5 to console for quick view
    print("\n--- Top 5 Runs ---")
    print(header)
    for i in range(min(5, len(completed_runs))):
        run = completed_runs[i]
        print(f"{i+1:<6} | {run['loss']:<12.4f} | {run['step']:<6} | {run['name']}")

# --- Usage Example ---
if __name__ == "__main__":
    # Update this path to the specific timestamp folder you want to analyze
    # Example: "train_log/20251201_210753_sweep_noPnorm_b1ortho_kR_sched"
    
    # You can pass the directory as an argument or hardcode it here
    import sys
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        # CHANGE THIS to your specific log directory
        target_dir = "train_log/20251201_210753_sweep_noPnorm_b1ortho_kR_sched"

    generate_training_report(target_dir)