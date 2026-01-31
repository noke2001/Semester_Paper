#!/bin/bash
#SBATCH --job-name=Baseline_Suite
#SBATCH --account=ls_math
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_baseline-%j.out
#SBATCH --error=slurm_baseline-%j.err

# Example Usage:
# $ sbatch run_baseline_suite.sh 379

# --- PATHS ---
CSV_FILE="openml_suite_tasks.csv"
CACHE_PATH="./openml_cache"
PROJECT_ROOT="./main"
ABS_ROOT=$(readlink -f "$PROJECT_ROOT")

# --- INPUT VALIDATION ---
SUITE_ID_INPUT=$1

if [ -z "$SUITE_ID_INPUT" ]; then
    echo "Error: No Suite ID provided."
    echo "Usage: sbatch run_adv_suite.sh <SUITE_ID>"
    exit 1
fi

if [ ! -f "$CSV_FILE" ]; then
    echo "Error: CSV file '$CSV_FILE' not found."
    exit 1
fi

APP_CMD="apptainer exec --bind /etc/OpenCL/vendors --bind "$ABS_ROOT:$ABS_ROOT" --cleanenv --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 --env XDG_CACHE_HOME=$CACHE_PATH --env NUMBA_CACHE_DIR=/tmp/numba_cache --env PYTHONPATH=$PROJECT_ROOT SMAC_optuna.sif python3"

echo "--- Starting Batch Run for Suite: $SUITE_ID_INPUT ---"
echo "--- Date: $(date) ---"

tail -n +2 "$CSV_FILE" | awk -F, -v target_suite="$SUITE_ID_INPUT" '$1 == target_suite {print $2}' | while read -r TASK_ID; do

    # Remove potential whitespace/newlines
    TASK_ID=$(echo "$TASK_ID" | tr -d '[:space:]')

    if [ -z "$TASK_ID" ]; then
        continue
    fi

    echo "=========================================================="
    echo "Processing SUITE: $SUITE_ID_INPUT | TASK: $TASK_ID"
    echo "Started at: $(date)"
    echo "=========================================================="

    # --- EXECUTION ---
    $APP_CMD -u $PROJECT_ROOT/baseline_experiment.py \
        --suite_id $SUITE_ID_INPUT \
        --task_id $TASK_ID \
        --result_folder $PROJECT_ROOT/smac_results/

    # Check exit status
    if [ $? -ne 0 ]; then
        echo "!!! WARNING: Task $TASK_ID failed. Proceeding to next task..."
    else
        echo ">>> Task $TASK_ID completed successfully."
    fi

    # Tiny sleep to let file buffers flush and system settle before next heavy task
    sleep 5

done

echo "--- All tasks for Suite $SUITE_ID_INPUT finished at $(date) ---"