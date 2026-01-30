#!/bin/bash
#SBATCH --job-name=Neural_Suite_SMAC
#SBATCH --account=ls_math
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1G
#SBATCH --time=110:00:00
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=slurm_neural_suite_%j.out
#SBATCH --error=slurm_neural_suite_%j.err

# Example Usage:
# '''(bash)
# $ sbatch run_neural_suite.sh 334
# '''

# --- CONFIGURATION ---
CSV_FILE="openml_suite_tasks.csv"
CACHE_PATH="./openml_cache"
PROJECT_ROOT="./main"
ABS_ROOT=$(readlink -f "$PROJECT_ROOT")

# --- INPUT VALIDATION ---
SUITE_ID_INPUT=$1

if [ -z "$SUITE_ID_INPUT" ]; then
    echo "Error: No Suite ID provided."
    echo "Usage: sbatch run_neural_suite.sh <SUITE_ID>"
    exit 1
fi

if [ ! -f "$CSV_FILE" ]; then
    echo "Error: CSV file '$CSV_FILE' not found."
    exit 1
fi

echo "--- Starting Batch Run for Suite: $SUITE_ID_INPUT ---"

# --- PARSE CSV AND EXECUTE ---
# Skip the header -> Filter by Suite ID -> Loop through Tasks
tail -n +2 "$CSV_FILE" | awk -F, -v target_suite="$SUITE_ID_INPUT" '$1 == target_suite {print $2}' | while read -r TASK_ID; do
    
    # Remove any potential whitespace/newlines from TASK_ID
    TASK_ID=$(echo "$TASK_ID" | tr -d '[:space:]')
    
    if [ -z "$TASK_ID" ]; then
        continue
    fi

    echo "------------------------------------------------"
    echo "Processing SUITE: $SUITE_ID_INPUT | TASK: $TASK_ID"
    echo "------------------------------------------------"

    # --- EXECUTION ---
    # Using the specific binds and env vars from your Neural script
    apptainer exec --nv \
        --bind /etc/OpenCL/vendors \
        --bind "$ABS_ROOT:$ABS_ROOT" \
        --cleanenv \
        --env PYTHONNOUSERSITE=1 \
        --env HF_HUB_OFFLINE=1 \
        --env XDG_CACHE_HOME=$CACHE_PATH \
        --env NUMBA_CACHE_DIR=/tmp/numba_cache \
        --env PYTHONPATH=$PROJECT_ROOT \
        SMAC_optuna.sif \
        python3 -u $PROJECT_ROOT/neural_experiment_smac.py \
        --suite_id $SUITE_ID_INPUT \
        --task_id $TASK_ID \
        --result_folder $PROJECT_ROOT/smac_results/

    # Check exit status
    if [ $? -ne 0 ]; then
        echo "WARNING: Task $TASK_ID failed. Proceeding to next task..."
    else
        echo "Task $TASK_ID completed successfully."
    fi

done

echo "--- All tasks for Suite $SUITE_ID_INPUT finished ---"
