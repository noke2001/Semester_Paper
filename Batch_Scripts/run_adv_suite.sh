#!/bin/bash
#SBATCH --job-name=SMAC_Adv_Suite
#SBATCH --account=ls_math
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=120:00:00
#SBATCH --output=slurm_adv_suite_%j.out
#SBATCH --error=slurm_adv_suite_%j.err

# NOTE: 
# - GPUs removed (Code is CPU-optimized for stability)
# - CPUs set to 4 (Prevent R/SMAC Thread Overload/Deadlock)
# - Wall time increased for long sequences

# Example Usage:
# $ sbatch run_adv_suite.sh 337

# --- CONFIGURATION ---
CSV_FILE="openml_suite_tasks.csv"
CACHE_PATH="./openml_cache"
PROJECT_ROOT="./main"
ABS_ROOT=$(readlink -f "$PROJECT_ROOT")

echo "--- Cleaning Python Cache to Force Code Update ---"
find $PROJECT_ROOT -name "__pycache__" -type d -exec rm -rf {} +
find $PROJECT_ROOT -name "*.pyc" -delete
echo "--- Cache Cleaned ---"

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

# --- WORK IN PROGRESS ---
# --- THREADING CONFIGURATION (CRITICAL FOR STABILITY) ---
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export VECLIB_MAXIMUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1
# export R_SIGNAL_HANDLERS=0

# --- APPTAINER SETUP ---
# CPU-Only Mode (No --nv flag)
# APP_CMD="apptainer exec --env SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK --bind /etc/OpenCL/vendors --bind "$ABS_ROOT:$ABS_ROOT" --cleanenv --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 --env XDG_CACHE_HOME=$CACHE_PATH --env NUMBA_CACHE_DIR=/tmp/numba_cache --env PYTHONPATH=$PROJECT_ROOT SMAC_optuna.sif python3"
APP_CMD="apptainer exec --bind /etc/OpenCL/vendors --bind "$ABS_ROOT:$ABS_ROOT" --cleanenv --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 --env XDG_CACHE_HOME=$CACHE_PATH --env NUMBA_CACHE_DIR=/tmp/numba_cache --env PYTHONPATH=$PROJECT_ROOT SMAC_optuna.sif python3"

echo "--- Starting Batch Run for Suite: $SUITE_ID_INPUT ---"
# echo "--- Mode: Advanced Models / CPU (4 Cores / Serial) ---"
echo "--- Date: $(date) ---"

# --- PARSE CSV AND EXECUTE ---
# Skip header -> Filter by Suite ID -> Loop through Tasks
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
    # We use the $APP_CMD variable defined above which contains the Safe Config
    $APP_CMD -u $PROJECT_ROOT/adv_trial_all_smac.py \
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