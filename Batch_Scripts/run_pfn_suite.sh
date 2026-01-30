#!/bin/bash
#SBATCH --job-name=TabPFN_Suite
#SBATCH --account=ls_math
#SBATCH --gpus=rtx_4090:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=12G
#SBATCH --time=120:00:00
#SBATCH --output=slurm_tabpfn_suite-%j.out
#SBATCH --error=slurm_tabpfn_suite-%j.err

# Example usage:
# ``` (bash)
#  sbatch run_pfn_clean_suite.sh 334
# ```

# --- INPUT HANDLING ---
# Check if an argument was provided. "$1" is the first command line argument.
if [ -z "$1" ]; then
    echo "=================================================="
    echo " [ERROR] Missing SUITE_ID Argument "
    echo "=================================================="
    echo "Usage: sbatch run_pfn_clean_suite.sh <SUITE_ID>"
    echo "Example: sbatch run_pfn_clean_suite.sh 337"
    exit 1
fi

# Assign the argument to the variable
SUITE_ID="$1"

# --- CONFIGURATION ---
PROJECT_ROOT="./main"
ABS_ROOT=$(readlink -f "$PROJECT_ROOT")
CSV_FILE="${PROJECT_ROOT}/openml_suite_tasks.csv"

# Weights Paths
W_CLASS="./tabpfn_weights/v2.5/tabpfn-v2.5-classifier-v2.5_large-samples.ckpt"
W_REGRESS="./tabpfn_weights/v2.5/tabpfn-v2.5-regressor-v2.5_default.ckpt"

echo "=================================================="
echo "Starting Batch Run for SUITE_ID: $SUITE_ID"
echo "=================================================="

# 1. Extract all TASK_IDs for this Suite from the CSV
TASK_LIST=$(awk -F, -v s="$SUITE_ID" '$1==s {print $2}' "$CSV_FILE" | tr -d '\r')

# Check if we found any tasks
if [ -z "$TASK_LIST" ]; then
    echo "ERROR: No tasks found for Suite $SUITE_ID in $CSV_FILE"
    echo "Please check the ID and the CSV file."
    exit 1
fi

echo "Found Tasks: $TASK_LIST"
echo "--------------------------------------------------"

# 2. Loop through each task
for TASK_ID in $TASK_LIST; do

    echo -e "\n[$(date +'%H:%M:%S')] >>> Processing Task: $TASK_ID"

    # --- AUTO-DETECTION LOGIC (Per Task) ---
    TASK_TYPE=$(awk -F, -v s="$SUITE_ID" -v t="$TASK_ID" '$1==s && $2==t {print $6}' "$CSV_FILE" | tr -d '\r')
    
    # Determine weights based on task type
    if [[ "$TASK_TYPE" == *"Classification"* ]]; then
        WEIGHTS_FILE="$W_CLASS"
        echo "    -> Type: CLASSIFICATION (Large-Samples Weights)"
    elif [[ "$TASK_TYPE" == *"Regression"* ]]; then
        WEIGHTS_FILE="$W_REGRESS"
        echo "    -> Type: REGRESSION (Default Weights)"
    else
        echo "    -> [WARNING] Unknown type '$TASK_TYPE' for Task $TASK_ID. Skipping..."
        continue
    fi

    # --- EXECUTION ---
    apptainer exec --nv --fakeroot \
        --bind /etc/OpenCL/vendors \
        --bind "$ABS_ROOT:$ABS_ROOT" \
        --cleanenv \
        --env PYTHONNOUSERSITE=1 \
        --env TABPFN_DISABLE_TELEMETRY=1 \
        --env PYTORCH_ALLOC_CONF=max_split_size_mb:1024 \
        --env PYTHONPATH=$PROJECT_ROOT \
        --env HF_HUB_OFFLINE=1 \
        --env NUMBA_CACHE_DIR=/tmp/numba_cache \
        --env TABPFN_WEIGHTS_PATH="${WEIGHTS_FILE}" \
        SMAC_optuna.sif \
        python3 -u ${PROJECT_ROOT}/TabPFN_experiment_smac.py \
        --suite_id ${SUITE_ID} \
        --task_id ${TASK_ID} \
        --result_folder ${PROJECT_ROOT}/smac_results/
    
    echo "    -> Task $TASK_ID Finished."
    echo "--------------------------------------------------"

done

echo "=================================================="
echo "All tasks in Suite $SUITE_ID completed."
echo "=================================================="
