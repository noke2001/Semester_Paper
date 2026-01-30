#!/bin/bash
#SBATCH --job-name=TabPFN_Exp
#SBATCH --account=ls_math
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_tabpfn-%j.out
#SBATCH --error=slurm_tabpfn-%j.err

# --- TASK CONFIGURATION ---
TASK_ID="189356"
SUITE_ID="379"

# --- CONFIGURATION ---
PROJECT_ROOT="./main"
ABS_ROOT=$(readlink -f "$PROJECT_ROOT")
CSV_FILE="openml_suite_tasks.csv"

# Define your available weights
W_CLASS="./tabpfn_weights/v2.5/tabpfn-v2.5-classifier-v2.5_large-samples.ckpt"
W_REGRESS="./tabpfn_weights/v2.5/tabpfn-v2.5-regressor-v2.5_default.ckpt"

echo "--- Setup ---"
echo "Task: $TASK_ID | Suite: $SUITE_ID"

# --- AUTO-DETECTION LOGIC ---
# 1. Use awk to find the row where Col 1 (Suite) and Col 2 (Task) match.
# 2. Print Column 6 (Type).
# 3. 'tr -d' removes potential Windows carriage returns (\r) which break bash.
TASK_TYPE=$(awk -F, -v s="$SUITE_ID" -v t="$TASK_ID" '$1==s && $2==t {print $6}' "$CSV_FILE" | tr -d '\r')

echo "Detected Type: '$TASK_TYPE'"

if [[ "$TASK_TYPE" == *"Classification"* ]]; then
    WEIGHTS_FILE="$W_CLASS"
    echo "--> Mode: CLASSIFICATION (Using Large-Samples Weights)"
elif [[ "$TASK_TYPE" == *"Regression"* ]]; then
    WEIGHTS_FILE="$W_REGRESS"
    echo "--> Mode: REGRESSION (Using Default Weights)"
else
    echo "ERROR: Could not find Task $TASK_ID in Suite $SUITE_ID in $CSV_FILE"
    echo "Please check your CSV file and IDs."
    exit 1
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

echo "--- Job Finished ---"
