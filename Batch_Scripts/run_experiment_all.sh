#!/bin/bash
#SBATCH --job-name=SMAC_Adv
#SBATCH --account=ls_math
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_all-%j.out
#SBATCH --error=slurm_all-%j.err

# --- TASK CONFIGURATION ---
TASK_ID="7"
SUITE_ID="379"

# --- PATHS ---
CACHE_PATH="./openml_cache"
PROJECT_ROOT="./main"
ABS_ROOT=$(readlink -f "$PROJECT_ROOT")
CSV_FILE="openml_suite_tasks.csv"

# --- TabPFN Weights ---
W_CLASS="./tabpfn_weights/v2.5/tabpfn-v2.5-classifier-v2.5_large-samples.ckpt"
W_REGRESS="./tabpfn_weights/v2.5/tabpfn-v2.5-regressor-v2.5_default.ckpt"

# --- Apptainer CMD ---
APP_CMD="apptainer exec --nv --bind /etc/OpenCL/vendors --bind "$ABS_ROOT:$ABS_ROOT" --cleanenv --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 --env XDG_CACHE_HOME=$CACHE_PATH --env NUMBA_CACHE_DIR=/tmp/numba_cache --env PYTHONPATH=$PROJECT_ROOT SMAC_optuna.sif python3"

echo "==============================================="
echo "Date: $(date)"
echo "Task ID: $TASK_ID"
echo "Suite ID: $SUITE_ID"
echo "Node: $HOSTNAME"
echo "==============================================="

# --- Baseline Models ---
echo "==============================================="
echo "===== Running Baseline Models Experiment ======"
echo "==============================================="

$APP_CMD -u $PROJECT_ROOT/baseline_experiment.py \
    --suite_id $SUITE_ID \
    --task_id $TASK_ID \
    --result_folder $PROJECT_ROOT/smac_results/

# --- Tree Ensemble Models ---
echo "==============================================="
echo "==== Running Tree Models Experiment (SMAC) ===="
echo "==============================================="
$APP_CMD -u $PROJECT_ROOT/tree_experiment_smac.py \
    --suite_id $SUITE_ID \
    --task_id $TASK_ID \
    --result_folder $PROJECT_ROOT/smac_results/

# --- Neural Network Models ---
echo "==============================================="
echo "===== Running NN Models Experiment (SMAC) ====="
echo "==============================================="
$APP_CMD -u $PROJECT_ROOT/neural_experiment_smac.py \
    --suite_id $SUITE_ID \
    --task_id $TASK_ID \
    --result_folder $PROJECT_ROOT/smac_results/

# --- TabPFN Model ---
echo "==============================================="
echo "======= Running TabPFN Model Experiment ======="
echo "==============================================="
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

# --- Advanced Models ---
echo "=============================================="
echo "== Running Advanced Model Experiment (SMAC) =="
echo "=============================================="
$APP_CMD -u $PROJECT_ROOT/adv_trial_all_smac.py \
    --suite_id $SUITE_ID \
    --task_id $TASK_ID \
    --result_folder $PROJECT_ROOT/smac_results/

echo "=============================================="
echo "================ Job Finished ================"
echo "=============================================="
