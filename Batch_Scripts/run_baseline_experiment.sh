#!/bin/bash
#SBATCH --job-name=Baseline_Exp
#SBATCH --account=ls_math
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_baseline-%j.out
#SBATCH --error=slurm_baseline-%j.err

# --- TASK CONFIGURATION ---
TASK_ID="361110"
SUITE_ID="334"

# --- PATHS ---
CACHE_PATH="./openml_cache"
PROJECT_ROOT="./main"
ABS_ROOT=$(readlink -f "$PROJECT_ROOT")

APP_CMD="apptainer exec --bind /etc/OpenCL/vendors --bind "$ABS_ROOT:$ABS_ROOT" --cleanenv --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 --env XDG_CACHE_HOME=$CACHE_PATH --env NUMBA_CACHE_DIR=/tmp/numba_cache --env PYTHONPATH=$PROJECT_ROOT SMAC_optuna.sif python3"
echo "--- Running Baseline Experiment ---"
echo "Date: $(date)"
echo "Task ID: $TASK_ID"
echo "Suite ID: $SUITE_ID"
echo "Node: $HOSTNAME"

$APP_CMD -u $PROJECT_ROOT/baseline_experiment.py \
    --suite_id $SUITE_ID \
    --task_id $TASK_ID \
    --result_folder $PROJECT_ROOT/smac_results/