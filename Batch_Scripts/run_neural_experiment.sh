#!/bin/bash
#SBATCH --job-name=Neural_Exp_SMAC
#SBATCH --account=ls_math
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=48:00:00
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=slurm_neural-%j.out
#SBATCH --error=slurm_neural-%j.err

# --- TASK CONFIGURATION ---
TASK_ID="361110"
SUITE_ID="334"

# --- PATHS ---
CACHE_PATH="./openml_cache"
PROJECT_ROOT="./main"
ABS_ROOT=$(readlink -f "$PROJECT_ROOT")

# --- Apptainer CMD ---
APP_CMD="apptainer exec --nv --bind /etc/OpenCL/vendors --bind "$ABS_ROOT:$ABS_ROOT" --cleanenv --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 --env XDG_CACHE_HOME=$CACHE_PATH --env NUMBA_CACHE_DIR=/tmp/numba_cache --env PYTHONPATH=$PROJECT_ROOT SMAC_optuna.sif python3"

echo "--- Running Experiment ---"
echo "Task ID: $TASK_ID"
echo "Suite ID: $SUITE_ID"

$APP_CMD $PROJECT_ROOT/neural_experiment_smac.py --suite_id $SUITE_ID --task_id $TASK_ID --result_folder $PROJECT_ROOT/smac_results/

echo "--- Job Finished ---"
