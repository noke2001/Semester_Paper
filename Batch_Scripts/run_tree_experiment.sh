#!/bin/bash
#SBATCH --job-name=Tree_Exp
#SBATCH --account=ls_math
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=1G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_tree-%j.out
#SBATCH --error=slurm_tree-%j.err

# --- If GPU ---
# Note: 
# - that this will not work for the larger datasets due to OOM errors
# - only accellerates LGBM training, RandomForest not impacted

# #SBATCH --partition=gpu
# #SBATCH --gpus=1
# #SBATCH --gres=gpumem:20GB

# --- PATHS ---
CACHE_PATH="./openml_cache"
PROJECT_ROOT="./main"

# --- TASK ---
TASK_ID="361110"
SUITE_ID="334"

# --- Apptainer CMD ---
APP_CMD="apptainer exec --nv --bind /etc/OpenCL/vendors --cleanenv --env PYTHONNOUSERSITE=1 --env NUMBA_CACHE_DIR=/tmp/numba_cache --env XDG_CACHE_HOME=$CACHE_PATH SMAC_optuna.sif python3"

echo "--- Running Experiment ---"
echo "Task ID: $TASK_ID"
echo "Suite ID: $SUITE_ID"

$APP_CMD $PROJECT_ROOT/tree_experiment_smac.py --suite_id $SUITE_ID --task_id $TASK_ID --result_folder $PROJECT_ROOT/smac_results/

echo "--- Job Finished ---"
