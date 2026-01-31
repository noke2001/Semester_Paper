#!/bin/bash
#SBATCH --job-name=SMAC_Adv
#SBATCH --account=ls_math
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_adv-%j.out
#SBATCH --error=slurm_adv-%j.err

# --- TASK CONFIGURATION ---
TASK_ID="361110"
SUITE_ID="334"

# --- PATHS ---
CACHE_PATH="./openml_cache"
PROJECT_ROOT="./main"
ABS_ROOT=$(readlink -f "$PROJECT_ROOT")

# --- THREADING CONFIGURATION ---
export OMP_NUM_THREADS=1 #$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=1 #$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1 #$SLURM_CPUS_PER_TASK
export R_SIGNAL_HANDLERS=0

# --- Apptainer CMD ---
# removed --nv flag for CPU only
APP_CMD="apptainer exec --env SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK --bind /etc/OpenCL/vendors --bind "$ABS_ROOT:$ABS_ROOT" --cleanenv --env PYTHONNOUSERSITE=1 --env HF_HUB_OFFLINE=1 --env XDG_CACHE_HOME=$CACHE_PATH --env NUMBA_CACHE_DIR=/tmp/numba_cache --env PYTHONPATH=$PROJECT_ROOT SMAC_optuna.sif python3"

echo "--- Running Advanced Models Experiment (SMAC) ---"
echo "Date: $(date)"
echo "Task ID: $TASK_ID"
echo "Suite ID: $SUITE_ID"
echo "Node: $HOSTNAME"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"

$APP_CMD -u $PROJECT_ROOT/adv_trial_all_smac.py \
    --suite_id $SUITE_ID \
    --task_id $TASK_ID \
    --result_folder $PROJECT_ROOT/smac_results/

echo "--- Job Finished ---"
