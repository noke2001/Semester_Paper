#!/bin/bash
#SBATCH --job-name=ApptainerPython 
#SBATCH --cpus-per-task=4 
#SBATCH --mem-per-cpu=8G 
#SBATCH --time=04:00:00 
#SBATCH --output=slurm-%j.out 
#SBATCH --error=slurm-%j.err 

CACHE_PATH="./openml_cache"
FUNCTION_ROOT="./main/helper_functions"

apptainer exec --cleanenv --env PYTHONNOUSERSITE=1 --env NUMBA_CACHE_DIR=/tmp/numba_cache --env XDG_CACHE_HOME=$CACHE_PATH SMAC_optuna.sif python3 $FUNCTION_ROOT/download_data.py --task_id 361082 --suite_id 336

echo "--- Job Finished ---"
