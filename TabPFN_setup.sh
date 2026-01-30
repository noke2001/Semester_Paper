#!/bin/bash
#SBATCH --job-name=TabPFN_SMAC
#SBATCH --account=ls_math
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# --- PATHS ---
CACHE_PATH="/cluster/home/pdamota/openml_cache"
PROJECT_ROOT="./semester_project_smac/master_thesis-main/master_thesis-main"
FLAT_MODEL_DIR="${CACHE_PATH}/tabpfn_flat"

# --- HOST PREP ---
# Ensure directories exist
mkdir -p "${CACHE_PATH}/optunahub"
mkdir -p "${FLAT_MODEL_DIR}"

# --- TASK ---
TASK_ID="361093"
SUITE_ID="335"

# --- Apptainer CMD ---
# We bind the 'tabpfn_flat' folder to '/opt/flat_model' inside the container
APP_CMD="apptainer exec --nv --fakeroot \
    --bind /etc/OpenCL/vendors \
    --bind /cluster/home/pdamota \
    --bind ${CACHE_PATH}/optunahub:/opt/optunahub_cache \
    --bind ${FLAT_MODEL_DIR}:/opt/flat_model \
    --cleanenv \
    --env PYTHONNOUSERSITE=1 \
    --env TABPFN_DISABLE_TELEMETRY=1 \
    --env NUMBA_CACHE_DIR=/tmp/numba_cache \
    --env PYTORCH_ALLOC_CONF=max_split_size_mb:1024 \
    --env PYTHONPATH=$PROJECT_ROOT \
    SMAC_optuna.sif python3 -u"

# --- THE MONKEYPATCH SCRIPT ---
WRAPPER_SCRIPT="
import sys
import os
import shutil
import huggingface_hub
import runpy
import threading
import time

# --- Heartbeat Loop ---
def heartbeat_loop():
    while True:
        print(f'[HEARTBEAT] Job is alive... {time.strftime(\"%X\")}')
        time.sleep(60)

t = threading.Thread(target=heartbeat_loop, daemon=True)
t.start()

# --- The Download Patch ---
original_download = huggingface_hub.hf_hub_download

def patched_download(*args, **kwargs):
    filename = kwargs.get('filename', '')
    staging_dir = '/opt/optunahub_cache/patch_staging'
    os.makedirs(staging_dir, exist_ok=True)
    src = None
    
    # 1. Intercept Classifier
    if 'classifier' in filename and 'ckpt' in filename:
        print(f'[PATCH] Intercepted CLASSIFIER request: {filename}')
        src = '/opt/flat_model/tabpfn-v2.5-classifier-v2.5_default.ckpt'

    # 2. Intercept Regressor (NEW!)
    elif 'regressor' in filename and 'ckpt' in filename:
        print(f'[PATCH] Intercepted REGRESSOR request: {filename}')
        src = '/opt/flat_model/tabpfn-v2.5-regressor-v2.5_default.ckpt'

    # 3. Intercept Config
    elif 'config.json' in filename:
        print(f'[PATCH] Intercepted CONFIG request: {filename}')
        src = '/opt/flat_model/config.json'
        
    # If we found a match, copy it to the staging area and return that path
    if src and os.path.exists(src):
        dst = os.path.join(staging_dir, os.path.basename(src))
        # Only copy if destination doesn't exist or is different size
        if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
             print(f'[PATCH] Copying {src} -> {dst}')
             shutil.copy2(src, dst)
        else:
             print(f'[PATCH] Using existing cached file: {dst}')
        return dst

    print(f'[PATCH] Warning: Unhandled file request: {filename} - Passing to original downloader (will likely fail if offline)')
    return original_download(*args, **kwargs)

print('--- Applying Hugging Face Download Patch (Regressor + Classifier) ---')
huggingface_hub.hf_hub_download = patched_download

# Setup Arguments
sys.argv = [
    'TabPFN_experiment_smac.py',
    '--suite_id', '$SUITE_ID',
    '--task_id', '$TASK_ID',
    '--result_folder', '$PROJECT_ROOT/smac_results/'
]

# Run the Real Script
print('--- Starting Main Experiment Script ---')
target_script = '$PROJECT_ROOT/TabPFN_experiment_smac.py'
runpy.run_path(target_script, run_name='__main__')
"

echo "--- Running Experiment ---"
echo "Task ID: $TASK_ID"
echo "Suite ID: $SUITE_ID"

# Execute the wrapper script
$APP_CMD -c "$WRAPPER_SCRIPT"

echo "--- Job Finished ---"