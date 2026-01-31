#!/usr/bin/env python3
import os
import glob
import Repo.main.adv_trial_smac as adv_trial_smac
from utils_exp import generate_base_command, generate_run_commands

# ——— configuration ———
suites = [334, 335, 336, 337, 379]
SEED = 10
RESULT_FOLDER = 'adv5_results'

def main():
    command_list = []
    base     = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base, "original_data")

    for suite_id in suites:
        # look for folders like "original_data/336_361088"
        pattern = os.path.join(data_dir, f"{suite_id}_*")
        # pick only the first matching folder
        folders = sorted(glob.glob(pattern))
        if not folders:
            print(f"No data folder found for suite {suite_id!r}", flush=True)
            continue
        folder = folders[0]
        task_id = int(os.path.basename(folder).split("_", 1)[1])

        if suite_id == 379 and task_id == 168337:
                env_prefix = ""   # allow Numba JIT for speed
        else:
            env_prefix = (
            "export NUMBA_DISABLE_JIT=1 && "
            "export NUMBA_CACHE_DIR=/tmp/numba_nocache && "
            )


        # build the python command to call advanced_experiment.main()
        python_cmd = generate_base_command(
            adv_trial_smac,
            flags={
                'suite_id':     suite_id,
                'task_id':      task_id,
                'result_folder':RESULT_FOLDER,
                'seed':         SEED
            }
        )

        command_list.append(env_prefix + python_cmd)

    # submit via sbatch on Euler
    generate_run_commands(
        command_list,
        promt=False,
        num_cpus=1,
        mem=32 * 1024,
        duration="47:59:00",
        mode="euler"
    )

if __name__ == "__main__":
    main()
