# launcher_tabpfn.py

import os
import glob
import neural_experiment_engr as neural_experiment
from utils_exp import generate_base_command, generate_run_commands

# the list of suite_ids you care about
suites = [334, 337]
SEED = 10  # or whatever default you like

RESULT_DIR = 'engression_results'

def main():
    command_list = []
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base, "original_data")

    for suite_id in suites:
        pattern = os.path.join(data_dir, f"{suite_id}_*")
        for folder in glob.glob(pattern):
            task_id = int(os.path.basename(folder).split("_", 1)[1])

            if suite_id == 379 and task_id == 168337:
                env_prefix = ""   # allow Numba JIT for speed
            else:
                env_prefix = (
                "export NUMBA_DISABLE_JIT=1 && "
                "export NUMBA_CACHE_DIR=/tmp/numba_nocache && "
                )


            python_cmd = generate_base_command(
                neural_experiment,
                flags={
                    'suite_id': suite_id,
                    'task_id': task_id,
                    'result_folder': RESULT_DIR,
                    'seed': SEED
                }
            )

            cmd = env_prefix + python_cmd
            command_list.append(cmd)

    generate_run_commands(
        command_list,
        promt=False,
        num_cpus=1,
        num_gpus=1,
        mem=32 * 1024,
        duration="71:59:00",
        mode="euler"
    )

if __name__ == "__main__":
    main()
