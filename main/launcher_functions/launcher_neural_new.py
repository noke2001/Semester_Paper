# launcher_neural.py

import os
import glob
import neural_experiment
# NOTE: We no longer need generate_run_commands from utils_exp
from utils_exp import generate_base_command

# the list of suite_ids you care about
suites = [335]
SEED = 10

RESULT_DIR = 'neural_results'
COMMAND_FILE = "neural_commands.txt" # The output file for our commands

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

    # --- KEY CHANGE ---
    # Instead of calling generate_run_commands, we write the list to a file.
    with open(COMMAND_FILE, "w") as f:
        for cmd in command_list:
            f.write(f"{cmd}\n")

    print(f"Generated {len(command_list)} commands in '{COMMAND_FILE}'")
    print("Next step: Use 'sbatch submit_array.sh' to run these commands as a job array.")


if __name__ == "__main__":
    main()