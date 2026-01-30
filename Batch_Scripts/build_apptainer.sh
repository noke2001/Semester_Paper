#!/bin/bash
#SBATCH --job-name=Build_Container
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=2:00:00
#SBATCH --output=slurm_build-%j.out

# example usage:
# sbatch build_apptainer.sh

# use scratch for temporary storage
export APPTAINER_TMPDIR="/PATH/TO/SCRATCH"
#
cd /PATH/TO/SCRATCH
# run build
apptainer build --fakeroot ./SMAC_optuna.sif ~/SMAC_pip.def > ./build.log 2>&1

