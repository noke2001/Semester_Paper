# Semester_Project_Extrapolation_Methods

# Overview
This repository is structured into a couple sections:
- /Batch_Scripts/ contains all the ready to fire batch script to run individual experiments or entire suites.
- /main/ conatins all the python scripts used to run the experiments.
- /main/original_data/ has to be populated by the user using the download_data.py script or manually downloading and indexing the OpenML datasets.
- /main/helper_functions/ contains all sorts of useful functions such as the download_data.py function above.
- /main/src/ contains all the wrappers for the models used in the experiments.
- /openml_cache/ contains all the local cache required for the apptainer to run these models locally.
- /tabpfn_weights/v2.5 has to be populated with the TabPFN model's weights. These can be downloaded on their HuggingFace page. Note that you may need to update the paths in the run_pfn_[exp/suite].sh script.

# Building/Using Containers
The SMAC_optuna.sif container can either be downloaded from (https://zenodo.org/records/18431873), or it can be built from scratch using the SMAC_pip.def file. 
> [!WARNING]
> Before running any experiments or suites, make sure that you have the OpenCL drivers installed on your system in /etc/OpenCL/vendors.
> Before running TabPFN experiments or suites, make sure that you have downloaded the weights from (https://huggingface.co/Prior-Labs/tabpfn_2_5/tree/main), and save them in the ./tabpfn_weights/ directory.

Note that you can install the OpenCL drivers by running the following commands (here for ubuntu/debian):
```bash
sudo apt-gat update
sudo apt-gat install ocl-icd-libopencl1 opencl-headers clinfo
sudo apt-get install intel-opencl-icd # If no NVidia GPU
sudo apt-get install nvidia-opencl-icd # If NVidia GPU available (recommended for deep-learning models)
```


# Setting up the SMAC Sampler on Euler
Note the architecture of containers on Euler:
- Login Node (Has internet, low RAM): Use for downloading/converting data
- Compute Node (No internet, high RAM): Use for running experiments
- Cache: (Bridge between these): Download files here so the compute node doesn't crash trying to access the web.

## One-time Setup
### (i) Vendorize the SMAC Sampler
This has to be done because the compute node cannot connect to GitHub/OptunaHub
```bash
cd $PROJECT_ROOT
wget -O smac_sampler.py https://raw.githubusercontent.com/optuna/optunahub-registry/main/package/samplers/smac_sampler/sampler.py
```
### (ii) Ensure [type]_experiment_smac.py imports from the local file, not the library:
```python
from smac_sampler import SMACSampler
```

## Phase 2 (Compute Node):
### The SLURM Scripts: 
All the pre-made scripts can all be found in the ./Batch_Scripts directory.
If you want to run a single experiment, you have to edit the run_[type]_experiment.sh file. Change the sbatch command at the top of the file to reflect your system usage, and change the following lines:
```bash
# --- TASK CONFIGURATION ---
TASK_ID=<YOUR_TASK_ID_HERE>
SUITE_ID=<YOUR_SUITE_ID_HERE>
```
and then submit the job:
```bash
sbatch run_[type]_experiment.sh
```
If you want to run a whole suite, you can use the run_[type]_suite.sh files. Just specify which suite you want to run in the shell:
```bash
sbatch run_[type]_suite.sh <YOUR_SUITE_ID_HERE>
```

### Tasks/Suites
Choose a valid task_id/suite_id comination from openml_suite_tasks.csv.

## Common Errors & Fixes

| Error Message | Cause | Fix |
| :--- | :--- | :--- |
| **Killed** | Login node ran out of RAM processing data. | Ensure you are running `wget` (manual download) first, then the processing script. |
| **OpenMLHashException / Checksum error** | Firewall blocked download; file is corrupted/empty. | Delete the file in cache and repeat Phase 1 (Manual Download). |
| **Invalid layout of the ARFF file** | File is an HTML error page, not data. | Delete the file and check the URL in Phase 1. |
| **OSError: Read-only file system** | Script trying to write to `/opt` or default cache. | Ensure `XDG_CACHE_HOME` is set in the Apptainer command. |
| **Connection refused (GitHub/OpenML)** | Running internet commands on Compute Node. | Move that step to Phase 1 (Login Node). |
| **SMACSampler missing argument: search_space** | SMAC needs params upfront. | Define `search_space = {...}` dictionary in your Python script. |