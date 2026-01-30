from huggingface_hub import hf_hub_download
import os

# 1. PASTE YOUR TOKEN HERE
MY_TOKEN = "hf_..." #replace with your actual token

print("Starting download to global Hugging Face cache...")

# Download Classifier
# We do NOT specify 'local_dir'. We let it go to ~/.cache/huggingface
# This ensures the library finds it automatically later.
path_c = hf_hub_download(
    repo_id="Prior-Labs/tabpfn_2_5",
    filename="tabpfn-v2.5-classifier-v2.5_default.ckpt",
    token=MY_TOKEN
)
print(f"Classifier downloaded to: {path_c}")

# Download Regressor
path_r = hf_hub_download(
    repo_id="Prior-Labs/tabpfn_2_5",
    filename="tabpfn-v2.5-regressor-v2.5_default.ckpt",
    token=MY_TOKEN
)
print(f"Regressor downloaded to: {path_r}")