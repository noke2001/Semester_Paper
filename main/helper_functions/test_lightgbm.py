import lightgbm as lgb
import torch
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

def test_gpu():
    print("--- GPU Test ---")
    print(f"GPU is available? {torch.cuda.is_available()}")
    print(f"Current torch device: {torch.cuda.current_device()}")
    if torch.cuda.is_available() == True :
        print(torch.cuda.get_device_name(0))

    print("--- LGBM Test ---")
    # 1. Create a dummy dataset large enough to benefit from GPU
    X, y = make_regression(n_samples=10000, n_features=20, random_state=42)
    
    # 2. Define parameters for GPU
    # We use 'device': 'gpu' and a small number of iterations
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'device': 'cuda',     # changed from 'gpu'
        'gpu_platform_id': 0, # Usually 0
        'gpu_device_id': 0,   # Usually 0
        'verbose': 1
    }
    
    train_data = lgb.Dataset(X, label=y)
    
    print("Attempting to train with GPU...")
    try:
        gbm = lgb.train(params, train_data, num_boost_round=10)
        print("\nSUCCESS: LightGBM successfully trained using the GPU!")
    except Exception as e:
        print("\nFAILURE: LightGBM could not initialize GPU.")
        print(f"Error message: {e}")
        
        # Check for common OpenCL errors
        if "OpenCL" in str(e):
            print("\nHint: This usually means OpenCL headers are missing or the GPU drivers are not visible to the container.")

if __name__ == "__main__":
    test_gpu()