import os
import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn.constants import ModelVersion

class TabPFNClassifierWrapper:
    def __init__(self, random_state=0, ignore_pretraining_limits=True, device=None, model_version=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Check for Environment Variable for Custom Weights
        # This allows us to inject the path from Slurm without changing python arguments
        custom_weights_path = os.environ.get("TABPFN_WEIGHTS_PATH", None)

        if custom_weights_path and os.path.exists(custom_weights_path):
            print(f"[Wrapper] Loading Custom Weights from: {custom_weights_path}")
            self.model = TabPFNClassifier(
                device=device,
                random_state=random_state,
                model_path=custom_weights_path,
                ignore_pretraining_limits=ignore_pretraining_limits
            )
        elif model_version is not None:
            self.model = TabPFNClassifier.create_default_for_version(model_version)
            self.model.device = device
            self.model.random_state = random_state
            self.model.ignore_pretraining_limits = ignore_pretraining_limits
        else:
            # Default behavior (will try to download if offline mode isn't set)
            self.model = TabPFNClassifier(
                random_state=random_state,
                device=device,
                ignore_pretraining_limits=ignore_pretraining_limits
            )

    def fit(self, X, y):
        X_np = np.asarray(pd.DataFrame(X).fillna(0), dtype=np.float32)
        y_np = np.asarray(y).ravel().astype(int)
        self.model.fit(X_np, y_np)
        return self

    def predict_proba(self, X, batch_size: int = 64):
        X_df = pd.DataFrame(X).fillna(0)
        n = X_df.shape[0]
        probs = []
        for i in range(0, n, batch_size):
            chunk = X_df.iloc[i : i + batch_size].to_numpy(dtype=np.float32)
            probs_chunk = self.model.predict_proba(chunk)
            probs.append(probs_chunk)
        return np.vstack(probs)

    def predict(self, X, batch_size: int = 64):
        probs = self.predict_proba(X, batch_size=batch_size)
        if probs.shape[1] == 2:
            return (probs[:, 1] >= 0.5).astype(int)
        else:
            return np.argmax(probs, axis=1)


class TabPFNRegressorWrapper:
    def __init__(self, random_state=0, ignore_pretraining_limits=True, device=None, model_version=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. Check for Environment Variable (use a separate one for Regressor if needed, 
        # or share if you swap them in the bash script)
        # Assuming you change the env var in bash depending on the task
        custom_weights_path = os.environ.get("TABPFN_WEIGHTS_PATH", None)
            
        if custom_weights_path and os.path.exists(custom_weights_path):
            print(f"[Wrapper] Loading Custom Regressor Weights from: {custom_weights_path}")
            self.model = TabPFNRegressor(
                device=device,
                random_state=random_state,
                model_path=custom_weights_path,
                ignore_pretraining_limits=ignore_pretraining_limits
            )
        elif model_version is not None:
            self.model = TabPFNRegressor.create_default_for_version(model_version)
            self.model.device = device
            self.model.random_state = random_state
            self.model.ignore_pretraining_limits = ignore_pretraining_limits
        else:
            self.model = TabPFNRegressor(
                random_state=random_state,
                device=device,
                ignore_pretraining_limits=ignore_pretraining_limits
            )

    def fit(self, X, y):
        X_np = np.asarray(pd.DataFrame(X).fillna(0), dtype=np.float32)
        y_np = np.asarray(y).ravel().astype(float)
        self.model.fit(X_np, y_np)
        return self

    def predict(self, X, batch_size: int = 64):
        X_df = pd.DataFrame(X).fillna(0)
        n = X_df.shape[0]
        preds = []
        for i in range(0, n, batch_size):
            chunk = X_df.iloc[i : i + batch_size].to_numpy(dtype=np.float32)
            preds_chunk = self.model.predict(chunk)
            preds.append(preds_chunk)
        return np.concatenate(preds, axis=0)