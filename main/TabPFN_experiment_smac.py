#!/usr/bin/env python3
"""
TabPFN Experiment script mirroring baseline_experiment.py structure,
with robust missing-value handling, bounded scaling, and silent categorical encoding.
"""
import argparse
import os
import random
import warnings
import time
import sys

import numpy as np
import pandas as pd
import torch

from src.loader import load_dataset_offline, clean_data, standardize_data
from sklearn.model_selection import train_test_split
from src.extrapolation_methods import (
    random_split,
    mahalanobis_split,
    umap_split,
    kmeans_split,
    gower_split,
    kmedoids_split,
    spatial_depth_split
)
from src.evaluation_metrics import (
    evaluate_rmse,
    evaluate_crps,
    evaluate_accuracy,
    evaluate_log_loss
)
from src.models.TabPFN_model import (
    TabPFNClassifierWrapper,
    TabPFNRegressorWrapper
)

# --- experiment constants ---
SEED = 10
smplr = "SMAC" # "TPE" ... [more can be added later]

# --- suite configuration ---
SUITE_CONFIG = {
    "regression_numerical":             {"suite_id":336, "task_type":"regression",      "data_type":"numerical"},
    "classification_numerical":         {"suite_id":337, "task_type":"classification", "data_type":"numerical"},
    "regression_numerical_categorical":{"suite_id":335, "task_type":"regression",      "data_type":"numerical_categorical"},
    "classification_numerical_categorical":{
        "suite_id":334, "task_type":"classification","data_type":"numerical_categorical"
    },
    "tabzilla": {"suite_id":379, "task_type":"classification",      "data_type":None}
}
EXTRAPOLATION_METHODS = {
    "numerical":            [random_split, mahalanobis_split, kmeans_split, umap_split, spatial_depth_split],
    "numerical_categorical":[random_split, gower_split, kmedoids_split, umap_split]
}

def find_config(suite_id):
    for cfg in SUITE_CONFIG.values():
        if cfg["suite_id"] == suite_id:
            return cfg.copy()
    raise ValueError(f"No suite config for suite_id={suite_id}")

def split_dataset(split_fn, X, y):
    """Handle random_split (6 outputs) and other 2-output splitters uniformly."""
    out = split_fn(X, y) if split_fn is random_split else split_fn(X)
    if isinstance(out, tuple) and len(out) == 6:
        X_tr, _, y_tr, _, X_te, y_te = out
    else:
        train_idx, test_idx = out
        X_tr, X_te = X.loc[train_idx], X.loc[test_idx]
        y_tr, y_te = y.loc[train_idx], y.loc[test_idx]
    return X_tr, y_tr, X_te, y_te

def print_tabpfn_version_info():
    """
    Instantiates a dummy TabPFN wrapper to verify package version 
    and loaded weight file path.
    """
    print("\n" + "="*40)
    print("      TABPFN VERSION DIAGNOSTICS")
    print("="*40)
    
    try:
        import tabpfn
        print(f"[*] Python Package Version: {tabpfn.__version__}")
        
        # Import your wrapper specifically to test exactly what the experiment uses
        from src.models.TabPFN_model import TabPFNClassifierWrapper
        
        # Initialize on CPU to avoid allocating GPU memory just for this check
        print("[*] Initializing wrapper to check weights...")
        dummy_wrapper = TabPFNClassifierWrapper(device='cpu')
        
        # Access the inner TabPFN model (self.model)
        if hasattr(dummy_wrapper, 'model'):
            inner_model = dummy_wrapper.model
            
            # Check for the model path attribute (standard in v2/v6+)
            if hasattr(inner_model, 'model_path'):
                path = inner_model.model_path
                print(f"[*] Model Weights Path: {path}")
                
                if "v2.5" in str(path):
                    print("--> RESULT: You are using TabPFN v2.5 (Latest/Default)")
                elif "v2" in str(path) and "v2.5" not in str(path):
                    print("--> RESULT: You are using TabPFN v2.0")
                else:
                    print("--> RESULT: Custom or Legacy Weights")
            else:
                print("[!] Could not find 'model_path' attribute. Likely TabPFN v1 (older).")
        else:
            print("[!] Wrapper does not expose .model attribute.")
            
    except Exception as e:
        print(f"[!] Error during version check: {e}")
        import traceback
        traceback.print_exc()
        
    print("="*40 + "\n", flush=True)

def main():
    print_tabpfn_version_info()
    
    print(f"--- Starting Experiment at {time.strftime('%X')} ---", flush=True)
    
    # --- GPU CHECK ---
    if torch.cuda.is_available():
        print(f"SUCCESS: GPU Detected: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("WARNING: No GPU detected. Running on CPU.", flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--suite_id",     type=int,   required=True)
    parser.add_argument("--task_id",      type=int,   required=True)
    parser.add_argument("--seed",         type=int,   default=SEED)
    parser.add_argument("--result_folder",type=str,   required=True)
    args = parser.parse_args()

    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = find_config(args.suite_id)
    is_regression = (cfg["task_type"] == "regression")

    print(f"Loading data for Suite {args.suite_id}, Task {args.task_id}...", flush=True)

    # load data
    X_full, y_full, cat_ind, attr_names = load_dataset_offline(
        args.suite_id, args.task_id
    )

    if cfg["data_type"] is None:
        if hasattr(cat_ind, "any"):
            has_categorical = bool(getattr(cat_ind, "any")())
        else:
            has_categorical = any(cat_ind)
        cfg["data_type"] = "numerical_categorical" if has_categorical else "numerical"
      
    methods = EXTRAPOLATION_METHODS[cfg["data_type"]]
    print(f"Data type: {cfg['data_type']}. Methods to run: {[m.__name__ for m in methods]}", flush=True)

    MAX_SAMPLES = 50000 # was 12000 before
    if len(X_full) > MAX_SAMPLES:
        print(f"Downsampling from {len(X_full)} to {MAX_SAMPLES} samples...", flush=True)
        X_full, _, y_full, _= train_test_split(
            X_full, y_full, 
            train_size=MAX_SAMPLES,
            stratify=y_full, 
            random_state=args.seed
        )

    # Attempt to clean data
    X, X_clean, y = clean_data(
        X_full, y_full, cat_ind, attr_names,
        task_type=cfg["task_type"]
    )

    # --- FALLBACK: IF CLEAN_DATA KILLED THE DATASET ---
    if X_clean.shape[1] == 0 or X.shape[1] == 0:
        print(f"  -> [WARNING] clean_data removed all features! Falling back to raw X_full.", flush=True)
        
        X = X_full.copy()
        y = y_full.copy()
        
        # Manual minimal cleaning: Drop columns that are obviously IDs (all unique strings)
        cols_to_drop = []
        for col in X.columns:
            if X[col].dtype == 'object' and X[col].nunique() == len(X):
                cols_to_drop.append(col)
        
        if cols_to_drop:
            print(f"  -> Dropping ID-like columns: {cols_to_drop}", flush=True)
            X.drop(columns=cols_to_drop, inplace=True)

        X_clean = X.copy()
    # ----------------------------------------------------

    # --- SAFETY CHECK: FINAL STOP ---
    if X.shape[1] == 0:
        print(f"  -> [ERROR] Dataset {args.task_id} has 0 features. Cannot proceed.", flush=True)
        return

    if args.task_id in (361082, 361088, 361099) and is_regression:
        y = np.log(y)

    # --- FIX: Ensure y is 1D ---
    if isinstance(y, pd.DataFrame):
        y = y.squeeze()

    # --- CLASS HANDLING ---
    if not is_regression:
        if not isinstance(y, pd.Series):
            y = pd.Series(y, index=X.index)
            
        n_classes = y.nunique()
        if n_classes > 10:
            print(f"  -> [WARNING] Dataset has {n_classes} classes. TabPFN supports max 10.", flush=True)
            print("  -> Filtering dataset to keep only the top 10 most frequent classes...", flush=True)
            
            top_10_classes = y.value_counts().head(10).index.tolist()
            mask = y.isin(top_10_classes)
            
            original_len = len(X)
            X = X.loc[mask]
            X_clean = X_clean.loc[mask]
            y = y.loc[mask]
            print(f"  -> reduced samples from {original_len} to {len(X)}", flush=True)
            
            if len(X) == 0:
                print("  -> [ERROR] Filtering resulted in empty dataset. Exiting.", flush=True)
                return

        # --- CRITICAL FIX: LABEL ENCODING (Strings 'R','L' -> Ints 0,1) ---
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        # Encode y to integers (0, 1, 2...)
        y_encoded = le.fit_transform(y)
        # Put it back in a Series with the original index so splitters work
        y = pd.Series(y_encoded, index=y.index)
        print(f"  -> Encoded target labels. Classes mapped: {dict(zip(le.classes_, range(len(le.classes_))))}", flush=True)
    # ------------------------------------------

    # --- SAFETY: PREPARE X FOR SPLITTING METHODS ---
    X_for_splitting = X_clean.copy()
    
    # 1. Try to coerce object columns to numeric
    if X_for_splitting.shape[1] > 0:
        for col in X_for_splitting.columns:
            if X_for_splitting[col].dtype == 'object':
                try:
                    converted = pd.to_numeric(X_for_splitting[col], errors='coerce')
                    if converted.notna().sum() > 0:
                        X_for_splitting[col] = converted
                except (ValueError, TypeError):
                    pass

    # 2. Select numeric columns
    X_split_numeric = X_for_splitting.select_dtypes(include=[np.number])

    # 3. If result is empty (dataset is purely categorical), we MUST encode
    if X_split_numeric.shape[1] == 0:
        print("  -> [INFO] Dataset appears categorical. One-Hot Encoding for split generation.", flush=True)
        
        if X_for_splitting.shape[1] == 0:
             print("  -> [ERROR] X_clean became empty (0 columns). Cannot split.", flush=True)
             return

        try:
            X_split_numeric = pd.get_dummies(X_for_splitting, drop_first=True)
        except ValueError:
            print("  -> [ERROR] get_dummies failed. Features are likely empty or all-NaN.", flush=True)
            return
            
        if X_split_numeric.shape[1] == 0:
            print("  -> [ERROR] Encoding resulted in 0 features. Cannot split.", flush=True)
            return
    
    # 4. Fill NaNs
    X_for_splitting = X_split_numeric.fillna(0)
    # ---------------------------------------------------------------------

    # prepare output
    out_dir = os.path.join(args.result_folder, f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{args.suite_id}_{args.task_id}_tabpfn_50k.csv")

    records = []
    total_methods = len(methods)
    
    for i, split_fn in enumerate(methods, 1):
        name_split = split_fn.__name__
        print(f"\n[{i}/{total_methods}] Processing split: {name_split}...", flush=True)
        start_time = time.time()
        
        try:
            # Use the strictly numeric/encoded X for splitting calculations
            X_tr_clean, y_tr, X_te_clean, y_te = split_dataset(split_fn, X_for_splitting, y)
        except Exception as e:
            print(f"  -> Split failed: {e}", flush=True)
            continue
            
        train_idx = X_tr_clean.index
        test_idx  = X_te_clean.index

        if len(train_idx) == 0 or len(test_idx) == 0:
            print("  -> Empty train or test set. Skipping.", flush=True)
            continue

        # --- PREPARE DATA FOR TABPFN ---
        X_loop = X.copy()

        # Robust encoding
        X_loop = pd.get_dummies(X_loop, drop_first=True).astype('float32')
        X_loop = X_loop.fillna(0)

        if X_loop.shape[1] == 0:
            print(f"  -> Skipping {name_split}: no features left.", flush=True)
            continue

        non_dummy_cols = X_loop.columns.tolist()

        X_tr = X_loop.loc[train_idx]
        X_te = X_loop.loc[test_idx]
        
        X_tr_p, X_te_p = standardize_data(X_tr, X_te, non_dummy_cols)

        X_tr_p = X_tr_p.fillna(0)
        X_te_p = X_te_p.fillna(0)

        MAX_TABPFN_FEATS = 500 # was 100
        if X_tr_p.shape[1] > MAX_TABPFN_FEATS:
            keep = X_tr_p.columns[:MAX_TABPFN_FEATS]
            X_tr_p = X_tr_p[keep]
            X_te_p = X_te_p[keep]

        y_tr_arr = y_tr.to_numpy().ravel()
        y_te_arr = y_te.to_numpy().ravel()

        print(f"  -> Fitting model on {len(X_tr_p)} samples, {X_tr_p.shape[1]} features...", flush=True)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        try:
            if is_regression:
                model = TabPFNRegressorWrapper(random_state=args.seed, device=device)
                model.fit(X_tr_p, y_tr_arr)
                y_pred = model.predict(X_te_p)
                rmse = evaluate_rmse(y_te_arr, y_pred)
                
                residuals = y_tr_arr - model.predict(X_tr_p)
                sigma = np.std(residuals)
                if sigma == 0: sigma = 1e-6 
                
                crps = evaluate_crps(
                    y_te_arr,
                    y_pred,
                    np.full_like(y_pred, sigma)
                )
                records += [
                    {"suite_id": args.suite_id, "task_id": args.task_id,
                     "split_method": name_split, "model": "TabPFNRegressor",
                     "metric": m, "value": v}
                    for m, v in [("RMSE", rmse), ("CRPS", crps)]
                ]
            else:
                model = TabPFNClassifierWrapper(random_state=args.seed, device=device)
                model.fit(X_tr_p, y_tr_arr)

                probs = model.predict_proba(X_te_p)
                if probs.shape[1] == 2:
                    preds = (probs[:, 1] >= 0.5).astype(int)
                else:
                    preds = np.argmax(probs, axis=1)

                acc = evaluate_accuracy(y_te_arr, preds)
                ll  = evaluate_log_loss(y_te_arr, probs)
                records += [
                    {
                        "suite_id":     args.suite_id,
                        "task_id":      args.task_id,
                        "split_method": name_split,
                        "model":        "TabPFNClassifier",
                        "metric":       m,
                        "value":        v
                    }
                    for m, v in [("Accuracy", acc), ("LogLoss", ll)]
                ]
        except Exception as e:
            print(f"  -> Model Error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue
        
        elapsed = time.time() - start_time
        print(f"  -> Done in {elapsed:.1f}s", flush=True)

    pd.DataFrame(records).to_csv(out_file, index=False)
    print(f"\nSaved TabPFN results to {out_file}", flush=True)
    print(f"--- Experiment Finished at {time.strftime('%X')} ---", flush=True)

if __name__ == "__main__":
    main()