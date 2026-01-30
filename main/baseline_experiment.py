#!/usr/bin/env python3
# src/experiments/baseline_experiment.py

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import numba
numba.config.DISABLE_JIT = True


import time
import contextlib


# --- your loader & utils ---
from src.loader import load_dataset_offline, clean_data, standardize_data
from src.new_extrapolation import (
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
from src.models.Baseline_models import (
    LinearRegressor,
    LogisticRegressor,
    ConstantPredictor
)

SEED = 10

SUITE_CONFIG = {
    "regression_numerical":            {"suite_id":336, "task_type":"regression",     "data_type":"numerical"},
    "classification_numerical":        {"suite_id":337, "task_type":"classification", "data_type":"numerical"},
    "regression_numerical_categorical":{"suite_id":335, "task_type":"regression",     "data_type":"numerical_categorical"},
    "classification_numerical_categorical":{
        "suite_id":334, "task_type":"classification","data_type":"numerical_categorical"
    },
    "tabzilla": {"suite_id":379, "task_type":"classification",     "data_type":None}
}
EXTRAPOLATION_METHODS = {
    "numerical":            [random_split, mahalanobis_split, kmeans_split, umap_split, spatial_depth_split],
    "numerical_categorical":[random_split, gower_split, kmedoids_split, umap_split]
}

@contextlib.contextmanager
def log_time(stage_name:str):
    t0 = time.perf_counter()
    yield
    t1 = time.perf_counter()
    print(f"[TIMING] {stage_name}: {t1 - t0:.2f} seconds")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite_id",     type=int,   required=True)
    parser.add_argument("--task_id",      type=int,   required=True)
    parser.add_argument("--seed",         type=int,   default=SEED)
    parser.add_argument("--result_folder",type=str,   required=True)
    parser.add_argument("--split_method", type=str, choices=[fn.__name__ for l in EXTRAPOLATION_METHODS.values() for fn in l], default=None)
    args = parser.parse_args()

    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg     = find_config(args.suite_id)
    is_regression  = (cfg["task_type"] == "regression")

    # load & clean data
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

    if args.split_method is not None:
        methods = [fn for fn in methods if fn.__name__ == args.split_method]
        if not methods:
            raise ValueError(f"No split method named {args.split_method}")



    MAX_SAMPLES = 10000
    if len(X_full) > MAX_SAMPLES:
        X_full, _, y_full, _= train_test_split(
            X_full, y_full, 
            train_size=MAX_SAMPLES,
            stratify=y_full, 
            random_state=args.seed
        )

    X, X_clean, y = clean_data(
        X_full, y_full, cat_ind, attr_names,
        task_type=cfg["task_type"]
    )


    if args.task_id in (361082, 361088, 361099) and is_regression:
        y = np.log(y)

    # prepare output
    out_dir = os.path.join(args.result_folder, f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{args.suite_id}_{args.task_id}_baseline.csv")

    if is_regression:
        model_constructors = [("LinearRegressor", LinearRegressor), ("ConstantPredictor", ConstantPredictor)]
    else:
        model_constructors = [("LogisticRegressor", LogisticRegressor), ("ConstantPredictor", ConstantPredictor)]

    records = []
    import traceback


    for split_fn in methods:
        name_split = split_fn.__name__
        print(f"\n---\n[DEBUG] About to run split: {name_split}")
        print("  X_clean.shape =", X_clean.shape)
        print("  X_clean.dtypes:\n", X_clean.dtypes.value_counts())
        with log_time(f"Splitting data with {name_split}"):
            try:
                X_tr_clean, y_tr, X_te_clean, y_te = split_dataset(split_fn, X_clean, y)
            except Exception as e:
                print(f"[ERROR] {name_split} threw {type(e).__name__}: {e}")
                traceback.print_exc()
                continue
        train_idx = X_tr_clean.index
        test_idx  = X_te_clean.index

        print(f"[DEBUG] {name_split} →  train={len(X_tr_clean)}  test={len(X_te_clean)}")
        if len(X_tr_clean)==0 or len(X_te_clean)==0:
            print(f"[WARNING] {name_split} produced an empty split!  thresh may be wrong.")

        X_loop = X.copy()

        dummy_cols = X_loop.select_dtypes(include=['bool','category','object','string']).columns
        for col in dummy_cols:
            if X_loop[col].nunique() != X_loop.loc[train_idx, col].nunique():
                X_loop = X_loop.drop(col, axis=1)

        if X_loop.shape[1] == 0:
            print(f"Skipping {name_split} split for {args.suite_id}_{args.task_id}: no valid features left after dummy removal.")
            continue

        non_dummy_cols = X_loop.select_dtypes(
            exclude=['bool','category','object','string']
        ).columns.tolist()

        X_loop = pd.get_dummies(X_loop, drop_first=True).astype('float32')

        X_tr = X_loop.loc[train_idx]
        X_te = X_loop.loc[test_idx]

        with log_time(f"Standardizing data for {name_split}"):
            X_tr_p, X_te_p = standardize_data(X_tr, X_te, non_dummy_cols)

        y_tr_arr = y_tr.to_numpy().ravel()
        y_te_arr = y_te.to_numpy().ravel()

        with log_time(f"Fitting models for {name_split}"):
            for mdl_name, Mdl in model_constructors:
                model = Mdl()
                model.fit(X_tr_p, y_tr_arr)

                if is_regression:
                    # regression metrics
                    y_pred_test = model.predict(X_te_p)
                    y_pred_train = model.predict(X_tr_p)
                    sigma = np.std(y_tr_arr - y_pred_train)
                    sigma_arr = np.full_like(y_pred_test, sigma, dtype=float)
                    records.extend([
                        {"suite_id": args.suite_id, "task_id": args.task_id,
                        "split_method": name_split, "model": mdl_name,
                        "metric": "RMSE", "value": evaluate_rmse(y_te_arr, y_pred_test)},
                        {"suite_id": args.suite_id, "task_id": args.task_id,
                        "split_method": name_split, "model": mdl_name,
                        "metric": "CRPS", "value": evaluate_crps(y_te_arr, y_pred_test, sigma_arr)}
                    ])
                else:
                    # classification metrics
                    preds = model.predict(X_te_p)
                    probs = model.predict_proba(X_te_p)
                    records.extend([
                        {"suite_id": args.suite_id, "task_id": args.task_id,
                        "split_method": name_split, "model": mdl_name,
                        "metric": "Accuracy", "value": evaluate_accuracy(y_te_arr, preds)},
                        {"suite_id": args.suite_id, "task_id": args.task_id,
                        "split_method": name_split, "model": mdl_name,
                        "metric": "LogLoss", "value": evaluate_log_loss(y_te_arr, probs)}
                ])

    # save all results
    df = pd.DataFrame.from_records(records)
    df.to_csv(out_file, index=False)
    print(f"Saved baseline results to {out_file}")


if __name__ == "__main__":
    main()
