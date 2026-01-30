import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.model_selection import train_test_split
from properscoring import crps_gaussian

from src.loader import (
    load_dataset_offline, clean_data,
    standardize_data
)
from src.extrapolation_methods import (
    random_split, mahalanobis_split, umap_split,
    kmeans_split, gower_split, kmedoids_split, spatial_depth_split
)
from src.evaluation_metrics import (
    evaluate_rmse,
    evaluate_accuracy, evaluate_log_loss
)
from src.models.Tree_models import (
    RandomForestRegressor, RandomForestClassifier,
    LGBMRegressor, LGBMClassifier
)
from properscoring import crps_gaussian

# --- experiment constants ---
SEED      = 10
N_TRIALS  = 100           # Optuna trials per model/metric
VAL_RATIO = 0.2          # fraction of train fold for validation

# --- suite configuration ---
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
    args = parser.parse_args()

    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg     = find_config(args.suite_id)
    is_reg  = (cfg["task_type"] == "regression")

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

    MAX_SAMPLES = 12000
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


    if args.task_id in (361082, 361088, 361099) and is_reg:
        y = np.log(y)

    # prepare output
    out_dir = os.path.join(args.result_folder, f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{args.suite_id}_{args.task_id}_trees_debug.csv")

    records = []



    for split_fn in methods:
        name_split = split_fn.__name__

        try:
            X_tr_clean, y_tr, X_te_clean, y_te = split_dataset(split_fn, X_clean, y)
        except Exception as e:
            print(f"Skipping {name_split}: first split failed: {e}")
            continue
        if X_tr_clean.shape[1] == 0 or X_te_clean.shape[1] == 0:
            print(f"Skipping {name_split}: no features after split")
            continue

        try:
            X_train_clean_unfilt, _, X_val_clean_unfilt, _ = split_dataset(split_fn, X_tr_clean, y_tr)
        except Exception as e:
            print(f"Skipping {name_split}: validation split failed: {e}")
            continue
        if X_train_clean_unfilt.shape[1] == 0 or X_val_clean_unfilt.shape[1] == 0:
            print(f"Skipping {name_split}: no features in train/validation")
            continue

        X_loop = X.copy()
        dummy_cols = X_loop.select_dtypes(include=['bool','category','object','string']).columns
        for col in dummy_cols:
            if X_loop[col].nunique() != X_loop.loc[X_train_clean_unfilt.index, col].nunique():
                X_loop = X_loop.drop(col, axis=1)

        if X_loop.shape[1] == 0:
            print(f"Skipping {name_split}: no columns left after dropping unseen dummies")
            continue

        non_dummy_cols = X_loop.select_dtypes(exclude=['bool','category','object','string']).columns.tolist()
        X_loop = pd.get_dummies(X_loop, drop_first=True).astype('float32')

        X_tr_p = X_loop.loc[X_tr_clean.index]
        X_te_p = X_loop.loc[X_te_clean.index]

        X_train = X_tr_p.loc[X_train_clean_unfilt.index]
        X_val   = X_tr_p.loc[X_val_clean_unfilt.index]

        X_train_, X_val = standardize_data(X_train, X_val, non_dummy_cols)
        X_tr_p, X_te_p  = standardize_data(X_tr_p, X_te_p, non_dummy_cols)


        y_train_ = y_tr.loc[X_train_clean_unfilt.index].to_numpy().ravel()
        y_val    = y_tr.loc[X_val_clean_unfilt.index].to_numpy().ravel()
        y_tr_arr = y_tr.to_numpy().ravel()
        y_te_arr = y_te.to_numpy().ravel()


        def obj_rmse(trial, ModelClass):
            if ModelClass is RandomForestRegressor:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators',100,500),
                    'max_depth':    trial.suggest_int('max_depth',1,30),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf',10,100),
                    'max_features': trial.suggest_float('max_features',0,1)
                }
            else:
                params = {
                    'learning_rate': trial.suggest_float('learning_rate',1e-4,0.5,log=True),
                    'n_estimators':  trial.suggest_int('n_estimators',100,500),
                    'max_depth':     trial.suggest_int('max_depth',1,30),
                    'num_leaves':    2**10,
                    'reg_lambda':    trial.suggest_float('reg_lambda',1e-8,10,log=True),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 100)
                }
            model = ModelClass(**params)
            model.fit(X_train_.to_numpy(), y_train_)
            y_pred = model.predict(X_val)
            return np.sqrt(np.mean((y_val - y_pred)**2))

        def obj_crps(trial, ModelClass):
            if ModelClass is RandomForestRegressor:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators',100,500),
                    'max_depth':    trial.suggest_int('max_depth',1,30),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf',10,100),
                    'max_features': trial.suggest_float('max_features',0,1)
                }
            else:
                params = {
                    'learning_rate': trial.suggest_float('learning_rate',1e-4,0.5,log=True),
                    'n_estimators':  trial.suggest_int('n_estimators',100,500),
                    'max_depth':     trial.suggest_int('max_depth',1,30),
                    'num_leaves':    2**10,
                    'reg_lambda':    trial.suggest_float('reg_lambda',1e-8,10,log=True),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 100)
                }
            model = ModelClass(**params)
            model.fit(X_train_.to_numpy(), y_train_)
            resid = y_train_ - model.predict(X_train_.to_numpy())
            sigma = np.std(resid)
            y_pred = model.predict(X_val)
            crps_vals = crps_gaussian(y_val, y_pred, sigma)
            return float(np.mean(crps_vals))

        def obj_accuracy(trial, ModelClass):
            if ModelClass is RandomForestClassifier:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators',100,500),
                    'max_depth':    trial.suggest_int('max_depth',1,30),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf',10,100),
                    'max_features': trial.suggest_float('max_features',0,1)
                }
            else:
                params = {
                    'learning_rate': trial.suggest_float('learning_rate',1e-4,0.5,log=True),
                    'n_estimators':  trial.suggest_int('n_estimators',100,500),
                    'max_depth':     trial.suggest_int('max_depth',1,30),
                    'num_leaves':    2**10,
                    'reg_lambda':    trial.suggest_float('reg_lambda',1e-8,10,log=True),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 100)
                }
            model = ModelClass(**params)
            model.fit(X_train_.to_numpy(), y_train_)
            unique_train = np.unique(y_train_)
            if unique_train.size < 2:
                const = unique_train[0]
                return float(np.mean(y_val == const))
            preds = model.predict(X_val)
            return float(np.mean(preds == y_val))
        def obj_logloss(trial, ModelClass):
            if ModelClass is RandomForestClassifier:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators',100,500),
                    'max_depth':    trial.suggest_int('max_depth',1,30),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf',10,100),
                    'max_features': trial.suggest_float('max_features',0,1)
                }
            else:
                params = {
                    'learning_rate': trial.suggest_float('learning_rate',1e-4,0.5,log=True),
                    'n_estimators':  trial.suggest_int('n_estimators',100,500),
                    'max_depth':     trial.suggest_int('max_depth',1,30),
                    'num_leaves':    2**10,
                    'reg_lambda':    trial.suggest_float('reg_lambda',1e-8,10,log=True),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 100)
                }
            model = ModelClass(**params)
            model.fit(X_train_.to_numpy(), y_train_)
            proba = model.predict_proba(X_val.to_numpy())
            if proba.ndim != 2 or proba.shape[1] < 2:
                raise ValueError(f"{ModelClass.__name__}.predict_proba returned shape {proba.shape}")
            probs = proba[:, 1]

            return evaluate_log_loss(y_val, probs)

        studies = {}
        for ModelClass, isClassifier in [
            (RandomForestRegressor, False),
            (RandomForestClassifier, True),
            (LGBMRegressor, False),
            (LGBMClassifier, True)
        ]:
            if isClassifier != (not is_reg):
                continue  

            if is_reg:
                metric_obj_list = [
                    ('RMSE',      obj_rmse,   'minimize'),
                    ('CRPS',      obj_crps,   'minimize'),
                ]
            else:
                metric_obj_list = [
                    ('Accuracy',  obj_accuracy, 'maximize'),
                    ('LogLoss',   obj_logloss,  'minimize'),
                ]
            for metric, fn, direction in metric_obj_list:
                study = optuna.create_study(
                    sampler=optuna.samplers.TPESampler(seed=args.seed),
                    direction=direction
                )
                study.optimize(lambda t: fn(t, ModelClass), n_trials=N_TRIALS)
                studies[(ModelClass.__name__, metric)] = study


        for (model_name, tuned_for), study in studies.items():
            try:
                best_params = study.best_params
            except ValueError:
                print(f"No successful trials for {model_name} @ {split_fn.__name__}, skipping.")
                continue
            if model_name in ("LGBMRegressor", "LGBMClassifier"):
                best_params["num_leaves"] = 2**10
            Constructor = globals()[model_name]
            model = Constructor(**best_params)
            model.fit(X_tr_p.to_numpy(), y_tr_arr)

            y_pred = model.predict(X_te_p.to_numpy())
            rmse  = evaluate_rmse(y_te_arr, y_pred) if is_reg else np.nan
            if is_reg:
                resid = y_tr_arr - model.predict(X_tr_p.to_numpy())
                sigma = np.std(resid)
                crps_vals = crps_gaussian(y_te_arr, y_pred, sigma)
                crps  = float(np.mean(crps_vals))
            else:
                crps  = np.nan

            if not is_reg:
                acc = evaluate_accuracy(y_te_arr, y_pred)

                proba_test = model.predict_proba(X_te_p.to_numpy())
                if proba_test.ndim != 2 or proba_test.shape[1] < 2:
                    raise ValueError(f"{ModelClass.__name__}.predict_proba returned shape {proba_test.shape}")
                probs = proba_test[:, 1]

                ll = evaluate_log_loss(y_te_arr, probs)
            else:
                acc = ll = np.nan


            if tuned_for == 'RMSE':
                records.append({
                    'suite_id':     args.suite_id,
                    'task_id':      args.task_id,
                    'split_method': name_split,
                    'model':        model_name,
                    'metric':       'RMSE',
                    'value':        rmse
                })
            elif tuned_for == 'CRPS':
                records.append({
                    'suite_id':     args.suite_id,
                    'task_id':      args.task_id,
                    'split_method': name_split,
                    'model':        model_name,
                    'metric':       'CRPS',
                    'value':        crps
                })
            elif tuned_for == 'Accuracy':
                records.append({
                    'suite_id':     args.suite_id,
                    'task_id':      args.task_id,
                    'split_method': name_split,
                    'model':        model_name,
                    'metric':       'Accuracy',
                    'value':        acc
                })
            elif tuned_for == 'LogLoss':
                records.append({
                    'suite_id':     args.suite_id,
                    'task_id':      args.task_id,
                    'split_method': name_split,
                    'model':        model_name,
                    'metric':       'LogLoss',
                    'value':        ll
                })


    # save
    pd.DataFrame.from_records(records).to_csv(out_file, index=False)
    print(f"Saved tree-based results to {out_file}")

if __name__ == "__main__":
    main()
