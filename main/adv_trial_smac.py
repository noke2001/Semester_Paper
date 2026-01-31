import argparse
import os
import random
import gc

import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.model_selection import train_test_split
from properscoring import crps_gaussian, crps_ensemble

import smac 
from smac_sampler import SMACSampler
from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution

from src.loader import (
    load_dataset_offline, clean_data,
    standardize_data, prepare_for_split
)
from src.extrapolation_methods import (
    random_split, mahalanobis_split, umap_split,
    kmeans_split, gower_split, kmedoids_split, spatial_depth_split
)
from src.evaluation_metrics import (
    evaluate_crps, evaluate_log_loss, evaluate_accuracy
)
from src.models.Advanced_models import (
    DistributionalRandomForestRegressor,
    LightGBMLSSRegressor)

# --- experiment constants ---
SEED      = 10
N_TRIALS  = 100        # full 100 trials as in original paper
VAL_RATIO = 0.2        # 20% for validation
QUANTILE_SAMPLES = 100

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

    MAX_SAMPLES = 10000
    if len(X_full) > MAX_SAMPLES:
        X_full, _, y_full, _ = train_test_split(
            X_full, y_full,
            train_size=MAX_SAMPLES,
            stratify=y_full if not is_reg else None,
            random_state=args.seed
        )

    X, X_clean, y = clean_data(
        X_full, y_full, cat_ind, attr_names,
        task_type=cfg["task_type"]
    )
    if args.task_id in (361082, 361088, 361099) and is_reg:
        y = np.log(y)

    out_dir = os.path.join(args.result_folder, f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{args.suite_id}_{args.task_id}_advanced.csv")

    records = []
    quantiles = list(np.random.uniform(0,1,QUANTILE_SAMPLES))

  
    X_ready = prepare_for_split(X_clean)
    for split_fn in methods:
        studies = {}
        study = None
        model = None
        name_split = split_fn.__name__
        
        try:
            X_tr_clean, y_tr, X_te_clean, y_te = split_dataset(split_fn, X_ready, y)
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
        X_loop = X_loop.fillna(0)

        X_tr_p = X_loop.loc[X_tr_clean.index]
        X_te_p = X_loop.loc[X_te_clean.index]

        X_train = X_tr_p.loc[X_train_clean_unfilt.index]
        X_val   = X_tr_p.loc[X_val_clean_unfilt.index]

        X_train_, X_val = standardize_data(X_train, X_val, non_dummy_cols)
        X_tr_p, X_te_p  = standardize_data(X_tr_p, X_te_p, non_dummy_cols)

        y_train_ = y_tr.loc[X_train_clean_unfilt.index].to_numpy().ravel()
        y_val    = y_tr.loc[X_val_clean_unfilt.index].to_numpy().ravel()

        X_tr_p = X_tr_p.fillna(0)
        X_te_p = X_te_p.fillna(0)
        X_train_ = X_train_.fillna(0)
        X_val = X_val.fillna(0)

        y_tr_arr = y_tr.to_numpy().ravel()
        y_te_arr = y_te.to_numpy().ravel()

        def obj_crps_drf(trial):
            params = {
                'num_trees':     trial.suggest_int('num_trees', 100, 500),
                'mtry':          trial.suggest_int('mtry', 1, X_train_.shape[1]),
                'min_node_size':trial.suggest_int('min_node_size', 10, 100),
                'seed':          args.seed
            }
            model = DistributionalRandomForestRegressor(**params)
            model.fit(X_train_, y_train_)
            y_q = model.predict_quantiles(X_val, quantiles=quantiles)
            vals = crps_ensemble(y_val, y_q.quantile.squeeze(1))
            return float(np.mean(vals))

        def obj_crps_lss(trial):
            params = {
                'learning_rate':    trial.suggest_float('learning_rate', 1e-4, 0.5, log=True),
                'n_estimators':     trial.suggest_int('n_estimators', 100, 500),
                'reg_lambda':       trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'max_depth':        trial.suggest_int('max_depth', 1, 30),
                'min_child_samples':trial.suggest_int('min_child_samples', 10, 100),
                'num_leaves':       2**10
            }
            model = LightGBMLSSRegressor(**params)
            model.fit(X_train_.to_numpy(), y_train_)
            pr = model.predict_parameters(X_val)
            return float(np.mean(crps_gaussian(y_val, pr['loc'], pr['scale'])))


        tasks = []
        if is_reg:
            tasks.append(("DRF", obj_crps_drf, 'minimize', 'CRPS', DistributionalRandomForestRegressor))
            tasks.append(("DGBT", obj_crps_lss, 'minimize', 'CRPS', LightGBMLSSRegressor))
        
        search_space = get_search_space(tasks)

        studies = {}
        for name, fn, direction, metric, ModelClass in tasks:
            study = optuna.create_study(
                sampler=SMACSampler(search_space=search_space, n_trials=args.n_trials, seed=args.seed),
                direction=direction
            )
            study.optimize(fn, n_trials=N_TRIALS)
            studies[name] = (study, metric, ModelClass)

        for name, (study, metric, ModelClass) in studies.items():
            try:
                try:
                    best = study.best_params
                except ValueError:
                    print(f"Skipping {name} on split {name_split}: No successful trials in hyperparameter search.")
                    continue
                
                # Simplified model creation since GAMs are removed
                model = ModelClass(**best)

                inp_tr = X_tr_p if not isinstance(model, LightGBMLSSRegressor) else X_tr_p.to_numpy()
                out_tr = y_tr_arr

                try:
                    model.fit(inp_tr, out_tr)
                except np.linalg.LinAlgError as e:
                    print(f"Skipping model {name} on split {name_split} because SVD failed on final fit: {e}")
                    continue
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"Skipping model {name} on split {name_split} due to Python-level out of memory error")
                        continue
                    else:
                        raise 

                if metric == 'CRPS':
                    if isinstance(model, DistributionalRandomForestRegressor):
                        yq = model.predict_quantiles(X_te_p, quantiles=quantiles)
                        vals = crps_ensemble(y_te_arr, yq.quantile.squeeze(1))
                    else:
                        pr = model.predict_parameters(X_te_p)
                        vals = crps_gaussian(y_te_arr, pr['loc'], pr['scale'])
                    val = float(np.mean(vals))
                elif metric == 'RMSE':
                    preds = model.predict(X_te_p if not isinstance(model, LightGBMLSSRegressor) else X_te_p.to_numpy())
                    val = float(np.sqrt(np.mean((y_te_arr - preds)**2)))
                elif metric == 'LogLoss':
                    probs = model.predict_proba(X_te_p)
                    val = float(evaluate_log_loss(y_te_arr, probs))
                elif metric == 'Accuracy':
                    preds = model.predict(X_te_p)
                    val = evaluate_accuracy(y_te_arr, preds)
                else:
                    raise ValueError(f"Unknown metric: {metric}")

                records.append({
                    'suite_id':     args.suite_id,
                    'task_id':      args.task_id,
                    'split_method': name_split,
                    'model':        name,
                    'metric':       metric,
                    'value':        val,
                })

            except Exception as e:
                print(f"!!! A critical error occurred for model '{name}' on split '{name_split}'. Skipping. Error: {e}")
                records.append({
                    'suite_id':     args.suite_id,
                    'task_id':      args.task_id,
                    'split_method': name_split,
                    'model':        name,
                    'metric':       metric,
                    'value':        np.nan, 
                })
                continue

        del studies, study, model
        gc.collect()

    pd.DataFrame.from_records(records).to_csv(out_file, index=False)
    print(f"Saved advanced-model results to {out_file}")


if __name__ == '__main__':
    main()