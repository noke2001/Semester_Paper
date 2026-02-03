import argparse
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.model_selection import train_test_split
from properscoring import crps_gaussian

# might work
import gc

# new imports
import smac
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution
from smac_sampler import SMACSampler

from src.loader import (
    load_dataset_offline, clean_data,
    standardize_data, prepare_for_split
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

# --- experiment constants ---
SEED      = 10
N_TRIALS  = 100            
VAL_RATIO = 0.2            
# MAX_SAMPLES = 12000
MAX_SAMPLES = 60000

# Global Hardware Detection
USE_GPU = torch.cuda.is_available()

# --- Progress Tracking Helper ---
class ProgressTracker:
    def __init__(self, current_method_idx, total_methods, n_trials, model_name, metric):
        self.current_method_idx = current_method_idx
        self.total_methods = total_methods
        self.n_trials = n_trials
        self.model_name = model_name
        self.metric = metric

    def callback(self, study, trial):
        # Print every 10 trials, or the very first trial
        display_trial = trial.number + 1
        if display_trial % 10 == 0 or display_trial == 1:
            print(f"  [Progress] Split {self.current_method_idx}/{self.total_methods} | "
                  f"{self.model_name} ({self.metric}) | Trial {display_trial}/{self.n_trials}")

# --- suite configuration ---
SUITE_CONFIG = {
    "regression_numerical":             {"suite_id":336, "task_type":"regression",       "data_type":"numerical"},
    "classification_numerical":         {"suite_id":337, "task_type":"classification", "data_type":"numerical"},
    "regression_numerical_categorical":{"suite_id":335, "task_type":"regression",       "data_type":"numerical_categorical"},
    "classification_numerical_categorical":{
        "suite_id":334, "task_type":"classification","data_type":"numerical_categorical"
    },
    "tabzilla": {"suite_id":379, "task_type":"classification", "data_type":None}
}
EXTRAPOLATION_METHODS = {
    "numerical":             [random_split, mahalanobis_split, kmeans_split, umap_split, spatial_depth_split],
    "numerical_categorical":[random_split, gower_split, kmedoids_split, umap_split]
}

def find_config(suite_id):
    for cfg in SUITE_CONFIG.values():
        if cfg["suite_id"] == suite_id:
            return cfg.copy()
    raise ValueError(f"No suite config for suite_id={suite_id}")

def split_dataset(split_fn, X, y):
    out = split_fn(X, y) if split_fn is random_split else split_fn(X)
    if isinstance(out, tuple) and len(out) == 6:
        X_tr, _, y_tr, _, X_te, y_te = out
    else:
        train_idx, test_idx = out
        X_tr, X_te = X.loc[train_idx], X.loc[test_idx]
        y_tr, y_te = y.loc[train_idx], y.loc[test_idx]
    return X_tr, y_tr, X_te, y_te

def get_search_space(model_class, n_samples):
    safe_max_bin_high = max(256, n_samples)
    if "RandomForest" in model_class.__name__:
        return {
            "n_estimators": IntDistribution(100, 500),
            "max_depth": IntDistribution(1, 30),
            "min_samples_leaf": IntDistribution(10, 100),
            "max_features": FloatDistribution(0.1, 1.0)
        }
    else: # LightGBM
        return {
            "learning_rate": FloatDistribution(0.001, 1.0, log=True),
            "min_child_samples": IntDistribution(1, 1000, log=True),
            "reg_lambda": FloatDistribution(1e-8, 1000.0, log=True),
            "max_bin": IntDistribution(255, safe_max_bin_high, log=True),
            "subsample": FloatDistribution(0.5, 1.0),
            "colsample_bytree": FloatDistribution(0.5, 1.0),
            "max_depth": CategoricalDistribution([-1]),
            "num_leaves": IntDistribution(2, 1024, log=True)
        }

def get_optim_params(trial, model_class, n_samples):
    safe_max_bin_high = max(256, n_samples)
    if "RandomForest" in model_class.__name__:
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth':    trial.suggest_int('max_depth', 1, 30),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100),
            'max_features': trial.suggest_float('max_features', 0.1, 1.0),
            'n_jobs': -1,
            'random_state': SEED
        }
    else: # LightGBM
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 1.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 1000, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1000.0, log=True),
            'max_bin': trial.suggest_int('max_bin', 255, safe_max_bin_high, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'num_leaves': trial.suggest_int('num_leaves', 2, 1024, log=True),
            'max_depth': trial.suggest_categorical('max_depth', [-1]), 
            'n_estimators': 100,
            'verbose': -1,
            'random_state': SEED
        }
        if USE_GPU:
            params['device'] = 'cuda'
        else:
            params['n_jobs'] = -1
        return params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite_id",      type=int,   required=True)
    parser.add_argument("--task_id",       type=int,   required=True)
    parser.add_argument("--seed",          type=int,   default=SEED)
    parser.add_argument("--result_folder", type=str,   required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg      = find_config(args.suite_id)
    is_reg   = (cfg["task_type"] == "regression")
    
    print(f"Task type: {cfg['task_type']} | Device: {'GPU' if USE_GPU else 'CPU'}")

    X_full, y_full, cat_ind, attr_names = load_dataset_offline(args.suite_id, args.task_id)
    
    if isinstance(y_full, pd.DataFrame):
        y_full = y_full.iloc[:, 0] 
    elif isinstance(y_full, np.ndarray):
        y_full = pd.Series(y_full.ravel())

    if cfg["data_type"] is None:
        has_categorical = bool(getattr(cat_ind, "any")()) if hasattr(cat_ind, "any") else any(cat_ind)
        cfg["data_type"] = "numerical_categorical" if has_categorical else "numerical"
      
    methods = EXTRAPOLATION_METHODS[cfg["data_type"]]
    num_methods = len(methods)

    if len(X_full) > MAX_SAMPLES:
        stratify = y_full if (not is_reg and np.min(np.unique(y_full, return_counts=True)[1]) >= 2) else None
        X_full, _, y_full, _= train_test_split(X_full, y_full, train_size=MAX_SAMPLES, stratify=stratify, random_state=args.seed)

    X, X_clean, y = clean_data(X_full, y_full, cat_ind, attr_names, task_type=cfg["task_type"])
    
    if isinstance(y, (pd.DataFrame)): y = y.iloc[:, 0]
    elif isinstance(y, np.ndarray): y = pd.Series(y.ravel(), index=X_clean.index)

    if args.task_id in (361082, 361088, 361099) and is_reg: y = np.log(y)

    out_dir = os.path.join(args.result_folder, f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{args.suite_id}_{args.task_id}_trees_50k.csv")

    X_ready = prepare_for_split(X_clean)
    if X_ready.shape[1] == 0:
        X_source = X if X.shape[1] > 0 else X_full.loc[y.index]
        X_ready = pd.get_dummies(X_source, drop_first=True).astype('float32')
        X_ready.index = y.index 

    records = []
    y_series = pd.Series(y, index=X_ready.index)

    for m_idx, split_fn in enumerate(methods, 1):
        name_split = split_fn.__name__
        print(f"\n>>> Processing Split Method {m_idx}/{num_methods}: {name_split}")

        try:
            X_tr_clean, y_tr, X_te_clean, y_te = split_dataset(split_fn, X_ready, y_series)
            X_train_clean_unfilt, _, X_val_clean_unfilt, _ = split_dataset(split_fn, X_tr_clean, y_tr)
        except Exception as e:
            print(f"Skipping {name_split}: Split failed - {e}")
            continue

        X_loop = pd.get_dummies(X.copy(), drop_first=True).astype('float32').fillna(0)
        X_tr_p = X_loop.loc[X_tr_clean.index]
        X_te_p = X_loop.loc[X_te_clean.index]
        X_train = X_loop.loc[X_train_clean_unfilt.index]
        X_val   = X_loop.loc[X_val_clean_unfilt.index]
        
        non_dummy_cols = [c for c in X_loop.columns if c in X_ready.columns and X_ready[c].dtype.kind in 'if']
        X_train_, X_val = standardize_data(X_train, X_val, non_dummy_cols)
        X_tr_p, X_te_p  = standardize_data(X_tr_p, X_te_p, non_dummy_cols)

        # Caching: Convert to contiguous numpy arrays to save overhead
        X_train_np = np.ascontiguousarray(X_train_.fillna(0).to_numpy(), dtype=np.float32)
        X_val_np   = np.ascontiguousarray(X_val.fillna(0).to_numpy(), dtype=np.float32)
        X_tr_p_np  = np.ascontiguousarray(X_tr_p.fillna(0).to_numpy(), dtype=np.float32)
        X_te_p_np  = np.ascontiguousarray(X_te_p.fillna(0).to_numpy(), dtype=np.float32)

        y_train_arr = y_tr.loc[X_train_clean_unfilt.index].to_numpy().ravel()
        y_val_arr   = y_tr.loc[X_val_clean_unfilt.index].to_numpy().ravel()
        y_tr_arr    = y_tr.to_numpy().ravel()
        y_te_arr    = y_te.to_numpy().ravel()

        n_samples_train = len(y_train_arr)

        def obj_rmse(trial, ModelClass):
            params = get_optim_params(trial, ModelClass, n_samples_train)
            # 1. Force safety params
            if USE_GPU and "LGBM" in ModelClass.__name__:
                params['max_bin'] = 63
                params['gpu_use_dp'] = False
                
            model = ModelClass(**params)
            model.fit(X_train_np, y_train_arr)
            score = np.sqrt(np.mean((y_val_arr - model.predict(X_val_np))**2))
            
            # 2. NUCLEAR CLEANUP
            del model
            gc.collect() # Force Python to release references
            return score

        def obj_crps(trial, ModelClass):
            params = get_optim_params(trial, ModelClass, n_samples_train)
            model = ModelClass(**params); model.fit(X_train_np, y_train_arr)
            sigma = np.std(y_train_arr - model.predict(X_train_np))
            return float(np.mean(crps_gaussian(y_val_arr, model.predict(X_val_np), sigma)))

        def obj_accuracy(trial, ModelClass):
            params = get_optim_params(trial, ModelClass, n_samples_train)
            model = ModelClass(**params); model.fit(X_train_np, y_train_arr)
            return float(np.mean(model.predict(X_val_np) == y_val_arr))

        def obj_logloss(trial, ModelClass):
             params = get_optim_params(trial, ModelClass, n_samples_train)
             model = ModelClass(**params); model.fit(X_train_np, y_train_arr)
             # return evaluate_log_loss(y_val_arr, model.predict_proba(X_val_np)[:, 1])
             return evaluate_log_loss(y_val_arr, model.predict_proba(X_val_np))

        studies = {}
        
        # --- CHANGED ORDER: LGBM first, then Random Forest ---
        model_list = [
            (LGBMRegressor, False), 
            (LGBMClassifier, True), 
            (RandomForestRegressor, False), 
            (RandomForestClassifier, True)
        ]
        
        for ModelClass, isClassifier in model_list:
            if isClassifier != (not is_reg): continue  
            
            search_space = get_search_space(ModelClass, n_samples_train)
            metrics = [('RMSE', obj_rmse, 'minimize'), ('CRPS', obj_crps, 'minimize')] if is_reg else [('Accuracy', obj_accuracy, 'maximize'), ('LogLoss', obj_logloss, 'minimize')]

            for metric, fn, direction in metrics:
                model_name = ModelClass.__name__
                tracker = ProgressTracker(m_idx, num_methods, N_TRIALS, model_name, metric)
                
                sampler = SMACSampler(search_space=search_space, n_trials=N_TRIALS, seed=SEED)
                study = optuna.create_study(sampler=sampler, direction=direction)
                optuna.logging.set_verbosity(optuna.logging.WARNING)
                
                study.optimize(lambda t: fn(t, ModelClass), n_trials=N_TRIALS, callbacks=[tracker.callback])
                studies[(model_name, metric)] = study

        for (model_name, tuned_for), study in studies.items():
            try:
                best_params = study.best_params
                if "RandomForest" in model_name:
                    best_params.update({'n_jobs': -1, 'random_state': SEED})
                else:
                    best_params.update({'n_estimators': 100, 'random_state': SEED, 'device': 'cuda' if USE_GPU else 'cpu', 'gpu_use_dp': False})
                    if not USE_GPU: best_params['n_jobs'] = -1

                model = globals()[model_name](**best_params)
                model.fit(X_tr_p_np, y_tr_arr)
                y_pred = model.predict(X_te_p_np)
                
                res = {'suite_id': args.suite_id, 'task_id': args.task_id, 'split_method': name_split, 'model': model_name, 'metric': tuned_for}
                if is_reg:
                    res.update({'test_rmse': evaluate_rmse(y_te_arr, y_pred), 'test_crps': float(np.mean(crps_gaussian(y_te_arr, y_pred, np.std(y_tr_arr - model.predict(X_tr_p_np))))), 'test_acc': np.nan, 'test_logloss': np.nan})
                else:
                    res.update({'test_rmse': np.nan, 'test_crps': np.nan, 'test_acc': evaluate_accuracy(y_te_arr, y_pred), 'test_logloss': evaluate_log_loss(y_te_arr, model.predict_proba(X_te_p_np)[:, 1])})
                records.append(res)
            except Exception as e:
                print(f"Eval failed for {model_name} {tuned_for}: {e}")

    pd.DataFrame.from_records(records).to_csv(out_file, index=False)
    print(f"\nFinal results saved to {out_file}")

if __name__ == "__main__":
    main()
