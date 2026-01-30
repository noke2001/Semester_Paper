import argparse
import os
import random
import gc
import sys

import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.model_selection import train_test_split
# We still need crps_ensemble for Engression (complex distribution), 
# but we can drop crps_gaussian as we now have a GPU version.
from properscoring import crps_ensemble

# --- NEW IMPORTS ---
import smac
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution
from smac_sampler import SMACSampler

from src.loader import (
    load_dataset_offline,
    clean_data,
    standardize_data
)
from src.extrapolation_methods import (
    random_split,
    mahalanobis_split,
    umap_split,
    kmeans_split,
    gower_split,
    kmedoids_split,
    spatial_depth_split
)
from src.evaluation_metrics_gpu import (
    evaluate_rmse,
    evaluate_accuracy,
    evaluate_log_loss,
    evaluate_crps 
)
from src.models.Neural_models import (
    EngressionRegressor,
    EngressionClassifier,
    MLPRegressor,
    MLPClassifier,
    ResNetRegressor,
    ResNetClassifier,
    FTTrans_Regressor,
    FTTrans_Classifier
)

# --- PROGRESS TRACKER CLASS ---
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
            sys.stdout.flush() 

# --- CONFIGURATION ---
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on {DEVICE}")

SEED = 10
N_TRIALS = 100
VAL_RATIO = 0.2
N_SAMPLES = 100
PATIENCE = 40
N_EPOCHS = 1000
BATCH_SIZE = 11000 

SUITE_CONFIG = {
    "regression_numerical":             {"suite_id":336, "task_type":"regression",       "data_type":"numerical"},
    "classification_numerical":         {"suite_id":337, "task_type":"classification", "data_type":"numerical"},
    "regression_numerical_categorical": {"suite_id":335, "task_type":"regression",       "data_type":"numerical_categorical"},
    "classification_numerical_categorical":{
        "suite_id":334, "task_type":"classification","data_type":"numerical_categorical"
    },
    "tabzilla": {"suite_id":379, "task_type":"classification",       "data_type":None}
}

EXTRAPOLATION_METHODS = {
    "numerical":             [random_split, mahalanobis_split, kmeans_split, umap_split, spatial_depth_split],
    "numerical_categorical": [random_split, gower_split, kmedoids_split, umap_split]
}

# --- SEARCH SPACES ---

def get_engression_space():
    return {
        "learning_rate": FloatDistribution(1e-4, 1e-2, log=True),
        "num_epochs": IntDistribution(100, 1000),
        "num_layer": IntDistribution(2, 5),
        "hidden_dim": IntDistribution(100, 500),
        "resblock": CategoricalDistribution([True, False]),
    }

def get_mlp_space():
    return {
        "n_blocks": IntDistribution(1, 5),
        "d_block": IntDistribution(10, 500),
        "dropout": FloatDistribution(0, 1),
        "learning_rate": FloatDistribution(1e-4, 5e-2, log=True),
        "weight_decay": FloatDistribution(1e-8, 1e-3, log=True),
    }

def get_resnet_space():
    return {
        "n_blocks": IntDistribution(1, 5),
        "d_block": IntDistribution(10, 500),
        "dropout1": FloatDistribution(0, 1),
        "dropout2": FloatDistribution(0, 1),
        "d_hidden_multiplier": FloatDistribution(0.5, 3.0),
        "learning_rate": FloatDistribution(1e-4, 5e-2, log=True),
        "weight_decay": FloatDistribution(1e-8, 1e-3, log=True),
    }

def get_fttrans_space():
    return {
        "n_blocks": IntDistribution(1, 5),
        "d_block_multiplier": IntDistribution(1, 25),
        "attention_n_heads": IntDistribution(1, 20),
        "attention_dropout": FloatDistribution(0, 1),
        "ffn_d_hidden_multiplier": FloatDistribution(0.5, 3.0),
        "ffn_dropout": FloatDistribution(0, 1),
        "residual_dropout": FloatDistribution(0, 1),
        "learning_rate": FloatDistribution(1e-4, 5e-2, log=True),
        "weight_decay": FloatDistribution(1e-8, 1e-3, log=True),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite_id",     type=int,   required=True)
    parser.add_argument("--task_id",      type=int,   required=True)
    parser.add_argument("--seed",         type=int,   default=SEED)
    parser.add_argument("--result_folder",type=str,   required=True)
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = find_config(args.suite_id)
    is_reg = (cfg["task_type"] == "regression")

    # Load & Clean Data
    X_full, y_full, cat_ind, attr_names = load_dataset_offline(args.suite_id, args.task_id)

    if cfg["data_type"] is None:
        if hasattr(cat_ind, "any"):
            has_categorical = bool(getattr(cat_ind, "any")())
        else:
            has_categorical = any(cat_ind)
        cfg["data_type"] = "numerical_categorical" if has_categorical else "numerical"
      
    methods = EXTRAPOLATION_METHODS[cfg["data_type"]]
    num_methods = len(methods)

    # --- INCREASED LIMIT (Safe for large datasets) ---
    TOTAL_LIMIT = 100000 
    
    # We only try to subsample if the dataset is actually larger than the limit.
    if len(X_full) > TOTAL_LIMIT:
        X_full, _, y_full, _ = train_test_split(
            X_full, y_full,
            train_size=TOTAL_LIMIT,
            stratify=None if is_reg else y_full,
            random_state=args.seed,
        )

    X, X_clean, y = clean_data(X_full, y_full, cat_ind, attr_names, task_type=cfg["task_type"])
      
    num_classes = 0
    if not is_reg:
        # Robust label encoding
        if hasattr(y, "values"):
            y_flat = y.values.ravel()
        else:
            y_flat = np.asarray(y).ravel()
        y_encoded, uniques = pd.factorize(y_flat, sort=True)
        y = pd.Series(y_encoded, index=y.index)
        num_classes = len(uniques)
        print(f"  [Info] Re-encoded target variable. {num_classes} classes found.")

    if is_reg and args.task_id in (361082, 361088, 361099):
        y = np.log(y)

    out_dir = os.path.join(args.result_folder, f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{args.suite_id}_{args.task_id}_neurals_50k.csv")

    records = []

    # --- MAIN LOOP OVER SPLITS ---
    for m_idx, split_fn in enumerate(methods, 1):
        name_split = split_fn.__name__
        print(f"\n>>> Processing Split Method {m_idx}/{num_methods}: {name_split}")

        # 1. First Split (Train+Val / Test)
        try:
            X_tr_clean, y_tr, X_te_clean, y_te = split_dataset(split_fn, X_clean, y)
        except Exception as e:
            print(f"Skipping {name_split}: first split failed: {e}")
            continue
        
        if X_tr_clean.shape[1] == 0 or X_te_clean.shape[1] == 0:
            print(f"Skipping {name_split}: no features after split")
            continue

        # 2. Second Split (Train / Validation)
        try:
            X_train_clean_unfilt, _, X_val_clean_unfilt, _ = split_dataset(split_fn, X_tr_clean, y_tr)
        except Exception as e:
            print(f"Skipping {name_split}: validation split failed: {e}")
            continue
        
        # --- ROBUST SIZE ENFORCEMENT ---
        TARGET_TRAIN = 50000
        TARGET_VAL = 10000

        # min() ensures we never ask for more rows than exist.
        limit_tr = min(len(X_train_clean_unfilt), TARGET_TRAIN)
        limit_val = min(len(X_val_clean_unfilt), TARGET_VAL)

        # If dataset is too small to have ANY training or validation data
        if limit_tr == 0 or limit_val == 0:
            print(f"Skipping {name_split}: Dataset too small (Train: {limit_tr}, Val: {limit_val})")
            continue

        train_indices = X_train_clean_unfilt.index[:limit_tr]
        val_indices = X_val_clean_unfilt.index[:limit_val]
        
        X_train_clean_unfilt = X_train_clean_unfilt.loc[train_indices]
        X_val_clean_unfilt = X_val_clean_unfilt.loc[val_indices]

        print(f"  [Info] Sizes used -> Train: {len(X_train_clean_unfilt)}, Val: {len(X_val_clean_unfilt)}")

        # 3. Preprocessing
        X_loop = X.copy()
        dummy_cols = X_loop.select_dtypes(include=['bool','category','object','string']).columns
        
        # Safe check for dummy columns
        if len(X_train_clean_unfilt) > 0:
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

        # Prevent standardization crash if validation set is empty
        if len(X_train) > 0 and len(X_val) > 0:
             X_train_, X_val = standardize_data(X_train, X_val, non_dummy_cols)
             X_tr_p, X_te_p  = standardize_data(X_tr_p, X_te_p, non_dummy_cols)
        else:
             print(f"Skipping {name_split}: Data became empty during processing.")
             continue

        X_tr_p = X_tr_p.fillna(0)
        X_te_p = X_te_p.fillna(0)
        X_train_ = X_train_.fillna(0)
        X_val = X_val.fillna(0)

        y_train_ = y_tr.loc[X_train_clean_unfilt.index].to_numpy().ravel()
        y_val    = y_tr.loc[X_val_clean_unfilt.index].to_numpy().ravel()
        y_tr_arr = y_tr.to_numpy().ravel()
        y_te_arr = y_te.to_numpy().ravel()

        # --- MODEL DEFINITIONS ---
        
        # ========================================
        # 1. ENGRESSION LOOP (COMMENTED OUT)
        # ========================================
        # engression_space = get_engression_space()
        # if is_reg:
        #     metrics_to_run = [('RMSE', 'minimize'), ('CRPS', 'minimize')]
        #     ModelCls = EngressionRegressor
        # else:
        #     metrics_to_run = [('Accuracy', 'maximize'), ('LogLoss', 'minimize')]
        #     ModelCls = EngressionClassifier

        # for metric, direction in metrics_to_run:
        #    tracker = ProgressTracker(m_idx, num_methods, N_TRIALS, "Engression", metric)

        #    def obj_engression(trial):
        #        params = trial.params.copy()
        #        try:
        #            kwargs = {
        #                "batch_size": BATCH_SIZE, "seed": args.seed, "device": DEVICE 
        #            }
        #            if not is_reg: kwargs["num_classes"] = num_classes

        #            model = ModelCls(**params, **kwargs)
        #            model.fit(X_train_, y_train_)

        #            score = np.nan
        #            if metric == 'RMSE':
        #                pred = model.predict(X_val)
        #                score = evaluate_rmse(y_val, pred)
        #            elif metric == 'CRPS':
        #                y_val_samples = model.predict_samples(X_val, sample_size=N_SAMPLES).cpu().numpy()
        #                crps_vals = [crps_ensemble(y_val[i], y_val_samples[i]) for i in range(len(y_val))]
        #                score = float(np.mean(crps_vals))
        #            elif metric == 'Accuracy':
        #                preds = model.predict(X_val)
        #                score = evaluate_accuracy(y_val, preds)
        #            elif metric == 'LogLoss':
        #                probs = model.predict_proba(X_val)
        #                score = evaluate_log_loss(y_val, probs)
        #             
        #            if np.isnan(score) or np.isinf(score):
        #                return float('inf') if direction == 'minimize' else float('-inf')
        #            return score

        #        except Exception as e:
        #            return float('inf') if direction == 'minimize' else float('-inf')

        #    # Optimization
        #    try:
        #        sampler = SMACSampler(search_space=engression_space, n_trials=N_TRIALS, seed=args.seed)
        #        study = optuna.create_study(sampler=sampler, direction=direction)
        #        optuna.logging.set_verbosity(optuna.logging.WARNING)
        #        study.optimize(obj_engression, n_trials=N_TRIALS, callbacks=[tracker.callback])
        #    except Exception as e:
        #        print(f"Optuna Engression failed: {e}")
        #        continue

        #    # Refit & Evaluate
        #    try:
        #        try:
        #             best_params = study.best_params
        #        except ValueError:
        #             print(f"Skipping Engression ({metric}): All trials failed.")
        #             continue

        #        kwargs = {
        #             "batch_size": BATCH_SIZE, "seed": args.seed, "device": DEVICE
        #        }
        #        if not is_reg: kwargs["num_classes"] = num_classes

        #        model = ModelCls(**best_params, **kwargs)
        #        model.fit(X_tr_p, y_tr_arr)

        #        val_score = np.nan
        #        if is_reg:
        #            if metric == 'RMSE':
        #                pred = model.predict(X_te_p)
        #                val_score = evaluate_rmse(y_te_arr, pred)
        #            elif metric == 'CRPS':
        #                y_te_samples = model.predict_samples(X_te_p, sample_size=N_SAMPLES).cpu().numpy()
        #                crps_vals = [crps_ensemble(y_te_arr[i], y_te_samples[i]) for i in range(len(y_te_arr))]
        #                val_score = float(np.mean(crps_vals))
        #        else:
        #            if metric == 'Accuracy':
        #                preds = model.predict(X_te_p)
        #                val_score = evaluate_accuracy(y_te_arr, preds)
        #            elif metric == 'LogLoss':
        #                probs = model.predict_proba(X_te_p)
        #                val_score = evaluate_log_loss(y_te_arr, probs)
        #         
        #        records.append({
        #            "suite_id": args.suite_id, "task_id": args.task_id,
        #            "split_method": name_split, "model": "Engression",
        #            "metric": metric, "value": val_score
        #        })
        #    except Exception as e:
        #        print(f"Refit failed for Engression ({metric}): {e}")
        #     
        #    del model, study
        #    torch.cuda.empty_cache()
        #    gc.collect()


        # ----------------------------------------
        # 2. DEEP MODELS LOOP
        # ----------------------------------------
        
        deep_models = []
        if is_reg:
            # deep_models.append(("MLP", MLPRegressor, get_mlp_space())) # EXCLUDED
            deep_models.append(("ResNet", ResNetRegressor, get_resnet_space()))
            deep_models.append(("FTTransformer", FTTrans_Regressor, get_fttrans_space()))
        else:
            # deep_models.append(("MLP", MLPClassifier, get_mlp_space())) # EXCLUDED
            deep_models.append(("ResNet", ResNetClassifier, get_resnet_space()))
            deep_models.append(("FTTransformer", FTTrans_Classifier, get_fttrans_space()))

        for model_name, ModelClass, search_space in deep_models:
            
            if is_reg:
                metrics_loop = [('RMSE', 'minimize'), ('CRPS', 'minimize')]
            else:
                metrics_loop = [('Accuracy', 'maximize'), ('LogLoss', 'minimize')]

            for metric, direction in metrics_loop:

                tracker = ProgressTracker(m_idx, num_methods, N_TRIALS, model_name, metric)
                
                def obj_deep(trial):
                    params = trial.params.copy()
                    ckpt_name = f"ckpt_{model_name}_{metric}_{trial.number}.pt"
                    checkpoint_path = os.path.join(out_dir, ckpt_name)

                    kwargs = {
                        "n_epochs": N_EPOCHS, "patience": PATIENCE, "batch_size": BATCH_SIZE,
                        "seed": args.seed, "checkpoint_path": checkpoint_path, "device": DEVICE 
                    }
                    if not is_reg: kwargs["num_classes"] = num_classes
                    if "FTTransformer" in model_name and "device" in kwargs: del kwargs["device"]

                    try:
                        model = ModelClass(**params, **kwargs)
                        model.fit(X_train_, y_train_, X_val, y_val)
                        
                        score = np.nan
                        if is_reg:
                            if metric == 'RMSE':
                                pred = model.predict(X_val)
                                score = evaluate_rmse(y_val, pred)
                            elif metric == 'CRPS':
                                tr_loader, _, _ = model.prepare_data(X_train_, y_train_, X_val, y_val, None, None)
                                mu, sigma = model.predict_with_uncertainty(X_val, tr_loader)
                                score = evaluate_crps(y_val, mu, sigma)
                        else:
                            if metric == 'Accuracy':
                                preds = model.predict(X_val)
                                score = evaluate_accuracy(y_val, preds)
                            elif metric == 'LogLoss':
                                probs = model.predict_proba(X_val)
                                score = evaluate_log_loss(y_val, probs)
                        
                        if os.path.exists(checkpoint_path): os.remove(checkpoint_path)
                        
                        if np.isnan(score) or np.isinf(score):
                            return float('inf') if direction == 'minimize' else float('-inf')
                        return score

                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print(f"OOM for {model_name} trial {trial.number}. Pruning.")
                            torch.cuda.empty_cache()
                            raise optuna.TrialPruned()
                        return float('inf') if direction == 'minimize' else float('-inf')
                    except Exception as e:
                        return float('inf') if direction == 'minimize' else float('-inf')

                try:
                    sampler = SMACSampler(search_space=search_space, n_trials=N_TRIALS, seed=args.seed)
                    study = optuna.create_study(sampler=sampler, direction=direction)
                    optuna.logging.set_verbosity(optuna.logging.WARNING)
                    study.optimize(obj_deep, n_trials=N_TRIALS, callbacks=[tracker.callback])
                except Exception as e:
                    print(f"Optimization loop failed for {model_name} - {metric}: {e}")
                    continue

                try:
                    try:
                        best_params = study.best_params
                    except ValueError:
                        print(f"Skipping {model_name} ({metric}): All trials failed.")
                        continue

                    ckpt_final = os.path.join(out_dir, f"final_{model_name}_{metric}.pt")
                    kwargs = {
                        "n_epochs": N_EPOCHS, "patience": PATIENCE, "batch_size": BATCH_SIZE,
                        "seed": args.seed, "checkpoint_path": ckpt_final, "device": DEVICE
                    }
                    if not is_reg: kwargs["num_classes"] = num_classes
                    if "FTTransformer" in model_name and "device" in kwargs: del kwargs["device"]

                    model = ModelClass(**best_params, **kwargs)
                    model.fit(X_tr_p, y_tr_arr, None, None) 

                    val_score = np.nan
                    if is_reg:
                        if metric == 'RMSE':
                            pred = model.predict(X_te_p)
                            val_score = evaluate_rmse(y_te_arr, pred)
                        elif metric == 'CRPS':
                            final_tr_loader, _, _ = model.prepare_data(X_tr_p, y_tr_arr, None, None, None, None)
                            mu, sigma = model.predict_with_uncertainty(X_te_p, final_tr_loader)
                            val_score = evaluate_crps(y_te_arr, mu, sigma)
                    else:
                        if metric == 'Accuracy':
                            preds = model.predict(X_te_p)
                            val_score = evaluate_accuracy(y_te_arr, preds)
                        elif metric == 'LogLoss':
                            probs = model.predict_proba(X_te_p)
                            val_score = evaluate_log_loss(y_te_arr, probs)

                    records.append({
                        "suite_id": args.suite_id, "task_id": args.task_id,
                        "split_method": name_split, "model": model_name,
                        "metric": metric, "value": val_score
                    })
                    
                    if os.path.exists(ckpt_final): os.remove(ckpt_final)
                        
                except Exception as e:
                    print(f"Refit failed for {model_name} - {metric}: {e}")

                del model, study
                torch.cuda.empty_cache()
                gc.collect()

    print(f"Total records: {len(records)}")
    pd.DataFrame(records).to_csv(out_file, index=False)
    print(f"Saved neural results to {out_file}")

if __name__ == "__main__":
    main()