import argparse
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import random
import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.model_selection import train_test_split
from properscoring import crps_gaussian, crps_ensemble
from src.loader import (
    load_dataset_offline,
    clean_data,
    standardize_data
)
from src.extrapolation_methods_tabz import (
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
    evaluate_accuracy,
    evaluate_log_loss,
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

# Experiment constants
SEED = 10
N_TRIALS = 100  
VAL_RATIO = 0.2  
N_SAMPLES = 100  
PATIENCE = 40  
N_EPOCHS = 1000  
BATCH_SIZE = 1024



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

    
    if len(X_full) > 12000:
        X_full, _, y_full, _ = train_test_split(
            X_full,
            y_full,
            train_size=12000,
            stratify=None if is_reg else y_full,
            random_state=args.seed,
        )
    X, X_clean, y = clean_data(
        X_full, y_full, cat_ind, attr_names, task_type=cfg["task_type"]
    )

    if not is_reg:
        num_classes = y.nunique()

    if is_reg and args.task_id in (361082, 361088, 361099):
        y = np.log(y)

    # Prepare output
    out_dir = os.path.join(args.result_folder, f"seed_{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{args.suite_id}_{args.task_id}_engression.csv")

    records = []

    # Loop over splits
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
        X_loop = X_loop.fillna(0)

        X_tr_p = X_loop.loc[X_tr_clean.index]
        X_te_p = X_loop.loc[X_te_clean.index]

        X_train = X_tr_p.loc[X_train_clean_unfilt.index]
        X_val   = X_tr_p.loc[X_val_clean_unfilt.index]

        X_train_, X_val = standardize_data(X_train, X_val, non_dummy_cols)
        X_tr_p, X_te_p  = standardize_data(X_tr_p, X_te_p, non_dummy_cols)

        X_tr_p = X_tr_p.fillna(0)
        X_te_p = X_te_p.fillna(0)
        X_train_ = X_train_.fillna(0)
        X_val = X_val.fillna(0)

        y_train_ = y_tr.loc[X_train_clean_unfilt.index].to_numpy().ravel()
        y_val    = y_tr.loc[X_val_clean_unfilt.index].to_numpy().ravel()
        y_tr_arr = y_tr.to_numpy().ravel()
        y_te_arr = y_te.to_numpy().ravel()

   
        if is_reg:
            # RMSE
            def eng_rmse(trial):
                checkpoint_path = os.path.join(  # Unique checkpoint path
                    out_dir, f"checkpoint_engression_rmse_{trial.number}.pt"
                )
                m = EngressionRegressor(
                    learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                    num_epochs=trial.suggest_int("num_epochs", 100, 1000),
                    num_layer=trial.suggest_int("num_layer", 2, 5),
                    hidden_dim=trial.suggest_int("hidden_dim", 100, 500),
                    resblock=trial.suggest_categorical("resblock", [True, False]),
                    batch_size=BATCH_SIZE,
                    seed=args.seed,
                )
                
                m.fit(X_train_, y_train_)
                mu = m.predict(X_val)
                return evaluate_rmse(y_val, mu)

            study_er = optuna.create_study(
                sampler=optuna.samplers.TPESampler(seed=args.seed),
                direction="minimize",
            )
            study_er.optimize(eng_rmse, n_trials=N_TRIALS)

            # CRPS
            def eng_crps(trial):
                checkpoint_path = os.path.join(  # Unique checkpoint path
                    out_dir, f"checkpoint_engression_crps_{trial.number}.pt"
                )
                m = EngressionRegressor(
                    learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                    num_epochs=trial.suggest_int("num_epochs", 100, 1000),
                    num_layer=trial.suggest_int("num_layer", 2, 5),
                    hidden_dim=trial.suggest_int("hidden_dim", 100, 500),
                    resblock=trial.suggest_categorical("resblock", [True, False]),
                    batch_size=BATCH_SIZE,
                    seed=args.seed,
                )

                m.fit(X_train_, y_train_)

                y_val_samples = m.predict_samples(X_val, sample_size=N_SAMPLES) 
                crps_values = [
                    crps_ensemble(y_val[i], y_val_samples[i])
                    for i in range(len(y_val))
                ]
                return float(np.mean(crps_values))

            study_ec = optuna.create_study(
                sampler=optuna.samplers.TPESampler(seed=args.seed),
                direction="minimize",
            )
            study_ec.optimize(eng_crps, n_trials=N_TRIALS)

            # Refit & evaluate Engression
            for name, study, metric in [
                ("Engression", study_er, "RMSE"),
                ("Engression", study_ec, "CRPS"),
            ]:
                try:
                    bp = study.best_params
                except ValueError:
                    print(f"Skipping {name}: no completed trials.")
                    continue
                checkpoint_path = os.path.join(
                    out_dir, f"checkpoint_engression_{metric.lower()}.pt"
                )  # Use a consistent name here
                m = EngressionRegressor(
                    learning_rate=bp["learning_rate"],
                    num_epochs=bp["num_epochs"],
                    num_layer=bp["num_layer"],
                    hidden_dim=bp["hidden_dim"],
                    resblock=bp["resblock"],
                    batch_size=BATCH_SIZE,
                    seed=args.seed,
                )

                m.fit(X_tr_p, y_tr_arr)
                if metric == "RMSE":
                    mu_test = m.predict(X_te_p)
                    val = evaluate_rmse(y_te_arr, mu_test)
                else:
                    y_test_samples = m.predict_samples(X_te_p, sample_size=N_SAMPLES)
                    crps_values = [
                        crps_ensemble(y_te_arr[i], y_test_samples[i])
                        for i in range(len(y_te_arr))
                    ]
                    val = float(np.mean(crps_values))
                    
                records.append(
                    {
                        "suite_id": args.suite_id,
                        "task_id": args.task_id,
                        "split_method": name_split,
                        "model": name,
                        "metric": metric,
                        "value": val,
                    }
                )
        else:
            # Engression classification
            def eng_acc(trial):
                checkpoint_path = os.path.join(  # Unique checkpoint path
                    out_dir, f"checkpoint_engression_acc_{trial.number}.pt"
                )
                m = EngressionClassifier(
                    learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                    num_epochs=trial.suggest_int("num_epochs", 100, 1000),
                    num_layer=trial.suggest_int("num_layer", 2, 5),
                    hidden_dim=trial.suggest_int("hidden_dim", 100, 500),
                    resblock=trial.suggest_categorical("resblock", [True, False]),
                    batch_size=BATCH_SIZE,
                    seed=args.seed,
                )
                # EngressionClassifier.fit only takes train data
                m.fit(X_train_, y_train_)
                preds = m.predict(X_val)
                return evaluate_accuracy(y_val, preds)

            def eng_ll(trial):
                checkpoint_path = os.path.join(  # Unique checkpoint path
                    out_dir, f"checkpoint_engression_ll_{trial.number}.pt"
                )
                m = EngressionClassifier(
                    learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
                    num_epochs=trial.suggest_int("num_epochs", 100, 1000),
                    num_layer=trial.suggest_int("num_layer", 2, 5),
                    hidden_dim=trial.suggest_int("hidden_dim", 100, 500),
                    resblock=trial.suggest_categorical("resblock", [True, False]),
                    batch_size=BATCH_SIZE,
                    seed=args.seed,
                )
                # EngressionClassifier.fit only takes train data
                m.fit(X_train_, y_train_)
                probs = m.predict_proba(X_val)
                return evaluate_log_loss(y_val, probs)

            st_acc = optuna.create_study(
                sampler=optuna.samplers.TPESampler(seed=args.seed),
                direction="maximize",
            )
            st_acc.optimize(eng_acc, n_trials=N_TRIALS)
            st_ll = optuna.create_study(
                sampler=optuna.samplers.TPESampler(seed=args.seed),
                direction="minimize",
            )
            st_ll.optimize(eng_ll, n_trials=N_TRIALS)
            for name, study, metric in [
                ("Engression", st_acc, "Accuracy"),
                ("Engression", st_ll, "LogLoss"),
            ]:
                try:
                    bp = study.best_params
                except ValueError:
                    print(f"Skipping {name}: no completed trials.")
                    continue
                checkpoint_path = os.path.join(
                    out_dir, f"checkpoint_engression_{metric.lower()}.pt"
                )
                m = EngressionClassifier(
                    learning_rate=bp["learning_rate"],
                    num_epochs=bp["num_epochs"],
                    num_layer=bp["num_layer"],
                    hidden_dim=bp["hidden_dim"],
                    resblock=bp["resblock"],
                    batch_size=BATCH_SIZE,
                    seed=args.seed,
                )
                m.fit(X_tr_p, y_tr_arr)
                if metric == "Accuracy":
                    preds = m.predict(X_te_p)
                    val = evaluate_accuracy(y_te_arr, preds)
                else:
                    probs = m.predict_proba(X_te_p)
                    val = evaluate_log_loss(y_te_arr, probs)
                records.append(
                    {
                        "suite_id": args.suite_id,
                        "task_id": args.task_id,
                        "split_method": name_split,
                        "model": name,
                        "metric": metric,
                        "value": val,
                    }
                )

            
                

    print(len(records), "records in total")
    pd.DataFrame(records).to_csv(out_file, index=False)
    print(f"Saved engression results to {out_file}")


if __name__ == "__main__":
    main()
