import os, random, re
import numpy as np
import pandas as pd
import torch
from umap import UMAP
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
import lightgbm as lgbm
import optuna
from sklearn.ensemble import RandomForestRegressor as _RFRegressor
from src.models.Tree_models import RandomForestRegressor, LGBMRegressor
from src.loader import load_dataset_offline

def run_task_umap(suite_id: int, task_id: int, result_folder: str, seed: int = 10):
    # reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # load & clean data offline
    X_full, y_full, cat_ind, attr_names = load_dataset_offline(suite_id, task_id)
    # apply clean_data logic inline
    X, X_clean, y = X_full.copy(), X_full.copy(), y_full.copy()
    for col, ind in zip(attr_names, cat_ind):
        if ind and X[col].nunique() > 20:
            X = X.drop(col, axis=1)
    X_clean = X.copy()
    for col, ind in zip(attr_names, cat_ind):
        if not ind:
            if X[col].nunique() < 10:
                X = X.drop(col, axis=1)
                X_clean = X_clean.drop(col, axis=1)
            elif X[col].value_counts(normalize=True).max() > 0.7:
                X_clean = X_clean.drop(col, axis=1)
    corr = X_clean.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_feats = [c for c in upper.columns if any(upper[c] > 0.9)]
    X_clean = X_clean.drop(drop_feats, axis=1)
    X = X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+','', x))

    # optional log-transform
    if task_id in (361082, 361088, 361099):
        y = np.log(y)

    # UMAP global split
    umap = UMAP(n_components=2, random_state=10)
    X_umap = umap.fit_transform(X_clean)
    dist = pd.Series(euclidean_distances(X_umap).mean(axis=1), index=X_clean.index)
    thr = dist.quantile(0.8)
    close_idx = dist[dist < thr].index
    far_idx   = dist[dist >= thr].index

    # UMAP local split on train candidates
    X_train_clean = X_clean.loc[close_idx]
    X_train_full  = X.loc[close_idx]
    X_test_all    = X.loc[far_idx]
    y_train_full  = y.loc[close_idx]
    y_test_all    = y.loc[far_idx]
    X_umap2 = umap.fit_transform(X_train_clean)
    dist2 = pd.Series(euclidean_distances(X_umap2).mean(axis=1), index=X_train_clean.index)
    thr2 = dist2.quantile(0.8)
    close2 = dist2[dist2 < thr2].index
    far2   = dist2[dist2 >= thr2].index

    # drop inconsistent categorical
    X_prune = X.copy()
    dummy_cols = X_prune.select_dtypes(include=['bool','category','object','string']).columns
    for col in dummy_cols:
        if X_prune[col].nunique() != X_prune.loc[close2, col].nunique():
            X_prune = X_prune.drop(col, axis=1)
    non_dummy = X_prune.select_dtypes(exclude=['bool','category','object','string']).columns

    # one-hot full dataset
    X_dum = pd.get_dummies(X_prune, drop_first=True).astype('float32')
    X_train_all = X_dum.loc[close_idx]
    X_test_all  = X_dum.loc[far_idx]
    X_train_    = X_dum.loc[close2]
    X_val       = X_dum.loc[far2]
    y_train_    = y.loc[close2].to_numpy().ravel()
    y_val       = y.loc[far2].   to_numpy().ravel()

    # standardize nested
    scaler = StandardScaler()
    X_train_[non_dummy] = scaler.fit_transform(X_train_[non_dummy])
    X_val  [non_dummy] = scaler.transform(X_val[non_dummy])

    # standardize outer
    scaler2 = StandardScaler()
    X_train_all[non_dummy] = scaler2.fit_transform(X_train_all[non_dummy])
    X_test_all [non_dummy] = scaler2.transform(X_test_all[non_dummy])

    # hyperparameter tuning with Optuna
    N_TRIALS = 100

    def boosted(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.5, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 1, 30),
            'num_leaves': 2**10,
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100)
        }
        model = lgbm.LGBMRegressor(**params)
        model.fit(X_train_, y_train_)
        y_pred = model.predict(X_val)
        return np.sqrt(np.mean((y_val - y_pred) ** 2))

    study_boost = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
    study_boost.optimize(boosted, n_trials=N_TRIALS)
    best_boost = LGBMRegressor(**study_boost.best_params, num_leaves=2**10)

    def rf(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 1, 30),
            'max_features': trial.suggest_float('max_features', 0.0, 1.0),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100)
        }
        model = RandomForestRegressor(**params)
        model.fit(X_train_, y_train_)
        y_pred = model.predict(X_val)
        return np.sqrt(np.mean((y_val - y_pred) ** 2))

    study_rf = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
    study_rf.optimize(rf, n_trials=N_TRIALS)
    best_rf = RandomForestRegressor(**study_rf.best_params)

    # retrain on full train and evaluate on test
    best_boost.fit(X_train_all, y_train_full.to_numpy().ravel())
    y_pred_boost = best_boost.predict(X_test_all)
    rmse_boost = np.sqrt(mean_squared_error(y_test_all, y_pred_boost))

    best_rf.fit(X_train_all, y_train_full.to_numpy().ravel())
    y_pred_rf = best_rf.predict(X_test_all)
    rmse_rf = np.sqrt(mean_squared_error(y_test_all, y_pred_rf))

    # save results
    os.makedirs(result_folder, exist_ok=True)
    df = pd.DataFrame([{
        'suite_id': suite_id,
        'task_id': task_id,
        'model': 'LGBM',
        'RMSE': rmse_boost
    },{
        'suite_id': suite_id,
        'task_id': task_id,
        'model': 'RF',
        'RMSE': rmse_rf
    }])
    df.to_csv(os.path.join(result_folder, f"{suite_id}_{task_id}_umap_tuned.csv"), index=False)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--suite_id', type=int, required=True)
    p.add_argument('--task_id', type=int, required=True)
    p.add_argument('--result_folder', type=str, required=True)
    p.add_argument('--seed', type=int, default=10)
    args = p.parse_args()
    run_task_umap(args.suite_id, args.task_id, args.result_folder, args.seed)
