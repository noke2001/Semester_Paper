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
from sklearn.preprocessing import LabelEncoder, StandardScaler

'''
This scripts is reproducing entirely umap_decomposition_335_rmse. It is using the sklearn models.

'''

seed = 10 

def run_task_umap(suite_id: int, task_id: int, result_folder: str, seed: int = 10):
    # reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    X, y, cat_ind, attr_names = load_dataset_offline(suite_id, task_id)



    for col in [attribute for attribute, indicator in zip(attr_names, cat_ind) if indicator]:
        if len(X[col].unique()) > 20:
            X = X.drop(col, axis=1)

    X_clean=X.copy()
    for col in [attribute for attribute, indicator in zip(attr_names, cat_ind) if not indicator]:
        if len(X[col].unique()) < 10:
            X = X.drop(col, axis=1)
            X_clean = X_clean.drop(col, axis=1)
        elif X[col].value_counts(normalize=True).max() > 0.7:
            X_clean = X_clean.drop(col, axis=1)

    # Find features with absolute correlation > 0.9
    corr_matrix = X_clean.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]

    # Drop one of the highly correlated features from X_clean
    X_clean = X_clean.drop(high_corr_features, axis=1)

    # Rename columns to avoid problems with LGBM
    X = X.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))


    # Transform y to int type, to then be able to apply BCEWithLogitsLoss
    # Create a label encoder
    le = LabelEncoder()
    # Fit the label encoder and transform y to get binary labels
    y_encoded = le.fit_transform(y)
    # Convert the result back to a pandas Series
    y = pd.Series(y_encoded, index=y.index)

    # Apply UMAP decomposition
    umap = UMAP(n_components=2, random_state=42)
    X_umap = umap.fit_transform(X_clean)

    # calculate the Euclidean distance matrix
    euclidean_dist_matrix = euclidean_distances(X_umap)

    # calculate the Euclidean distance for each data point
    euclidean_dist = np.mean(euclidean_dist_matrix, axis=1)

    euclidean_dist = pd.Series(euclidean_dist, index=X_clean.index)
    far_index = euclidean_dist.index[np.where(euclidean_dist >= np.quantile(euclidean_dist, 0.8))[0]]
    close_index = euclidean_dist.index[np.where(euclidean_dist < np.quantile(euclidean_dist, 0.8))[0]]

    X_train_clean = X_clean.loc[close_index,:]
    X_train = X.loc[close_index,:]
    X_test = X.loc[far_index,:]
    y_train = y.loc[close_index]
    y_test = y.loc[far_index]

    # Apply UMAP decomposition on the training set
    X_umap_train = umap.fit_transform(X_train_clean)

    # calculate the Euclidean distance matrix for the training set
    euclidean_dist_matrix_train = euclidean_distances(X_umap_train)

    # calculate the Euclidean distance for each data point in the training set
    euclidean_dist_train = np.mean(euclidean_dist_matrix_train, axis=1)

    euclidean_dist_train = pd.Series(euclidean_dist_train, index=X_train_clean.index)
    far_index_train = euclidean_dist_train.index[np.where(euclidean_dist_train >= np.quantile(euclidean_dist_train, 0.8))[0]]
    close_index_train = euclidean_dist_train.index[np.where(euclidean_dist_train < np.quantile(euclidean_dist_train, 0.8))[0]]

    # Check if categorical variables have the same cardinality in X and X_train_, and remove the ones that don't
    dummy_cols = X.select_dtypes(['bool', 'category', 'object', 'string']).columns
    X_train = X.loc[close_index,:]
    X_train_ = X_train.loc[close_index_train,:]
    for col in dummy_cols:
        if len(X[col].unique()) != len(X_train_[col].unique()):
            X = X.drop(col, axis=1)
            
    # Convert data to PyTorch tensors
    # Modify X_train_, X_val, X_train, and X_test to have dummy variables
    non_dummy_cols = X.select_dtypes(exclude=['bool', 'category', 'object', 'string']).columns
    X = pd.get_dummies(X, drop_first=True).astype('float32')

    X_train = X.loc[close_index,:]
    X_test = X.loc[far_index,:]
    y_train = y.loc[close_index]
    y_test = y.loc[far_index]

    X_train_ = X_train.loc[close_index_train,:]
    X_val = X_train.loc[far_index_train,:]
    y_train_ = y_train.loc[close_index_train]
    y_val = y_train.loc[far_index_train]

    # Standardize the data for non-dummy variables
    mean_X_train_ = np.mean(X_train_[non_dummy_cols], axis=0)
    std_X_train_ = np.std(X_train_[non_dummy_cols], axis=0)
    X_train_[non_dummy_cols] = (X_train_[non_dummy_cols] - mean_X_train_) / std_X_train_
    X_val = X_val.copy()
    X_val[non_dummy_cols] = (X_val[non_dummy_cols] - mean_X_train_) / std_X_train_

    mean_X_train = np.mean(X_train[non_dummy_cols], axis=0)
    std_X_train = np.std(X_train[non_dummy_cols], axis=0)
    X_train[non_dummy_cols] = (X_train[non_dummy_cols] - mean_X_train) / std_X_train
    X_test = X_test.copy()
    X_test[non_dummy_cols] = (X_test[non_dummy_cols] - mean_X_train) / std_X_train

    # hyperparameter tuning with Optuna
    N_TRIALS = 100
    seed = 10

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
    best_boost = lgbm.LGBMRegressor(**study_boost.best_params, num_leaves=2**10)

    def rf(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 1, 30),
            'max_features': trial.suggest_float('max_features', 0.0, 1.0),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100)
        }
        model = _RFRegressor(**params)
        model.fit(X_train_, y_train_)
        y_pred = model.predict(X_val)
        return np.sqrt(np.mean((y_val - y_pred) ** 2))

    study_rf = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed))
    study_rf.optimize(rf, n_trials=N_TRIALS)
    best_rf = _RFRegressor(**study_rf.best_params)

    # retrain on full train and evaluate on test
    best_boost.fit(X_train, y_train.to_numpy().ravel())
    y_pred_boost = best_boost.predict(X_test)
    rmse_boost = np.sqrt(mean_squared_error(y_test, y_pred_boost))

    best_rf.fit(X_train, y_train.to_numpy().ravel())
    y_pred_rf = best_rf.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

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
    df.to_csv(os.path.join(result_folder, f"{suite_id}_{task_id}_umap_pure_copy.csv"), index=False)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--suite_id', type=int, required=True)
    p.add_argument('--task_id', type=int, required=True)
    p.add_argument('--result_folder', type=str, required=True)
    p.add_argument('--seed', type=int, default=10)
    args = p.parse_args()
    run_task_umap(args.suite_id, args.task_id, args.result_folder, args.seed)
