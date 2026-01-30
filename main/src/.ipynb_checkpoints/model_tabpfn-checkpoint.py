import os
import pandas as pd
import numpy as np
import openml
import argparse

parser = argparse.ArgumentParser(description="TabPFN Experiment")
parser.add_argument("--suite", type=int, required=True,
                    help="Suite ID (e.g., 337 for classification on numerical features)")
parser.add_argument("--task", type=int, required=True,
                    help="Task ID to evaluate")
args = parser.parse_args()

print(f"Suite ID: {args.suite}")
print(f"Task ID: {args.task}")


from src.loader import load_dataset, clean_data, standardize_data
from src.extrapolation_methods import (
    random_split,
    mahalanobis_split,
    umap_split,
    kmeans_split,
    gower_split,
    kmedoids_split 
)
from src.evaluation_metrics import evaluate_accuracy, evaluate_log_loss, evaluate_rmse, evaluate_crps

from tabpfn import TabPFNClassifier, TabPFNRegressor


SUITE_CONFIG = {
    "regression_numerical": { "suite_id": 336, "tasks": [361088] },
    "classification_numerical": { "suite_id": 337, "tasks": [361273] },
    "regression_numerical_categorical": { "suite_id": 335, "tasks": [361287] },
    "classification_numerical_categorical": { "suite_id": 334, "tasks": [361110] }
}


EXTRAPOLATION_METHODS = {
    "numerical": [mahalanobis_split, kmeans_split, umap_split],
    "numerical_categorical": [gower_split, kmedoids_split, umap_split],
}


def get_extrapolation_fn(suite_key: str):
    if suite_key in ["regression_numerical", "classification_numerical"]:
        return EXTRAPOLATION_METHODS["numerical"]
    else:
        return EXTRAPOLATION_METHODS["numerical_categorical"]


def experiment(suite_id: int, task_id: int):
    
    config = SUITE_CONFIG[suite_id]
    task_type = config["task_type"]  # 'classification' or 'regression'
    data_type = config["data_type"]  # 'numerical' or 'numerical_categorical'

    print(f"Running experiment on task {task_id} from suite {suite_id}...")
    # Load the dataset using your loader.py functions.
    X, y, cat_indicator, attr_names = load_dataset(task_id)
    X, X_clean, y = clean_data(X, y, cat_indicator, attr_names)

    # Choose the extrapolation method according to the data type.
    split_fn = get_extrapolation_fn(data_type)
    print(f"Using extrapolation splitting method: {split_fn.__name__}")
    close_idx, far_idx = split_fn(X_clean)

    # Use the indices on the original data (with proper column names)
    X_train = X.loc[close_idx]
    y_train = y.loc[close_idx]
    X_test = X.loc[far_idx]
    y_test = y.loc[far_idx]

    # Standardize the data
    X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)

    # Instantiate the appropriate TabPFN model based on task type.
    if task_type == "classification":
        model = TabPFNClassifier(random_state=42, ignore_pretraining_limits=True)
    else:
        model = TabPFNRegressor(random_state=42)

    print("Training TabPFN model...")
    model.fit(X_train_scaled, y_train)

    print("Generating predictions on extrapolation set...")
    if task_type == "classification":
        y_pred_proba = model.predict_proba(X_test_scaled)
        # For binary classification, threshold probability at 0.5
        if len(np.unique(y_train)) == 2:
            y_pred = (y_pred_proba[:, 1] >= 0.5).astype(int)
        else:
            y_pred = np.argmax(y_pred_proba, axis=1)
        acc = evaluate_accuracy(np.array(y_test), y_pred)
        ll = evaluate_log_loss(np.array(y_test), y_pred_proba)
        print("TabPFN Classification Results:")
        print(f"Accuracy: {acc}")
        print(f"Log Loss: {ll}")
    else:
        # Regression prediction is assumed to be a continuous output.
        y_pred = model.predict(X_test_scaled)
        rmse = evaluate_rmse(np.array(y_test), y_pred)
        print("TabPFN Regression Results:")
        print(f"RMSE: {rmse}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TabPFN Experiment")
    parser.add_argument("--suite", type=int, required=True, help="Suite ID (e.g., 337 for Classification on numerical features)")
    parser.add_argument("--task", type=int, required=True, help="Task ID to evaluate")
    args = parser.parse_args()
    experiment(args.suite, args.task)

