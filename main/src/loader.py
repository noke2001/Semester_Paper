import numpy as np
import pandas as pd
import re
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

# ==========================================
# --- CONFIGURATION ------------------------
# ==========================================
# Change this value to control the default loading limit.
# Set to None to load all data by default.
DEFAULT_MAX_SAMPLES = 50000 
# ==========================================

def load_dataset_offline(suite_id, task_id, max_samples=DEFAULT_MAX_SAMPLES, seed=42):
    base_dir  = os.path.join(os.path.dirname(__file__), "..", "original_data")
    subfolder = f"{suite_id}_{task_id}"
    X_path    = os.path.join(base_dir, subfolder, f"{suite_id}_{task_id}_X.csv")
    y_path    = os.path.join(base_dir, subfolder, f"{suite_id}_{task_id}_y.csv")
    cat_path  = os.path.join(base_dir, subfolder, f"{suite_id}_{task_id}_categorical_indicator.npy")

    # Optimization: If a specific task is huge and a limit is set, read only necessary rows
    # (This prevents loading 10GB into RAM just to throw away 90% of it)
    if (suite_id, task_id) == (379, 168337) and max_samples is not None:
        X = pd.read_csv(X_path, nrows=max_samples)
        y = pd.read_csv(y_path, nrows=max_samples)
    else:
        X = pd.read_csv(X_path)
        y = pd.read_csv(y_path)

    categorical_indicator = np.load(cat_path)
    attribute_names       = X.columns.tolist()

    # Apply the limit (if max_samples is not None)
    if max_samples is not None and len(X) > max_samples:
        print(f"[Loader] Subsampling from {len(X)} to {max_samples} (defined in loader defaults)...", flush=True)
        idx = np.random.RandomState(seed).choice(len(X), size=max_samples, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)

    return X, y, categorical_indicator, attribute_names

def load_dataset_offline_extra(suite_id, task_id, max_samples=DEFAULT_MAX_SAMPLES, seed=42):
    """
    Extra loader with reservoir sampling for massive datasets.
    """
    base_dir  = os.path.join(os.path.dirname(__file__), "..", "original_data")
    subfolder = f"{suite_id}_{task_id}"
    X_path    = os.path.join(base_dir, subfolder, f"{suite_id}_{task_id}_X.csv")
    y_path    = os.path.join(base_dir, subfolder, f"{suite_id}_{task_id}_y.csv")
    cat_path  = os.path.join(base_dir, subfolder, f"{suite_id}_{task_id}_categorical_indicator.npy")

    # Reservoir sampling logic only triggers if max_samples is strictly set
    if (suite_id, task_id) == (379, 168337) and max_samples is not None:
        print(f"⚠️  Reservoir‐sampling {max_samples} rows from {suite_id}/{task_id}", flush=True)
        rng = np.random.RandomState(seed)
        reservoir_X = []
        reservoir_y = []
        total = 0

        for chunk_X, chunk_y in zip(
            pd.read_csv(X_path, chunksize=50_000, low_memory=False),
            pd.read_csv(y_path, chunksize=50_000, low_memory=False)
        ):
            y_vals = chunk_y.iloc[:, 0].to_numpy()
            for i, row in enumerate(chunk_X.itertuples(index=False, name=None)):
                if total < max_samples:
                    reservoir_X.append(row)
                    reservoir_y.append(y_vals[i])
                else:
                    j = rng.randint(0, total + 1)
                    if j < max_samples:
                        reservoir_X[j] = row
                        reservoir_y[j] = y_vals[i]
                total += 1

        cols = pd.read_csv(X_path, nrows=0).columns.tolist()
        X = pd.DataFrame(reservoir_X, columns=cols)
        y = pd.Series(reservoir_y, name=cols[0])
    else:
        # Standard load
        X = pd.read_csv(X_path)
        y = pd.read_csv(y_path)

    categorical_indicator = np.load(cat_path)
    attribute_names       = X.columns.tolist()

    if max_samples is not None and len(X) > max_samples:
        idx = np.random.RandomState(seed).choice(len(X), size=max_samples, replace=False)
        X = X.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)

    return X, y, categorical_indicator, attribute_names


def clean_data(X, y, categorical_indicator, attribute_names, task_type=None):
    # (Unchanged)
    for col, indicator in zip(attribute_names, categorical_indicator):
        if indicator and X[col].nunique() > 20:
            X = X.drop(col, axis=1)
    X_clean = X.copy()
    for col, indicator in zip(attribute_names, categorical_indicator):
        if not indicator:
            if X[col].nunique() < 10:
                X = X.drop(col, axis=1)
                X_clean = X_clean.drop(col, axis=1)
            elif X[col].value_counts(normalize=True).max() > 0.7:
                X_clean = X_clean.drop(col, axis=1)

    corr_matrix = X_clean.select_dtypes(include=[np.number]).corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_features = [col for col in upper_tri.columns if any(upper_tri[col] > 0.9)]
    X_clean = X_clean.drop(high_corr_features, axis=1)

    const_cols = [c for c in X_clean.columns if X_clean[c].nunique() <= 1]
    logging.info(f"Dropping {len(const_cols)} constant cols: {const_cols}")
    X_clean = X_clean.drop(const_cols, axis=1)

    X = X.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    
    if task_type == "classification":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y).ravel(), index=y.index)

    return X, X_clean, y

def standardize_data(X_train: pd.DataFrame, X_test: pd.DataFrame, non_dummy_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    # (Unchanged)
    X_tr = X_train.copy()
    X_te = X_test.copy()

    mean = X_tr[non_dummy_cols].mean(axis=0)
    std  = X_tr[non_dummy_cols].std(axis=0, ddof=0)

    non_zero_std_cols = std[std > 1e-9].index
    zero_std_cols     = std[std <= 1e-9].index

    if not zero_std_cols.empty:
        print(f"INFO (standardize_data): Found and handling {len(zero_std_cols)} constant-value columns: {list(zero_std_cols)}")

    X_tr[non_zero_std_cols] = (X_tr[non_zero_std_cols] - mean[non_zero_std_cols]) / std[non_zero_std_cols]
    X_te[non_zero_std_cols] = (X_te[non_zero_std_cols] - mean[non_zero_std_cols]) / std[non_zero_std_cols]

    if not zero_std_cols.empty:
        X_tr[zero_std_cols] = X_tr[zero_std_cols] - mean[zero_std_cols]
        X_te[zero_std_cols] = X_te[zero_std_cols] - mean[zero_std_cols]

    return X_tr, X_te

def prepare_for_split(df: pd.DataFrame) -> pd.DataFrame:
    # (Unchanged)
    X0 = df.copy()
    cat_cols = df.select_dtypes(include=['category', 'object', 'bool', 'string']).columns
    if not cat_cols.empty:
        X0[cat_cols] = X0[cat_cols].apply(lambda col: col.astype('category').cat.codes)
    X0 = X0.astype('float64')
    return X0.fillna(0.0)

def preprocess_data(X_train, X_test, X_clean=None):
    # (Unchanged)
    if X_clean is not None:
        cat_cols = X_clean.select_dtypes(include=['object','category','bool', 'string']).columns
        for col in cat_cols:
            if not set(X_train[col].unique()) >= set(X_clean[col].unique()):
                X_train = X_train.drop(col, axis=1)
                X_test  = X_test.drop(col, axis=1)
    return X_train, X_test