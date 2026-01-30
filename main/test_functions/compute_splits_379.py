import pickle
from src.loader import load_dataset_offline, clean_data
from src.extrapolation_methods import (random_split, gower_split, kmedoids_split,
                                       umap_split, mahalanobis_split, spatial_depth_split)

X_full, y_full, cat_ind, attr_names = load_dataset_offline(379, 168337)
_, X_clean, y_clean = clean_data(X_full, y_full, cat_ind, attr_names, task_type="auto")

splits = {}
for fn in [random_split, gower_split, kmedoids_split,
           umap_split, mahalanobis_split, spatial_depth_split]:
    # Most split_fns return either (train, test) or 6-tuple
    out = fn(X_clean, y_clean) if fn is random_split else fn(X_clean)
    # normalize to (train_idx, test_idx)
    if len(out) == 6:
        train_idx = out[0]
        test_idx  = out[5]
    else:
        train_idx, test_idx = out
    splits[fn.__name__] = (train_idx, test_idx)
    print(f"Saved split {fn.__name__}")

with open("splits_379_168337.pkl", "wb") as f:
    pickle.dump(splits, f)
