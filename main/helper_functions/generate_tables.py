import pandas as pd
import glob
import numpy as np
import os

def generate_tables_fixed():
    # --- Configuration ---
    BASE_DIR = "/cluster/home/pdamota/"
    METADATA_FILE = os.path.join(BASE_DIR, "openml_suite_tasks.csv")
    RESULTS_DIR = os.path.join(BASE_DIR, "semester_project_smac/master_thesis-main/master_thesis-main/smac_results/seed_10/")
    OUTPUT_DIR = os.path.join(BASE_DIR, "semester_project_smac/tables/")

    COLUMN_ORDER = [
        "Const.", "Engression", "FT-Trans.", "GP", "GBT", 
        "Log. Regr.", "MLP", "RF", "ResNet", "TabPFN"
    ]

    NAME_MAPPING = {
        "Constant": "Const.", "DummyClassifier": "Const.", "DummyRegressor": "Const.",
        "Engression": "Engression",
        "FTTransformer": "FT-Trans.",
        "GaussianProcess": "GP", "GaussianProcessClassifier": "GP", "GaussianProcessRegressor": "GP",
        "GradientBoosting": "GBT", "GradientBoostingClassifier": "GBT", "GradientBoostingRegressor": "GBT",
        "LGBMClassifier": "GBT", "LGBMRegressor": "GBT",
        "LogisticRegression": "Log. Regr.", "LinearRegression": "Log. Regr.",
        "MLP": "MLP", "MLPClassifier": "MLP", "MLPRegressor": "MLP",
        "RandomForest": "RF", "RandomForestClassifier": "RF", "RandomForestRegressor": "RF",
        "ResNet": "ResNet",
        "TabPFN": "TabPFN", "TabPFNClassifier": "TabPFN", "TabPFNRegressor": "TabPFN"
    }

    METRIC_CONFIG = {
        "Accuracy": {"lower_is_better": False, "label": "Avg. acc."},
        "LogLoss":  {"lower_is_better": True,  "label": "Avg. logloss"},
        "root_mean_squared_error": {"lower_is_better": True, "label": "Avg. RMSE"},
        "CRPS": {"lower_is_better": True, "label": "Avg. CRPS"}
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load Metadata ---
    print(f"Loading metadata from {METADATA_FILE}...")
    try:
        meta_df = pd.read_csv(METADATA_FILE)
        meta_df.columns = [c.strip() for c in meta_df.columns]
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return

    suites = meta_df['Suite ID'].unique()

    for suite_id in suites:
        # Determine Target Metrics
        suite_meta = meta_df[meta_df['Suite ID'] == suite_id]
        
        target_metrics = []
        if suite_meta['Type'].str.contains("Classification").any():
            target_metrics = ["Accuracy", "LogLoss"]
        else:
            target_metrics = ["root_mean_squared_error", "CRPS"]

        print(f"\n--- Processing Suite {suite_id} ---")
        
        # Load Files
        search_pattern = os.path.join(RESULTS_DIR, f"{suite_id}_*.csv")
        files = glob.glob(search_pattern)
        
        if not files:
            print(f"    No files found.")
            continue

        # Loop Metrics
        for current_metric in target_metrics:
            print(f"    > Generating tables for {current_metric}...")
            
            dfs = []
            for f in files:
                try:
                    df = pd.read_csv(f)
                    
                    # --- Map 'value' column if missing ---
                    if 'value' not in df.columns:
                        if current_metric == "Accuracy" and 'test_acc' in df.columns:
                            df['value'] = df['test_acc']
                            df['metric'] = 'Accuracy'
                        elif current_metric == "LogLoss" and 'test_logloss' in df.columns:
                            df['value'] = df['test_logloss']
                            df['metric'] = 'LogLoss'
                        elif current_metric == "root_mean_squared_error" and 'test_rmse' in df.columns:
                            df['value'] = df['test_rmse']
                            df['metric'] = 'root_mean_squared_error'
                        elif current_metric == "CRPS" and 'test_crps' in df.columns:
                             df['value'] = df['test_crps']
                             df['metric'] = 'CRPS'

                    # --- Filter Rows ---
                    if 'suite_id' in df.columns:
                        df = df[df['suite_id'] == suite_id]
                    
                    # --- Filter by Metric ---
                    if 'metric' in df.columns:
                        # Normalize RMSE variations
                        if current_metric == "root_mean_squared_error":
                            # Normalize widely used variations to standard name
                            if "RMSE" in df['metric'].values:
                                df.loc[df['metric'] == "RMSE", 'metric'] = "root_mean_squared_error"
                            if "test_rmse" in df['metric'].values:
                                df.loc[df['metric'] == "test_rmse", 'metric'] = "root_mean_squared_error"
                        
                        df = df[df['metric'] == current_metric]

                    if not df.empty and 'value' in df.columns:
                        keep_cols = ['task_id', 'split_method', 'model', 'value']
                        dfs.append(df[keep_cols])

                except Exception:
                    continue

            if not dfs:
                print(f"      No data found for {current_metric}.")
                continue

            full_df = pd.concat(dfs, ignore_index=True)
            full_df['model'] = full_df['model'].map(NAME_MAPPING).fillna(full_df['model'])

            # --- Check found splits ---
            unique_splits = full_df['split_method'].unique()
            print(f"      Found splits: {unique_splits}") # Debug print
            
            for split_method in unique_splits:
                sub_df = full_df[full_df['split_method'] == split_method]
                
                # Pivot
                pivot = sub_df.pivot_table(
                    index="task_id", 
                    columns="model", 
                    values="value", 
                    aggfunc="first"
                )
                
                # Column Cleanup
                valid_cols = [c for c in COLUMN_ORDER if c in pivot.columns]
                pivot = pivot[valid_cols]
                pivot = pivot.dropna(axis=1, how='all')

                if pivot.empty:
                    print(f"      Skipping {split_method} (Empty table)")
                    continue

                # --- Stats ---
                config = METRIC_CONFIG.get(current_metric, {"lower_is_better": True, "label": f"Avg. {current_metric}"})
                lower_is_better = config["lower_is_better"]
                avg_label = config["label"]

                if lower_is_better:
                    best_per_task = pivot.min(axis=1)
                    diff_func = lambda col: (col - best_per_task)
                    rank_asc = True
                else:
                    best_per_task = pivot.max(axis=1)
                    diff_func = lambda col: (best_per_task - col)
                    rank_asc = False

                avg_diff = pivot.apply(lambda col: (100 * diff_func(col)).mean())
                avg_rank = pivot.rank(axis=1, ascending=rank_asc).mean()
                avg_val = pivot.mean()

                summary_df = pd.DataFrame({
                    "Avg. diff.": avg_diff,
                    avg_label: avg_val,
                    "Avg. rank.": avg_rank
                }).T
                
                final_df = pd.concat([pivot, summary_df])
                final_df.reset_index(inplace=True)
                final_df.columns.values[0] = "task\_id"

                safe_metric = current_metric.replace("root_mean_squared_error", "RMSE")
                out_filename = f"{suite_id}_{split_method}_{safe_metric}.tex"
                out_path = os.path.join(OUTPUT_DIR, out_filename)
                
                with open(out_path, "w") as f:
                    f.write(final_df.to_latex(index=False, float_format="%.3f", na_rep="NaN"))
                
                print(f"      Saved {out_filename}")

if __name__ == "__main__":
    generate_tables_fixed()