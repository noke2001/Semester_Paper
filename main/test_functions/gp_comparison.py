import pandas as pd
from pathlib import Path

# --- Configuration ---

# 1. Define the main folders for each model configuration.
CONFIG_FOLDERS = [
    'gp_fitc_results',
    'gp_full_scale_results',
    'gp_vecchia_results'
]

# 2. Define metrics optimization direction.
METRICS_OPTIMIZATION_DIRECTION = {
    'LogLoss': 'lower', 'RMSE': 'lower', 'CRPS': 'lower', 'Accuracy': 'higher'
}

# --- Main Script ---

def analyze_final_structure():
    """
    Loads data from the final, specified folder structure:
    e.g., 'gp_fitc_results/seed_10/data.csv'
    """
    all_data = []

    print("🔎 Starting analysis...")
    # Loop through each configuration folder
    for config_name in CONFIG_FOLDERS:
        # Create a path to the specific 'seed_10' subfolder
        seed_path = Path(config_name) / 'seed_10'

        if not seed_path.is_dir():
            print(f"⚠️ Warning: Directory '{seed_path}' not found. Skipping.")
            continue

        # Use .rglob('*.csv') to find all CSV files within the seed folder
        csv_files = list(seed_path.rglob('*.csv'))
        if not csv_files:
            print(f"Info: No CSV files found in '{seed_path}'.")
            continue
            
        print(f"   - Loading {len(csv_files)} files from '{seed_path}'...")

        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                # Assign the configuration name from the top-level folder
                df['configuration'] = config_name
                all_data.append(df)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    if not all_data:
        print("\n❌ No data was loaded. Please check your folder names and file locations.")
        return

    # --- Analysis (This part remains the same) ---
    master_df = pd.concat(all_data, ignore_index=True)
    print(f"\n✅ Successfully combined data from {len(master_df)} experiments.")

    summary = master_df.groupby(['configuration', 'metric'])['value'].agg(['mean', 'std', 'count']).reset_index()

    print("\n--- 🏆 Performance Summary & Rankings ---")
    unique_metrics = summary['metric'].unique()

    for metric in unique_metrics:
        direction = METRICS_OPTIMIZATION_DIRECTION.get(metric)
        if not direction:
            print(f"\n⚠️ Metric '{metric}' not defined. Skipping ranking.")
            continue

        is_ascending = (direction == 'lower')
        metric_ranking = summary[summary['metric'] == metric].sort_values(
            by='mean', ascending=is_ascending
        ).reset_index(drop=True)

        print(f"\n--- Results for {metric} (best is {direction}) ---")
        print(metric_ranking.to_string())


# Run the analysis
if __name__ == "__main__":
    analyze_final_structure()