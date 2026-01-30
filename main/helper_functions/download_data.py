import os
import argparse
import openml
import numpy as np
import pandas as pd

# 1. Force OpenML to use the cache dir from the environment
cache_dir = os.environ.get("XDG_CACHE_HOME", "/cluster/home/pdamota/openml_cache")
openml.config.cache_directory = cache_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, required=True, help="Task ID to process")
    parser.add_argument("--suite_id", type=int, default=336, help="Suite ID")
    args = parser.parse_args()

    suite_id = args.suite_id
    task_id = args.task_id

    print(f"--- Processing ONLY Task {task_id} (Suite {suite_id}) ---")
    
    # [FIX] Get the absolute path of the directory containing THIS script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # [FIX] Force output to be inside the repository folder
    output_path = os.path.join(script_dir, "original_data", f"{suite_id}_{task_id}")
    
    print(f"   Target Output Folder: {output_path}")
    os.makedirs(output_path, exist_ok=True)

    try:
        # 4. Get Task Metadata
        task = openml.tasks.get_task(task_id, download_data=False)
        dataset = task.get_dataset()

        print(f"   Dataset Name: {dataset.name}")
        print("   Loading data from cache...")
        
        # 5. Load Data
        X, y, categorical_indicator, _ = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )

        print("   Saving to CSV...")
        X.to_csv(os.path.join(output_path, f"{suite_id}_{task_id}_X.csv"), index=False)
        y.to_csv(os.path.join(output_path, f"{suite_id}_{task_id}_y.csv"), index=False)
        np.save(os.path.join(output_path, f"{suite_id}_{task_id}_categorical_indicator.npy"), categorical_indicator)
        
        print(f"✅ Done. Files saved to: {output_path}")

    except Exception as e:
        print(f"❌ Error: {e}")
        raise e

if __name__ == '__main__':
    main()