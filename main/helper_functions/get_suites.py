import openml
import pandas as pd

def get_suite_tasks(suite_id):
    print(f"Fetching details for Suite {suite_id}...")
    try:
        # Fetch the suite object from OpenML
        suite = openml.study.get_suite(suite_id)
        
        # Get the list of task IDs included in this suite
        task_ids = suite.tasks
        
        print(f"Found {len(task_ids)} tasks in Suite {suite_id}.\n")
        
        task_data = []
        
        # Iterate through tasks to get friendly dataset names
        # We process in batches or just list them to be fast
        for t_id in task_ids:
            try:
                # We can get basic task info without downloading the full dataset
                task = openml.tasks.get_task(t_id, download_data=False)
                dataset = openml.datasets.get_dataset(task.dataset_id, download_data=False)
                
                task_data.append({
                    "Suite ID": suite_id,
                    "Task ID": t_id,
                    "Dataset ID": task.dataset_id,
                    "Dataset Name": dataset.name,
                    "Target Feature": task.target_name,
                    "Type": str(task.task_type)
                })
                print(f" - Task {t_id}: {dataset.name}")
            except Exception as e:
                print(f" - Task {t_id}: Error fetching details ({e})")
                
        return pd.DataFrame(task_data)
        
    except Exception as e:
        print(f"Error fetching suite {suite_id}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Define the suites we want to look up
    # 336: Regression (Numerical)
    # 337: Classification (Numerical)
    # 335: Regression (Numerical + Categorical)
    # 334: Classification (Numerical + Categorical)
    # 379: Additional Benchmark (likely Hard or TabZilla subset)
    suites_to_fetch = [334, 335, 336, 337, 379]
    
    all_results = []
    
    for s_id in suites_to_fetch:
        df = get_suite_tasks(s_id)
        all_results.append(df)
        print("-" * 50)

    if all_results:
        final_df = pd.concat(all_results)
        
        # Save to CSV for easy reference
        final_df.to_csv("openml_suite_tasks.csv", index=False)
        print("\nSuccessfully saved task list to 'openml_suite_tasks.csv'")
        
        # Print a clean summary table
        print("\n=== Summary Table ===")
        print(final_df[['Suite ID', 'Task ID', 'Dataset Name']].to_string(index=False))