import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def generate_analysis_plots(input_filepath: str, output_dir: str):
    """
    Loads consolidated results, correctly calculates relative differences
    on a per-task basis, and generates a separate summary plot for each
    split method and metric.
    """
    try:
        df = pd.read_csv(input_filepath)
        print(f"Successfully loaded {input_filepath} with {len(df)} records.")
    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{input_filepath}'")
        return

    metric_configs = {
        'RMSE': {'lower_is_better': True, 'y_label': 'Avg. Relative Difference to Best (RMSE) (%)'},
        'Accuracy': {'lower_is_better': False, 'y_label': 'Avg. Relative Difference from Best (Accuracy) (%)'},
        'LogLoss': {'lower_is_better': True, 'y_label': 'Avg. Relative Difference to Best (LogLoss) (%)'},
        'CRPS': {'lower_is_better': True, 'y_label': 'Avg. Relative Difference to Best (CRPS) (%)'}
    }
    
    split_methods = df['split_method'].unique()
    os.makedirs(output_dir, exist_ok=True)

    for split_method in split_methods:
        for metric, config in metric_configs.items():
            
            analysis_df = df[(df['split_method'] == split_method) & (df['metric'] == metric)].copy()

            if analysis_df.empty:
                continue

            print(f"\nProcessing: '{split_method}' with metric '{metric}'...")

            # --- START OF THE CRITICAL FIX ---
            # Correctly calculate relative difference on a per-task, per-split basis.
            
            # This will store the properly calculated differences
            processed_groups = []
            
            # Group by each individual task
            for task_id, group in analysis_df.groupby(['task_id']):
                
                # Find the best score *within this task's group*
                if config['lower_is_better']:
                    best_score = group['value'].min()
                else:
                    best_score = group['value'].max()

                # Avoid division by zero if the best score is 0
                if best_score == 0:
                    # Assign 0 difference if value is also 0, otherwise 1 (100% diff)
                    group['relative_diff'] = np.where(group['value'] == 0, 0, 1)
                else:
                    if config['lower_is_better']:
                        group['relative_diff'] = (group['value'] - best_score) / best_score
                    else:
                        group['relative_diff'] = (best_score - group['value']) / best_score
                
                processed_groups.append(group)
            
            # Combine the processed groups back into a single dataframe
            if not processed_groups:
                print("No data to process after grouping. Skipping plot.")
                continue
                
            analysis_df = pd.concat(processed_groups)
            # --- END OF THE CRITICAL FIX ---


            # Aggregate results for plotting
            summary_df = analysis_df.groupby('model')['relative_diff'].mean().dropna() * 100
            summary_df = summary_df.reset_index(name='mean_relative_diff_pct')

            if summary_df.empty:
                print("WARNING: Summary DataFrame is empty. Plot will be empty.")
                continue

            # Generate and Save the Plot
            plt.figure(figsize=(10, 8))
            sns.set_style("whitegrid")
            
            method_order = ['constant', 'linear_regression', 'GAM', 'rf', 'boosted_trees', 'engression', 'MLP', 'ResNet', 'FTTransformer']
            summary_df['model'] = pd.Categorical(summary_df['model'], categories=method_order, ordered=True)
            summary_df.sort_values('model', inplace=True)
            
            summary_df['model'] = summary_df['model'].replace({
                'rf': 'Random Forest', 
                'boosted_trees': 'Boosted Trees', 
                'linear_regression': 'Linear Regression', 
                'FTTransformer': 'FT-Transformer'
            })

            plot_title = f'Model Performance for {split_method.replace("_split", "").title()} Extrapolation'
            sns.scatterplot(data=summary_df, x='model', y='mean_relative_diff_pct', color='black', s=120, legend=False)
            
            plt.ylabel(config['y_label'])
            plt.xlabel('Model')
            plt.title(plot_title)
            plt.xticks(rotation=45, ha='right')
            # Set y-axis limit to match reference plots
            if metric == "RMSE" or metric == "CRPS":
                 plt.ylim(0, 50)

            plt.tight_layout()
            
            plot_filename = os.path.join(output_dir, f"summary_{split_method}_{metric}.png")
            plt.savefig(plot_filename)
            plt.close()
            print(f"Plot saved to {plot_filename}")

if __name__ == '__main__':
    input_csv_file = 'combined_results.csv'
    output_plot_dir = 'PICTURES'
    
    generate_analysis_plots(input_filepath=input_csv_file, output_dir=output_plot_dir)