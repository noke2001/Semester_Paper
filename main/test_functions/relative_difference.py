import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox

try:
    master_df = pd.read_csv('combined_results.csv')
    print("Successfully loaded 'combined_results.csv'.")
except FileNotFoundError:
    print("Error: 'combined_results.csv' not found. Please make sure it is in the same directory.")
    exit()

SUITE_ID = 335             
split_method_to_analyze = 'gower_split'
metric_to_analyze = 'RMSE'

analysis_df = master_df[
    (master_df['suite_id'] == SUITE_ID) &
    (master_df['split_method'] == split_method_to_analyze) &
    (master_df['metric'] == metric_to_analyze)
].copy()

if analysis_df.empty:
    print(f"Error: No data found for split_method='{split_method_to_analyze}' and metric='{metric_to_analyze}'.")
    exit()

print(f"Filtered data for '{split_method_to_analyze}' and metric '{metric_to_analyze}'.")




lowest_rmses = analysis_df.groupby(['suite_id','task_id'])['value'].transform('min')

analysis_df['relative_diff'] = (analysis_df['value'] - lowest_rmses) / lowest_rmses
analysis_df['relative_diff'].fillna(0, inplace=True) # Handle any division-by-zero cases


df = analysis_df.pivot_table(index='model', columns='task_id', values='relative_diff')



df.index.name = 'Method'
mean = df.mean(axis=1)
median = df.median(axis=1)
std = df.std(axis=1)
result_df = pd.DataFrame({'Mean': mean, 'Median': median, 'Standard Deviation': std})
result_df.index.name = 'Method'
result_df.reset_index(inplace=True)

# Reorder the methods
method_order = ['ConstantPredictor', 'LinearRegressor', 'TabPFNRegressor', 'RandomForestRegressor', 'LGBMRegressor']
result_df['Method'] = pd.Categorical(result_df['Method'], categories=method_order, ordered=True)

# Sort the dataframe by the reordered method column
result_df.sort_values('Method', inplace=True)

result_df['Method'] = result_df['Method'].replace({'RandomForestRegressor': 'random forest', 'LGBMRegressor': 'boosted trees', 'LinearRegressor': 'linear regression', 'TabPFNRegressor': 'TabPFN', 'ConstantPredictor': 'constant'})
result_df['Mean'] = 100 * result_df['Mean']

# Plot the scatterplot
plt.figure(figsize=(10, 8)) # Added to ensure a good plot size
sns.scatterplot(data=result_df, x='Method', y='Mean', color='black')

# Removing the ylabel
plt.ylabel('Average relative difference to the best test RMSE (in %)')
plt.ylim(0, 400)

# Adding labels and title
plt.xlabel('Method')
plt.title('Gower Split')

# Rotating x-axis labels vertically
plt.xticks(rotation=45, ha='right')

# Set the figure size
fig = plt.gcf()  # Get the current figure
fig_size = fig.get_size_inches()  # Get the size of the figure

plt.tight_layout()

os.makedirs('PICTURES', exist_ok=True)
plt.savefig('PICTURES/clustering_RMSE_only_num_features_relative_differences.png')

print("\nAnalysis complete. Plot has been generated and saved.")
plt.show()