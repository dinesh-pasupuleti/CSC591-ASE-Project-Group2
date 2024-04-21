import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the results from the CSV file
df_results = pd.read_csv('data/Wine_quality_mse.csv')

# Prepare the grid for plotting; choose a manageable number of depths, like every 5th depth
unique_depths = df_results['max_depth'].unique()
selected_depths = unique_depths[::5]  # adjust the stride as necessary for visualization

# Create the plot grid
n_cols = 3  # adjust based on preference and screen resolution
n_rows = int(np.ceil(len(selected_depths) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4), sharex=True, sharey=True)
axes = axes.flatten()

# Plot each depth in its subplot
for ax, depth in zip(axes, selected_depths):
    subset = df_results[df_results['max_depth'] == depth]
    ax.plot(subset['n_estimators'], subset['mse'], marker='o', linestyle='-', markersize=4)
    ax.set_title(f'Max Depth = {depth}')
    ax.set_xlabel('n_estimators')
    ax.set_ylabel('MSE')
    ax.grid(True)

# Adjust layout and add a general title
fig.tight_layout(pad=2.0)
fig.suptitle('MSE vs n_estimators for selected max_depths', fontsize=16)

# Hide unused axes if any
for i in range(len(selected_depths), len(axes)):
    fig.delaxes(axes[i])

plt.show()
