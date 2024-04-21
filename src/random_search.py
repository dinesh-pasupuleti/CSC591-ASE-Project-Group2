import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
import csv
import time

# Load data (assuming 'data/Wine_quality.csv' is your data file path)
start = time.time()
filename = "Wine_quality"
data = pd.read_csv(f"data/{filename}.csv")

# Normalize the dataset based on column suffixes
for col in data.columns:
    if col.endswith("+") or col.endswith("-"):
        data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
        data[col] = 1 - data[col]

# Calculate the 'd2h' column based on the specified distance calculation
calculate_distance = lambda row: round(
    math.sqrt(
        sum((row[col] ** 2) for col in data.columns if col.endswith("+") or col.endswith("-"))
        / sum(1 for col in data.columns if col.endswith("+") or col.endswith("-"))
    ),
    3,
)
data['d2h'] = data.apply(calculate_distance, axis=1)

# Drop columns ending with '+' or '-'
columns_to_drop = [col for col in data.columns if col.endswith('+') or col.endswith('-')]
data.drop(columns=columns_to_drop, inplace=True)

# Split data into features (X) and target (y)
X = data.drop(columns=['d2h'])
y = data['d2h']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': np.random.choice(np.arange(1, 201), size=75, replace=False),
    'max_depth': np.random.choice(np.arange(1, 31), size=8, replace=False),
}

# Store MSE results for different hyperparameter combinations
mse_results = []

with open(f'data/randomsearch/{filename}_mse_results.csv', 'w', newline='') as csvfile:
    fieldnames = ['n_estimators', 'max_depth', 'mse']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Perform grid search over the parameter grid
    for params in ParameterGrid(param_grid):
        # Initialize random forest regressor with current hyperparameters
        rf = RandomForestRegressor(**params)

        # Train the model
        rf.fit(X_train, y_train)

        # Predict on the testing set
        y_pred = rf.predict(X_test)

        # Calculate mean squared error (MSE)
        mse = mean_squared_error(y_test, y_pred)

        # Store MSE along with corresponding hyperparameters
        mse_results.append((params['n_estimators'], params['max_depth'], mse))
        
        writer.writerow(
            {
                'n_estimators': params['n_estimators'],
                'max_depth': params['max_depth'],
                'mse': mse,
            }
        )

        # Print current hyperparameters and MSE
        print(f"n_estimators: {params['n_estimators']}, max_depth: {params['max_depth']}, MSE: {mse}")

# Convert the MSE results into a pandas DataFrame
results_df = pd.DataFrame(mse_results, columns=['n_estimators', 'max_depth', 'MSE'])

# Display summary statistics of the MSE values
summary_stats = results_df['MSE'].describe()

# Print the summary statistics
print("\nSummary Statistics for MSE:")
print(summary_stats)

# Plot the MSE values against hyperparameters
# plt.figure()
# plt.tricontourf(results_df['n_estimators'], results_df['max_depth'], results_df['MSE'], cmap='viridis')
# plt.colorbar(label='Mean Squared Error')
# plt.xlabel('n_estimators')
# plt.ylabel('max_depth')
# plt.title('MSE for Random Forest Regression Hyperparameters')
# plt.show()

# Find the best combination with the minimum MSE
best_combination = results_df.loc[results_df['MSE'].idxmin()]

# Print the best combination and its corresponding MSE
print("\nBest Combination:")
print(best_combination)
print(f"Time taken: {time.time() - start}")
