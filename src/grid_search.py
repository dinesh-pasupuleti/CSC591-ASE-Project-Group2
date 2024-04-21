import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import csv

# Load data, specifying '?' as NaN values
data = pd.read_csv("data/SS.csv", na_values="?")

# Identify and exclude columns ending with 'X'
columns_to_exclude = [col for col in data.columns if col.endswith('X')]
data.drop(columns=columns_to_exclude, inplace=True)

# Preprocess data
for col in data.columns:
    if col.endswith("+") or col.endswith("-"):
        # Apply transformation to numeric columns ending with '+' or '-'
        data[col] = 1 - (data[col] - data[col].min()) / (
            data[col].max() - data[col].min()
        )


def calculate_distance(row):
    cols_to_include = [
        col for col in data.columns if col.endswith("+") or col.endswith("-")
    ]
    return round(
        math.sqrt(sum((row[col] ** 2) for col in cols_to_include))
        / len(cols_to_include),
        3,
    )


# Apply the calculate_distance function row-wise to create the 'd2h' column
data['d2h'] = data.apply(calculate_distance, axis=1)


# Drop columns ending with '+' or '-'
columns_to_drop = [
    col for col in data.columns if col.endswith('+') or col.endswith('-')
]
data.drop(columns=columns_to_drop, inplace=True)
print(data)
# Drop rows with NaN values (those that originally contained '?')
data.dropna(inplace=True)

# Split data into X and y
X = data.drop(columns=['d2h'])
y = data['d2h']


# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define hyperparameter ranges
param_grid = {
    'n_estimators': np.linspace(1, 100, 100, dtype=int),
    'max_depth': np.linspace(1, 20, 20, dtype=int),
}

# Initialize list to store MSE results
mse_results = []

# Open CSV file for writing MSE results
with open('mse_results.csv', 'w', newline='') as csvfile:
    fieldnames = ['n_estimators', 'max_depth', 'mse']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Loop over parameter grid and evaluate each combination
    for params in ParameterGrid(param_grid):
        # Initialize RandomForestRegressor with current hyperparameters
        rf = RandomForestRegressor(**params, random_state=42)

        # Train the model
        rf.fit(X_train, y_train)

        # Predict on the test set
        y_pred = rf.predict(X_test)

        # Calculate mean squared error
        mse = mean_squared_error(y_test, y_pred)

        # Store hyperparameters and MSE in results list
        mse_results.append((params['n_estimators'], params['max_depth'], mse))

        # Write hyperparameters and MSE to CSV file
        writer.writerow(
            {
                'n_estimators': params['n_estimators'],
                'max_depth': params['max_depth'],
                'mse': mse,
            }
        )

        # Print current hyperparameters and MSE
        print(
            f"n_estimators: {params['n_estimators']}, max_depth: {params['max_depth']}, MSE: {mse}"
        )

print("MSE results saved to mse_results.csv")

# Extract data for visualization from the CSV file
df_results = pd.read_csv('mse_results.csv')
estimators = df_results['n_estimators']
depths = df_results['max_depth']
mse_values = df_results['mse']

# Create 3D scatter plot
# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(estimators, depths, mse_values, c=mse_values, cmap='viridis')
# ax.set_xlabel('n_estimators')
# ax.set_ylabel('max_depth')
# ax.set_zlabel('Mean Squared Error')
# plt.title('MSE for Random Forest Regression Hyperparameters')
# plt.colorbar(ax.collections[0], label='Mean Squared Error')
# plt.show()

# # Alternatively, create a heatmap
# plt.figure()
# plt.tricontourf(estimators, depths, mse_values, cmap='viridis')
# plt.colorbar(label='Mean Squared Error')
# plt.xlabel('n_estimators')
# plt.ylabel('max_depth')
# plt.title('MSE for Random Forest Regression Hyperparameters')
# plt.show()

# Find the best combination with minimum MSE
best_params = df_results.loc[df_results['mse'].idxmin()]
print(
    f"\nBest combination - (n_estimators: {best_params['n_estimators']}, max_depth: {best_params['max_depth']}), Min MSE: {best_params['mse']}"
)
