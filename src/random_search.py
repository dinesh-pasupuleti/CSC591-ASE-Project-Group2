import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("data/Wine_quality.csv")

for col in data.columns:
    if col.endswith("+") or col.endswith("-"):
        data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
        data[col] = 1 - data[col]
    elif col.endswith("-"):
        data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())


data["d2h"] = None


calculate_distance = lambda row: round(
    math.sqrt(
        sum(
            (row[col] ** 2)
            for col in data.columns
            if col.endswith("+") or col.endswith("-")
        )
        / sum(1 for col in data.columns if col.endswith("+") or col.endswith("-"))
    ),
    3,
)
data['d2h'] = data.apply(calculate_distance, axis=1)


cols = list(data.columns)

columns_to_drop = [
    col for col in data.columns if col.endswith('+') or col.endswith('-')
]
data = data.drop(columns=columns_to_drop)


X = data.drop(columns=['d2h'])
y = data['d2h']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestRegressor()

# Define hyperparameter ranges
param_grid = {
    'n_estimators': np.random.choice(np.arange(1, 201), size=10, replace=False),
    'max_depth': np.random.choice(np.arange(1, 31), size=5, replace=False),
}

mse_results = []

for params in ParameterGrid(param_grid):
    # Initialize random forest regressor with current hyperparameters
    rf = RandomForestRegressor(**params)

    # Train the model
    rf.fit(X_train, y_train)

    # Predict on the training set
    y_pred = rf.predict(X_test)

    # Calculate mean squared error
    mse = mean_squared_error(y_test, y_pred)

    # Store MSE along with corresponding hyperparameters
    mse_results.append((params['n_estimators'], params['max_depth'], mse))

    # Print current hyperparameters and MSE
    print(
        f"n_estimators: {params['n_estimators']}, max_depth: {params['max_depth']}, MSE: {mse}"
    )

# Extract data for visualization
estimators = [result[0] for result in mse_results]
depths = [result[1] for result in mse_results]
mse_values = [result[2] for result in mse_results]

# Create 3D scatter plot (optional)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(estimators, depths, mse_values)
ax.set_xlabel('n_estimators')
ax.set_ylabel('max_depth')
ax.set_zlabel('Mean Squared Error')
plt.show()

# Alternatively, create a heatmap
plt.figure()
plt.tricontourf(estimators, depths, mse_values, cmap='viridis')
plt.colorbar(label='Mean Squared Error')
plt.xlabel('n_estimators')
plt.ylabel('max_depth')
plt.title('MSE for Random Forest Regression Hyperparameters')
plt.show()


# Find the best combination with minimum MSE
n_estimators, max_depth, best_mse = min(mse_results, key=lambda x: x[2])

# Print the best combination and its MSE
print(f"\nBest combination - {(n_estimators, max_depth)}, Min MSE: {best_mse}")
