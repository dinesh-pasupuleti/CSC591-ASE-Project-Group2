import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import time  # Import time to measure the runtime of the TPE

def collect_tpe_mse_values(X, y, num_evals=30, random_state=42):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Define the search space for Hyperopt
    space = {
        'n_estimators': hp.quniform('n_estimators', 50, 500, 10),
        'max_depth': hp.quniform('max_depth', 5, 20, 1),
    }

    # Define the objective function for Hyperopt
    def objective(params):
        params['n_estimators'] = int(params['n_estimators'])
        params['max_depth'] = int(params['max_depth'])
        model = RandomForestRegressor(**params, random_state=random_state)
        model.fit(X_train, y_train)
        mse = mean_squared_error(y_test, model.predict(X_test))
        return {'loss': mse, 'status': STATUS_OK}

    # Run Hyperopt with timing
    start_time = time.time()
    trials = Trials()
    best_hyperopt = fmin(
        fn=objective, space=space, algo=tpe.suggest, max_evals=num_evals, trials=trials
    )
    end_time = time.time()
    mse_hyperopt = [trial['result']['loss'] for trial in trials.trials]

    # Return the MSE values, the best configuration, and the runtime
    runtime = end_time - start_time
    return mse_hyperopt, best_hyperopt, runtime

# Load data
# Load data, specifying '?' as NaN values
data = pd.read_csv("data/Wine_quality.csv", na_values="?")

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

# Drop rows with NaN values (those that originally contained '?')
data.dropna(inplace=True)

# Split data into X and y
X = data.drop(columns=['d2h'])
y = data['d2h']

# Collect TPE MSE values, best configuration, and runtime
mse_hyperopt, best_hyperopt, runtime = collect_tpe_mse_values(X, y)

# Determine the minimum MSE achieved
min_mse = min(mse_hyperopt)

print(f"Best TPE Configuration: {best_hyperopt}")
print(f"Minimum MSE Achieved: {min_mse}")
print(f"TPE Runtime: {runtime:.2f} seconds")

