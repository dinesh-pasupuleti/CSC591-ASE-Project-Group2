import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from scipy import stats

def collect_mse_values(X, y, num_evals=30, random_state=42):
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

    # Run Hyperopt
    trials = Trials()
    best_hyperopt = fmin(
        fn=objective, space=space, algo=tpe.suggest, max_evals=num_evals, trials=trials
    )
    mse_hyperopt = [trial['result']['loss'] for trial in trials.trials]

    # Setup for Randomized Search
    param_dist = {
        'n_estimators': np.arange(50, 501, 10),
        'max_depth': np.arange(5, 21, 1),
    }
    model = RandomForestRegressor(random_state=random_state)
    random_search = RandomizedSearchCV(
        model, param_dist, n_iter=num_evals, cv=5, scoring='neg_mean_squared_error', random_state=random_state
    )
    random_search.fit(X_train, y_train)
    mse_random_search = -random_search.cv_results_['mean_test_score']

    return mse_hyperopt, mse_random_search

# Load your data
# Load data, specifying '?' as NaN values
data = pd.read_csv("data/SS-A.csv", na_values="?")

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

# Collect MSE values from both tuning methods
mse_hyperopt, mse_random_search = collect_mse_values(X, y)

# Shapiro-Wilk Test for normality
print("Normality Test (Hyperopt):", stats.shapiro(mse_hyperopt))
print("Normality Test (Random Search):", stats.shapiro(mse_random_search))

# Mann-Whitney U Test for non-parametric comparison
u_stat, p_value = stats.mannwhitneyu(mse_hyperopt, mse_random_search, alternative='two-sided')
print("U-statistic:", u_stat, "P-value:", p_value)

# Calculate effect size r
r_effect_size = 1 - (2 * u_stat) / (len(mse_hyperopt) * len(mse_random_search))
print("Effect Size r:", r_effect_size)

# Plot the evolution of mean squared error across iterations
plt.figure(figsize=(10, 6))
plt.plot(mse_hyperopt, marker='o', linestyle='-', color='b', label='Hyperopt MSE')
plt.plot(mse_random_search, marker='x', linestyle='-', color='r', label='Random Search MSE')
plt.title('Evolution of Mean Squared Error')
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.show()
