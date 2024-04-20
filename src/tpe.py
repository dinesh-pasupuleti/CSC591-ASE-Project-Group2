import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from hyperopt import hp, fmin, tpe, Trials


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

data.to_csv("data.csv", sep=",", index=False)

X = data.drop(columns=['d2h'])
y = data['d2h']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def objective(params):
    # Convert 'n_estimators' and 'max_depth' parameters to integers
    params['n_estimators'] = int(params['n_estimators'])
    params['max_depth'] = int(params['max_depth'])

    # Create RandomForestRegressor with given parameters
    model = RandomForestRegressor(**params, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate mean squared error
    mse = mean_squared_error(y_test, y_pred)

    return mse


space = {
    'n_estimators': hp.quniform(
        'n_estimators', 50, 500, 10
    ),  # Integer values between 50 and 500
    'max_depth': hp.quniform('max_depth', 5, 20, 1),  # Integer values between 5 and 20
}

trials = Trials()  # Keep track of the trials
best_params = fmin(
    fn=objective,  # Objective function to minimize
    space=space,  # Search space
    algo=tpe.suggest,  # Optimization algorithm (TPE)
    max_evals=100,  # Maximum number of evaluations
    trials=trials,
)  # Trials object to track the process

# Extract the loss values (mse) from the trials
losses = [trial['result']['loss'] for trial in trials.trials]

# Plot the evolution of mean squared error across iterations
plt.figure(figsize=(10, 6))
plt.plot(losses, marker='o', linestyle='-', color='b')
plt.title('Evolution of Mean Squared Error')
plt.xlabel('Iteration')
plt.ylabel('Mean Squared Error')
plt.grid(True)
plt.show()

# Print the best parameters found
print("Best Parameters:", best_params)

# Evaluate the final model with the best parameters on the test set
best_params['n_estimators'] = int(best_params['n_estimators'])
best_params['max_depth'] = int(best_params['max_depth'])

final_model = RandomForestRegressor(**best_params, random_state=42)
final_model.fit(X_train, y_train)

y_pred_final = final_model.predict(X_test)
mse_final = mean_squared_error(y_test, y_pred_final)
print("Final MSE on Test Set:", mse_final)
