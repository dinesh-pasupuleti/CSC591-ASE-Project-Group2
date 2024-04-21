import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from skopt import BayesSearchCV
from skopt.space import Integer

# Load data
data = pd.read_csv("data/Wine_quality.csv")

# Preprocess data
for col in data.columns:
    if col.endswith("+") or col.endswith("-"):
        data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
        data[col] = 1 - data[col]

data["d2h"] = data.apply(
    lambda row: round(
        math.sqrt(
            sum(
                (row[col] ** 2)
                for col in data.columns
                if col.endswith("+") or col.endswith("-")
            )
        )
        / sum(1 for col in data.columns if col.endswith("+") or col.endswith("-"))
    ),
    axis=1,
)

# Drop unnecessary columns
columns_to_drop = [
    col for col in data.columns if col.endswith('+') or col.endswith('-')
]
data = data.drop(columns=columns_to_drop)

# Split data into features (X) and target (y)
X = data.drop(columns=['d2h'])
y = data['d2h']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Define objective function for optimization
def objective_function(params):
    model = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse


# Define search space for Bayesian optimization
search_space = {'n_estimators': Integer(50, 500), 'max_depth': Integer(5, 20)}

# Initialize BayesSearchCV optimizer
opt = BayesSearchCV(
    estimator=RandomForestRegressor(),
    search_spaces=search_space,
    n_iter=100,  # Number of iterations
    cv=5,  # Number of cross-validation folds
    scoring='neg_mean_squared_error',  # Optimization metric
    random_state=42,
)

# Perform Bayesian optimization
opt.fit(X_train, y_train)

# Retrieve best parameters and best model
best_params = opt.best_params_
best_model = opt.best_estimator_

print("Best Parameters:", best_params)

# Evaluate the final model with the best parameters on the test set
y_pred_final = best_model.predict(X_test)
mse_final = mean_squared_error(y_test, y_pred_final)
print("Final MSE on Test Set:", mse_final)

# Plot optimization history
mean_scores = opt.cv_results_['mean_test_score']
x_iters = np.arange(1, len(mean_scores) + 1)

plt.figure(figsize=(10, 6))
plt.plot(x_iters, mean_scores, marker='o', linestyle='-', color='b')
plt.title('Bayesian Optimization with RandomForestRegressor')
plt.xlabel('Iteration')
plt.ylabel('Mean Validation Score (Negative MSE)')
plt.grid(True)
plt.show()
