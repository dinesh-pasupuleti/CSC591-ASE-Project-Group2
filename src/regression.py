import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

data = pd.read_csv("data/Wine_quality.csv")

for col in data.columns:
    if col.endswith("+") or col.endswith("-"):
        data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
        data[col] = 1 - data[col]
    elif col.endswith("-"):
        data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
        

data["d2h"] = None


calculate_distance = lambda row: round(math.sqrt(sum((row[col] ** 2) for col in data.columns if col.endswith("+") or col.endswith("-")) / sum(1 for col in data.columns if col.endswith("+") or col.endswith("-"))), 3)
data['d2h'] = data.apply(calculate_distance, axis=1)


cols = list(data.columns)    

columns_to_drop = [col for col in data.columns if col.endswith('+') or col.endswith('-')]
data = data.drop(columns=columns_to_drop)

data.to_csv("data.csv", sep=",", index=False)

X = data.drop(columns=['d2h'])
y = data['d2h']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# k = 10
# knn_regressor = KNeighborsRegressor(n_neighbors=k, algorithm="kd_tree", weights="distance", p=2)
# knn_regressor.fit(X_train, y_train)
# y_pred = knn_regressor.predict(X_test)

# elastic_net = ElasticNet(alpha=1.0, l1_ratio=0.5)
# elastic_net.fit(X_train, y_train)
# y_pred = elastic_net.predict(X_test)

# mse = mean_squared_error(y_test, y_pred)
# print(f"{round(mse, 7)}")

for alpha in range(1, 100):
    l1_ratio = 0.0
    while l1_ratio <= 1.00:
        elastic_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        elastic_net.fit(X_train, y_train)
        y_pred = elastic_net.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        print(f"{alpha} \t {l1_ratio:.2f} \t {round(mse, 7)}")
        
        l1_ratio += 0.01