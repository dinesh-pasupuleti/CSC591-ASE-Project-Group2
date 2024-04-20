from utils import *
from config import *
from data import DATA
import math
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def main():
    saved_options = {}
    fails = 0

    for key, value in cli(settings(help)).items():
        the[key] = value
        saved_options[key] = value

    if the['help']:
        print(help)
    else:
        for action, _ in egs.items():
            if the['todo'] == 'all' or the['todo'] == action:
                for key, value in saved_options.items():
                    the[key] = value

                global Seed
                Seed = the['seed']

                if egs[action]() == False:
                    fails += 1
                    print('❌ fail:', action)
                else:
                    print('✅ pass:', action)


def print_class_percentages(data):
    class_counts = {}
    total_rows = len(data.rows)

    for row in data.rows:
        class_label = row.cells[data.cols.all[-1].at]
        class_counts[class_label] = class_counts.get(class_label, 0) + 1

    print("     Class         \t    Percentage   ")
    print("------------------ \t ----------------")
    for class_label, count in class_counts.items():
        percentage = (count / total_rows) * 100
        print(f"{class_label.ljust(25)} \t {percentage:.2f}%")


if __name__ == '__main__':
    main()

    file_name = "Wine_quality"

    params = pd.read_csv(f'data/{file_name}_mse.csv')
    paramsX = params.drop(columns=["mse"])
    paramsX.to_csv("data/paramsX.csv", index=False)

    paramsX = DATA('data/paramsX.csv')

    os.remove('data/paramsX.csv')

    data1 = pd.read_csv(f"data/{file_name}.csv")

    for col in data1.columns:
        if col.endswith("+") or col.endswith("-"):
            data1[col] = (data1[col] - data1[col].min()) / (
                data1[col].max() - data1[col].min()
            )
            data1[col] = 1 - data1[col]
        elif col.endswith("-"):
            data1[col] = (data1[col] - data1[col].min()) / (
                data1[col].max() - data1[col].min()
            )

    data1["d2h"] = None

    calculate_distance = lambda row: round(
        math.sqrt(
            sum(
                (row[col] ** 2)
                for col in data1.columns
                if col.endswith("+") or col.endswith("-")
            )
            / sum(1 for col in data1.columns if col.endswith("+") or col.endswith("-"))
        ),
        3,
    )
    data1['d2h'] = data1.apply(calculate_distance, axis=1)

    cols = list(data1.columns)

    columns_to_drop = [
        col for col in data1.columns if col.endswith('+') or col.endswith('-')
    ]
    data1 = data1.drop(columns=columns_to_drop)
    data1.to_csv(f"data/{file_name}_processed.csv", sep=",", index=False)

    iters = 20
    total_error = 0

    for _ in range(iters):
        lite = paramsX.gate(4, 10, 0.5, file_name)

        X, y = [], []

        for row in lite:
            n, m = row.cells[0], row.cells[1]
            X.append([n, m])
            mse_value = params.loc[
                (params['n_estimators'] == row.cells[0])
                & (params['max_depth'] == row.cells[1]),
                'mse',
            ].values[0]
            y.append(mse_value)

        print(X)
        print(y)

        lr = LinearRegression()
        lr.fit(X, y)

        X_test = params.drop("mse", axis=1)
        y_test = params["mse"]
        y_pred = lr.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        total_error += mse

        print(mse)

    avg_error = total_error / iters
    print(avg_error)

    os.remove(f"data/{file_name}_processed.csv")
