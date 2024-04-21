from cols import COLS
from rows import ROW
from utils import *
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd


class DATA:
    def __init__(self, src, fun=None):
        self.rows = []
        self.cols = None
        self.mse_data = None
        if isinstance(src, str):
            csv(src, self.add)
        else:
            # for x in src or []:
            self.add(src, fun)

    def add(self, t, fun=None):
        row = t if isinstance(t, ROW) and t.cells else ROW(t)
        if self.cols:
            if fun:
                fun(row)
            self.rows.append(self.cols.add(row))
        else:
            self.cols = COLS(row)

    def mid(self, cols=None):
        u = {}
        for col in cols or self.cols.all:
            u[col.at] = col.mid()
        return ROW(u)

    def div(self, cols=None):
        u = {}
        for col in cols or self.cols.all:
            u[col.at] = col.div()
        return ROW(u)

    def stats(self, cols=None, fun=None, nDivs=None):
        u = {".N": len(self.rows)}
        for col in self.cols.y if cols is None else [self.cols.names[c] for c in cols]:
            cur_col = self.cols.all[col]
            u[cur_col.txt] = (
                round(getattr(cur_col, fun or "mid")(), nDivs)
                if nDivs
                else getattr(cur_col, fun or "mid")()
            )
        return u

    def shuffle(self, items):
        return random.sample(items, len(items))

    def get_mse_value(self, row):
        mse_value = self.mse_data.loc[
            (self.mse_data['n_estimators'] == row.cells[0])
            & (self.mse_data['max_depth'] == row.cells[1]),
            'mse',
        ].values[0]
        return mse_value

    def gate(self, budget0, budget, some, file_name):

        self.mse_data = pd.read_csv(f"data/{file_name}_mse.csv")

        rows = self.shuffle(self.rows)
        lite = rows[:budget0]
        dark = rows[budget0:]

        for i in range(budget):
            print(f"processing {i}")
            lite.sort(key=lambda row: self.get_mse_value(row))
            n = int(len(lite) ** some)
            best, rest = lite[:n], lite[n:]
            todo = self.split(best, rest, lite, dark)
            lite.append(dark.pop(todo))
        return lite

    def split(self, best, rest, lite, dark):
        max_score = float('-inf')

        best_data = DATA(self.cols.names)
        for row in best:
            best_data.add(row)

        rest_data = DATA(self.cols.names)
        for row in rest:
            rest_data.add(row)

        for i, row in enumerate(dark):
            b = row.like(best_data, len(lite), 2)
            r = row.like(rest_data, len(lite), 2)
            score = abs(b + r) / (abs(b - r) + 1e-30)
            if score > max_score:
                max_score, todo = score, i
        return todo
