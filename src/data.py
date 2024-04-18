from cols import COLS
from rows import ROW
from utils import *
import random
import math

class DATA:
    def __init__(self, src, fun=None): 
        self.rows = []
        self.cols = None
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
            self.cols=COLS(row)

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
        for col in (self.cols.y if cols is None else [self.cols.names[c] for c in cols]):
            cur_col = self.cols.all[col]
            u[cur_col.txt] = round(getattr(cur_col, fun or "mid")(), nDivs) if nDivs else getattr(cur_col, fun or "mid")()
        return u
   
    def shuffle(self, items):
        return random.sample(items, len(items))
    
    def gate(self, budget0, budget, some):
        heaven = 1.0
        rows = self.shuffle(self.rows)
        print("1. top6", [[row.cells[x] for x in self.cols.y.keys()] for row in rows[:6]])
        print("2. top50", [[row.cells[x] for x in self.cols.y.keys()] for row in rows[:50]])

        rows.sort(key=lambda row: self.distance2heaven(row, heaven))
        print("3. most", [rows[0].cells[x] for x in list(self.cols.y.keys())])

        rows = self.shuffle(self.rows)
        lite = rows[:budget0]
        dark = rows[budget0:]

        for _ in range(budget):
            lite.sort(key=lambda row: self.distance2heaven(row, heaven))
            n = int(len(lite)**some)
            best, rest = lite[:n], lite[n:]
            todo, selected = self.split(best, rest, lite, dark)
            
            dark_sample = random.sample(dark, budget0+1)
            print("4: rand", [dark_sample[len(dark_sample)//2].cells[x] for x in self.cols.y.keys()])
            if len(selected.rows) > 0:
                print("5: mid", [selected.rows[len(selected.rows)//2].cells[x] for x in self.cols.y.keys()])
            print("6: top", [best[0].cells[x] for x in self.cols.y.keys()])
            
            lite.append(dark.pop(todo))


    def distance2heaven(self, row, heaven):
        norm = lambda c, x: (x - c.lo) / (c.hi - c.lo)
        return math.sqrt(sum((heaven - norm(col, row.cells[y]))**2 for y, col in self.cols.y.items()) / len(self.cols.y))

    def split(self, best, rest, lite, dark):
        selected = DATA(self.cols.names)
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
            if b > r:
                selected.add(row)
            score = abs(b + r) / abs(b - r)
            if score > max_score:
                max_score, todo = score, i
        return todo, selected
