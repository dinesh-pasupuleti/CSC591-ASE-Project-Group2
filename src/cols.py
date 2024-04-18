from num import *
from sym import *

class COLS:
    def __init__(self, row):
        self.x, self.y, self.all = {}, {}, []
        self.klass = None
        for at, txt in enumerate(row.cells):
            col = NUM(txt, at) if txt[0].isupper() else SYM(txt, at) # Determine column type
            self.all.append(col) # Add to all columns
            if not txt.endswith("X"):
                if txt.endswith("!"):
                    self.klass = col # Set klass column
                if txt.endswith(("!", "+", "-")):
                    self.y[at] = col  # Add to dependent columns
                else:
                    self.x[at] = col  # Add to independent columns
        self.names = row.cells

    def add(self, row):
        # Updates columns with new row data
        for cols in (self.x, self.y):
            for col in cols.values():
                col.add(row.cells[col.at])
        return row