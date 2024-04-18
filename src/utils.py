import sys
import re
from pathlib import Path
from config import *

def coerce(s):
    if s.isdigit():
        return int(s)
    elif '.' in s and s.replace('.', '').isdigit():
        return float(s)
    elif s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    elif s.lower() == 'nil':
        return None
    else:
        return s.strip()

def settings(s):
    t = {}
    pat = r"[-][-]([\S]+)[^\n]+= ([\S]+)"
    return dict(re.findall(pat, s))


def cli(options):
    args = sys.argv[1:]
    for k, v in options.items():
        for n, x in enumerate(args):
            if x == '-' + k[0] or x == '--' + k:
                v = 'true' if v == 'false' else 'false' if v == 'true' else args[n + 1]
        options[k] = coerce(v)
    return options


def eg(key, str, fun):
    egs[key] = fun
    global help
    help = help + '  -g '+ key + '\t' + str + '\n'
    

def csv(sFilename, fun):
    sFilename = Path(sFilename)
    if sFilename.exists() and sFilename.suffix == '.csv':
        t = []
        with open(sFilename.absolute(), 'r', encoding='utf-8') as file:
            for line in file:
                row = list(map(coerce, line.strip().split(',')))
                t.append(row)
                fun(row)
    else:
        print("File path does not exist OR File not csv, given path: ", sFilename.absolute())
        return
    

