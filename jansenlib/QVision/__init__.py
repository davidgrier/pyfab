import json
import os

path = os.path.abspath(__file__)
path = '/'.join(path.split('/')[:-1])

with open(os.path.join(path, '.QVision'), 'r') as f:
    d = json.load(f)

if d['QVision'] == 'QHVM':
    from .QHVM import QHVM as QVision

all = [QVision]
