import numpy as np
from pytest import approx

from bathygrid.bgrid import *
from bathygrid.maingrid import *
from bathygrid.tile import Tile


x = np.arange(0, 5000, 100, dtype=np.float64)
y = np.arange(50000, 55000, 100, dtype=np.float64)
x, y = np.meshgrid(x, y)
x = x.ravel()
y = y.ravel()
z = np.linspace(20, 30, num=x.size).astype(np.float32)
tvu = np.linspace(1, 2, num=x.size).astype(np.float32)
thu = np.linspace(0.5, 1, num=x.size).astype(np.float32)

dtyp = [('x', np.float64), ('y', np.float64), ('z', np.float32), ('tvu', np.float32), ('thu', np.float32)]
data1 = np.empty(len(x), dtype=dtyp)
data1['x'] = x
data1['y'] = y
data1['z'] = z
data1['tvu'] = tvu
data1['thu'] = thu


