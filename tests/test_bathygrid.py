import numpy as np

from bathygrid import BathyGrid


x = np.arange(0, 5000, 100, dtype=np.float64)
y = np.arange(50000, 55000, 100, dtype=np.float64)
x, y = np.meshgrid(x, y)
x = x.ravel()
y = y.ravel()
z = np.linspace(20, 30, num=x.size).astype(np.float32)

dtyp = [('x', np.float64), ('y', np.float64), ('z', np.float32)]
data1 = np.empty(len(x), dtype=dtyp)
data1['x'] = x
data1['y'] = y
data1['z'] = z

x = np.arange(3000, 8000, 100, dtype=np.float64)
y = np.arange(52000, 57000, 100, dtype=np.float64)
x, y = np.meshgrid(x, y)
x = x.ravel()
y = y.ravel()
z = np.linspace(20, 30, num=x.size).astype(np.float32)

dtyp = [('x', np.float64), ('y', np.float64), ('z', np.float32)]
data2 = np.empty(len(x), dtype=dtyp)
data2['x'] = x
data2['y'] = y
data2['z'] = z


def test_bathygrid():
    bg = BathyGrid(tile_size=1024)
    bg.add_points(data1, 'test1')
    bg.remove_points('test1')