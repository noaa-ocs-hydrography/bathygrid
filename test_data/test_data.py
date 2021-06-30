import os
import numpy as np
from bathygrid.utilities import remove_with_permissionserror


x = np.arange(0, 1000, 100, dtype=np.float64)
y = np.arange(0, 1000, 100, dtype=np.float64)
x, y = np.meshgrid(x, y)
x = x.ravel()
y = y.ravel()
z = np.linspace(20, 30, num=x.size).astype(np.float32)
tvu = np.linspace(1, 2, num=x.size).astype(np.float32)
thu = np.linspace(0.5, 1, num=x.size).astype(np.float32)

dtyp = [('x', np.float64), ('y', np.float64), ('z', np.float32), ('tvu', np.float32), ('thu', np.float32)]
smalldata = np.empty(len(x), dtype=dtyp)
smalldata['x'] = x
smalldata['y'] = y
smalldata['z'] = z
smalldata['tvu'] = tvu
smalldata['thu'] = thu


x = np.arange(0, 5000, 100, dtype=np.float64)
y = np.arange(50000, 55000, 100, dtype=np.float64)
x, y = np.meshgrid(x, y)
x = x.ravel()
y = y.ravel()
z = np.linspace(20, 30, num=x.size).astype(np.float32)
tvu = np.linspace(1, 2, num=x.size).astype(np.float32)
thu = np.linspace(0.5, 1, num=x.size).astype(np.float32)

smalldata2 = np.empty(len(x), dtype=dtyp)
smalldata2['x'] = x
smalldata2['y'] = y
smalldata2['z'] = z
smalldata2['tvu'] = tvu
smalldata2['thu'] = thu


x = np.arange(3000, 8000, 100, dtype=np.float64)
y = np.arange(52000, 57000, 100, dtype=np.float64)
x, y = np.meshgrid(x, y)
x = x.ravel()
y = y.ravel()
z = np.linspace(20, 30, num=x.size).astype(np.float32)
tvu = np.linspace(1, 2, num=x.size).astype(np.float32)
thu = np.linspace(0.5, 1, num=x.size).astype(np.float32)

smalldata3 = np.empty(len(x), dtype=dtyp)
smalldata3['x'] = x
smalldata3['y'] = y
smalldata3['z'] = z
smalldata3['tvu'] = tvu
smalldata3['thu'] = thu


x = np.arange(3000, 8000, 10, dtype=np.float64)
y = np.arange(52000, 57000, 10, dtype=np.float64)
x, y = np.meshgrid(x, y)
x = x.ravel()
y = y.ravel()
z = np.linspace(500, 5000, num=x.size).astype(np.float32)
tvu = np.linspace(1, 2, num=x.size).astype(np.float32)
thu = np.linspace(0.5, 1, num=x.size).astype(np.float32)

deepdata = np.empty(len(x), dtype=dtyp)
deepdata['x'] = x
deepdata['y'] = y
deepdata['z'] = z
deepdata['tvu'] = tvu
deepdata['thu'] = thu


x = np.arange(800, 1200, 100, dtype=np.float64)
y = np.arange(800, 1200, 100, dtype=np.float64)
x, y = np.meshgrid(x, y)
x = x.ravel()
y = y.ravel()
z = np.linspace(5, 30, num=x.size).astype(np.float32)
tvu = np.linspace(1, 2, num=x.size).astype(np.float32)
thu = np.linspace(0.5, 1, num=x.size).astype(np.float32)

closedata = np.empty(len(x), dtype=dtyp)
closedata['x'] = x
closedata['y'] = y
closedata['z'] = z
closedata['tvu'] = tvu
closedata['thu'] = thu

x = np.arange(800, 1200, 100, dtype=np.float64)
y = np.arange(800, 1200, 100, dtype=np.float64)
x, y = np.meshgrid(x, y)
x = x.ravel()
y = y.ravel()
z = np.linspace(5, 30, num=x.size).astype(np.float32)

onlyzdata = np.empty(len(x), dtype=dtyp)
onlyzdata['x'] = x
onlyzdata['y'] = y
onlyzdata['z'] = z

x = np.array([200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 400, 600])
y = np.array([600, 550, 500, 470, 440, 410, 400, 410, 440, 470, 500, 550, 600, 800, 800])
x = x.ravel()
y = y.ravel()
z = np.linspace(5, 30, num=x.size).astype(np.float32)
tvu = np.linspace(1, 2, num=x.size).astype(np.float32)
thu = np.linspace(0.5, 1, num=x.size).astype(np.float32)

dtyp = [('x', np.float64), ('y', np.float64), ('z', np.float32), ('tvu', np.float32), ('thu', np.float32)]
smileyface = np.empty(len(x), dtype=dtyp)
smileyface['x'] = x
smileyface['y'] = y
smileyface['z'] = z
smileyface['tvu'] = tvu
smileyface['thu'] = thu

x = np.array([-73.0024, -73.0022, -73.0020, -73.0018, -73.0016, -73.0014, -73.0012, -73.0010, -73.0008, -73.0006,
              -73.0004, -73.0002, -73.0000, -73.0016, -73.0008])
y = np.array([30.0008, 30.0006, 30.0004, 30.0003, 30.0002, 30.0001, 30.00005, 30.0001, 30.0002, 30.0003,
              30.0004, 30.0006, 30.0008, 30.0016, 30.0016])
x = x.ravel()
y = y.ravel()
z = np.linspace(5, 30, num=x.size).astype(np.float32)
tvu = np.linspace(1, 2, num=x.size).astype(np.float32)
thu = np.linspace(0.5, 1, num=x.size).astype(np.float32)

dtyp = [('x', np.float64), ('y', np.float64), ('z', np.float32), ('tvu', np.float32), ('thu', np.float32)]
geographicsmileyface = np.empty(len(x), dtype=dtyp)
geographicsmileyface['x'] = x
geographicsmileyface['y'] = y
geographicsmileyface['z'] = z
geographicsmileyface['tvu'] = tvu
geographicsmileyface['thu'] = thu


def get_grid_data():
    depth = np.linspace(10, 20, 20)
    tvu = np.linspace(1, 2, 20)
    thu = np.linspace(0.5, 1.5, 20)
    cell_indices = np.array([3, 1, 0, 2, 1, 0, 0, 7, 7, 2, 5, 4, 5, 4, 5, 6, 5, 6, 3, 3])
    grid = np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]])
    tvugrid = np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]])
    thugrid = np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]])
    return depth, tvu, thu, cell_indices, grid, tvugrid, thugrid


def get_test_path():
    pth = os.path.join(os.path.dirname(__file__), 'grid')
    if os.path.exists(pth):
        remove_with_permissionserror(pth)
    return pth


