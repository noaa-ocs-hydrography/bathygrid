# bathygrid

A tiled bathymetric point cloud gridding engine built in Python3.  Has the following features:

1. Swappable gridding methods powered by Numba for speed (currently supports 'mean' and 'shoalest', see algorithms.py)
2. Sequential gridding or grid in parallel with Dask client (see utilities.py)
3. Single and Variable Resolution with swappable tile types (srgrid and vrgridtile, see maingrid.py)
4. Multiple Backends for different storage types (currently only NumpyGrid, see backends.py)
5. Export to csv, geotiff and BAG (requires GDAL, see maingrid.py)
6. Add and remove points from the grid using tags (see bgrid.py, remove_points and add_points)

bathygrid currently requires a structured numpy array or xarray dataset with 'z', 'tvu', 'thu' data names:

```
import numpy as np

x = np.arange(800, 1200, 100, dtype=np.float64)
y = np.arange(800, 1200, 100, dtype=np.float64)
x, y = np.meshgrid(x, y)
x = x.ravel()
y = y.ravel()
z = np.linspace(5, 30, num=x.size).astype(np.float32)
tvu = np.linspace(1, 2, num=x.size).astype(np.float32)
thu = np.linspace(0.5, 1, num=x.size).astype(np.float32)

dtyp = [('x', np.float64), ('y', np.float64), ('z', np.float32), ('tvu', np.float32), ('thu', np.float32)]
data = np.empty(len(x), dtype=dtyp)
data['x'] = x
data['y'] = y
data['z'] = z
data['tvu'] = tvu
data['thu'] = thu
```

Or to illustrate the xarray Dataset example

```
import xarray as xr
import numpy as np

x = np.arange(800, 1200, 100, dtype=np.float64)
y = np.arange(800, 1200, 100, dtype=np.float64)
x, y = np.meshgrid(x, y)
x = x.ravel()
y = y.ravel()
z = np.linspace(5, 30, num=x.size).astype(np.float32)
tvu = np.linspace(1, 2, num=x.size).astype(np.float32)
thu = np.linspace(0.5, 1, num=x.size).astype(np.float32)

data = xr.Dataset({'x': (x), 'y': (y), 'z': (z), 'tvu': (tvu), 'thu': (thu)})
```

bathygrid can either be initialized with an output path (if you want to write to disk as you add data) or without (if 
you want to work entirely within memory).  If you include an output path, data will be continuously written to disk as 
points are added and removed and whenever .grid() is called.

```
from bathygrid.convenience import create_grid

# a single resolution grid that is entirely within computer memory
bg = create_grid(grid_type='single_resolution')
...
# can be saved to disk later
bg.save('output/folder/path')

# a single resolution grid that is flushed to memory on adding/removing/gridding data, does not require .save()
bg = create_grid(folder_path='output/folder/path', grid_type='single_resolution')
```

Add some new points to the dataset, use a tag to identify those points later and some other metdata

```
bg = create_grid(grid_type='single_resolution')
# add points from two multibeam lines, EPSG:26917 with vertical reference 'waterline'
bg.add_points(data, 'test1', ['line1', 'line2'], 26917, 'waterline')
bg.points_count
Out: 16
assert not bg.is_empty
```

Remove those same points with the tag, note the grid is now empty!

```
# remove those test1 tagged points, grid is now empty
bg.remove_points('test1')
bg.points_count
Out: 0
assert bg.is_empty
```

Same works for vr grids (variable resolution tile grids are just nested single resolution grids)

```
bg = create_grid(grid_type='variable_resolution_tile')
# add points from two multibeam lines, EPSG:26917 with vertical reference 'waterline'
bg.add_points(data, 'test1', ['line1', 'line2'], 26917, 'waterline')
bg.points_count
Out: 16
assert not bg.is_empty
bg.remove_points('test1')
bg.points_count
Out: 0
assert bg.is_empty
```

Now (after adding points), we can grid the data.  You can either pick a resolution, or let bathygrid decide for you with 
the built in depth-to-resolution lookup tables.

```
# you don't have to do this, but just to show you the lookup
from bathygrid.bgrid import depth_resolution_lookup
depth_resolution_lookup
Out: 
{20: 0.5,
 40: 1.0,
 60: 2.0,
 80: 4.0,
 160: 8.0,
 320: 16.0,
 640: 32.0,
 1280: 64.0,
 2560: 128.0,
 5120: 256.0,
 10240: 512.0,
 20480: 1024.0}

bg = create_grid(grid_type='variable_resolution_tile')
bg.add_points(data, 'test1', ['line1', 'line2'], 26917, 'waterline')
# grid by looking up the mean depth of each tile to determine resolution
bg.grid()
Out: array([0.5, 1. ])  # two resolutions used throughout grid
bg.resolutions
Out: array([0.5, 1. ])
```

And finally we can export the data to one of the accepted formats.

```
out_bag = os.path.join(bg.output_folder, 'outtiff.bag')
bg.export(out_bag, export_format='bag')

# check to see if the new bags are written
new_bag = os.path.join(bg.output_folder, 'outtiff_0.5.bag')
new_bag_two = os.path.join(bg.output_folder, 'outtiff_1.0.bag')
assert os.path.exists(new_bag)
assert os.path.exists(new_bag_two)
```
