import numpy as np
from bathygrid.algorithms import *
from bathygrid.maingrid import *
import matplotlib.pyplot as plt


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

bg = SRGrid(tile_size=1024)
bg.add_points(smileyface, 'test1', ['line1', 'line2'], 26917, 'waterline')
bg.grid(resolution=64)
plt.scatter(smileyface['x'], smileyface['y'], c=smileyface['z'])
bg.plot()

plt.figure()
x, y, lyrdata, newmins, newmaxs = bg.return_surf_xyz('depth', 64.0, False)
lat2d, lon2d = np.meshgrid(y, x)
data_m = np.ma.array(lyrdata[0], mask=np.isnan(lyrdata[0]))
plt.pcolormesh(lon2d, lat2d, data_m.T, vmin=data_m.min(), vmax=data_m.max())


def trial_data():
    number_of_points = 100000
    depth = np.linspace(10, 20, number_of_points)
    vunc = np.linspace(1, 2, number_of_points)
    hunc = np.linspace(1, 2, number_of_points)
    cell_indices = np.random.randint(0, 10000, number_of_points)
    grid = np.full((100, 100), np.nan)
    vuncgrid = np.full((100, 100), np.nan)
    huncgrid = np.full((100, 100), np.nan)
    return depth, vunc, hunc, cell_indices, grid, vuncgrid, huncgrid


def trial_grid_mean_numba():
    depth, tvu, thu, cell_indices, grid, tvugrid, thugrid = trial_data()
    nb_grid_mean(depth, cell_indices, grid, tvu, thu, tvugrid, thugrid)


def trial_grid_mean_numpy():
    depth, tvu, thu, cell_indices, grid, tvugrid, thugrid = trial_data()
    np_grid_mean(depth, cell_indices, grid, tvu, thu, tvugrid, thugrid)


def trial_grid_shoal_numba():
    depth, tvu, thu, cell_indices, grid, tvugrid, thugrid = trial_data()
    nb_grid_shoalest(depth, cell_indices, grid, tvu, thu, tvugrid, thugrid)


def trial_grid_shoal_numpy():
    depth, tvu, thu, cell_indices, grid, tvugrid, thugrid = trial_data()
    np_grid_shoalest(depth, cell_indices, grid, tvu, thu, tvugrid, thugrid)


if __name__ == '__main__':
    import timeit

    trial_grid_mean_numba()  # run once to skip the overhead around import/compiling
    trial_grid_shoal_numba()  # run once to skip the overhead around import/compiling

    print('Numba mean: {}'.format(timeit.timeit(trial_grid_mean_numba, number=2)))
    print('Numpy mean: {}'.format(timeit.timeit(trial_grid_mean_numpy, number=2)))
    print('Numba shoal: {}'.format(timeit.timeit(trial_grid_shoal_numba, number=2)))
    print('Numpy shoal: {}'.format(timeit.timeit(trial_grid_shoal_numpy, number=2)))



from bathygrid.convenience import load_grid
surf = load_grid(r"C:\collab\dasktest\data_dir\EM2040c_NRT2\srgrid_mean_auto")
surf.export(r"C:\collab\dasktest\data_dir\EM2040c_NRT2\test.tif", export_format='BAG')