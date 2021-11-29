import numpy as np

from bathygrid.utilities import is_power_of_two, bin2d_with_indices
from test_data.test_data import realdata


def test_bin2d_with_indices():
    resolution = 2
    x, y, z, tvu, thu = realdata['x'], realdata['y'], realdata['z'], realdata['tvu'], realdata['thu']
    cell_edges_x = np.arange(403734, 403790, resolution)
    cell_edges_y = np.arange(4122656, 4122716, resolution)
    cell_indices = bin2d_with_indices(x, y, cell_edges_x, cell_edges_y)

    grid_shape = (cell_edges_y.shape[0] - 1, cell_edges_x.shape[0] - 1)
    ucol, urow = np.unravel_index(cell_indices, grid_shape)
    assert (np.abs(cell_edges_x[urow] - x) < resolution).all()
    assert (np.abs(cell_edges_y[ucol] - y) < resolution).all()
