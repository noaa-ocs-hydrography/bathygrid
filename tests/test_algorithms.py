import numpy as np
from bathygrid.algorithms import *


def get_data():
    depth = np.linspace(10, 20, 20)
    unc = np.linspace(1, 2, 20)
    cell_indices = np.array([3, 1, 0, 2, 1, 0, 0, 7, 7, 2, 5, 4, 5, 4, 5, 6, 5, 6, 3, 3])
    grid = np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]])
    uncgrid = np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]])
    return depth, unc, cell_indices, grid, uncgrid


def test_grid_mean():
    depth, unc, cell_indices, grid, uncgrid = get_data()
    dpthgrid, uncgrid = nb_grid_mean(depth, unc, cell_indices, grid, uncgrid)
