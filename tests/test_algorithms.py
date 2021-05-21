import numpy as np
from bathygrid.algorithms import *


def get_data():
    depth = np.linspace(10, 20, 20)
    tvu = np.linspace(1, 2, 20)
    thu = np.linspace(0.5, 1.5, 20)
    cell_indices = np.array([3, 1, 0, 2, 1, 0, 0, 7, 7, 2, 5, 4, 5, 4, 5, 6, 5, 6, 3, 3])
    grid = np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]])
    tvugrid = np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]])
    thugrid = np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]])
    return depth, tvu, thu, cell_indices, grid, tvugrid, thugrid


def test_grid_mean():
    depth, tvu, thu, cell_indices, grid, tvugrid, thugrid = get_data()
    nb_grid_mean(depth, tvu, thu, cell_indices, grid, tvugrid, thugrid)
    dpthgrid = np.round(grid, 3)
    tvugrid = np.round(tvugrid, 3)
    thugrid = np.round(thugrid, 3)
    unique_indices = np.unique(cell_indices)
    for unq in unique_indices:
        assert np.round(np.mean(depth[cell_indices == unq]), 3) == dpthgrid.flat[unq]
        assert np.round(np.mean(tvu[cell_indices == unq]), 3) == tvugrid.flat[unq]
        assert np.round(np.mean(thu[cell_indices == unq]), 3) == thugrid.flat[unq]


def test_grid_shoalest():
    depth, tvu, thu, cell_indices, grid, tvugrid, thugrid = get_data()
    nb_grid_shoalest(depth, tvu, thu, cell_indices, grid, tvugrid, thugrid)
    dpthgrid = np.round(grid, 3)
    tvugrid = np.round(tvugrid, 3)
    thugrid = np.round(thugrid, 3)
    unique_indices = np.unique(cell_indices)
    for unq in unique_indices:
        min_depth_idx = depth[cell_indices == unq].argmin()
        assert np.round(depth[cell_indices == unq][min_depth_idx], 3) == dpthgrid.flat[unq]
        assert np.round(tvu[cell_indices == unq][min_depth_idx], 3) == tvugrid.flat[unq]
        assert np.round(thu[cell_indices == unq][min_depth_idx], 3) == thugrid.flat[unq]
