from bathygrid.algorithms import *
from test_data.test_data import get_grid_data


def test_grid_mean():
    depth, tvu, thu, cell_indices, grid, tvugrid, thugrid = get_grid_data()
    nb_grid_mean(depth, cell_indices, grid, tvu, thu, tvugrid, thugrid)
    dpthgrid = np.round(grid, 3)
    tvugrid = np.round(tvugrid, 3)
    thugrid = np.round(thugrid, 3)
    unique_indices = np.unique(cell_indices)
    for unq in unique_indices:
        assert np.round(np.mean(depth[cell_indices == unq]), 3) == dpthgrid.flat[unq]
        assert np.round(np.mean(tvu[cell_indices == unq]), 3) == tvugrid.flat[unq]
        assert np.round(np.mean(thu[cell_indices == unq]), 3) == thugrid.flat[unq]


def test_grid_shoalest():
    depth, tvu, thu, cell_indices, grid, tvugrid, thugrid = get_grid_data()
    nb_grid_shoalest(depth, cell_indices, grid, tvu, thu, tvugrid, thugrid)
    dpthgrid = np.round(grid, 3)
    tvugrid = np.round(tvugrid, 3)
    thugrid = np.round(thugrid, 3)
    unique_indices = np.unique(cell_indices)
    for unq in unique_indices:
        min_depth_idx = depth[cell_indices == unq].argmin()
        assert np.round(depth[cell_indices == unq][min_depth_idx], 3) == dpthgrid.flat[unq]
        assert np.round(tvu[cell_indices == unq][min_depth_idx], 3) == tvugrid.flat[unq]
        assert np.round(thu[cell_indices == unq][min_depth_idx], 3) == thugrid.flat[unq]
