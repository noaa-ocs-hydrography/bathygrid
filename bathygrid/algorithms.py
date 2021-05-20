import numba
import numpy as np


@numba.jit(nopython=True, nogil=True)
def nb_grid_mean(depth: np.array, uncertainty: np.array, cell_indices: np.array, grid: np.ndarray, uncertainty_grid: np.ndarray):
    unique_indices = np.unique(cell_indices)
    for uniq in iter(unique_indices):
        msk = cell_indices == uniq
        idx = cell_indices[msk]
        grid.flat[uniq] = np.mean(depth[idx])
        uncertainty_grid.flat[uniq] = np.mean(uncertainty[idx])
    return grid, uncertainty_grid


def np_grid_mean(depth: np.array, uncertainty: np.array, cell_indices: np.array, grid: np.ndarray, uncertainty_grid: np.ndarray):
    unique_indices = np.unique(cell_indices)
    for uniq in iter(unique_indices):
        msk = cell_indices == uniq
        idx = cell_indices[msk]
        grid.flat[uniq] = np.mean(depth[idx])
        uncertainty_grid.flat[uniq] = np.mean(uncertainty[idx])
    return grid, uncertainty_grid

