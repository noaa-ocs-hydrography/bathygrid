import numba
import numpy as np


@numba.jit(nopython=True, nogil=True)
def grid_mean(depth: np.array, uncertainty: np.array, cell_indices: np.array, grid: np.ndarray, uncertainty_grid: np.ndarray):
    unique_indices = np.unique(cell_indices)
    flatgrid = grid.ravel()
    flatunc = uncertainty_grid.ravel()
    for uniq in unique_indices:
        msk = cell_indices == uniq
        flatgrid[uniq] = np.mean(depth[cell_indices[msk]])
        flatunc[uniq] = np.mean(uncertainty[cell_indices[msk]])
    return flatgrid.reshape(grid.shape), flatunc.reshape(grid.shape)
