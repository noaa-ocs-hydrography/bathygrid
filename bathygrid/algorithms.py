import numba
import numpy as np


@numba.jit(nopython=True, nogil=True)
def nb_grid_mean(depth: np.array, tvu: np.array, thu: np.array, cell_indices: np.array, grid: np.ndarray, tvu_grid: np.ndarray, thu_grid: np.ndarray):
    unique_indices = np.unique(cell_indices)
    for uniq in iter(unique_indices):
        msk = cell_indices == uniq
        grid.flat[uniq] = np.mean(depth[msk])
        tvu_grid.flat[uniq] = np.mean(tvu[msk])
        thu_grid.flat[uniq] = np.mean(thu[msk])
    return grid, tvu_grid, thu_grid


def np_grid_mean(depth: np.array, tvu: np.array, thu: np.array, cell_indices: np.array, grid: np.ndarray, tvu_grid: np.ndarray, thu_grid: np.ndarray):
    unique_indices = np.unique(cell_indices)
    for uniq in iter(unique_indices):
        msk = cell_indices == uniq
        grid.flat[uniq] = np.mean(depth[msk])
        tvu_grid.flat[uniq] = np.mean(tvu[msk])
        thu_grid.flat[uniq] = np.mean(thu[msk])
    return grid, tvu_grid, thu_grid


@numba.jit(nopython=True, nogil=True)
def nb_grid_shoalest(depth: np.array, tvu: np.array, thu: np.array, cell_indices: np.array, grid: np.ndarray, tvu_grid: np.ndarray, thu_grid: np.ndarray):
    unique_indices = np.unique(cell_indices)
    for uniq in iter(unique_indices):
        msk = cell_indices == uniq
        min_depth_idx = depth[msk].argmin()
        grid.flat[uniq] = depth[msk][min_depth_idx]
        tvu_grid.flat[uniq] = tvu[msk][min_depth_idx]
        thu_grid.flat[uniq] = thu[msk][min_depth_idx]
    return grid, tvu_grid, thu_grid


def np_grid_shoalest(depth: np.array, tvu: np.array, thu: np.array, cell_indices: np.array, grid: np.ndarray, tvu_grid: np.ndarray, thu_grid: np.ndarray):
    unique_indices = np.unique(cell_indices)
    for uniq in iter(unique_indices):
        msk = cell_indices == uniq
        min_depth_idx = depth[msk].argmin()
        grid.flat[uniq] = depth[msk][min_depth_idx]
        tvu_grid.flat[uniq] = tvu[msk][min_depth_idx]
        thu_grid.flat[uniq] = thu[msk][min_depth_idx]
    return grid, tvu_grid, thu_grid
