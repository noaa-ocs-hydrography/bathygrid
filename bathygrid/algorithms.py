import numba
import numpy as np


@numba.jit(nopython=True, nogil=True)
def nb_grid_mean(depth: np.array, tvu: np.array, thu: np.array, cell_indices: np.array, grid: np.ndarray, tvu_grid: np.ndarray, thu_grid: np.ndarray):
    """
    Numba version of np_grid_mean
    """

    unique_indices = np.unique(cell_indices)
    for uniq in iter(unique_indices):
        msk = cell_indices == uniq
        grid.flat[uniq] = np.mean(depth[msk])
        tvu_grid.flat[uniq] = np.mean(tvu[msk])
        thu_grid.flat[uniq] = np.mean(thu[msk])
    return grid, tvu_grid, thu_grid


def np_grid_mean(depth: np.array, tvu: np.array, thu: np.array, cell_indices: np.array, grid: np.ndarray, tvu_grid: np.ndarray, thu_grid: np.ndarray):
    """
    Arithmetic mean of each cell depth/uncertainty point values.

    Parameters
    ----------
    depth
        1d array of point depth values
    tvu
        1d array of point vertical uncertainty values
    thu
        1d array of point horizontal uncertainty values
    cell_indices
        1d index of which cell each point belongs to
    grid
        empty 2d grid of depth values
    tvu_grid
        empty 2d grid of vertical uncertainty values
    thu_grid
        empty 2d grid of horizontal uncertainty values

    Returns
    -------
    np.ndarray
        empty 2d grid of depth values
    np.ndarray
        empty 2d grid of vertical uncertainty values
    np.ndarray
        empty 2d grid of horizontal uncertainty values
    """

    unique_indices = np.unique(cell_indices)
    for uniq in iter(unique_indices):
        msk = cell_indices == uniq
        grid.flat[uniq] = np.mean(depth[msk])
        tvu_grid.flat[uniq] = np.mean(tvu[msk])
        thu_grid.flat[uniq] = np.mean(thu[msk])
    return grid, tvu_grid, thu_grid


@numba.jit(nopython=True, nogil=True)
def nb_grid_shoalest(depth: np.array, tvu: np.array, thu: np.array, cell_indices: np.array, grid: np.ndarray, tvu_grid: np.ndarray, thu_grid: np.ndarray):
    """
    Numba version of np_grid_shoalest
    """

    unique_indices = np.unique(cell_indices)
    for uniq in iter(unique_indices):
        msk = cell_indices == uniq
        min_depth_idx = depth[msk].argmin()
        grid.flat[uniq] = depth[msk][min_depth_idx]
        tvu_grid.flat[uniq] = tvu[msk][min_depth_idx]
        thu_grid.flat[uniq] = thu[msk][min_depth_idx]
    return grid, tvu_grid, thu_grid


def np_grid_shoalest(depth: np.array, tvu: np.array, thu: np.array, cell_indices: np.array, grid: np.ndarray, tvu_grid: np.ndarray, thu_grid: np.ndarray):
    """
    Calculate the shoalest depth value of all points in each cell.  Take that depth/uncertainty value and use it for
    that grid cell.  Do that for all grid cells.

    Parameters
    ----------
    depth
        1d array of point depth values
    tvu
        1d array of point vertical uncertainty values
    thu
        1d array of point horizontal uncertainty values
    cell_indices
        1d index of which cell each point belongs to
    grid
        empty 2d grid of depth values
    tvu_grid
        empty 2d grid of vertical uncertainty values
    thu_grid
        empty 2d grid of horizontal uncertainty values

    Returns
    -------
    np.ndarray
        empty 2d grid of depth values
    np.ndarray
        empty 2d grid of vertical uncertainty values
    np.ndarray
        empty 2d grid of horizontal uncertainty values
    """

    unique_indices = np.unique(cell_indices)
    for uniq in iter(unique_indices):
        msk = cell_indices == uniq
        min_depth_idx = depth[msk].argmin()
        grid.flat[uniq] = depth[msk][min_depth_idx]
        tvu_grid.flat[uniq] = tvu[msk][min_depth_idx]
        thu_grid.flat[uniq] = thu[msk][min_depth_idx]
    return grid, tvu_grid, thu_grid
