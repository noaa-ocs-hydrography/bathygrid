import numba
import numpy as np


@numba.jit(nopython=True, nogil=True)
def nb_grid_mean(depth: np.array, cell_indices: np.array, grid: np.ndarray, tvu: np.array = None, thu: np.array = None, tvu_grid: np.ndarray = None, thu_grid: np.ndarray = None):
    """
    Numba version of np_grid_mean
    """

    tvu_enabled = False
    thu_enabled = False
    if tvu is not None and tvu_grid is not None:
        tvu_enabled = True
    if thu is not None and thu_grid is not None:
        thu_enabled = True

    unique_indices = np.unique(cell_indices)
    for uniq in iter(unique_indices):
        msk = cell_indices == uniq
        grid.flat[uniq] = np.mean(depth[msk])
        if tvu_enabled:
            tvu_grid.flat[uniq] = np.mean(tvu[msk])
        if thu_enabled:
            thu_grid.flat[uniq] = np.mean(thu[msk])
    return grid, tvu_grid, thu_grid


def np_grid_mean(depth: np.array, cell_indices: np.array, grid: np.ndarray, tvu: np.array = None, thu: np.array = None, tvu_grid: np.ndarray = None, thu_grid: np.ndarray = None):
    """
    Arithmetic mean of each cell depth/uncertainty point values.

    Parameters
    ----------
    depth
        1d array of point depth values
    cell_indices
        1d index of which cell each point belongs to
    grid
        empty 2d grid of depth values
    tvu
        Optional, 1d array of point vertical uncertainty values
    thu
        Optional, 1d array of point horizontal uncertainty values
    tvu_grid
        Optional, empty 2d grid of vertical uncertainty values
    thu_grid
        Optional, empty 2d grid of horizontal uncertainty values

    Returns
    -------
    np.ndarray
        empty 2d grid of depth values
    np.ndarray
        empty 2d grid of vertical uncertainty values
    np.ndarray
        empty 2d grid of horizontal uncertainty values
    """

    tvu_enabled = False
    thu_enabled = False
    if tvu is not None and tvu_grid is not None:
        tvu_enabled = True
    if thu is not None and thu_grid is not None:
        thu_enabled = True

    unique_indices = np.unique(cell_indices)
    for uniq in iter(unique_indices):
        msk = cell_indices == uniq
        grid.flat[uniq] = np.mean(depth[msk])
        if tvu_enabled:
            tvu_grid.flat[uniq] = np.mean(tvu[msk])
        if thu_enabled:
            thu_grid.flat[uniq] = np.mean(thu[msk])
    return grid, tvu_grid, thu_grid


@numba.jit(nopython=True, nogil=True)
def nb_grid_shoalest(depth: np.array, cell_indices: np.array, grid: np.ndarray, tvu: np.array = None, thu: np.array = None, tvu_grid: np.ndarray = None, thu_grid: np.ndarray = None):
    """
    Numba version of np_grid_shoalest
    """

    tvu_enabled = False
    thu_enabled = False
    if tvu is not None and tvu_grid is not None:
        tvu_enabled = True
    if thu is not None and thu_grid is not None:
        thu_enabled = True

    unique_indices = np.unique(cell_indices)
    for uniq in iter(unique_indices):
        msk = cell_indices == uniq
        min_depth_idx = depth[msk].argmin()
        grid.flat[uniq] = depth[msk][min_depth_idx]
        if tvu_enabled:
            tvu_grid.flat[uniq] = tvu[msk][min_depth_idx]
        if thu_enabled:
            thu_grid.flat[uniq] = thu[msk][min_depth_idx]
    return grid, tvu_grid, thu_grid


def np_grid_shoalest(depth: np.array, cell_indices: np.array, grid: np.ndarray, tvu: np.array = None, thu: np.array = None, tvu_grid: np.ndarray = None, thu_grid: np.ndarray = None):
    """
    Calculate the shoalest depth value of all points in each cell.  Take that depth/uncertainty value and use it for
    that grid cell.  Do that for all grid cells.

    Parameters
    ----------
    depth
        1d array of point depth values
    cell_indices
        1d index of which cell each point belongs to
    grid
        empty 2d grid of depth values
    tvu
        Optional, 1d array of point vertical uncertainty values
    thu
        Optional, 1d array of point horizontal uncertainty values
    tvu_grid
        Optional, empty 2d grid of vertical uncertainty values
    thu_grid
        Optional, empty 2d grid of horizontal uncertainty values

    Returns
    -------
    np.ndarray
        empty 2d grid of depth values
    np.ndarray
        empty 2d grid of vertical uncertainty values
    np.ndarray
        empty 2d grid of horizontal uncertainty values
    """

    tvu_enabled = False
    thu_enabled = False
    if tvu is not None and tvu_grid is not None:
        tvu_enabled = True
    if thu is not None and thu_grid is not None:
        thu_enabled = True

    unique_indices = np.unique(cell_indices)
    for uniq in iter(unique_indices):
        msk = cell_indices == uniq
        min_depth_idx = depth[msk].argmin()
        grid.flat[uniq] = depth[msk][min_depth_idx]
        if tvu_enabled:
            tvu_grid.flat[uniq] = tvu[msk][min_depth_idx]
        if thu_enabled:
            thu_grid.flat[uniq] = thu[msk][min_depth_idx]
    return grid, tvu_grid, thu_grid
