import numba
import numpy as np


def np_grid_mean(depth: np.array, cell_indices: np.array, grid: np.ndarray, tvu: np.array = np.array([]),
                 thu: np.array = np.array([]), tvu_grid: np.ndarray = np.array([[]]), thu_grid: np.ndarray = np.array([[]])):
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
    if tvu.size > 0 and tvu_grid.size > 0:
        tvu_enabled = True
    if thu.size > 0 and thu_grid.size > 0:
        thu_enabled = True

    unique_indices = np.unique(cell_indices)
    for uniq in iter(unique_indices):
        msk = cell_indices == uniq
        # this is required for numba to work
        #urow, ucol = my_unravel_index(uniq, grid.shape)
        urow, ucol = np.unravel_index(uniq, grid.shape)
        grid[urow, ucol] = np.mean(depth[msk])
        if tvu_enabled:
            tvu_grid[urow, ucol] = np.mean(tvu[msk])
        if thu_enabled:
            thu_grid[urow, ucol] = np.mean(thu[msk])
    return grid, tvu_grid, thu_grid


def np_grid_shoalest(depth: np.array, cell_indices: np.array, grid: np.ndarray, tvu: np.array = np.array([]),
                     thu: np.array = np.array([]), tvu_grid: np.ndarray = np.array([]), thu_grid: np.ndarray = np.array([])):
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
    if tvu.size > 0 and tvu_grid.size > 0:
        tvu_enabled = True
    if thu.size > 0 and thu_grid.size > 0:
        thu_enabled = True

    unique_indices = np.unique(cell_indices)
    for uniq in iter(unique_indices):
        msk = cell_indices == uniq
        # this is required for numba to work
        # urow, ucol = my_unravel_index(uniq, grid.shape)
        urow, ucol = np.unravel_index(uniq, grid.shape)
        min_depth_idx = depth[msk].argmin()
        grid[urow, ucol] = depth[msk][min_depth_idx]
        if tvu_enabled:
            tvu_grid[urow, ucol] = tvu[msk][min_depth_idx]
        if thu_enabled:
            thu_grid[urow, ucol] = thu[msk][min_depth_idx]
    return grid, tvu_grid, thu_grid


@numba.njit
def my_unravel_index(index, shape):
    """
    No current implementation for numpy unravel_index in numba, I found this mostly on the numba gitter.
    """
    sizes = np.zeros(len(shape), dtype=np.int64)
    result = np.zeros(len(shape), dtype=np.int64)
    sizes[-1] = 1
    for i in range(len(shape) - 2, -1, -1):
        sizes[i] = sizes[i + 1] * shape[i + 1]
    remainder = index
    for i in range(len(shape)):
        result[i] = remainder // sizes[i]
        remainder %= sizes[i]
    return result

# # numba-ized version of np_grid_mean
# nb_grid_mean = numba.jit(nopython=True, nogil=True, cache=True)(np_grid_mean)
# # precompile the algorithm to avoid slowdown on the first run
# nb_grid_mean(np.array([1]), np.array([0]), np.array([[np.nan]]))
#
# # numba-ized version of np_grid_shoalest
# nb_grid_shoalest = numba.jit(nopython=True, nogil=True, cache=True)(np_grid_shoalest)
# # precompile the algorithm to avoid slowdown on the first run
# nb_grid_shoalest(np.array([1]), np.array([0]), np.array([[np.nan]]))