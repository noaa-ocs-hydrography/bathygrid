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

    cell_sort = np.argsort(cell_indices)
    unique_indices, uidx, ucounts = np.unique(cell_indices[cell_sort], return_index=True, return_counts=True)
    urow, ucol = np.unravel_index(unique_indices, grid.shape)

    depth_sum = np.add.reduceat(depth[cell_sort], uidx, axis=0)
    depth_mean = depth_sum / ucounts
    grid[urow, ucol] = depth_mean
    if tvu_enabled:
        tvu_sum = np.add.reduceat(tvu[cell_sort], uidx, axis=0)
        tvu_mean = tvu_sum / ucounts
        tvu_grid[urow, ucol] = tvu_mean
    if thu_enabled:
        thu_sum = np.add.reduceat(thu[cell_sort], uidx, axis=0)
        thu_mean = thu_sum / ucounts
        thu_grid[urow, ucol] = thu_mean

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

    cell_sort = np.argsort(cell_indices)
    unique_indices, uidx, ucounts = np.unique(cell_indices[cell_sort], return_index=True, return_counts=True)
    urow, ucol = np.unravel_index(unique_indices, grid.shape)

    depth_min = np.minimum.reduceat(depth[cell_sort], uidx, axis=0)
    grid[urow, ucol] = depth_min

    depth_idx = None
    if tvu_enabled or thu_enabled:
        depth_sort = np.argsort(depth)
        depth_idx = np.searchsorted(depth[depth_sort], depth_min)
    if tvu_enabled:
        tvu_shoalest = tvu[depth_idx]
        tvu_grid[urow, ucol] = tvu_shoalest
    if thu_enabled:
        thu_shoalest = thu[depth_idx]
        thu_grid[urow, ucol] = thu_shoalest

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