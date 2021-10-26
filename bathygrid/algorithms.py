import numba
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def np_grid_mean(depth: np.array, cell_indices: np.array, grid: np.ndarray, density_grid: np.ndarray, tvu: np.array = np.array([]),
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
    density_grid
        empty 2d grid of density values
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
    density_grid[urow, ucol] = ucounts
    if tvu_enabled:
        tvu_sum = np.add.reduceat(tvu[cell_sort], uidx, axis=0)
        tvu_mean = tvu_sum / ucounts
        tvu_grid[urow, ucol] = tvu_mean
    if thu_enabled:
        thu_sum = np.add.reduceat(thu[cell_sort], uidx, axis=0)
        thu_mean = thu_sum / ucounts
        thu_grid[urow, ucol] = thu_mean

    return grid, tvu_grid, thu_grid


def np_grid_shoalest(depth: np.array, cell_indices: np.array, grid: np.ndarray, density_grid: np.ndarray, tvu: np.array = np.array([]),
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
    density_grid
        empty 2d grid of density values
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
    density_grid[urow, ucol] = ucounts

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


def _calculate_slopes(x: np.array, y: np.array, z: np.array, cell_indices: np.array, grid_x: np.array, grid_y: np.array,
                      visualize: bool = False):
    """
    Perform least squares regression to get plane equation of best fit plane for each grid cell.  grid cells are defined
    by the provided (grid_x, grid_y) edge values.

    Optionally visualize the least squares planes using matplotlib wireframe

    Parameters
    ----------
    x
        1d x values
    y
        1d y values
    z
        1d z values
    cell_indices
        1d index of which cell each point belongs to
    grid_x
        1d x values for the grid edges
    grid_y
        1d y values for the grid edges
    visualize
        if True, plots the points and planes

    Returns
    -------
    np.ndarray
        (m,n) grid of slopes in the x direction for each cell/plane, m = len(grid_x) - 1, n = len(grid_y) - 1
    np.ndarray
        (m,n) grid of slopes in the x direction for each cell/plane, m = len(grid_x) - 1, n = len(grid_y) - 1
    """

    x_slope_grid = np.zeros((grid_x.shape[0] - 1, grid_y.shape[0] - 1), dtype=np.float32)
    y_slope_grid = np.zeros((grid_x.shape[0] - 1, grid_y.shape[0] - 1), dtype=np.float32)

    if visualize:
        plt.figure()
        ax = plt.subplot(111, projection='3d')
        ax.scatter(x, y, z, color='b')
        lstsq_grid = np.full((grid_x.shape[0], grid_y.shape[0]), np.float32(np.nan), dtype=np.float32)
        lstsq_x, lstsq_y = np.meshgrid(grid_x, grid_y)

    cell_sort = np.argsort(cell_indices)
    unique_indices, uidx, ucounts = np.unique(cell_indices[cell_sort], return_index=True, return_counts=True)
    urow, ucol = np.unravel_index(unique_indices, x_slope_grid.shape)

    z_split = np.split(z[cell_sort], uidx)[1:]
    x_split = np.split(x[cell_sort], uidx)[1:]
    y_split = np.split(y[cell_sort], uidx)[1:]

    for row, col, x_cell, y_cell, z_cell in zip(urow, ucol, x_split, y_split, z_split):
        # following equation of plane ax + by + c = z we set up the Ax = B matrices and do the least squares fit
        a_data = np.column_stack([x_cell, y_cell, np.ones(x_cell.shape[0], dtype=np.float32)])
        b_data = np.column_stack([z_cell])
        fit, residual, rnk, s = np.linalg.lstsq(a_data, b_data, rcond=None)

        minz = fit[0] * grid_x[row] + fit[1] * grid_y[row] + fit[2]
        maxz_xdirect = fit[0] * grid_x[row + 1] + fit[1] * grid_y[row] + fit[2]
        maxz_ydirect = fit[0] * grid_x[row] + fit[1] * grid_y[row + 1] + fit[2]
        x_slope_grid[row, col] = (maxz_xdirect - minz) / (grid_x[row + 1] - grid_x[row])
        y_slope_grid[row, col] = (maxz_ydirect - minz) / (grid_y[row + 1] - grid_y[row])
        if visualize:
            lstsq_grid[row, col] = fit[0] * lstsq_x[row, col] + fit[1] * lstsq_y[row, col] + fit[2]
    if visualize:
        ax.plot_wireframe(lstsq_x, lstsq_y, lstsq_grid, color='k')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
    return x_slope_grid, y_slope_grid


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