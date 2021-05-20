import numpy as np
from bathygrid.algorithms import *


def trial_data():
    number_of_points = 100000
    depth = np.linspace(10, 20, number_of_points)
    unc = np.linspace(1, 2, number_of_points)
    cell_indices = np.random.randint(0, 10000, number_of_points)
    grid = np.full((100, 100), np.nan)
    uncgrid = np.full((100, 100), np.nan)
    return depth, unc, cell_indices, grid, uncgrid


def trial_grid_mean_numba():
    depth, unc, cell_indices, grid, uncgrid = trial_data()
    nb_grid_mean(depth, unc, cell_indices, grid, uncgrid)


def trial_grid_mean_numpy():
    depth, unc, cell_indices, grid, uncgrid = trial_data()
    np_grid_mean(depth, unc, cell_indices, grid, uncgrid)


if __name__ == '__main__':
    import timeit

    trial_grid_mean_numba()  # run once to skip the overhead around import/compiling

    print('Numba mean: {}'.format(timeit.timeit(trial_grid_mean_numba, number=2)))
    print('Numpy mean: {}'.format(timeit.timeit(trial_grid_mean_numpy, number=2)))

