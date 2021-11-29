from bathygrid.algorithms import *
from bathygrid.utilities import is_power_of_two, bin2d_with_indices
from test_data.test_data import get_grid_data, realdata
from pytest import approx


def test_grid_mean():
    depth, tvu, thu, cell_indices, grid, density_grid, tvugrid, thugrid = get_grid_data()
    np_grid_mean(depth, cell_indices, grid, density_grid, tvu, thu, tvugrid, thugrid)
    dpthgrid = np.round(grid, 3)
    tvugrid = np.round(tvugrid, 3)
    thugrid = np.round(thugrid, 3)
    unique_indices, ucounts = np.unique(cell_indices, return_counts=True)
    for unq, ucnt in zip(unique_indices, ucounts):
        assert np.round(np.mean(depth[cell_indices == unq]), 3) == dpthgrid.flat[unq]
        assert np.round(np.mean(tvu[cell_indices == unq]), 3) == tvugrid.flat[unq]
        assert np.round(np.mean(thu[cell_indices == unq]), 3) == thugrid.flat[unq]
        assert density_grid.flat[unq] == ucnt


def test_grid_mean_onlydepth():
    depth, tvu, thu, cell_indices, grid, density_grid, tvugrid, thugrid = get_grid_data()
    np_grid_mean(depth, cell_indices, grid, density_grid)
    dpthgrid = np.round(grid, 3)
    unique_indices, ucounts = np.unique(cell_indices, return_counts=True)
    for unq, ucnt in zip(unique_indices, ucounts):
        assert np.round(np.mean(depth[cell_indices == unq]), 3) == dpthgrid.flat[unq]
        assert density_grid.flat[unq] == ucnt


# def test_grid_numba_numpy_mean():
#     depth, tvu, thu, cell_indices, grid, tvugrid, thugrid = get_grid_data()
#     nb_grid_mean(depth, cell_indices, grid, tvu, thu, tvugrid, thugrid)
#     dpthgrid = np.round(grid, 3)
#     tvugrid = np.round(tvugrid, 3)
#     thugrid = np.round(thugrid, 3)
#
#     npdepth, nptvu, npthu, npcell_indices, npgrid, nptvugrid, npthugrid = get_grid_data()
#     np_grid_mean(npdepth, npcell_indices, npgrid, nptvu, npthu, nptvugrid, npthugrid)
#     npdpthgrid = np.round(npgrid, 3)
#     nptvugrid = np.round(nptvugrid, 3)
#     npthugrid = np.round(npthugrid, 3)
#
#     unique_indices = np.unique(cell_indices)
#     for unq in unique_indices:
#         assert npdpthgrid.flat[unq] == dpthgrid.flat[unq]
#         assert nptvugrid.flat[unq] == tvugrid.flat[unq]
#         assert npthugrid.flat[unq] == thugrid.flat[unq]


def test_grid_shoalest():
    depth, tvu, thu, cell_indices, grid, density_grid, tvugrid, thugrid = get_grid_data()
    np_grid_shoalest(depth, cell_indices, grid, density_grid, tvu, thu, tvugrid, thugrid)
    dpthgrid = np.round(grid, 3)
    tvugrid = np.round(tvugrid, 3)
    thugrid = np.round(thugrid, 3)
    unique_indices, ucounts = np.unique(cell_indices, return_counts=True)
    for unq, ucnt in zip(unique_indices, ucounts):
        min_depth_idx = depth[cell_indices == unq].argmin()
        assert np.round(depth[cell_indices == unq][min_depth_idx], 3) == dpthgrid.flat[unq]
        assert np.round(tvu[cell_indices == unq][min_depth_idx], 3) == tvugrid.flat[unq]
        assert np.round(thu[cell_indices == unq][min_depth_idx], 3) == thugrid.flat[unq]
        assert density_grid.flat[unq] == ucnt


def test_grid_shoalest_onlydepth():
    depth, tvu, thu, cell_indices, grid, density_grid, tvugrid, thugrid = get_grid_data()
    np_grid_shoalest(depth, cell_indices, grid, density_grid)
    dpthgrid = np.round(grid, 3)
    unique_indices, ucounts = np.unique(cell_indices, return_counts=True)
    for unq, ucnt in zip(unique_indices, ucounts):
        min_depth_idx = depth[cell_indices == unq].argmin()
        assert np.round(depth[cell_indices == unq][min_depth_idx], 3) == dpthgrid.flat[unq]
        assert density_grid.flat[unq] == ucnt


# def test_grid_numba_numpy_shoalest():
#     depth, tvu, thu, cell_indices, grid, tvugrid, thugrid = get_grid_data()
#     nb_grid_shoalest(depth, cell_indices, grid, tvu, thu, tvugrid, thugrid)
#     dpthgrid = np.round(grid, 3)
#     tvugrid = np.round(tvugrid, 3)
#     thugrid = np.round(thugrid, 3)
#
#     npdepth, nptvu, npthu, npcell_indices, npgrid, nptvugrid, npthugrid = get_grid_data()
#     np_grid_shoalest(npdepth, npcell_indices, npgrid, nptvu, npthu, nptvugrid, npthugrid)
#     npdpthgrid = np.round(npgrid, 3)
#     nptvugrid = np.round(nptvugrid, 3)
#     npthugrid = np.round(npthugrid, 3)
#
#     unique_indices = np.unique(cell_indices)
#     for unq in unique_indices:
#         assert npdpthgrid.flat[unq] == dpthgrid.flat[unq]
#         assert nptvugrid.flat[unq] == tvugrid.flat[unq]
#         assert npthugrid.flat[unq] == thugrid.flat[unq]


def test_is_power_of_two():
    assert is_power_of_two(2**50)
    assert is_power_of_two(2**5)
    assert is_power_of_two(4)
    assert is_power_of_two(2)
    assert is_power_of_two(1)
    assert is_power_of_two(0.5)
    assert is_power_of_two(0.25)
    assert is_power_of_two(2 ** -3)
    assert is_power_of_two(2 ** -5)
    assert is_power_of_two(2 ** -8)
    assert is_power_of_two(2 ** -10)

    assert not is_power_of_two(1.99 ** 50)
    assert not is_power_of_two(2.01 ** 5)
    assert not is_power_of_two(4.104)
    assert not is_power_of_two(2.124)
    assert not is_power_of_two(1.00001)
    assert not is_power_of_two(0.5124)
    assert not is_power_of_two(0.2514512)
    assert not is_power_of_two(2.01 ** -3)
    assert not is_power_of_two(2.01 ** -5)
    assert not is_power_of_two(1.99 ** -8)
    assert not is_power_of_two(1.99 ** -10)


def test_grid_slopes():
    x, y, z, tvu, thu = realdata['x'], realdata['y'], realdata['z'], realdata['tvu'], realdata['thu']
    cell_edges_x = np.arange(403734, 403790, 8)
    cell_edges_y = np.arange(4122656, 4122712, 8)
    cell_indices = bin2d_with_indices(x, y, cell_edges_x, cell_edges_y)
    slpx, slpy = calculate_slopes(x, y, z, cell_indices, cell_edges_x, cell_edges_y, visualize=False)

    assert slpx[1][1] == approx(0.01538095)
    assert slpx[1][2] == approx(-0.010140932)
    assert slpx[1][3] == approx(-0.011870318)
    assert slpx[2][1] == approx(0.005293401)
    assert slpx[2][2] == approx(-0.006488629)
    assert slpx[2][3] == approx(-0.013622731)

    assert slpy[1][1] == approx(-0.00029286137)
    assert slpy[1][2] == approx(0.004407489)
    assert slpy[1][3] == approx(-0.0002697181)
    assert slpy[2][1] == approx(0.00396455)
    assert slpy[2][2] == approx(-0.002549838)
    assert slpy[2][3] == approx(-0.0002785997)

    assert slpx.shape == (cell_edges_x.shape[0] - 1, cell_edges_y.shape[0] - 1)
    assert slpy.shape == (cell_edges_x.shape[0] - 1, cell_edges_y.shape[0] - 1)
