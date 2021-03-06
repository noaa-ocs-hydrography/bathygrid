import numpy as np
from bathygrid.tile import SRTile
from test_data.test_data import smalldata, smileyface, realdata


def test_tile_setup():
    til = SRTile(0.0, 0.0, 1024)
    assert til.is_empty

    til.add_points(smalldata, 'test1')
    assert til.data.size == 100
    assert til.container == {'test1': [0, 100]}
    assert not til.is_empty

    assert til.min_x == 0
    assert til.max_x == 1024
    assert til.min_y == 0
    assert til.max_y == 1024

    assert til.width == 1024
    assert til.height == 1024
    assert til.name == '0.0_0.0'

    assert not til.cells
    assert not til.cell_edges_x
    assert not til.cell_edges_y
    assert not til.cell_indices

    til.remove_points('test1')
    assert til.is_empty


def test_tile_newgrid():
    til = SRTile(0.0, 0.0, 1024)
    til.new_grid(8, 'mean')

    assert til.cells[8]['depth'].shape == (1024 / 8, 1024 / 8)
    assert til.cell_edges_x[8][0] == 0
    assert til.cell_edges_x[8][-1] == 1024
    assert til.cell_edges_y[8][0] == 0
    assert til.cell_edges_y[8][-1] == 1024

    assert np.isnan(til.cells[8]['depth'][0][0])
    assert til.point_count_changed is False


def test_tile_addpoints():
    til = SRTile(0.0, 0.0, 1024)
    til.add_points(smalldata, 'test1')

    assert np.array_equal(smalldata, til.data)
    assert til.container == {'test1': [0, 100]}
    assert til.point_count_changed is True


def test_tile_single_resolution():
    til = SRTile(0.0, 0.0, 1024)
    til.add_points(smalldata, 'test1')
    til.grid('mean', 128.0)

    assert til.point_count_changed is False
    assert til.cells[128.0]['depth'].shape == (1024 / 128, 1024 / 128)
    assert np.array_equal(til.cells[128.0]['depth'], np.array([[20.556, 20.707, 20.808, 20.96 , 21.111, 21.212, 21.313, 21.414],
                                                               [22.071, 22.222, 22.323, 22.475, 22.626, 22.727, 22.828, 22.929],
                                                               [23.081, 23.232, 23.333, 23.485, 23.636, 23.737, 23.838, 23.939],
                                                               [24.596, 24.747, 24.848, 25.0, 25.152, 25.253, 25.354, 25.455],
                                                               [26.111, 26.263, 26.364, 26.515, 26.667, 26.768, 26.869, 26.97 ],
                                                               [27.121, 27.273, 27.374, 27.525, 27.677, 27.778, 27.879, 27.98 ],
                                                               [28.131, 28.283, 28.384, 28.535, 28.687, 28.788, 28.889, 28.99 ],
                                                               [29.141, 29.293, 29.394, 29.545, 29.697, 29.798, 29.899, 30.0]], dtype=np.float32))
    assert np.array_equal(til.cells[128.0]['vertical_uncertainty'], np.array([[1.056, 1.071, 1.081, 1.096, 1.111, 1.121, 1.131, 1.141],
                                                                              [1.207, 1.222, 1.232, 1.247, 1.263, 1.273, 1.283, 1.293],
                                                                              [1.308, 1.323, 1.333, 1.348, 1.364, 1.374, 1.384, 1.394],
                                                                              [1.46 , 1.475, 1.485, 1.5, 1.515, 1.525, 1.535, 1.545],
                                                                              [1.611, 1.626, 1.636, 1.652, 1.667, 1.677, 1.687, 1.697],
                                                                              [1.712, 1.727, 1.737, 1.753, 1.768, 1.778, 1.788, 1.798],
                                                                              [1.813, 1.828, 1.838, 1.854, 1.869, 1.879, 1.889, 1.899],
                                                                              [1.914, 1.929, 1.939, 1.955, 1.97, 1.98, 1.99, 2.0]], dtype=np.float32))
    assert np.array_equal(til.cells[128.0]['horizontal_uncertainty'], np.array([[0.528, 0.535, 0.54, 0.548, 0.556, 0.561, 0.566, 0.571],
                                                                                [0.604, 0.611, 0.616, 0.624, 0.631, 0.636, 0.641, 0.646],
                                                                                [0.654, 0.662, 0.667, 0.674, 0.682, 0.687, 0.692, 0.697],
                                                                                [0.73, 0.737, 0.742, 0.75 , 0.758, 0.763, 0.768, 0.773],
                                                                                [0.806, 0.813, 0.818, 0.826, 0.833, 0.838, 0.843, 0.848],
                                                                                [0.856, 0.864, 0.869, 0.876, 0.884, 0.889, 0.894, 0.899],
                                                                                [0.907, 0.914, 0.919, 0.927, 0.934, 0.939, 0.944, 0.949],
                                                                                [0.957, 0.965, 0.97, 0.977, 0.985, 0.99, 0.995, 1.0]], dtype=np.float32))
    assert np.array_equal(til.cell_indices[128.0], np.array([0, 0, 1, 2, 3, 3, 4, 5, 6, 7, 0, 0, 1, 2, 3, 3, 4,
                                                             5, 6, 7, 8, 8, 9, 10, 11, 11, 12, 13, 14, 15, 16, 16, 17, 18,
                                                             19, 19, 20, 21, 22, 23, 24, 24, 25, 26, 27, 27, 28, 29, 30, 31, 24,
                                                             24, 25, 26, 27, 27, 28, 29, 30, 31, 32, 32, 33, 34, 35, 35, 36, 37,
                                                             38, 39, 40, 40, 41, 42, 43, 43, 44, 45, 46, 47, 48, 48, 49, 50, 51,
                                                             51, 52, 53, 54, 55, 56, 56, 57, 58, 59, 59, 60, 61, 62, 63]))


def test_cell_indices_modification():
    til = SRTile(0.0, 0.0, 1024)
    til.add_points(smalldata, 'test1')
    assert not til.cell_indices
    assert til.point_count_changed is True
    til.grid('mean', 128.0)
    assert til.point_count_changed is False
    assert np.array_equal(til.cell_indices[128.0], np.array([0, 0, 1, 2, 3, 3, 4, 5, 6, 7, 0, 0, 1, 2, 3, 3, 4,
                                                             5, 6, 7, 8, 8, 9, 10, 11, 11, 12, 13, 14, 15, 16, 16, 17, 18,
                                                             19, 19, 20, 21, 22, 23, 24, 24, 25, 26, 27, 27, 28, 29, 30, 31, 24,
                                                             24, 25, 26, 27, 27, 28, 29, 30, 31, 32, 32, 33, 34, 35, 35, 36, 37,
                                                             38, 39, 40, 40, 41, 42, 43, 43, 44, 45, 46, 47, 48, 48, 49, 50, 51,
                                                             51, 52, 53, 54, 55, 56, 56, 57, 58, 59, 59, 60, 61, 62, 63]))
    til.remove_points('test1')
    assert til.point_count_changed is True
    assert not til.cell_indices[128.0].any()
    til.add_points(smalldata, 'test1')
    assert np.array_equal(til.cell_indices[128.0], np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                             -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                             -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                             -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                             -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                             -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]))
    til.grid('mean', 128.0)
    assert til.point_count_changed is False
    assert np.array_equal(til.cell_indices[128.0], np.array([0, 0, 1, 2, 3, 3, 4, 5, 6, 7, 0, 0, 1, 2, 3, 3, 4,
                                                             5, 6, 7, 8, 8, 9, 10, 11, 11, 12, 13, 14, 15, 16, 16, 17, 18,
                                                             19, 19, 20, 21, 22, 23, 24, 24, 25, 26, 27, 27, 28, 29, 30, 31, 24,
                                                             24, 25, 26, 27, 27, 28, 29, 30, 31, 32, 32, 33, 34, 35, 35, 36, 37,
                                                             38, 39, 40, 40, 41, 42, 43, 43, 44, 45, 46, 47, 48, 48, 49, 50, 51,
                                                             51, 52, 53, 54, 55, 56, 56, 57, 58, 59, 59, 60, 61, 62, 63]))


def test_cell_indices_append():
    til = SRTile(0.0, 0.0, 1024)
    til.add_points(smalldata, 'test1')
    til.grid('mean', 128.0)
    assert np.array_equal(til.cell_indices[128.0], np.array([0, 0, 1, 2, 3, 3, 4, 5, 6, 7, 0, 0, 1, 2, 3, 3, 4,
                                                             5, 6, 7, 8, 8, 9, 10, 11, 11, 12, 13, 14, 15, 16, 16, 17, 18,
                                                             19, 19, 20, 21, 22, 23, 24, 24, 25, 26, 27, 27, 28, 29, 30, 31, 24,
                                                             24, 25, 26, 27, 27, 28, 29, 30, 31, 32, 32, 33, 34, 35, 35, 36, 37,
                                                             38, 39, 40, 40, 41, 42, 43, 43, 44, 45, 46, 47, 48, 48, 49, 50, 51,
                                                             51, 52, 53, 54, 55, 56, 56, 57, 58, 59, 59, 60, 61, 62, 63]))
    til.add_points(smalldata, 'test2')
    assert np.array_equal(til.cell_indices[128.0], np.array([0, 0, 1, 2, 3, 3, 4, 5, 6, 7, 0, 0, 1, 2, 3, 3, 4,
                                                             5, 6, 7, 8, 8, 9, 10, 11, 11, 12, 13, 14, 15, 16, 16, 17, 18,
                                                             19, 19, 20, 21, 22, 23, 24, 24, 25, 26, 27, 27, 28, 29, 30, 31, 24,
                                                             24, 25, 26, 27, 27, 28, 29, 30, 31, 32, 32, 33, 34, 35, 35, 36, 37,
                                                             38, 39, 40, 40, 41, 42, 43, 43, 44, 45, 46, 47, 48, 48, 49, 50, 51,
                                                             51, 52, 53, 54, 55, 56, 56, 57, 58, 59, 59, 60, 61, 62, 63,
                                                             -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                             -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                             -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                             -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                             -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                                             -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]))
    til.grid('mean', 128.0)
    assert np.array_equal(til.cell_indices[128.0], np.array([0, 0, 1, 2, 3, 3, 4, 5, 6, 7, 0, 0, 1, 2, 3, 3, 4,
                                                             5, 6, 7, 8, 8, 9, 10, 11, 11, 12, 13, 14, 15, 16, 16, 17, 18,
                                                             19, 19, 20, 21, 22, 23, 24, 24, 25, 26, 27, 27, 28, 29, 30, 31, 24,
                                                             24, 25, 26, 27, 27, 28, 29, 30, 31, 32, 32, 33, 34, 35, 35, 36, 37,
                                                             38, 39, 40, 40, 41, 42, 43, 43, 44, 45, 46, 47, 48, 48, 49, 50, 51,
                                                             51, 52, 53, 54, 55, 56, 56, 57, 58, 59, 59, 60, 61, 62, 63, 0, 0,
                                                             1, 2, 3, 3, 4, 5, 6, 7, 0, 0, 1, 2, 3, 3, 4, 5, 6,
                                                             7, 8, 8, 9, 10, 11, 11, 12, 13, 14, 15, 16, 16, 17, 18, 19, 19,
                                                             20, 21, 22, 23, 24, 24, 25, 26, 27, 27, 28, 29, 30, 31, 24, 24, 25,
                                                             26, 27, 27, 28, 29, 30, 31, 32, 32, 33, 34, 35, 35, 36, 37, 38, 39,
                                                             40, 40, 41, 42, 43, 43, 44, 45, 46, 47, 48, 48, 49, 50, 51, 51, 52,
                                                             53, 54, 55, 56, 56, 57, 58, 59, 59, 60, 61, 62, 63]))
    til.remove_points('test1')
    assert np.array_equal(til.cell_indices[128.0], np.array([0, 0, 1, 2, 3, 3, 4, 5, 6, 7, 0, 0, 1, 2, 3, 3, 4,
                                                             5, 6, 7, 8, 8, 9, 10, 11, 11, 12, 13, 14, 15, 16, 16, 17, 18,
                                                             19, 19, 20, 21, 22, 23, 24, 24, 25, 26, 27, 27, 28, 29, 30, 31, 24,
                                                             24, 25, 26, 27, 27, 28, 29, 30, 31, 32, 32, 33, 34, 35, 35, 36, 37,
                                                             38, 39, 40, 40, 41, 42, 43, 43, 44, 45, 46, 47, 48, 48, 49, 50, 51,
                                                             51, 52, 53, 54, 55, 56, 56, 57, 58, 59, 59, 60, 61, 62, 63]))


def test_geotransform():
    til = SRTile(0.0, 0.0, 1024)
    til.add_points(smalldata, 'test1')
    til.grid('mean', 128.0)
    geo = til.get_geotransform(128.0)
    assert geo == [0.0, 128.0, 0, 1024.0, 0, -128.0]


def test_get_layers_by_name_params():
    til = SRTile(0.0, 0.0, 1024)
    til.add_points(smileyface, 'test1')
    til.grid('mean', 128.0)
    layer_data = til.get_layers_by_name('depth')
    assert np.isnan(layer_data[0][0])
    assert layer_data[3][2] == np.float32(9.464)
    layer_data = til.get_layers_by_name('depth', nodatavalue=1000000)
    assert layer_data[0][0] == np.float32(1000000)
    assert layer_data[3][2] == np.float32(9.464)
    layer_data = til.get_layers_by_name('depth', nodatavalue=1000000, z_positive_up=True)
    assert layer_data[0][0] == np.float32(1000000)
    assert layer_data[3][2] == np.float32(-9.464)


def test_resolution_by_density_old():
    til = SRTile(403744.0, 4122656.0, 32)
    til.add_points(realdata, 'test1')
    assert til.resolution_by_density_old() == 8.0
    # even if you start at a different resolution, you eventually get the same answer
    assert til.resolution_by_density_old(starting_resolution=0.5) == 8.0
    # provided starter resolution must be one of the valid powers of two
    try:
        til.resolution_by_density_old(starting_resolution=666)
        assert False
    except ValueError:
        assert True


def test_resolution_by_density():
    til = SRTile(403744.0, 4122656.0, 32)
    til.add_points(realdata, 'test1')
    # with a coarse resolution we see the result of the density based estimate
    assert til.resolution_by_density() == 16.0
    # with too fine a resolution, we see it default back to the old lookup resolution by density method
    assert til.resolution_by_density(starting_resolution=0.5) == 0.5
    try:
        til.resolution_by_density(starting_resolution=666)
        assert False
    except ValueError:
        assert True


def test_density():
    til = SRTile(0.0, 0.0, 1024)
    til.add_points(smalldata, 'test1')
    til.grid('mean', 128.0)
    expected_density_count = [4, 2, 2, 4, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 4, 2, 2, 4, 2,
                              2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1,
                              1, 2, 1, 1, 1, 1]
    expected_density_permeter = [0.000244, 0.000122, 0.000122, 0.000244, 0.000122, 0.000122, 0.000122, 0.000122,
                                 0.000122, 6.1e-05, 6.1e-05, 0.000122, 6.1e-05, 6.1e-05, 6.1e-05, 6.1e-05, 0.000122,
                                 6.1e-05, 6.1e-05, 0.000122, 6.1e-05, 6.1e-05, 6.1e-05, 6.1e-05, 0.000244, 0.000122,
                                 0.000122, 0.000244, 0.000122, 0.000122, 0.000122, 0.000122, 0.000122, 6.1e-05, 6.1e-05,
                                 0.000122, 6.1e-05, 6.1e-05, 6.1e-05, 6.1e-05, 0.000122, 6.1e-05, 6.1e-05, 0.000122,
                                 6.1e-05, 6.1e-05, 6.1e-05, 6.1e-05, 0.000122, 6.1e-05, 6.1e-05, 0.000122, 6.1e-05,
                                 6.1e-05, 6.1e-05, 6.1e-05, 0.000122, 6.1e-05, 6.1e-05, 0.000122, 6.1e-05, 6.1e-05,
                                 6.1e-05, 6.1e-05]
    expected_depth_after_formatting = [20.556, 20.707, 20.808, 20.96, 21.111, 21.212, 21.313, 21.414, 22.071, 22.222,
                                       22.323, 22.475, 22.626, 22.727, 22.828, 22.929, 23.081, 23.232, 23.333, 23.485,
                                       23.636, 23.737, 23.838, 23.939, 24.596, 24.747, 24.848, 25.0, 25.152, 25.253,
                                       25.354, 25.455, 26.111, 26.263, 26.364, 26.515, 26.667, 26.768, 26.869, 26.97,
                                       27.121, 27.273, 27.374, 27.525, 27.677, 27.778, 27.879, 27.98, 28.131, 28.283,
                                       28.384, 28.535, 28.687, 28.788, 28.889, 28.99, 29.141, 29.293, 29.394, 29.545,
                                       29.697, 29.798, 29.899, 30.0]
    assert til.density_count == expected_density_count
    assert np.array(til.density_per_square_meter).round(6).tolist() == expected_density_permeter
    count, depth = til.density_count_vs_depth
    assert count == expected_density_count
    assert np.round(np.array(depth), 3).tolist() == expected_depth_after_formatting
    permeter, depth = til.density_per_square_meter_vs_depth
    assert np.array(permeter).round(6).tolist() == expected_density_permeter
    assert np.round(np.array(depth), 3).tolist() == expected_depth_after_formatting


def test_coverage_area():
    til = SRTile(0.0, 0.0, 1024)
    til.add_points(smalldata, 'test1')
    til.grid('mean', 128.0)
    assert til.coverage_area_square_meters == til.width * til.height
    assert round(til.coverage_area_square_nm, 3) == 0.305
