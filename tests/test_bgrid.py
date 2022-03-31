import numpy as np
from pytest import approx

from bathygrid.bgrid import *
from bathygrid.maingrid import *
from bathygrid.tile import Tile
from test_data.test_data import smalldata2, smalldata3, deepdata, closedata, smileyface, onlyzdata, realdata
from bathygrid.utilities import utc_seconds_to_formatted_string


def test_SRGrid_setup():
    bg = SRGrid(tile_size=1024)
    assert bg.data is None
    assert bg.container == {}
    assert bg.container_timestamp == {}
    assert bg.epsg is None
    assert bg.vertical_reference is None


def test_SRGrid_add_points():
    bg = SRGrid(tile_size=1024)
    bg.add_points(smalldata2, 'test1', ['line1', 'line2'], 26917, 'waterline')
    assert not bg.is_empty
    assert bg.data is None  # after adding we clear the point data to free memory
    assert len(bg.tiles) == 6
    assert len(bg.tiles[0]) == 5
    for row in bg.tiles:
        for til in row:
            assert isinstance(til, Tile)
    assert bg.container == {'test1': ['line1', 'line2']}
    assert 'test1' in bg.container_timestamp
    assert bg.vertical_reference == 'waterline'
    assert bg.epsg == 26917

    tile = bg.tiles[0][0]
    assert not tile.is_empty
    assert tile.data.size == 22
    assert tile.points_count == 22
    assert tile.width == 1024
    assert tile.height == 1024
    assert tile.data['x'][0] == 0.0
    assert tile.data['y'][0] == 50000.0
    assert tile.data['z'][0] == approx(20.000)
    assert tile.min_x == 0.0
    assert tile.max_x == 1024.0
    assert tile.min_y == 49152.0
    assert tile.max_y == 50176.0
    assert tile.name == '0.0_49152.0'

    flat_tiles = bg.tiles.ravel()
    for tile in flat_tiles:
        if tile:
            points = tile.data
            for point in points:
                assert point['x'] >= tile.min_x
                assert point['x'] < tile.max_x
                assert point['y'] >= tile.min_y
                assert point['y'] < tile.max_y


def test_SRGrid_remove_points():
    bg = SRGrid(tile_size=1024)
    bg.add_points(smalldata2, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.remove_points('test1')

    assert bg.data is None
    assert bg.container == {}
    assert bg.container_timestamp == {}
    assert bg.tiles is None


def test_SRGrid_add_multiple_sources():
    bg = SRGrid(tile_size=1024)
    bg.add_points(smalldata2, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.add_points(smalldata3, 'test2', ['line3', 'line4'], 26917, 'waterline')

    # test2 had no points for this tile, should match the other add points test
    tile = bg.tiles[0][0]
    assert not tile.is_empty
    assert tile.data.size == 22
    assert tile.points_count == 22
    assert tile.width == 1024
    assert tile.height == 1024
    assert tile.data['x'][0] == 0.0
    assert tile.data['y'][0] == 50000.0
    assert tile.data['z'][0] == approx(20.000)
    assert tile.min_x == 0.0
    assert tile.max_x == 1024.0
    assert tile.min_y == 49152.0
    assert tile.max_y == 50176.0
    assert tile.name == '0.0_49152.0'
    assert tile.container == {'test1': [0, 22]}
    assert 'test1' in bg.container_timestamp

    # this tile has points from both containers
    tile = bg.tiles[3][3]
    assert not tile.is_empty
    assert tile.data.size == 200
    assert tile.points_count == 200
    assert tile.width == 1024
    assert tile.height == 1024
    assert tile.data['x'][0] == 3100.0
    assert tile.data['y'][0] == 52300.0
    assert tile.data['z'][0] == approx(24.725891)
    assert tile.min_x == 3072.0
    assert tile.max_x == 4096.0
    assert tile.min_y == 52224.0
    assert tile.max_y == 53248.0
    assert tile.name == '3072.0_52224.0'
    assert tile.container == {'test1': [0, 100], 'test2': [100, 200]}
    assert 'test1' in bg.container_timestamp
    assert 'test2' in bg.container_timestamp

    # removing points from this container will remove all test2 points from all tiles
    bg.remove_points('test2')

    tile = bg.tiles[3][3]
    assert not tile.is_empty
    assert tile.data.size == 100
    assert tile.points_count == 100
    assert tile.width == 1024
    assert tile.height == 1024
    assert tile.data['x'][0] == 3100.0
    assert tile.data['y'][0] == 52300.0
    assert tile.data['z'][0] == approx(24.725891)
    assert tile.min_x == 3072.0
    assert tile.max_x == 4096.0
    assert tile.min_y == 52224.0
    assert tile.max_y == 53248.0
    assert tile.name == '3072.0_52224.0'
    assert tile.container == {'test1': [0, 100]}
    assert 'test1' in bg.container_timestamp


def test_SRGrid_point_count_changed():
    bg = SRGrid(tile_size=1024)
    assert not bg.point_count_changed
    bg.add_points(smalldata2, 'test1', ['line1', 'line2'], 26917, 'waterline')
    assert bg.point_count_changed
    res = bg.grid(algorithm='mean')
    assert not bg.point_count_changed
    bg.add_points(smalldata3, 'test2', ['line3', 'line4'], 26917, 'waterline')
    assert bg.point_count_changed


def test_SRGrid_update():
    bg = SRGrid(tile_size=1024)
    bg.add_points(smalldata2, 'test1', ['line1', 'line2'], 26917, 'waterline')
    res = bg.grid(algorithm='mean')
    bg.add_points(smalldata3, 'test2', ['line3', 'line4'], 26917, 'waterline')
    assert bg.point_count_changed
    res = bg.grid(algorithm='mean', regrid_option='update')
    assert not bg.point_count_changed


def test_SRGrid_grid_mean():
    bg = SRGrid(tile_size=1024)
    bg.add_points(smalldata2, 'test1', ['line1', 'line2'], 26917, 'waterline')
    assert not bg.is_empty
    assert bg.no_grid

    res = bg.grid(algorithm='mean')
    assert not bg.no_grid
    assert res == [1.0]
    assert bg.data is None  # after adding we clear the point data to free memory
    lyrs = bg.get_layers_by_name(['depth', 'density', 'horizontal_uncertainty', 'vertical_uncertainty'])
    assert lyrs[0].size == lyrs[1].size == lyrs[2].size == lyrs[3].size == 31457280
    assert np.count_nonzero(~np.isnan(lyrs[0])) == np.count_nonzero(lyrs[1]) == np.count_nonzero(~np.isnan(lyrs[2])) == np.count_nonzero(~np.isnan(lyrs[3])) == 2500
    assert lyrs[0][848, 0] == 20.0
    assert lyrs[1][848, 0] == 1
    assert lyrs[2][848, 0] == 0.5
    assert lyrs[3][848, 0] == 1.0


def test_SRGrid_grid_shoalest():
    bg = SRGrid(tile_size=1024)
    bg.add_points(smalldata2, 'test1', ['line1', 'line2'], 26917, 'waterline')
    assert not bg.is_empty
    assert bg.no_grid

    res = bg.grid(algorithm='shoalest')
    assert not bg.no_grid
    assert res == [1.0]
    assert bg.data is None  # after adding we clear the point data to free memory
    lyrs = bg.get_layers_by_name(['depth', 'density', 'horizontal_uncertainty', 'vertical_uncertainty'])
    assert lyrs[0].size == lyrs[1].size == lyrs[2].size == lyrs[3].size == 31457280
    assert np.count_nonzero(~np.isnan(lyrs[0])) == np.count_nonzero(lyrs[1]) == np.count_nonzero(~np.isnan(lyrs[2])) == np.count_nonzero(~np.isnan(lyrs[3])) == 2500
    assert lyrs[0][848, 0] == 20.0
    assert lyrs[1][848, 0] == 1
    assert lyrs[2][848, 0] == 0.5
    assert lyrs[3][848, 0] == 1.0


def test_SRGrid_grid_cube():
    bg = SRGrid(tile_size=1024)
    bg.add_points(realdata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    assert not bg.is_empty
    assert bg.no_grid

    res = bg.grid(algorithm='cube', grid_parameters={'variance_selection': 'cube', 'method': 'local', 'iho_order': 'order1a'})
    assert not bg.no_grid
    assert res == [0.5]
    assert bg.data is None  # after adding we clear the point data to free memory
    lyrs = bg.get_layers_by_name(['depth', 'density', 'total_uncertainty', 'hypothesis_count', 'hypothesis_ratio'])
    assert lyrs[0].size == lyrs[1].size == lyrs[2].size == lyrs[3].size == lyrs[4].size == 4194304
    assert np.count_nonzero(~np.isnan(lyrs[0])) == np.count_nonzero(~np.isnan(lyrs[2])) == np.count_nonzero(lyrs[3]) == np.count_nonzero(~np.isnan(lyrs[4])) == 0
    assert lyrs[0][848, 0] == 20.0
    assert lyrs[1][848, 0] == 1
    assert lyrs[2][848, 0] == 0.5
    assert lyrs[3][848, 0] == 1.0


def test_auto_resolution_methods():
    bg = SRGrid(tile_size=1024)
    bg.add_points(smalldata2, 'test1', ['line1', 'line2'], 26917, 'waterline')
    assert not bg.is_empty
    assert bg.no_grid

    # start with the basic lookup-the-mean-depth-to-get-resolution method
    res = bg.grid(resolution=None, algorithm='mean', auto_resolution_mode='depth')
    assert bg.mean_depth == 25.0
    assert depth_resolution_lookup[20] == 0.5
    assert depth_resolution_lookup[40] == 1.0
    assert res == bg.resolutions == [1.0]
    # now illustrate the more complex density based method
    res = bg.grid(resolution=None, algorithm='mean', auto_resolution_mode='density')
    assert res == bg.resolutions == [1.0]


def test_SRGrid_get_layer_by_name():
    bg = SRGrid(tile_size=1024)
    bg.add_points(smalldata2, 'test1', ['line1', 'line2'], 26917, 'waterline')
    assert not bg.is_empty
    assert bg.no_grid

    res = bg.grid()
    assert not bg.no_grid
    assert res == [1.0]
    assert bg.data is None  # after adding we clear the point data to free memory
    lyrs = bg.get_layers_by_name(['depth', 'density', 'horizontal_uncertainty', 'vertical_uncertainty'])
    assert lyrs[0].size == lyrs[1].size == lyrs[2].size == lyrs[3].size == 31457280
    assert np.count_nonzero(~np.isnan(lyrs[0])) == np.count_nonzero(lyrs[1]) == np.count_nonzero(~np.isnan(lyrs[2])) == np.count_nonzero(~np.isnan(lyrs[3])) == 2500
    assert lyrs[0][848, 0] == 20.0
    assert lyrs[1][848, 0] == 1
    assert lyrs[2][848, 0] == 0.5
    assert lyrs[3][848, 0] == 1.0

    res = bg.grid(resolution=128, clear_existing=True)
    assert res == [128.0]

    lyrs = bg.get_layers_by_name(['depth', 'density', 'horizontal_uncertainty', 'vertical_uncertainty'], nodatavalue=1000000, z_positive_up=True)
    assert lyrs[0].size == lyrs[1].size == lyrs[2].size == lyrs[3].size == 1920
    assert np.count_nonzero(lyrs[0] != 1000000) == np.count_nonzero(lyrs[1]) == np.count_nonzero(lyrs[2] != 1000000) == np.count_nonzero(lyrs[3] != 1000000) == 1521
    assert lyrs[0][6, 0] == approx(-20.002, 0.001)
    assert lyrs[1][6, 0] == 2
    assert lyrs[2][6, 0] == 0.5
    assert lyrs[3][6, 0] == 1.0

    lyrs = bg.get_layers_by_name(['depth', 'density', 'horizontal_uncertainty', 'vertical_uncertainty'])
    assert lyrs[0].size == lyrs[1].size == lyrs[2].size == lyrs[3].size == 1920
    assert np.count_nonzero(~np.isnan(lyrs[0])) == np.count_nonzero(lyrs[1]) == np.count_nonzero(~np.isnan(lyrs[2])) == np.count_nonzero(~np.isnan(lyrs[3])) == 1521
    assert lyrs[0][6, 0] == approx(20.002, 0.001)
    assert lyrs[1][6, 0] == 2
    assert lyrs[2][6, 0] == 0.5
    assert lyrs[3][6, 0] == 1.0


def test_SRGrid_get_trimmed_layer():
    bg = SRGrid(tile_size=1024)
    bg.add_points(smalldata2, 'test1', ['line1', 'line2'], 26917, 'waterline')
    assert not bg.is_empty
    assert bg.no_grid

    res = bg.grid(resolution=128, clear_existing=True)
    assert res == [128.0]
    dpth = bg.get_layers_by_name('depth')
    dpth = dpth[0]
    dpth_trim, mins, maxs = bg.get_layers_trimmed('depth')
    dpth_trim = dpth_trim[0]

    assert dpth.shape == (48, 40)
    assert np.count_nonzero(~np.isnan(dpth)) == 1521
    assert dpth.size == 1920

    assert dpth_trim.shape == (39, 39)
    assert np.count_nonzero(~np.isnan(dpth_trim)) == 1521  # has the same populated cell count, just less empty space
    assert dpth_trim.size == 1521
    assert mins == [6, 0]
    assert maxs == [45, 39]


def test_VRGridTile_add_points():
    bg = VRGridTile(tile_size=1024, subtile_size=128)
    bg.add_points(smalldata2, 'test1', ['line1', 'line2'], 26917, 'waterline')
    assert not bg.is_empty
    assert bg.data is None  # after adding we clear the point data to free memory
    assert len(bg.tiles) == 6
    assert len(bg.tiles[0]) == 5
    for row in bg.tiles:
        for til in row:
            assert isinstance(til, BathyGrid)
    assert bg.container == {'test1': ['line1', 'line2']}
    assert 'test1' in bg.container_timestamp
    assert bg.vertical_reference == 'waterline'
    assert bg.epsg == 26917
    assert bg.min_x == 0.0
    assert bg.max_x == 5120.0
    assert bg.min_y == 49152.0
    assert bg.max_y == 55296.0
    assert np.array_equal(bg.tile_edges_x, np.array([0.0, 1024.0, 2048.0, 3072.0, 4096.0, 5120.0]))
    assert np.array_equal(bg.tile_edges_y, np.array([49152.0, 50176.0, 51200.0, 52224.0, 53248.0, 54272.0, 55296.0]))

    child_bg = bg.tiles[0][0]
    assert not child_bg.is_empty
    assert not child_bg.data
    assert len(child_bg.tiles) == 1024 / 128
    assert len(child_bg.tiles[0]) == 1024 / 128
    assert child_bg.tile_size == 128
    assert child_bg.min_x == 0.0
    assert child_bg.max_x == 1024.0
    assert child_bg.min_y == 49152.0
    assert child_bg.max_y == 50176.0
    assert np.array_equal(child_bg.tile_edges_x, np.array([0.0, 128.0, 256.0, 384.0, 512.0, 640.0, 768.0, 896.0, 1024.0]))
    assert np.array_equal(child_bg.tile_edges_y, np.array([49152.0, 49280.0, 49408.0, 49536.0, 49664.0, 49792.0, 49920.0, 50048.0, 50176.0]))

    assert not child_bg.tiles[0, :].any()
    assert not child_bg.tiles[1, :].any()
    assert not child_bg.tiles[2, :].any()
    assert not child_bg.tiles[3, :].any()
    assert not child_bg.tiles[4, :].any()
    assert not child_bg.tiles[5, :].any()
    child_bg_tile = child_bg.tiles[6, 0]
    assert child_bg_tile.points_count == 2
    assert child_bg_tile.width == 128
    assert child_bg_tile.height == 128
    assert child_bg_tile.data['x'][0] == 0.0
    assert child_bg_tile.data['y'][0] == 50000.0
    assert child_bg_tile.data['z'][0] == approx(20.000)
    assert child_bg_tile.min_x == 0.0
    assert child_bg_tile.max_x == 128.0
    assert child_bg_tile.min_y == 49920.0
    assert child_bg_tile.max_y == 50048.0
    assert child_bg_tile.name == '0.0_49920.0'

    flat_tiles = child_bg.tiles.ravel()
    for tile in flat_tiles:
        if tile:
            points = tile.data
            for point in points:
                assert point['x'] >= tile.min_x
                assert point['x'] < tile.max_x
                assert point['y'] >= tile.min_y
                assert point['y'] < tile.max_y


def test_VRGridTile_variable_rez_grid():
    bg = VRGridTile(tile_size=1024, subtile_size=128)
    bg.add_points(deepdata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid()
    assert np.array_equal(bg.resolutions, np.array([32.0, 64.0, 128.0]))

    expected_shape = [(192, 192), (96, 96), (48, 48)]
    expected_real = [471, 948, 1320]
    for cnt, resolution in enumerate(bg.resolutions):
        layers = bg.get_layers_by_name(['depth', 'density', 'horizontal_uncertainty', 'vertical_uncertainty'], resolution=resolution)
        assert layers[0].shape == layers[1].shape == layers[2].shape == expected_shape[cnt]
        assert np.count_nonzero(~np.isnan(layers[0])) == np.count_nonzero(layers[1]) == np.count_nonzero(~np.isnan(layers[2])) == expected_real[cnt]

    assert bg.tiles.shape == (6, 6)

    assert bg.tiles[0][0].resolutions == [32.0, 64.0]
    assert bg.tiles[0][0].tiles.shape == (8, 8)
    assert bg.tiles[0][0].tiles[0][0] is None
    assert bg.tiles[0][0].tiles[7][7].cells[64.0]['depth'][1][1] == approx(671.073, 0.001)
    assert bg.tiles[0][0].tiles[7][7].cells[64.0]['density'][1][1] == 49
    assert bg.tiles[0][0].tiles[7][7].cells[64.0]['horizontal_uncertainty'][1][1] == approx(0.519, 0.001)
    assert bg.tiles[0][0].tiles[7][7].cells[64.0]['vertical_uncertainty'][1][1] == approx(1.038, 0.001)

    assert bg.tiles[3][3].resolutions == [128.0]
    assert bg.tiles[3][3].tiles.shape == (8, 8)
    assert bg.tiles[3][3].tiles[7][7].cells[128.0]['depth'][0][0] == approx(3412.555, 0.001)
    assert bg.tiles[3][3].tiles[7][7].cells[128.0]['density'][0][0] == 169
    assert bg.tiles[3][3].tiles[7][7].cells[128.0]['horizontal_uncertainty'][0][0] == approx(0.824, 0.001)
    assert bg.tiles[3][3].tiles[7][7].cells[128.0]['vertical_uncertainty'][0][0] == approx(1.647, 0.001)


def test_SRGrid_backscatter():
    bg = SRGridZarr(tile_size=1024, is_backscatter=True)
    bg.add_points(onlyzdata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid(resolution=8.0)
    assert np.array_equal(bg.resolutions, np.array([8.0]))
    assert bg.layer_names == ['intensity', 'density']
    assert bg.cell_count == {8.0: 16}
    assert bg.coverage_area_square_meters == 1024.0


def test_grid_names():
    bg = VRGridTile(tile_size=1024, subtile_size=128)
    bg.add_points(smalldata2, 'test1', ['line1', 'line2'], 26917, 'waterline')
    assert bg.name == 'VRGridTile_Root'
    assert isinstance(bg.tiles[0][0], BathyGrid)
    assert bg.tiles[0][0].name == '0.0_49152.0'
    assert bg.tiles[3][3].name == '3072.0_52224.0'
    assert bg.tiles[2][4].name == '4096.0_51200.0'
    assert isinstance(bg.tiles[0][0].tiles[6][6], Tile)
    assert bg.tiles[0][0].tiles[6][6].name == '768.0_49920.0'
    assert bg.tiles[0][0].tiles[7][3].name == '384.0_50048.0'

    bg = SRGrid(tile_size=1024)
    bg.add_points(smalldata2, 'test1', ['line1', 'line2'], 26917, 'waterline')
    assert bg.name == 'SRGrid_Root'
    assert isinstance(bg.tiles[0][0], Tile)
    assert bg.tiles[0][0].name == '0.0_49152.0'
    assert bg.tiles[3][3].name == '3072.0_52224.0'
    assert bg.tiles[2][4].name == '4096.0_51200.0'


def test_return_layer_names():
    bg = SRGrid(tile_size=1024)
    bg.add_points(closedata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid()
    assert bg.return_layer_names() == ['depth', 'density', 'vertical_uncertainty', 'horizontal_uncertainty']

    bg = VRGridTile(tile_size=1024, subtile_size=128)
    bg.add_points(closedata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid()
    assert bg.return_layer_names() == ['depth', 'density', 'vertical_uncertainty', 'horizontal_uncertainty']


def test_only_z_data():
    bg = SRGrid(tile_size=1024)
    bg.add_points(onlyzdata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid()
    assert bg.cell_count == {0.5: 16}
    assert bg.coverage_area_square_meters == 4.0

    bg = VRGridTile(tile_size=1024, subtile_size=128)
    bg.add_points(onlyzdata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid()
    assert bg.cell_count == {0.25: 3, 0.5: 7, 1.0: 6}
    assert bg.coverage_area_square_meters == 7.937


def test_return_extents():
    bg = SRGrid(tile_size=1024)
    bg.add_points(closedata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid()
    assert bg.return_extents() == [[0.0, 0.0], [2048.0, 2048.0]]

    bg = VRGridTile(tile_size=1024, subtile_size=128)
    bg.add_points(closedata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid()
    assert bg.return_extents() == [[0.0, 0.0], [2048.0, 2048.0]]


def test_cell_count():
    bg = SRGrid(tile_size=1024)
    bg.add_points(closedata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid()
    assert bg.cell_count == {0.5: 16}
    assert bg.coverage_area_square_meters == 4.0

    bg = VRGridTile(tile_size=1024, subtile_size=128)
    bg.add_points(closedata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid()
    assert bg.cell_count == {0.25: 3, 0.5: 7, 1.0: 6}
    assert bg.coverage_area_square_meters == 7.937


def test_with_dask():
    bg = SRGrid(tile_size=1024)
    bg.add_points(closedata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid(use_dask=True)
    assert bg.client
    assert bg.cell_count == {0.5: 16}
    assert bg.coverage_area_square_meters == 4.0

    bg = VRGridTile(tile_size=1024, subtile_size=128)
    bg.add_points(closedata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid(use_dask=True)
    assert bg.client
    assert bg.cell_count == {0.25: 3, 0.5: 7, 1.0: 6}
    assert bg.coverage_area_square_meters == 7.937


def test_return_unique_containers():
    bg = SRGrid(tile_size=1024)
    bg.add_points(smileyface, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.add_points(smileyface, 'em2040_123_09_07_2020_0', ['line1', 'line2'], 26917, 'waterline')
    bg.add_points(smileyface, 'em2040_123_09_07_2020_1', ['line1', 'line2'], 26917, 'waterline')
    bg.add_points(smileyface, 'sjkof_sdjkfh_skodf', ['line1', 'line2'], 26917, 'waterline')
    bg.add_points(smileyface, 'someother_thing with stuff', ['line1', 'line2'], 26917, 'waterline')
    bg.add_points(smileyface, 'thiswill_be_messedup_123', ['line1', 'line2'], 26917, 'waterline')
    assert bg.return_unique_containers() == ['test1', 'em2040_123_09_07_2020', 'sjkof_sdjkfh_skodf',
                                             'someother_thing with stuff', 'thiswill_be_messedup']


def test_return_attribution():
    bg = SRGrid(tile_size=1024)
    bg.add_points(smileyface, 'test1', ['line1', 'line2'], 26917, 'waterline', min_time=1625235801.123, max_time=1625235901)
    bg.add_points(smileyface, 'em2040_123_09_07_2020_0', ['line1', 'line2'], 26917, 'waterline', min_time=1625123546.123, max_time=1625129546)
    bg.add_points(smileyface, 'em2040_123_09_07_2020_1', ['line1', 'line2'], 26917, 'waterline', min_time=1625465421.123, max_time=1625469421.15512)
    bg.add_points(smileyface, 'sjkof_sdjkfh_skodf', ['line1', 'line2'], 26917, 'waterline', min_time=1625265801.123, max_time=1625275801)
    bg.add_points(smileyface, 'someother_thing with stuff', ['line1', 'line2'], 26917, 'waterline', min_time=1625333333.123, max_time=1625333999)
    bg.add_points(smileyface, 'thiswill_be_messedup_123', ['line1', 'line2'], 26917, 'waterline', min_time=1625444444.123, max_time=1625444445)
    bg.grid(resolution=64)
    attr = bg.return_attribution()
    assert attr['grid_folder'] == ''
    assert attr['name'] == 'SRGrid_Root'
    assert attr['type'] == SRGrid
    assert attr['grid_resolution'] == 64
    assert attr['grid_algorithm'] == 'mean'
    assert attr['epsg'] == 26917
    assert attr['vertical_reference'] == 'waterline'
    assert attr['height'] == 6144.0
    assert attr['width'] == 6144.0
    assert attr['minimum_x'] == 0.0
    assert attr['maximum_x'] == 6144.0
    assert attr['minimum_y'] == 0.0
    assert attr['maximum_y'] == 6144.0
    assert attr['tile_size'] == 1024
    assert attr['subtile_size'] == 0
    assert attr['tile_count'] == 1
    assert attr['resolutions'] == [64.0]
    assert attr['storage_type'] == 'numpy'
    assert attr['source_test1']['multibeam_lines'] == ['line1', 'line2']
    assert attr['source_em2040_123_09_07_2020']['multibeam_lines'] == ['line1', 'line2']
    assert attr['source_someother_thing with stuff']['multibeam_lines'] == ['line1', 'line2']
    assert attr['source_thiswill_be_messedup_123']['multibeam_lines'] == ['line1', 'line2']
    assert attr['minimum_time_utc'] == utc_seconds_to_formatted_string(1625123546)
    assert attr['maximum_time_utc'] == utc_seconds_to_formatted_string(1625469421)


def test_get_geotransform():
    bg = SRGrid(tile_size=1024)
    bg.add_points(smalldata2, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid(resolution=64)
    assert bg.get_geotransform(64.0) == [0.0, 64.0, 0, 55296.0, 0, -64.0]

    bg = VRGridTile(tile_size=1024, subtile_size=128)
    bg.add_points(smalldata2, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid(resolution=64)
    assert bg.get_geotransform(64.0) == [0.0, 64.0, 0, 55296.0, 0, -64.0]


def test_get_tiles_by_resolution():
    bg = SRGrid(tile_size=1024)
    bg.add_points(smalldata2, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid(resolution=64)

    for geo, data_col, data_row, tile_cell_count, tdata in bg.get_tiles_by_resolution(64.0):
        assert geo == [0.0, 64.0, 0, 50176.0, 0, -64.0]
        assert geo == bg.tiles[0][0].get_geotransform(64.0)  # first tile
        assert tdata['depth'].shape == (16, 16)
        assert data_col == 0
        assert data_row == 0
        assert tile_cell_count == 16
        break

    bg = VRGridTile(tile_size=1024, subtile_size=128)
    bg.add_points(smalldata2, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid(resolution=64)

    for geo, data_col, data_row, tile_cell_count, tdata in bg.get_tiles_by_resolution(64.0):
        assert geo == [0.0, 64.0, 0, 50176.0, 0, -64.0]  # with vr the geotransform is going to be the first subgrid
        assert geo != bg.tiles[0][0].tiles[6][0].get_geotransform(64.0)  # first tile
        assert tdata['depth'].shape == (16, 16)
        assert tdata['depth'][13][1] == approx(20.004, 0.001)
        assert np.isnan(tdata['depth'][13][2])
        assert data_col == 0
        assert data_row == 0
        assert tile_cell_count == 16
        break

    for geo, data_col, data_row, tile_cell_count, tdata in bg.get_tiles_by_resolution(64.0, nodatavalue=1000000, z_positive_up=True):
        assert geo == [0.0, 64.0, 0, 50176.0, 0, -64.0]  # with vr the geotransform is going to be the first subgrid
        assert geo != bg.tiles[0][0].tiles[6][0].get_geotransform(64.0)  # first tile
        assert tdata['depth'].shape == (16, 16)
        assert tdata['depth'][13][1] == approx(-20.004, 0.001)
        assert tdata['depth'][13][2] == 1000000
        assert data_col == 0
        assert data_row == 0
        assert tile_cell_count == 16
        break


def test_get_chunks_of_tiles():
    bg = SRGrid(tile_size=1024)
    bg.add_points(smalldata2, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid(resolution=64)

    # default chunk width in grid_variables is 65536, so this is going to be one big chunk
    for geo, max_dimension, tdata in bg.get_chunks_of_tiles(64.0):
        assert geo == [0.0, 64.0, 0, 55296.0, 0, -64.0]
        assert bg.min_x == 0
        assert bg.max_y == 55296
        assert bg.width == 5120.0
        assert bg.height == 6144.0
        assert max_dimension == 6144.0
        assert list(tdata.keys()) == ['depth']
        assert tdata['depth'].shape == (80, 96)

    # lets try a smaller chunk to force chunks on the grid
    testfinaldata = []
    testfinalgeo = []
    testfinalmaxdim = []
    for geo, max_dimension, tdata in bg.get_chunks_of_tiles(64.0, override_maximum_chunk_dimension=2048):
        testfinaldata.append(tdata)
        testfinalgeo.append(geo)
        testfinalmaxdim.append(max_dimension)
    assert testfinalmaxdim[0] == 2048.0
    assert testfinalgeo[0] == [0.0, 64.0, 0, 50176.0, 0, -64.0]
    assert testfinaldata[0]['depth'].shape == (32, 16)

    bg = VRGridTile(tile_size=1024, subtile_size=128)
    bg.add_points(smalldata2, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid(resolution=64)

    for geo, max_dimension, tdata in bg.get_chunks_of_tiles(64.0):
        assert geo == [0.0, 64.0, 0, 55296.0, 0, -64.0]
        assert bg.min_x == 0
        assert bg.max_y == 55296  # geo is not at max_y, as there are NaNs we are leaving out with the fine subgrid size
        assert bg.width == 5120.0
        assert bg.height == 6144.0
        assert max_dimension == 6144.0
        assert list(tdata.keys()) == ['depth']
        assert tdata['depth'].shape == (80, 96)

    # lets try a smaller chunk to force chunks on the grid
    testfinaldata = []
    testfinalgeo = []
    testfinalmaxdim = []
    for geo, max_dimension, tdata in bg.get_chunks_of_tiles(64.0, override_maximum_chunk_dimension=2048):
        testfinaldata.append(tdata)
        testfinalgeo.append(geo)
        testfinalmaxdim.append(max_dimension)
    assert testfinalmaxdim[0] == 2048.0
    assert testfinalgeo[0] == [0.0, 64.0, 0, 50176.0, 0, -64.0]
    assert testfinaldata[0]['depth'].shape == (32, 16)


def test_layer_values_at_xy():
    bg = SRGrid(tile_size=1024)
    bg.add_points(smalldata2, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid(resolution=64)

    assert np.isnan(bg.tiles[0][0].cells[64.0]['depth'][12, 1])
    assert np.isnan(bg.tiles[0][0].cells[64.0]['depth'][12, 2])
    assert np.isnan(bg.tiles[0][0].cells[64.0]['depth'][12, 3])
    assert bg.tiles[0][0].cells[64.0]['depth'][13, 1] == approx(20.004, 0.001)
    assert np.isnan(bg.tiles[0][0].cells[64.0]['depth'][13, 2])
    assert bg.tiles[0][0].cells[64.0]['depth'][13, 3] == approx(20.008, 0.001)
    assert bg.tiles[0][0].cells[64.0]['depth'][14, 1] == approx(20.204, 0.001)
    assert np.isnan(bg.tiles[0][0].cells[64.0]['depth'][14, 2])
    assert bg.tiles[0][0].cells[64.0]['depth'][14, 3] == approx(20.208, 0.001)

    assert np.array_equal(bg.tiles[0][0].cell_edges_x[64.0][0:4], np.array([0.0, 64.0, 128.0, 192.0]))
    assert np.array_equal(bg.tiles[0][0].cell_edges_y[64.0][11:15], np.array([49856.0, 49920.0, 49984.0, 50048.0]))

    assert np.isnan(bg.layer_values_at_xy(0.0, 49983.9))
    assert bg.layer_values_at_xy(0.0, 49984.0) == approx(20.0, 0.001)
    assert bg.layer_values_at_xy(100.0, 50000.0) == approx(20.004, 0.001)
    assert bg.layer_values_at_xy(64.0, 49984.1) == approx(20.004, 0.001)
    assert bg.layer_values_at_xy(192.0, 49984.0) == approx(20.008, 0.001)
    assert bg.layer_values_at_xy(64.0, 50048.0) == approx(20.204, 0.001)
    assert bg.layer_values_at_xy(192.0, 50048.0) == approx(20.208, 0.001)

    assert bg.layer_values_at_xy([0, 64], [49984, 49984]) == approx(np.array([20.0, 20.004]), 0.001)
    outofboundscheck = bg.layer_values_at_xy([-10, 0, 64, 9999999999], [49984, 49984, 49984, 49984])
    assert outofboundscheck[1:3] == approx(np.array([20.0, 20.004]), 0.001)
    assert np.isnan(outofboundscheck[0])
    assert np.isnan(outofboundscheck[3])

    bg = VRGridTile(tile_size=1024, subtile_size=128)
    bg.add_points(deepdata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid()

    assert bg.tiles[0][0].tiles[7][7].cells[64.0]['depth'][0][0] == 612.5
    assert np.array_equal(bg.tiles[0][0].tiles[7][7].cell_edges_x[64.0], np.array([2944., 3008., 3072.]))
    assert np.array_equal(bg.tiles[0][0].tiles[7][7].cell_edges_y[64.0], np.array([52096., 52160., 52224.]))

    assert bg.tiles[3][3].tiles[0][0].cells[128.0]['depth'][0][0] == approx(2605.432, 0.001)
    assert np.array_equal(bg.tiles[3][3].tiles[0][0].cell_edges_x[128.0], np.array([5120., 5248.]))
    assert np.array_equal(bg.tiles[3][3].tiles[0][0].cell_edges_y[128.0], np.array([54272., 54400.]))

    assert bg.layer_values_at_xy(2945.0, 52097.0) == approx(612.5, 0.001)
    assert bg.layer_values_at_xy(5120.0, 54272.0) == approx(2605.432, 0.001)

    assert bg.layer_values_at_xy([2945, 5120], [52097, 54272]) == approx(np.array([612.5, 2605.432]), 0.001)
    outofboundscheck = bg.layer_values_at_xy([-100, 2945, 5120, 99999999999], [52097, 52097, 54272, 54272])
    assert outofboundscheck[1:3] == approx(np.array([612.5, 2605.432]), 0.001)
    assert np.isnan(outofboundscheck[0])
    assert np.isnan(outofboundscheck[3])


def test_density_properties():
    bg = SRGridZarr(tile_size=16)  # small tile size just to ensure this works with multiple tiles
    bg.add_points(realdata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid()
    expected_len = 232
    expected_depth_sum = 3430.543

    dcount = np.array(bg.density_count)
    assert dcount.size == expected_len
    assert dcount.sum() == 238
    dsm = np.array(bg.density_per_square_meter)
    assert dsm.size == expected_len
    assert dsm.sum() == 952.0
    dcount, dpth = bg.density_count_vs_depth
    dcount, dpth = np.array(dcount), np.array(dpth)
    assert dcount.size == expected_len
    assert dcount.sum() == 238
    assert dpth.size == expected_len
    assert dpth.sum().round(3) == expected_depth_sum
    dsm, dpth = bg.density_per_square_meter_vs_depth
    dsm, dpth = np.array(dsm), np.array(dpth)
    assert dsm.size == expected_len
    assert dsm.sum() == 952.0
    assert dpth.size == expected_len
    assert dpth.sum().round(3) == expected_depth_sum

    bg = VRGridTileZarr(tile_size=16, subtile_size=16)
    bg.add_points(realdata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid()
    expected_len = 232
    expected_depth_sum = 3430.543

    dcount = np.array(bg.density_count)
    assert dcount.size == expected_len
    assert dcount.sum() == 238
    dsm = np.array(bg.density_per_square_meter)
    assert dsm.size == expected_len
    assert dsm.sum() == 952.0
    dcount, dpth = bg.density_count_vs_depth
    dcount, dpth = np.array(dcount), np.array(dpth)
    assert dcount.size == expected_len
    assert dcount.sum() == 238
    assert dpth.size == expected_len
    assert dpth.sum().round(3) == expected_depth_sum
    dsm, dpth = bg.density_per_square_meter_vs_depth
    dsm, dpth = np.array(dsm), np.array(dpth)
    assert dsm.size == expected_len
    assert dsm.sum() == 952.0
    assert dpth.size == expected_len
    assert dpth.sum().round(3) == expected_depth_sum


def test_coverage_area():
    bg = SRGridZarr(tile_size=16)  # small tile size just to ensure this works with multiple tiles
    bg.add_points(realdata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid()
    assert bg.coverage_area_square_meters == 58.0
    assert round(bg.coverage_area_square_nm, 6) == .000017

    bg = VRGridTileZarr(tile_size=16, subtile_size=16)
    bg.add_points(realdata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid()
    assert bg.coverage_area_square_meters == 58.0
    assert round(bg.coverage_area_square_nm, 6) == .000017
