import numpy as np
from pytest import approx

from bathygrid.bgrid import *
from bathygrid.maingrid import *
from bathygrid.tile import Tile
from test_data.test_data import smalldata2, smalldata3, deepdata, closedata, smileyface, onlyzdata


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


def test_SRGrid_get_layer_by_name():
    bg = SRGrid(tile_size=1024)
    bg.add_points(smalldata2, 'test1', ['line1', 'line2'], 26917, 'waterline')
    assert not bg.is_empty
    assert bg.no_grid

    res = bg.grid()
    assert not bg.no_grid
    assert res == [1.0]
    assert bg.data is None  # after adding we clear the point data to free memory
    lyrs = bg.get_layers_by_name(['depth', 'horizontal_uncertainty', 'vertical_uncertainty'])
    assert lyrs[0].size == lyrs[1].size == lyrs[2].size == 31457280
    assert np.count_nonzero(~np.isnan(lyrs[0])) == np.count_nonzero(~np.isnan(lyrs[1])) == np.count_nonzero(~np.isnan(lyrs[2])) == 2500
    assert lyrs[0][848, 0] == 20.0
    assert lyrs[1][848, 0] == 0.5
    assert lyrs[2][848, 0] == 1.0

    res = bg.grid(resolution=128, clear_existing=True)
    assert res == [128.0]
    lyrs = bg.get_layers_by_name(['depth', 'horizontal_uncertainty', 'vertical_uncertainty'])
    assert lyrs[0].size == lyrs[1].size == lyrs[2].size == 1920
    assert np.count_nonzero(~np.isnan(lyrs[0])) == np.count_nonzero(~np.isnan(lyrs[1])) == np.count_nonzero(~np.isnan(lyrs[2])) == 1521
    assert lyrs[0][6, 0] == 20.002
    assert lyrs[1][6, 0] == 0.5
    assert lyrs[2][6, 0] == 1.0


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
        layers = bg.get_layers_by_name(['depth', 'horizontal_uncertainty', 'vertical_uncertainty'], resolution=resolution)
        assert layers[0].shape == layers[1].shape == layers[2].shape == expected_shape[cnt]
        assert np.count_nonzero(~np.isnan(layers[0])) == np.count_nonzero(~np.isnan(layers[1])) == np.count_nonzero(~np.isnan(layers[2])) == expected_real[cnt]

    assert bg.tiles.shape == (6, 6)

    assert bg.tiles[0][0].resolutions == [32.0, 64.0]
    assert bg.tiles[0][0].tiles.shape == (8, 8)
    assert bg.tiles[0][0].tiles[0][0] is None
    assert bg.tiles[0][0].tiles[7][7].cells[64.0]['depth'][1][1] == 671.073
    assert bg.tiles[0][0].tiles[7][7].cells[64.0]['horizontal_uncertainty'][1][1] == 0.519
    assert bg.tiles[0][0].tiles[7][7].cells[64.0]['vertical_uncertainty'][1][1] == 1.038

    assert bg.tiles[3][3].resolutions == [128.0]
    assert bg.tiles[3][3].tiles.shape == (8, 8)
    assert bg.tiles[3][3].tiles[7][7].cells[128.0]['depth'][0][0] == 3412.555
    assert bg.tiles[3][3].tiles[7][7].cells[128.0]['horizontal_uncertainty'][0][0] == 0.824
    assert bg.tiles[3][3].tiles[7][7].cells[128.0]['vertical_uncertainty'][0][0] == 1.647


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
    assert bg.return_layer_names() == ['depth', 'vertical_uncertainty', 'horizontal_uncertainty']

    bg = VRGridTile(tile_size=1024, subtile_size=128)
    bg.add_points(closedata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid()
    assert bg.return_layer_names() == ['depth', 'vertical_uncertainty', 'horizontal_uncertainty']


def test_only_z_data():
    bg = SRGrid(tile_size=1024)
    bg.add_points(onlyzdata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid()
    assert bg.cell_count == {0.5: 16}
    assert bg.coverage_area == 8

    bg = VRGridTile(tile_size=1024, subtile_size=128)
    bg.add_points(onlyzdata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid()
    assert bg.cell_count == {0.5: 10, 1.0: 6}
    assert bg.coverage_area == 11.0


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
    assert bg.coverage_area == 8

    bg = VRGridTile(tile_size=1024, subtile_size=128)
    bg.add_points(closedata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid()
    assert bg.cell_count == {0.5: 10, 1.0: 6}
    assert bg.coverage_area == 11.0


def test_with_dask():
    bg = SRGrid(tile_size=1024)
    bg.add_points(closedata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid(use_dask=True)
    assert bg.client
    assert bg.cell_count == {0.5: 16}
    assert bg.coverage_area == 8

    bg = VRGridTile(tile_size=1024, subtile_size=128)
    bg.add_points(closedata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid(use_dask=True)
    assert bg.client
    assert bg.cell_count == {0.5: 10, 1.0: 6}
    assert bg.coverage_area == 11.0


def test_gdal_preprocessing():
    bg = SRGrid(tile_size=1024)
    bg.add_points(smileyface, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid(resolution=64)
    data, geo_transform, bandnames = bg._gdal_preprocessing(resolution=64.0, nodatavalue=np.nan, z_positive_up=False,
                                                            layer_names=['depth'])
    trimdata, mins, maxs = bg.get_layers_trimmed('depth', 64.0)

    # data mirrored for gdal
    assert data[0][0][3] == trimdata[0][3][0]
    assert data[0][-1][3] == trimdata[0][3][-1]
    trimsize = (np.array(maxs) - np.array(mins)).tolist()
    assert trimsize[0] == data[0].shape[1]
    assert trimsize[1] == data[0].shape[0]
    assert trimsize[0] == trimdata[0].shape[0]
    assert trimsize[1] == trimdata[0].shape[1]
    assert geo_transform == [192.0, 64.0, 0, 832.0, 0, -64.0]
    assert bandnames == ['Depth']