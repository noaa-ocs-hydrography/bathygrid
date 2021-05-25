import numpy as np
from pytest import approx

from bathygrid.bgrid import *
from bathygrid.tile import Tile


x = np.arange(0, 5000, 100, dtype=np.float64)
y = np.arange(50000, 55000, 100, dtype=np.float64)
x, y = np.meshgrid(x, y)
x = x.ravel()
y = y.ravel()
z = np.linspace(20, 30, num=x.size).astype(np.float32)
tvu = np.linspace(1, 2, num=x.size).astype(np.float32)
thu = np.linspace(0.5, 1, num=x.size).astype(np.float32)

dtyp = [('x', np.float64), ('y', np.float64), ('z', np.float32), ('tvu', np.float32), ('thu', np.float32)]
data1 = np.empty(len(x), dtype=dtyp)
data1['x'] = x
data1['y'] = y
data1['z'] = z
data1['tvu'] = tvu
data1['thu'] = thu

x = np.arange(3000, 8000, 100, dtype=np.float64)
y = np.arange(52000, 57000, 100, dtype=np.float64)
x, y = np.meshgrid(x, y)
x = x.ravel()
y = y.ravel()
z = np.linspace(20, 30, num=x.size).astype(np.float32)
tvu = np.linspace(1, 2, num=x.size).astype(np.float32)
thu = np.linspace(0.5, 1, num=x.size).astype(np.float32)

data2 = np.empty(len(x), dtype=dtyp)
data2['x'] = x
data2['y'] = y
data2['z'] = z
data2['tvu'] = tvu
data2['thu'] = thu


def test_SRGrid_setup():
    bg = SRGrid(tile_size=1024)
    assert bg.data is None
    assert bg.container == {}
    assert bg.epsg is None
    assert bg.vertical_reference is None
    assert bg.min_grid_resolution is None
    assert bg.max_grid_resolution is None


def test_SRGrid_add_points():
    bg = SRGrid(tile_size=1024)
    bg.add_points(data1, 'test1', ['line1', 'line2'], 26917, 'waterline')
    assert not bg.is_empty
    assert bg.data is None  # after adding we clear the point data to free memory
    assert len(bg.tiles) == 6
    assert len(bg.tiles[0]) == 5
    for row in bg.tiles:
        for til in row:
            assert isinstance(til, Tile)
    assert bg.container == {'test1': ['line1', 'line2']}
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
    bg.add_points(data1, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.remove_points('test1')

    assert bg.data is None
    assert bg.container == {}
    assert bg.tiles is None


def test_SRGrid_add_multiple_sources():
    bg = SRGrid(tile_size=1024)
    bg.add_points(data1, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.add_points(data2, 'test2', ['line3', 'line4'], 26917, 'waterline')

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


def test_SRGrid_get_layer_by_name():
    bg = SRGrid(tile_size=1024)
    bg.add_points(data1, 'test1', ['line1', 'line2'], 26917, 'waterline')
    assert not bg.is_empty
    assert bg.no_grid

    res = bg.grid()
    assert not bg.no_grid
    assert res == 1.0
    assert bg.data is None  # after adding we clear the point data to free memory
    dpth = bg.get_layer_by_name('depth')
    thu = bg.get_layer_by_name('horizontal_uncertainty')
    tvu = bg.get_layer_by_name('vertical_uncertainty')
    assert dpth.size == thu.size == tvu.size == 31457280
    assert np.count_nonzero(~np.isnan(dpth)) == np.count_nonzero(~np.isnan(thu)) == np.count_nonzero(~np.isnan(tvu)) == 2500
    assert dpth[848, 0] == 20.0
    assert thu[848, 0] == 0.5
    assert tvu[848, 0] == 1.0

    res = bg.grid(resolution=128, clear_existing=True)
    assert res == 128.0
    dpth = bg.get_layer_by_name('depth')
    thu = bg.get_layer_by_name('horizontal_uncertainty')
    tvu = bg.get_layer_by_name('vertical_uncertainty')
    assert dpth.size == thu.size == tvu.size == 1920
    assert np.count_nonzero(~np.isnan(dpth)) == np.count_nonzero(~np.isnan(thu)) == np.count_nonzero(~np.isnan(tvu)) == 1521
    assert dpth[6, 0] == 20.002
    assert thu[6, 0] == 0.5
    assert tvu[6, 0] == 1.0


def test_SRGrid_get_trimmed_layer():
    bg = SRGrid(tile_size=1024)
    bg.add_points(data1, 'test1', ['line1', 'line2'], 26917, 'waterline')
    assert not bg.is_empty
    assert bg.no_grid

    res = bg.grid(resolution=128, clear_existing=True)
    assert res == 128.0
    dpth = bg.get_layer_by_name('depth')
    dpth_trim, mins, maxs = bg.get_layer_trimmed('depth')

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
    bg.add_points(data1, 'test1', ['line1', 'line2'], 26917, 'waterline')
    assert not bg.is_empty
    assert bg.data is None  # after adding we clear the point data to free memory
    assert len(bg.tiles) == 6
    assert len(bg.tiles[0]) == 5
    for row in bg.tiles:
        for til in row:
            assert isinstance(til, BathyGrid)
    assert bg.container == {'test1': ['line1', 'line2']}
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
