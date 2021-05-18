import numpy as np
from pytest import approx

from bathygrid.bathygrid import BathyGrid
from bathygrid.tile import Tile


x = np.arange(0, 5000, 100, dtype=np.float64)
y = np.arange(50000, 55000, 100, dtype=np.float64)
x, y = np.meshgrid(x, y)
x = x.ravel()
y = y.ravel()
z = np.linspace(20, 30, num=x.size).astype(np.float32)

dtyp = [('x', np.float64), ('y', np.float64), ('z', np.float32)]
data1 = np.empty(len(x), dtype=dtyp)
data1['x'] = x
data1['y'] = y
data1['z'] = z

x = np.arange(3000, 8000, 100, dtype=np.float64)
y = np.arange(52000, 57000, 100, dtype=np.float64)
x, y = np.meshgrid(x, y)
x = x.ravel()
y = y.ravel()
z = np.linspace(20, 30, num=x.size).astype(np.float32)

dtyp = [('x', np.float64), ('y', np.float64), ('z', np.float32)]
data2 = np.empty(len(x), dtype=dtyp)
data2['x'] = x
data2['y'] = y
data2['z'] = z


def test_bathygrid_setup():
    bg = BathyGrid(tile_size=1024)
    assert bg.point_data is None
    assert bg.container == {}
    assert bg.epsg is None
    assert bg.vertical_reference is None
    assert bg.min_grid_resolution is None
    assert bg.max_grid_resolution is None


def test_bathygrid_add_points():
    bg = BathyGrid(tile_size=1024)
    bg.add_points(data1, 'test1', ['line1', 'line2'], 26917, 'waterline')
    assert not bg.is_empty
    assert bg.point_data is None  # after adding we clear the point data to free memory
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
    assert tile.data.size == 20
    assert tile.points_count == 20
    assert tile.width == 1024
    assert tile.height == 1024
    assert tile.data['x'][0] == 100.0
    assert tile.data['y'][0] == 50000.0
    assert tile.data['z'][0] == approx(20.0040)
    assert tile.min_x == 0.0
    assert tile.max_x == 1024.0
    assert tile.min_y == 49152.0
    assert tile.max_y == 50176.0
    assert tile.name == '0.0_49152.0'


def test_bathygrid_remove_points():
    bg = BathyGrid(tile_size=1024)
    bg.add_points(data1, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.remove_points('test1')

    assert bg.point_data is None
    assert bg.container == {}
    assert bg.tiles is None


def test_bathygrid_add_multiple_sources():
    bg = BathyGrid(tile_size=1024)
    bg.add_points(data1, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.add_points(data2, 'test2', ['line3', 'line4'], 26917, 'waterline')

    # test2 had no points for this tile, should match the other add points test
    tile = bg.tiles[0][0]
    assert not tile.is_empty
    assert tile.data.size == 20
    assert tile.points_count == 20
    assert tile.width == 1024
    assert tile.height == 1024
    assert tile.data['x'][0] == 100.0
    assert tile.data['y'][0] == 50000.0
    assert tile.data['z'][0] == approx(20.0040)
    assert tile.min_x == 0.0
    assert tile.max_x == 1024.0
    assert tile.min_y == 49152.0
    assert tile.max_y == 50176.0
    assert tile.name == '0.0_49152.0'
    assert tile.container == {'test1': [0, 20]}

    # this tile has points from both containers
    tile = bg.tiles[3][3]
    assert not tile.is_empty
    assert tile.data.size == 120
    assert tile.points_count == 120
    assert tile.width == 1024
    assert tile.height == 1024
    assert tile.data['x'][0] == 3100.0
    assert tile.data['y'][0] == 50000.0
    assert tile.data['z'][0] == approx(20.12405)
    assert tile.min_x == 3072.0
    assert tile.max_x == 4096.0
    assert tile.min_y == 52224.0
    assert tile.max_y == 53248.0
    assert tile.name == '3072.0_52224.0'
    assert tile.container == {'test1': [0, 20], 'test2': [20, 120]}

    # removing points from this container will remove all test2 points from all tiles
    bg.remove_points('test2')

    tile = bg.tiles[3][3]
    assert not tile.is_empty
    assert tile.data.size == 20
    assert tile.points_count == 20
    assert tile.width == 1024
    assert tile.height == 1024
    assert tile.data['x'][0] == 3100.0
    assert tile.data['y'][0] == 50000.0
    assert tile.data['z'][0] == approx(20.12405)
    assert tile.min_x == 3072.0
    assert tile.max_x == 4096.0
    assert tile.min_y == 52224.0
    assert tile.max_y == 53248.0
    assert tile.name == '3072.0_52224.0'
    assert tile.container == {'test1': [0, 20]}
