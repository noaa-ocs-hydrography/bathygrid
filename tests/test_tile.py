import numpy as np
from bathygrid.tile import Tile


x = np.arange(0, 1000, 100, dtype=np.float64)
y = np.arange(0, 1000, 100, dtype=np.float64)
x, y = np.meshgrid(x, y)
x = x.ravel()
y = y.ravel()
z = np.linspace(20, 30, num=x.size).astype(np.float32)
unc = np.linspace(1, 2, num=x.size).astype(np.float32)

dtyp = [('x', np.float64), ('y', np.float64), ('z', np.float32), ('unc', np.float32)]
data = np.empty(len(x), dtype=dtyp)
data['x'] = x
data['y'] = y
data['z'] = z
data['unc'] = unc


def test_tile_setup():
    til = Tile(0.0, 0.0, 1024)
    assert til.is_empty

    til.add_points(data, 'test1')
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

    til.remove_points('test1')
    assert til.is_empty


def test_tile_newgrid():
    til = Tile(0.0, 0.0, 1024)
    grid, grid_x_edges, grid_y_edges = til.new_grid(8)

    assert grid.shape == (1024 / 8, 1024 / 8)
    assert grid_x_edges[0] == 0
    assert grid_x_edges[-1] == 1024
    assert grid_y_edges[0] == 0
    assert grid_y_edges[-1] == 1024

    assert np.isnan(grid[0][0])


def test_tile_single_resolution():
    til = Tile(0.0, 0.0, 1024)
    til.add_points(data, 'test1')
    til.grid_single_resolution(128.0, 'mean')
