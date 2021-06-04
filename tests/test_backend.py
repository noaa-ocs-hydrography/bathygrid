import numpy as np
from pytest import approx

from bathygrid.bgrid import *
from bathygrid.maingrid import *
from bathygrid.convenience import *
from bathygrid.tile import Tile
from test_data.test_data import closedata, get_test_path


def _expected_sr_data(bathygrid):
    rootpath = os.path.join(bathygrid.output_folder, bathygrid.name)
    assert os.path.exists(rootpath)
    numtiles = bathygrid.tiles.size
    fldrs = [fldr for fldr in os.listdir(rootpath) if os.path.isdir(os.path.join(rootpath, fldr))]
    assert len(fldrs) == numtiles
    assert os.path.exists(os.path.join(rootpath, 'metadata.json'))

    grid_tile = bathygrid.tiles[0][0]  # pull out random populated tile
    grid_tile_rootpath = os.path.join(rootpath, grid_tile.name)
    assert os.path.exists(grid_tile_rootpath)
    assert os.path.exists(os.path.join(grid_tile_rootpath, 'data'))
    assert os.path.exists(os.path.join(grid_tile_rootpath, 'metadata.json'))

    assert bathygrid.name == 'SRGrid_Root'
    assert bathygrid.output_folder
    assert bathygrid.container == {'test1': ['line1', 'line2']}
    assert bathygrid.min_x == 0.0
    assert bathygrid.min_y == 0.0
    assert bathygrid.max_x == 2048.0
    assert bathygrid.max_y == 2048.0
    assert bathygrid.epsg == 26917
    assert bathygrid.vertical_reference == 'waterline'
    assert bathygrid.sub_type == 'srtile'
    assert bathygrid.storage_type == 'numpy'


def _expected_vrgrid_data(bathygrid):
    rootpath = os.path.join(bathygrid.output_folder, bathygrid.name)
    assert os.path.exists(rootpath)
    numgrids = bathygrid.tiles.size
    fldrs = [fldr for fldr in os.listdir(rootpath) if os.path.isdir(os.path.join(rootpath, fldr))]
    assert len(fldrs) == numgrids
    assert os.path.exists(os.path.join(rootpath, 'metadata.json'))

    assert bathygrid.name == 'VRGridTile_Root'
    assert bathygrid.output_folder
    assert bathygrid.container == {'test1': ['line1', 'line2']}
    assert bathygrid.min_x == 0.0
    assert bathygrid.min_y == 0.0
    assert bathygrid.max_x == 2048.0
    assert bathygrid.max_y == 2048.0
    assert bathygrid.epsg == 26917
    assert bathygrid.vertical_reference == 'waterline'
    assert bathygrid.sub_type == 'grid'
    assert bathygrid.storage_type == 'numpy'

    subgrid = bathygrid.tiles[0][0]
    subgrid_rootpath = os.path.join(subgrid.output_folder, subgrid.name)
    assert os.path.exists(subgrid_rootpath)
    populated_tiles = [x for x in subgrid.tiles.ravel() if x is not None]
    fldrs = [fldr for fldr in os.listdir(subgrid_rootpath) if os.path.isdir(os.path.join(subgrid_rootpath, fldr))]
    assert len(fldrs) == len(populated_tiles)
    assert os.path.exists(os.path.join(subgrid_rootpath, 'metadata.json'))

    assert subgrid.name == '0.0_0.0'
    assert subgrid.output_folder
    assert subgrid.container == {'test1': ['Unknown']}
    assert subgrid.min_x == 0.0
    assert subgrid.min_y == 0.0
    assert subgrid.max_x == 1024.0
    assert subgrid.max_y == 1024.0
    assert not subgrid.epsg
    assert not subgrid.vertical_reference
    assert subgrid.sub_type == 'srtile'
    assert subgrid.storage_type == 'numpy'

    subgrid_tile = bathygrid.tiles[0][0].tiles[7][7]  # pull out random populated tile
    subgrid_tile_rootpath = os.path.join(subgrid_rootpath, subgrid_tile.name)
    assert os.path.exists(subgrid_tile_rootpath)
    assert os.path.exists(os.path.join(subgrid_tile_rootpath, 'data'))
    assert os.path.exists(os.path.join(subgrid_tile_rootpath, 'metadata.json'))


def test_basic_save():
    bg = SRGrid(tile_size=1024)
    bg.add_points(closedata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    gridfolder = get_test_path()
    bg.save(gridfolder)
    _expected_sr_data(bg)


def test_vrgrid_save():
    bg = VRGridTile(tile_size=1024, subtile_size=128)
    bg.add_points(closedata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    gridfolder = get_test_path()
    bg.save(gridfolder)
    _expected_vrgrid_data(bg)


def test_srgrid_create_with_output_folder():
    bg = SRGrid(tile_size=1024, output_folder=get_test_path())
    bg.add_points(closedata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    _expected_sr_data(bg)


def test_vrgrid_create_with_output_folder():
    bg = VRGridTile(tile_size=1024, subtile_size=128, output_folder=get_test_path())
    bg.add_points(closedata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    _expected_vrgrid_data(bg)


def test_srgrid_remove_points():
    bg = SRGrid(tile_size=1024, output_folder=get_test_path())
    bg.add_points(closedata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.remove_points('test1')
    assert bg.number_of_tiles == 0
    rootpath = os.path.join(bg.output_folder, bg.name)
    assert os.path.exists(rootpath)
    fldrs = [fldr for fldr in os.listdir(rootpath) if os.path.isdir(os.path.join(rootpath, fldr))]
    assert len(fldrs) == 0


def test_vrgrid_remove_points():
    bg = VRGridTile(tile_size=1024, subtile_size=128, output_folder=get_test_path())
    bg.add_points(closedata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.remove_points('test1')
    assert bg.number_of_tiles == 0
    rootpath = os.path.join(bg.output_folder, bg.name)
    assert os.path.exists(rootpath)
    fldrs = [fldr for fldr in os.listdir(rootpath) if os.path.isdir(os.path.join(rootpath, fldr))]
    assert len(fldrs) == 4


def test_srgrid_after_load():
    bg = SRGrid(tile_size=1024, output_folder=get_test_path())
    bg.add_points(closedata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg = load_grid(bg.output_folder)
    _expected_sr_data(bg)


def test_vrgrid_after_load():
    bg = VRGridTile(tile_size=1024, subtile_size=128, output_folder=get_test_path())
    bg.add_points(closedata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg = load_grid(bg.output_folder)
    _expected_vrgrid_data(bg)


def test_clear_data():
    # one last call to clear data
    get_test_path()
