import numpy as np
from pytest import approx

from bathygrid.bgrid import *
from bathygrid.maingrid import *
from bathygrid.convenience import *
from bathygrid.tile import Tile
from test_data.test_data import smalldata2, get_test_path


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


def _expected_vrgrid_data(bathygrid):
    rootpath = os.path.join(bathygrid.output_folder, bathygrid.name)
    assert os.path.exists(rootpath)
    numgrids = bathygrid.tiles.size
    fldrs = [fldr for fldr in os.listdir(rootpath) if os.path.isdir(os.path.join(rootpath, fldr))]
    assert len(fldrs) == numgrids
    assert os.path.exists(os.path.join(rootpath, 'metadata.json'))

    subgrid = bathygrid.tiles[0][0]
    subgrid_rootpath = os.path.join(subgrid.output_folder, subgrid.name)
    assert os.path.exists(subgrid_rootpath)
    populated_tiles = [x for x in subgrid.tiles.ravel() if x is not None]
    fldrs = [fldr for fldr in os.listdir(subgrid_rootpath) if os.path.isdir(os.path.join(subgrid_rootpath, fldr))]
    assert len(fldrs) == len(populated_tiles)
    assert os.path.exists(os.path.join(subgrid_rootpath, 'metadata.json'))

    subgrid_tile = bathygrid.tiles[0][0].tiles[6][5]  # pull out random populated tile
    subgrid_tile_rootpath = os.path.join(subgrid_rootpath, subgrid_tile.name)
    assert os.path.exists(subgrid_tile_rootpath)
    assert os.path.exists(os.path.join(subgrid_tile_rootpath, 'data'))
    assert os.path.exists(os.path.join(subgrid_tile_rootpath, 'metadata.json'))


def test_basic_save():
    bg = SRGrid(tile_size=1024)
    bg.add_points(smalldata2, 'test1', ['line1', 'line2'], 26917, 'waterline')
    gridfolder = get_test_path()
    bg.save(gridfolder)
    _expected_sr_data(bg)


def test_vrgrid_save():
    bg = VRGridTile(tile_size=1024, subtile_size=128)
    bg.add_points(smalldata2, 'test1', ['line1', 'line2'], 26917, 'waterline')
    gridfolder = get_test_path()
    bg.save(gridfolder)
    _expected_vrgrid_data(bg)


def test_srgrid_create_with_output_folder():
    bg = SRGrid(tile_size=1024, output_folder=get_test_path())
    bg.add_points(smalldata2, 'test1', ['line1', 'line2'], 26917, 'waterline')
    _expected_sr_data(bg)


def test_vrgrid_create_with_output_folder():
    bg = VRGridTile(tile_size=1024, subtile_size=128, output_folder=get_test_path())
    bg.add_points(smalldata2, 'test1', ['line1', 'line2'], 26917, 'waterline')
    _expected_vrgrid_data(bg)
