from pytest import approx

from bathygrid.maingrid import *
from bathygrid.convenience import *
from test_data.test_data import closedata, get_test_path, onlyzdata, smalldata
from bathygrid.utilities import utc_seconds_to_formatted_string


def _expected_sr_data(bathygrid):
    rootpath = os.path.join(bathygrid.output_folder, bathygrid.name)
    assert os.path.exists(rootpath)
    numtiles = np.count_nonzero(bathygrid.tiles)
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
    assert 'test1' in bathygrid.container_timestamp
    assert bathygrid.min_x == 0.0
    assert bathygrid.min_y == 0.0
    assert bathygrid.max_x == 2048.0
    assert bathygrid.max_y == 2048.0
    assert bathygrid.epsg == 26917
    assert bathygrid.vertical_reference == 'waterline'
    assert bathygrid.sub_type == 'srtile'
    assert bathygrid.storage_type == 'numpy'


def _expected_srgrid_griddata(bathygrid, include_uncertainties: bool = True):
    assert bathygrid.mean_depth == 17.5
    assert bathygrid.resolutions == np.array([1.0])
    assert bathygrid.grid_algorithm == 'mean'

    grid_tile = bathygrid.tiles[0][0]  # pull out random populated tile
    assert grid_tile.algorithm == 'mean'
    assert list(grid_tile.cells.keys()) == [1.0]
    assert grid_tile.cell_edges_x[1.0].shape[0] == 1025
    assert grid_tile.cell_edges_x[1.0][0].compute() == 0.0
    assert grid_tile.cell_edges_x[1.0][-1].compute() == 1024.0
    assert grid_tile.cell_edges_y[1.0].shape[0] == 1025
    assert grid_tile.cell_edges_y[1.0][0].compute() == 0.0
    assert grid_tile.cell_edges_y[1.0][-1].compute() == 1024.0
    assert np.array_equal(grid_tile.cell_indices[1.0].compute(), np.array([820000, 820100, 820200, 922400, 922500, 922600, 1024800, 1024900, 1025000]))
    assert grid_tile.data[0]['z'] == 5.0
    if include_uncertainties:
        assert grid_tile.data[0]['tvu'] == 1.0
        assert grid_tile.data[0]['thu'] == 0.5
    assert grid_tile.cells[1.0]['depth'].compute().flat[820000] == 5.0
    if include_uncertainties:
        assert grid_tile.cells[1.0]['vertical_uncertainty'].compute().flat[820000] == 1.0
        assert grid_tile.cells[1.0]['horizontal_uncertainty'].compute().flat[820000] == 0.5
    assert grid_tile.name == '0.0_0.0'
    assert grid_tile.points_count == 9
    assert grid_tile.height == 1024
    assert grid_tile.width == 1024


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
    assert 'test1' in bathygrid.container_timestamp
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
    assert 'test1' in bathygrid.container_timestamp
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


def _expected_vrgrid_griddata(bathygrid, include_uncertainties: bool = True):
    assert bathygrid.mean_depth == 17.5
    assert np.array_equal(bathygrid.resolutions, np.array([0.5, 1.0]))
    assert bathygrid.grid_algorithm == 'mean'

    subgrid = bathygrid.tiles[0][0]

    subgrid_tile = bathygrid.tiles[0][0].tiles[7][7]  # pull out random populated tile
    assert subgrid_tile.algorithm == 'mean'
    assert list(subgrid_tile.cells.keys()) == [0.5]
    assert subgrid_tile.cell_edges_x[0.5].shape[0] == 257
    assert subgrid_tile.cell_edges_x[0.5][0].compute() == 896.0
    assert subgrid_tile.cell_edges_x[0.5][-1].compute() == 1024.0
    assert subgrid_tile.cell_edges_y[0.5].shape[0] == 257
    assert subgrid_tile.cell_edges_y[0.5][0].compute() == 896.0
    assert subgrid_tile.cell_edges_y[0.5][-1].compute() == 1024.0
    assert np.array_equal(subgrid_tile.cell_indices[0.5].compute(), np.array([2056, 2256, 53256, 53456]))
    assert approx(subgrid_tile.data[0]['z'], 0.001) == 13.333
    if include_uncertainties:
        assert approx(subgrid_tile.data[0]['tvu'], 0.001) == 1.333
        assert approx(subgrid_tile.data[0]['thu'], 0.001) == 0.667
    assert approx(subgrid_tile.cells[0.5]['depth'].compute().flat[2056]) == 13.333
    if include_uncertainties:
        assert approx(subgrid_tile.cells[0.5]['vertical_uncertainty'].compute().flat[2056]) == 1.333
        assert approx(subgrid_tile.cells[0.5]['horizontal_uncertainty'].compute().flat[2056]) == 0.667
    assert subgrid_tile.name == '896.0_896.0'
    assert subgrid_tile.points_count == 4
    assert subgrid_tile.height == 128
    assert subgrid_tile.width == 128


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


def test_srgrid_grid_update():
    bg = SRGrid(tile_size=1024, output_folder=get_test_path())
    bg.add_points(closedata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.add_points(smalldata, 'test2', ['line1', 'line2'], 26917, 'waterline')
    bg.grid(resolution=1)
    assert bg.cell_count == {1.0: 112}
    bg.remove_points('test2')
    bg.grid(resolution=1)
    assert bg.cell_count == {1.0: 16}


def test_vrgrid_grid_update():
    bg = VRGridTile(tile_size=1024, subtile_size=128, output_folder=get_test_path())
    bg.add_points(closedata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.add_points(smalldata, 'test2', ['line1', 'line2'], 26917, 'waterline')
    bg.grid()
    assert bg.cell_count == {1.0: 106, 0.5: 6}
    bg.remove_points('test2')
    bg.grid()
    assert bg.cell_count == {0.5: 10, 1.0: 10}


def test_srgrid_after_load():
    bg = SRGrid(tile_size=1024, output_folder=get_test_path())
    bg.add_points(closedata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid(resolution=1)
    bg = load_grid(bg.output_folder)
    _expected_sr_data(bg)
    _expected_srgrid_griddata(bg)


def test_vrgrid_after_load():
    bg = VRGridTile(tile_size=1024, subtile_size=128, output_folder=get_test_path())
    bg.add_points(closedata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid()
    bg = load_grid(bg.output_folder)
    _expected_vrgrid_data(bg)
    _expected_vrgrid_griddata(bg)


def test_srgrid_onlyz():
    bg = SRGrid(tile_size=1024, output_folder=get_test_path())
    bg.add_points(onlyzdata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid(resolution=1)
    bg = load_grid(bg.output_folder)
    _expected_sr_data(bg)
    _expected_srgrid_griddata(bg, include_uncertainties=False)


def test_vrgrid_onlyz():
    bg = VRGridTile(tile_size=1024, subtile_size=128, output_folder=get_test_path())
    bg.add_points(onlyzdata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid()
    bg = load_grid(bg.output_folder)
    _expected_vrgrid_data(bg)
    _expected_vrgrid_griddata(bg, include_uncertainties=False)


def test_srgrid_with_dask():
    bg = SRGrid(tile_size=1024, output_folder=get_test_path())
    bg.add_points(closedata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid(resolution=1, use_dask=True)
    bg = load_grid(bg.output_folder)
    _expected_sr_data(bg)
    _expected_srgrid_griddata(bg)


def test_vrgrid_with_dask():
    bg = VRGridTile(tile_size=1024, subtile_size=128, output_folder=get_test_path())
    bg.add_points(closedata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid(use_dask=True)
    bg = load_grid(bg.output_folder)
    _expected_vrgrid_data(bg)
    _expected_vrgrid_griddata(bg)


def test_srgrid_export_csv():
    bg = SRGrid(tile_size=1024, output_folder=get_test_path())
    bg.add_points(closedata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid(resolution=1)
    out_csv = os.path.join(bg.output_folder, 'test.csv')
    bg.export(out_csv, export_format='csv')
    new_csv = os.path.join(bg.output_folder, 'test_1.0.csv')
    assert os.path.exists(new_csv)
    assert os.stat(new_csv).st_size == 2688415


def test_vrgrid_export_csv():
    bg = VRGridTile(tile_size=1024, subtile_size=128, output_folder=get_test_path())
    bg.add_points(closedata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid()
    out_csv = os.path.join(bg.output_folder, 'test.csv')
    bg.export(out_csv, export_format='csv')
    new_csv = os.path.join(bg.output_folder, 'test_0.5.csv')
    new_csv_two = os.path.join(bg.output_folder, 'test_1.0.csv')
    assert os.path.exists(new_csv)
    assert os.stat(new_csv).st_size == 7070367
    assert os.path.exists(new_csv_two)
    assert os.stat(new_csv_two).st_size == 1805338


def test_srgrid_export_tiff():
    bg = SRGrid(tile_size=1024, output_folder=get_test_path())
    bg.add_points(closedata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid(resolution=1)
    out_tif = os.path.join(bg.output_folder, 'outtiff.tif')
    bg.export(out_tif, export_format='geotiff')
    new_tif = os.path.join(bg.output_folder, 'outtiff_1.0_1.tif')
    assert os.path.exists(new_tif)


def test_vrgrid_export_tiff():
    bg = VRGridTile(tile_size=1024, subtile_size=128, output_folder=get_test_path())
    bg.add_points(closedata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid()
    out_tif = os.path.join(bg.output_folder, 'outtiff.tif')
    bg.export(out_tif, export_format='geotiff')
    out_tif = os.path.join(bg.output_folder, 'outtiff_0.5_1.tif')
    out_tif_two = os.path.join(bg.output_folder, 'outtiff_1.0_1.tif')
    assert os.path.exists(out_tif)
    assert os.path.exists(out_tif_two)


def test_srgrid_export_bag():
    bg = SRGrid(tile_size=1024, output_folder=get_test_path())
    bg.add_points(closedata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid(resolution=1)
    out_bag = os.path.join(bg.output_folder, 'outtiff.bag')
    bg.export(out_bag, export_format='bag')
    new_bag = os.path.join(bg.output_folder, 'outtiff_1.0_1.bag')
    assert os.path.exists(new_bag)


def test_vrgrid_export_bag():
    bg = VRGridTile(tile_size=1024, subtile_size=128, output_folder=get_test_path())
    bg.add_points(closedata, 'test1', ['line1', 'line2'], 26917, 'waterline')
    bg.grid()
    out_bag = os.path.join(bg.output_folder, 'outtiff.bag')
    bg.export(out_bag, export_format='bag')
    new_bag = os.path.join(bg.output_folder, 'outtiff_0.5_1.bag')
    new_bag_two = os.path.join(bg.output_folder, 'outtiff_1.0_1.bag')
    assert os.path.exists(new_bag)
    assert os.path.exists(new_bag_two)


def test_srgrid_tracks_minmax_time():
    bg = SRGrid(tile_size=1024, output_folder=get_test_path())
    bg.add_points(closedata, 'test1', ['line1', 'line2'], 26917, 'waterline', min_time=1625235801.123, max_time=1625235901)
    bg.add_points(smalldata, 'test2', ['line1', 'line2'], 26917, 'waterline', min_time=1625465421.123, max_time=1625469421.15512)
    bg.grid(resolution=1)
    assert bg.min_time == utc_seconds_to_formatted_string(int(min(1625235801.123, 1625465421.123)))
    assert bg.max_time == utc_seconds_to_formatted_string(int(max(1625235901, 1625469421.15512)))


def test_clear_data():
    # one last call to clear data
    get_test_path()

