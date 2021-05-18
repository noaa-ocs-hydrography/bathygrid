import os
from datetime import datetime
import numpy as np
import xarray as xr
from dask.array import Array
from typing import Union

from bathygrid.grids import Grid
from bathygrid.tile import Tile


def create_folder(output_directory: str, fldrname: str):
    """
    Generate a new folder with folder name fldrname in output_directory.  Will create output_directory if it does
    not exist.  If fldrname exists, will generate a folder with a time tag next to it instead.  Will always
    create a folder this way.

    Parameters
    ----------
    output_directory
        path to containing folder
    fldrname
        name of new folder to create

    Returns
    -------
    str
        path to the created folder
    """

    os.makedirs(output_directory, exist_ok=True)
    tstmp = datetime.now().strftime('%Y%m%d_%H%M%S')
    try:
        fldr_path = os.path.join(output_directory, fldrname)
        os.mkdir(fldr_path)
    except FileExistsError:
        fldr_path = os.path.join(output_directory, fldrname + '_{}'.format(tstmp))
        os.mkdir(fldr_path)
    return fldr_path


def bin2d_with_indices(x: np.array, y: np.array, x_edges: np.array, y_edges: np.array):
    """
    Started out using scipy binned_statistic_2d, but I found that it would append bins regardless of the number of bins
    you ask for (even when all points are inside the explicit bin edges) and the binnumber would be difficult to
    translate.  Since our bin edges are always sorted, a 2d binning isn't really that hard, so we do it using
    searchsorted for speed.

    Parameters
    ----------
    x
        x coordinate of the points, should be same shape as y (one dimensional)
    y
        y coordinate of the points, should be same shape as x (one dimensional)
    x_edges
    y_edges

    Returns
    -------

    """
    xshape = x_edges.shape[0] - 1  # edges will be one longer than the number of tiles in this dimension
    yshape = y_edges.shape[0] - 1
    base_indices = np.arange(xshape * yshape).reshape(xshape, yshape)
    x_idx = np.searchsorted(x_edges, x, side='left') - 1
    y_idx = np.searchsorted(y_edges, y, side='left') - 1
    return base_indices[x_idx, y_idx]


class BathyGrid(Grid):
    """
    Manage a rectangular grid of Tiles, each able to operate independently and in parallel.  BathyGrid automates the
    creation and updating of each Tile, which happens under the hood when you add or remove points.
    """
    def __init__(self, tile_size: int = 1024):
        super().__init__(tile_size=tile_size)
        self.point_data = None

        self.container = {}  # dict of container name, list of multibeam files
        self.epsg = None  # epsg code
        self.vertical_reference = None  # string identifier for the vertical reference

        self.min_grid_resolution = None
        self.max_grid_resolution = None

        self.layer_lookup = {'depth': 'z', 'vertical_uncertainty': 'tvu'}
        self.rev_layer_lookup = {'z': 'depth', 'tvu': 'vertical_uncertainty'}

    def _convert_dataset(self):
        """
        We currently convert xarray Dataset input into a numpy structured array.  Xarry Datasets appear to be rather
        slow in testing, I believe because they do some stuff under the hood with matching coordinates when you do
        basic operations.  Also, you can't do any fancy indexing with xarray Dataset, at best you can use slice with isel.

        For all these reasons, we just convert to numpy.
        """
        allowed_vars = ['x', 'y', 'z', 'tvu', 'thu']
        dtyp = [(varname, self.point_data[varname].dtype) for varname in allowed_vars if varname in self.point_data]
        empty_struct = np.empty(len(self.point_data['x']), dtype=dtyp)
        for varname, vartype in dtyp:
            empty_struct[varname] = self.point_data[varname].values
        self.point_data = empty_struct

    def _update_metadata(self, container_name: str = None, multibeam_file_list: list = None, epsg: int = None,
                         vertical_reference: str = None):
        """
        Update the bathygrid metadata for the new data

        Parameters
        ----------
        container_name
            the folder name of the converted data, equivalent to splitting the output_path variable in the kluster
            dataset
        multibeam_file_list
            list of multibeam files that exist in the data to add to the grid
        epsg
            epsg (or proj4 string) for the coordinate system of the data.  Proj4 only shows up when there is no valid
            epsg
        vertical_reference
            vertical reference of the data
        """

        if multibeam_file_list:
            self.container[container_name] = multibeam_file_list
        else:
            self.container[container_name] = ['Unknown']

        if self.epsg and (self.epsg != int(epsg)):
            raise ValueError('BathyGrid: Found existing coordinate system {}, new coordinate system {} must match'.format(self.epsg,
                                                                                                                          epsg))
        if self.vertical_reference and (self.vertical_reference != vertical_reference):
            raise ValueError('BathyGrid: Found existing vertical reference {}, new vertical reference {} must match'.format(self.vertical_reference,
                                                                                                                            vertical_reference))
        self.epsg = int(epsg)
        self.vertical_reference = vertical_reference

    def _validate_input_data(self):
        """
        Ensure you get a structured numpy array as the input dataset.  If dataset is an Xarray Dataset, we convert it to
        Numpy for performance reasons.
        """

        if type(self.point_data) in [np.ndarray, Array]:
            if not self.point_data.dtype.names:
                raise ValueError('BathyGrid: numpy array provided for data, but no names were found, array must be a structured array')
            if 'x' not in self.point_data.dtype.names or 'y' not in self.point_data.dtype.names:
                raise ValueError('BathyGrid: numpy structured array provided for data, but "x" or "y" not found in variable names')
            self.layernames = [self.rev_layer_lookup[var] for var in self.point_data.dtype.names if var in ['z', 'tvu']]
        elif type(self.point_data) == xr.Dataset:
            if 'x' not in self.point_data:
                raise ValueError('BathyGrid: xarray Dataset provided for data, but "x" or "y" not found in variable names')
            if len(self.point_data.dims) > 1:
                raise ValueError('BathyGrid: xarray Dataset provided for data, but found multiple dimensions, must be one dimensional: {}'.format(self.point_data.dims))
            self.layernames = [self.rev_layer_lookup[var] for var in self.point_data if var in ['z', 'tvu']]
            self._convert_dataset()  # internally we just convert xarray dataset to numpy for ease of use
        else:
            raise ValueError('QuadTree: numpy structured array or dask array with "x" and "y" as variable must be provided')

    def _update_base_grid(self):
        """
        If the user adds new points, we need to make sure that we don't need to extend the grid in a direction.
        Extending a grid will build a new existing_tile_index for where the old tiles need to go in the new grid, see
        _update_tiles.
        """
        # extend the grid for new data or start a new grid if there are no existing tiles
        if self.point_data is not None:
            if self.is_empty:  # starting a new grid
                self._init_from_extents(self.point_data['y'].min(), self.point_data['x'].min(), self.point_data['y'].max(),
                                        self.point_data['x'].max())
            else:
                self._update_extents(self.point_data['y'].min(), self.point_data['x'].min(), self.point_data['y'].max(),
                                     self.point_data['x'].max())

    def _update_tiles(self, container_name):
        """
        Pick up existing tiles and put them in the correct place in the new grid.  Then add the new points to all of
        the tiles.

        Parameters
        ----------
        container_name
            the folder name of the converted data, equivalent to splitting the output_path variable in the kluster
            dataset
        """

        if self.point_data is not None:
            if self.is_empty:  # build empty list the same size as the tile attribute arrays
                self.tiles = np.full(self.tile_x_origin.shape, None, dtype=object)
            else:
                new_tiles = np.full(self.tile_x_origin.shape, None, dtype=object)
                new_tiles[self.existing_tile_mask] = self.tiles.ravel()
                self.tiles = new_tiles
            self._add_points_to_tiles(container_name)

    def _add_points_to_tiles(self, container_name):
        """
        Add new points to the Tiles.  Will run bin2d to figure out which points go in which Tiles.  If there is no Tile
        where the points go, will build a new Tile and add the points to it.  Otherwise, adds the points to an existing
        Tile.  If the container_name is already in the Tile (we have previously added these points), the Tile will
        clear out old points and replace them with new.

        If for some reason the resulting state of the Tile is empty (no points in the Tile) we replace the Tile with None.

        Parameters
        ----------
        container_name
            the folder name of the converted data, equivalent to splitting the output_path variable in the kluster
            dataset
        """

        if self.point_data is not None:
            binnum = bin2d_with_indices(self.point_data['x'], self.point_data['y'], self.tile_edges_x, self.tile_edges_y)
            unique_locs = np.unique(binnum)
            flat_tiles = self.tiles.ravel()
            tilexorigin = self.tile_x_origin.ravel()
            tileyorigin = self.tile_y_origin.ravel()
            for ul in unique_locs:
                point_mask = binnum == ul
                pts = self.point_data[point_mask]
                if flat_tiles[ul] is None:
                    flat_tiles[ul] = Tile(tilexorigin[ul], tileyorigin[ul], self.tile_size)
                flat_tiles[ul].add_points(pts, container_name)
                if flat_tiles[ul].is_empty:
                    flat_tiles[ul] = None

    def add_points(self, data: Union[xr.Dataset, Array, np.ndarray], container_name: str,
                   multibeam_file_list: list = None, crs: int = None, vertical_reference: str = None):
        """
        Add new points to the grid.  Build new tiles to encapsulate those points, or add the points to existing Tiles
        if they fall within existing tile boundaries.

        Parameters
        ----------
        data
            Sounding data from Kluster.  Should contain at least 'x', 'y', 'z' variable names/data
        container_name
            the folder name of the converted data, equivalent to splitting the output_path variable in the kluster
            dataset
        multibeam_file_list
            list of multibeam files that exist in the data to add to the grid
        crs
            epsg (or proj4 string) for the coordinate system of the data.  Proj4 only shows up when there is no valid
            epsg
        vertical_reference
            vertical reference of the data
        """

        if isinstance(data, (Array, xr.Dataset)):
            data = data.compute()
        self.point_data = data
        self._validate_input_data()
        self._update_metadata(container_name, multibeam_file_list, crs, vertical_reference)
        self._update_base_grid()
        self._update_tiles(container_name)
        self.point_data = None  # points are in the tiles, clear this attribute to free up memory

    def remove_points(self, container_name: str = None):
        """
        We go through all the existing Tiles and remove the points associated with container_name

        Parameters
        ----------
        container_name
            the folder name of the converted data, equivalent to splitting the output_path variable in the kluster
            dataset
        """
        if container_name in self.container:
            self.container.pop(container_name)
            if not self.is_empty:
                flat_tiles = self.tiles.ravel()
                for tile in flat_tiles:
                    if tile:
                        tile.remove_container(container_name)
                        if tile.is_empty:
                            flat_tiles[flat_tiles == tile] = None
            if self.is_empty:
                self.tiles = None
