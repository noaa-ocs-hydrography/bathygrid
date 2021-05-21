import numpy as np
import xarray as xr
from dask.array import Array
from typing import Union

from bathygrid.grids import BaseGrid
from bathygrid.tile import SRTile
from bathygrid.utilities import bin2d_with_indices


class BathyGrid(BaseGrid):
    """
    Manage a rectangular grid of tiles, each able to operate independently and in parallel.  BathyGrid automates the
    creation and updating of each Tile, which happens under the hood when you add or remove points.
    """
    def __init__(self, min_x: float = 0, min_y: float = 0, max_x: float = 0, max_y: float = 0, tile_size: float = 1024.0,
                 set_extents_manually: bool = False):
        super().__init__(min_x=min_x, min_y=min_y, tile_size=tile_size)

        if set_extents_manually:
            self.min_x = min_x
            self.min_y = min_y
            self.max_x = max_x
            self.max_y = max_y

        self.epsg = None  # epsg code
        self.vertical_reference = None  # string identifier for the vertical reference

        self.min_grid_resolution = None
        self.max_grid_resolution = None

        self.layer_lookup = {'depth': 'z', 'vertical_uncertainty': 'tvu', 'horizontal_uncertainty': 'thu'}
        self.rev_layer_lookup = {'z': 'depth', 'tvu': 'vertical_uncertainty', 'thu': 'horizontal_uncertainty'}

        self.tile_class = SRTile

    def _convert_dataset(self):
        """
        inherited class can write code here to convert the input data
        """
        pass

    def _update_metadata(self, container_name: str = None, file_list: list = None, epsg: int = None,
                         vertical_reference: str = None):
        """
        inherited class can write code here to handle the metadata
        """
        pass

    def _validate_input_data(self):
        """
        inherited class can write code here to validate the input data
        """
        pass

    def _update_base_grid(self):
        """
        If the user adds new points, we need to make sure that we don't need to extend the grid in a direction.
        Extending a grid will build a new existing_tile_index for where the old tiles need to go in the new grid, see
        _update_tiles.
        """
        # extend the grid for new data or start a new grid if there are no existing tiles
        if self.data is not None:
            if self.is_empty:  # starting a new grid
                if self.can_grow:
                    self._init_from_extents(self.data['y'].min(), self.data['x'].min(), self.data['y'].max(),
                                            self.data['x'].max())
                else:
                    self._init_from_extents(self.min_y, self.min_x, self.max_y, self.max_x)
            elif self.can_grow:
                self._update_extents(self.data['y'].min(), self.data['x'].min(), self.data['y'].max(),
                                     self.data['x'].max())
            else:  # grid can't grow, so we just leave existing tiles where they are
                pass

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

        if self.data is not None:
            if self.is_empty:  # build empty list the same size as the tile attribute arrays
                self.tiles = np.full(self.tile_x_origin.shape, None, dtype=object)
            elif self.can_grow:
                new_tiles = np.full(self.tile_x_origin.shape, None, dtype=object)
                new_tiles[self.existing_tile_mask] = self.tiles.ravel()
                self.tiles = new_tiles
            else:  # grid can't grow, so we just leave existing tiles where they are
                pass
            self._add_points_to_tiles(container_name)

    def _add_points_to_tiles(self, container_name):
        """
        Add new points to the tiles.  Will run bin2d to figure out which points go in which tiles.  If there is no tile
        where the points go, will build a new tile and add the points to it.  Otherwise, adds the points to an existing
        tile.  If the container_name is already in the tile (we have previously added these points), the tile will
        clear out old points and replace them with new.

        If for some reason the resulting state of the tile is empty (no points in the tile) we replace the tile with None.

        Parameters
        ----------
        container_name
            the folder name of the converted data, equivalent to splitting the output_path variable in the kluster
            dataset
        """

        if self.data is not None:
            binnum = bin2d_with_indices(self.data['x'], self.data['y'], self.tile_edges_x, self.tile_edges_y)
            unique_locs = np.unique(binnum)
            flat_tiles = self.tiles.ravel()
            tilexorigin = self.tile_x_origin.ravel()
            tileyorigin = self.tile_y_origin.ravel()
            for ul in unique_locs:
                point_mask = binnum == ul
                pts = self.data[point_mask]
                if flat_tiles[ul] is None:
                    flat_tiles[ul] = self.tile_class(tilexorigin[ul], tileyorigin[ul], self.tile_size)
                flat_tiles[ul].add_points(pts, container_name)
                if flat_tiles[ul].is_empty:
                    flat_tiles[ul] = None

    def add_points(self, data: Union[xr.Dataset, Array, np.ndarray], container_name: str,
                   file_list: list = None, crs: int = None, vertical_reference: str = None):
        """
        Add new points to the grid.  Build new tiles to encapsulate those points, or add the points to existing tiles
        if they fall within existing tile boundaries.

        Parameters
        ----------
        data
            Sounding data from Kluster.  Should contain at least 'x', 'y', 'z' variable names/data
        container_name
            the folder name of the converted data, equivalent to splitting the output_path variable in the kluster
            dataset
        file_list
            list of multibeam files that exist in the data to add to the grid
        crs
            epsg (or proj4 string) for the coordinate system of the data.  Proj4 only shows up when there is no valid
            epsg
        vertical_reference
            vertical reference of the data
        """

        if isinstance(data, (Array, xr.Dataset)):
            data = data.compute()
        self.data = data
        self._validate_input_data()
        self._update_metadata(container_name, file_list, crs, vertical_reference)
        self._update_base_grid()
        self._update_tiles(container_name)
        self.data = None  # points are in the tiles, clear this attribute to free up memory

    def remove_points(self, container_name: str = None):
        """
        We go through all the existing tiles and remove the points associated with container_name

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
                        tile.remove_points(container_name)
                        if tile.is_empty:
                            flat_tiles[flat_tiles == tile] = None
            if self.is_empty:
                self.tiles = None

    def grid(self, resolution: float, algorithm: str):
        flat_tiles = self.tiles.ravel()
        for tile in flat_tiles:
            if tile:
                tile.grid(resolution, algorithm)


class SRGrid(BathyGrid):
    def __init__(self, min_x: float = 0, min_y: float = 0, tile_size: float = 1024.0):
        super().__init__(min_x=min_x, min_y=min_y, tile_size=tile_size)
        self.tile_class = SRTile
        self.can_grow = True

    def _convert_dataset(self):
        """
        We currently convert xarray Dataset input into a numpy structured array.  Xarry Datasets appear to be rather
        slow in testing, I believe because they do some stuff under the hood with matching coordinates when you do
        basic operations.  Also, you can't do any fancy indexing with xarray Dataset, at best you can use slice with isel.

        For all these reasons, we just convert to numpy.
        """
        allowed_vars = ['x', 'y', 'z', 'tvu', 'thu']
        dtyp = [(varname, self.data[varname].dtype) for varname in allowed_vars if varname in self.data]
        empty_struct = np.empty(len(self.data['x']), dtype=dtyp)
        for varname, vartype in dtyp:
            empty_struct[varname] = self.data[varname].values
        self.data = empty_struct

    def _update_metadata(self, container_name: str = None, file_list: list = None, epsg: int = None,
                         vertical_reference: str = None):
        """
        Update the bathygrid metadata for the new data

        Parameters
        ----------
        container_name
            the folder name of the converted data, equivalent to splitting the output_path variable in the kluster
            dataset
        file_list
            list of multibeam files that exist in the data to add to the grid
        epsg
            epsg (or proj4 string) for the coordinate system of the data.  Proj4 only shows up when there is no valid
            epsg
        vertical_reference
            vertical reference of the data
        """

        if file_list:
            self.container[container_name] = file_list
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

        if type(self.data) in [np.ndarray, Array]:
            if not self.data.dtype.names:
                raise ValueError('BathyGrid: numpy array provided for data, but no names were found, array must be a structured array')
            if 'x' not in self.data.dtype.names or 'y' not in self.data.dtype.names:
                raise ValueError('BathyGrid: numpy structured array provided for data, but "x" or "y" not found in variable names')
            self.layernames = [self.rev_layer_lookup[var] for var in self.data.dtype.names if var in ['z', 'tvu']]
        elif type(self.data) == xr.Dataset:
            if 'x' not in self.data:
                raise ValueError('BathyGrid: xarray Dataset provided for data, but "x" or "y" not found in variable names')
            if len(self.data.dims) > 1:
                raise ValueError('BathyGrid: xarray Dataset provided for data, but found multiple dimensions, must be one dimensional: {}'.format(self.data.dims))
            self.layernames = [self.rev_layer_lookup[var] for var in self.data if var in ['z', 'tvu']]
            self._convert_dataset()  # internally we just convert xarray dataset to numpy for ease of use
        else:
            raise ValueError('QuadTree: numpy structured array or dask array with "x" and "y" as variable must be provided')


class VRGridTile(SRGrid):
    def __init__(self, min_x: float = 0, min_y: float = 0, tile_size: int = 1024, subtile_size: int = 128):
        super().__init__(min_x=min_x, min_y=min_y, tile_size=tile_size)
        self.tile_class = BathyGrid
        self.can_grow = True
        self.subtile_size = subtile_size

    def _add_points_to_tiles(self, container_name):
        """
        Add new points to the tiles.  Will run bin2d to figure out which points go in which tiles.  If there is no tile
        where the points go, will build a new tile and add the points to it.  Otherwise, adds the points to an existing
        tile.  If the container_name is already in the tile (we have previously added these points), the tile will
        clear out old points and replace them with new.

        If for some reason the resulting state of the tile is empty (no points in the tile) we replace the tile with None.

        Parameters
        ----------
        container_name
            the folder name of the converted data, equivalent to splitting the output_path variable in the kluster
            dataset
        """

        if self.data is not None:
            binnum = bin2d_with_indices(self.data['x'], self.data['y'], self.tile_edges_x, self.tile_edges_y)
            unique_locs = np.unique(binnum)
            flat_tiles = self.tiles.ravel()
            tilexorigin = self.tile_x_origin.ravel()
            tileyorigin = self.tile_y_origin.ravel()
            for ul in unique_locs:
                point_mask = binnum == ul
                pts = self.data[point_mask]
                if flat_tiles[ul] is None:
                    flat_tiles[ul] = self.tile_class(min_x=tilexorigin[ul], min_y=tileyorigin[ul], max_x=tilexorigin[ul] + self.tile_size,
                                                     max_y=tileyorigin[ul] + self.tile_size, tile_size=self.subtile_size, set_extents_manually=True)
                flat_tiles[ul].add_points(pts, container_name)
                if flat_tiles[ul].is_empty:
                    flat_tiles[ul] = None
