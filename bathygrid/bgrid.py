import os
from dask.distributed import wait, progress
import matplotlib.pyplot as plt
from xml.etree import ElementTree as et
from xml.dom import minidom
import numpy as np
from dask.array import Array
import xarray as xr
from datetime import datetime
import h5py
from pyproj import CRS
from typing import Union

from bathygrid.grids import BaseGrid
from bathygrid.tile import SRTile, Tile
from bathygrid.utilities import bin2d_with_indices, dask_find_or_start_client, print_progress_bar, \
    utc_seconds_to_formatted_string, formatted_string_to_utc_seconds, create_folder, gdal_raster_create, return_gdal_version
from bathygrid.grid_variables import depth_resolution_lookup, maximum_chunk_dimension, sr_grid_root_names
from bathygrid.__version__ import __version__ as bathyvers


class BathyGrid(BaseGrid):
    """
    Manage a rectangular grid of tiles, each able to operate independently and in parallel.  BathyGrid automates the
    creation and updating of each Tile, which happens under the hood when you add or remove points.

    Used in the VRGridTile as the tiles of the master grid.  Each tile of the VRGridTile is a BathyGrid with tiles within
    that grid.
    """

    def __init__(self, min_x: float = 0, min_y: float = 0, max_x: float = 0, max_y: float = 0, tile_size: float = 1024.0,
                 set_extents_manually: bool = False, output_folder: str = '', is_backscatter: bool = False):
        super().__init__(min_x=min_x, min_y=min_y, tile_size=tile_size)

        if set_extents_manually:
            self.min_x = min_x
            self.min_y = min_y
            self.max_x = max_x
            self.max_y = max_y

        self.is_backscatter = is_backscatter
        self.mean_depth = 0.0

        self.epsg = None  # epsg code
        self.vertical_reference = None  # string identifier for the vertical reference
        self.min_time = ''
        self.max_time = ''
        self.resolutions = []
        self.container_timestamp = {}

        self.name = ''
        self.output_folder = output_folder
        self.subtile_size = 0
        self.grid_algorithm = ''
        self.grid_resolution = ''
        self.grid_parameters = None
        self.sub_type = 'srtile'
        self.storage_type = 'numpy'
        self.client = None
        self.version = bathyvers

    def __repr__(self):
        output = 'Bathygrid Version: {}\n'.format(self.version)
        output += 'Resolutions (meters): {}\n'.format(self.resolutions)
        output += 'Containers: {}\n'.format('\n'.join(list(self.container.keys())))
        output += 'Backscatter Mosaic: {}\n'.format(self.is_backscatter)
        if not self.is_backscatter:
            output += 'Mean Depth: {}\n'.format(self.mean_depth)
        else:
            output += 'Mean Intensity: {}\n'.format(self.mean_depth)
        try:
            output += 'Minimum Northing: {} '.format(self.min_y)
            output += 'Maximum Northing: {}\n'.format(self.max_y)
        except:
            output += 'Minimum Northing: Unknown '
            output += 'Maximum Northing: Unknown\n'
        try:
            output += 'Minimum Easting: {} '.format(self.min_x)
            output += 'Maximum Easting: {}\n'.format(self.max_x)
        except:
            output += 'Minimum Easting: Unknown '
            output += 'Maximum Easting: Unknown\n'
        return output

    @property
    def depth_key(self):
        """
        Return the string identifier for the layer that represents the z value in the bathygrid

        Returns
        -------
        str
            layer name for z value
        """

        if self.is_backscatter:
            return 'intensity'
        else:
            return 'depth'

    @property
    def no_grid(self):
        """
        Simple check to see if this instance contains gridded data or not.  Looks for the first existing Tile and checks
        if that Tile has a grid.

        Returns
        -------
        bool
            True if the BathyGrid instance contains no grids, False if it does contain a grid.
        """

        if self.tiles is None:
            return True

        for tile in self.tiles.flat:
            if tile:
                if isinstance(tile, BathyGrid):
                    for subtile in tile.tiles.flat:
                        if subtile:
                            tile = subtile
                if tile.cells:
                    return False
                else:
                    return True

    @property
    def has_tiles(self):
        """
        BathyGrids can either contain more BathyGrids or contain Tiles with point/gridded data.  This check determines
        whether or not this instance contains Tiles
        """

        if self.tiles is not None:
            for tile in self.tiles.flat:
                if tile:
                    if isinstance(tile, Tile):
                        return True
        return False

    @property
    def layer_names(self):
        """
        Get the existing layer names in the tiles by checking the first real tile
        """

        if self.tiles is not None:
            for tile in self.tiles.flat:
                if tile:
                    layernames = tile.layer_names
                    return layernames
        return []

    @property
    def cell_count(self):
        """
        Return the total cell count for each resolution, cells being the gridded values in each tile.
        """

        final_count = {}
        if self.tiles is not None:
            for tile in self.tiles.flat:
                if tile:
                    tcell = tile.cell_count
                    for rez in tcell:
                        if rez in final_count:
                            final_count[rez] += tcell[rez]
                        else:
                            final_count[rez] = tcell[rez]
        return final_count

    @property
    def density_count(self):
        """
        Return the number of soundings per cell in all populated cells as a one dimensional list of counts, cells being
        the gridded values in each tile.
        """

        density_values = []
        if self.tiles is not None:
            for tile in self.tiles.flat:
                if tile:
                    density_values.extend(tile.density_count)
        return density_values

    @property
    def density_per_square_meter(self):
        """
        Return the density per cell per square meter in all populated cells as a one dimensional list of counts, cells
        being the gridded values in each tile.
        """

        density_values = []
        if self.tiles is not None:
            for tile in self.tiles.flat:
                if tile:
                    density_values.extend(tile.density_per_square_meter)
        return density_values

    @property
    def density_count_vs_depth(self):
        """
        Return the number of soundings per cell and the depth value of the cell in all populated cells, as two lists
        """

        density_values = []
        depth_values = []
        if self.tiles is not None:
            for tile in self.tiles.flat:
                if tile:
                    density, depth = tile.density_count_vs_depth
                    density_values.extend(density)
                    depth_values.extend(depth)
        return density_values, depth_values

    @property
    def density_per_square_meter_vs_depth(self):
        """
        Return the density per cell per square meter and the depth value of the cell in all populated cells, as two lists
        """

        density_values = []
        depth_values = []
        if self.tiles is not None:
            for tile in self.tiles.flat:
                if tile:
                    density, depth = tile.density_per_square_meter_vs_depth
                    density_values.extend(density)
                    depth_values.extend(depth)
        return density_values, depth_values

    @property
    def coverage_area_square_meters(self):
        """
        Return the coverage area of this grid in square meters
        """

        area = 0.0
        if self.tiles is not None:
            for tile in self.tiles.flat:
                if tile:
                    area += tile.coverage_area_square_meters
        return area

    @property
    def coverage_area_square_nm(self):
        """
        Return the coverage area of this grid in square nautical miles
        """

        area = 0.0
        if self.tiles is not None:
            for tile in self.tiles.flat:
                if tile:
                    area += tile.coverage_area_square_nm
        return area

    @property
    def point_count_changed(self):
        """
        Return True if any of the tiles in this grid have a point_count_changed (which is set when points are added/removed)
        """

        if self.tiles is not None:
            for tile in self.tiles.flat:
                if tile:
                    if tile.point_count_changed:
                        return True
        return False

    @property
    def positive_up(self):
        if self.vertical_reference:
            if self.vertical_reference.find('height (h)",up') > -1:
                return True
            else:
                return False
        return False

    def get_geotransform(self, resolution: float):
        """
        Return the summation of the geotransforms for all tiles in this grid and a place holder for tile count
        [x origin, x pixel size, x rotation, y origin, y rotation, -y pixel size]
        """

        parent_transform = [np.float32(self.min_x), resolution, 0, np.float32(self.max_y), 0, -resolution]

        return parent_transform

    def _update_metadata(self, container_name: str = None, file_list: list = None, epsg: int = None,
                         vertical_reference: str = None, min_time: float = None, max_time: float = None):
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
        min_time
            Optional, the minimum time in UTC seconds, only included if you want to track the total min max time of the
            added data
        max_time
            Optional, the maximum time in UTC seconds, only included if you want to track the total min max time of the
            added data
        """

        if min_time:
            min_time = int(min_time)
            if self.min_time:
                mintime = min(formatted_string_to_utc_seconds(self.min_time), min_time)
                self.min_time = utc_seconds_to_formatted_string(mintime)
            else:
                self.min_time = utc_seconds_to_formatted_string(min_time)
        if max_time:
            max_time = int(max_time)
            if self.max_time:
                maxtime = max(formatted_string_to_utc_seconds(self.max_time), max_time)
                self.max_time = utc_seconds_to_formatted_string(maxtime)
            else:
                self.max_time = utc_seconds_to_formatted_string(max_time)

        if file_list:
            self.container[container_name] = file_list
        else:
            self.container[container_name] = ['Unknown']
        self.container_timestamp[container_name] = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

        if self.epsg and epsg:
            if self.epsg != int(epsg):
                raise ValueError('BathyGrid: Found existing coordinate system {}, new coordinate system {} must match'.format(self.epsg,
                                                                                                                              epsg))
        if epsg:
            self.epsg = int(epsg)
        if self.vertical_reference and (self.vertical_reference != vertical_reference):
            raise ValueError('BathyGrid: Found existing vertical reference {}, new vertical reference {} must match'.format(self.vertical_reference,
                                                                                                                            vertical_reference))
        self.vertical_reference = vertical_reference

    def _update_mean_depth(self):
        """
        Calculate the mean depth of all loaded points before they are loaded into tiles and cleared from this object
        """

        if self.data is None or not self.data['z'].any():
            self.mean_depth = 0.0
        else:
            self.mean_depth = np.round(self.data['z'].mean(), 3)

    def _calculate_resolution_lookup(self):
        """
        Use the depth resolution lookup to find the appropriate depth resolution band.  The lookup is the max depth and
        the resolution that applies.

        Returns
        -------
        float
            resolution to use at the existing mean_depth
        """

        if not self.mean_depth:
            raise ValueError('Bathygrid: Unable to calculate resolution when mean_depth is None')
        dpth_keys = list(depth_resolution_lookup.keys())
        # get next positive value in keys of resolution lookup
        range_index = np.argmax((np.array(dpth_keys) - self.mean_depth) > 0)
        calc_resolution = depth_resolution_lookup[dpth_keys[range_index]]
        # ensure that resolution does not exceed the tile size for obvious reasons
        clipped_rez = min(self.tile_size, calc_resolution)
        return float(clipped_rez)

    def resolution_by_density(self, starting_resolution: float = None):
        """
        Determine the resolution according to the density of the points in all tiles, returns the coarsest resolution
        determined for the tiles in this grid.  See tile.resolution_by_density

        Parameters
        ----------
        starting_resolution
            the first resolution to evaluate, will go up/down from this resolution in the iterative check

        Returns
        -------
        float
            coarsest resolution amongst all tiles
        """

        rez_options = []
        for row in self.tiles:
            for tile in row:
                if tile:
                    tile_rez = tile.resolution_by_density(starting_resolution)
                    rez_options.append(tile_rez)
        if rez_options:
            return max(rez_options)
        else:
            return 0.0

    def _build_tile(self, tile_x_origin: float, tile_y_origin: float):
        """
        Default tile of the BathyGrid is just a simple SRTile.  More sophisticated grids will override this method to
        return the tile of their choice

        Parameters
        ----------
        tile_x_origin
            x origin coordinate for the tile, in the same units as the BathyGrid
        tile_y_origin
            y origin coordinate for the tile, in the same units as the BathyGrid

        Returns
        -------
        SRTile
            empty SRTile for this origin / tile size
        """

        return SRTile(tile_x_origin, tile_y_origin, self.tile_size, self.is_backscatter)

    def _build_empty_tile_space(self):
        """
        Build a 2d array of NaN for the size of one of the tiles.
        """

        return np.full((self.tile_size, self.tile_size), np.nan)

    def _build_layer_grid(self, resolution: float, layername: str, nodatavalue: float = np.float32(np.nan)):
        """
        Build a 2d array of NaN for the size of the whole BathyGrid (given the provided resolution)

        Parameters
        ----------
        resolution
            float resolution that we want to use to build the grid
        layername
            select layername to use for building the grid
        nodatavalue
            fill layer grid with nodatavalue where there is no data
        """

        if layername in ['depth', 'intensity', 'vertical_uncertainty', 'horizontal_uncertainty', 'total_uncertainty',
                         'hypothesis_ratio']:
            # ensure nodatavalue is a float32
            nodatavalue = np.float32(nodatavalue)
        elif layername in ['density', 'hypothesis_count']:
            # density has to have an integer based nodatavalue
            try:
                nodatavalue = np.int(nodatavalue)
            except ValueError:
                nodatavalue = 0
        y_size = self.height / resolution
        x_size = self.width / resolution
        assert y_size.is_integer()
        assert x_size.is_integer()
        return np.full((int(y_size), int(x_size)), nodatavalue)

    def _convert_dataset(self):
        """
        inherited class can write code here to convert the input data
        """
        pass

    def _validate_input_data(self):
        """
        inherited class can write code here to validate the input data
        """
        pass

    def _save_grid(self):
        """
        inherited class can write code here to save the grid data, see backends
        """
        pass

    def _save_tile(self, tile: Tile, flat_index: int, only_points: bool = False, only_grid: bool = False):
        """
        inherited class can write code here to save the tile data, see backends
        """
        pass

    def _load_grid(self):
        """
        inherited class can write code here to load the grid data, see backends
        """
        pass

    def _load_tile(self, flat_index: int, only_points: bool = False, only_grid: bool = False):
        """
        inherited class can write code here to load the tile data, see backends
        """
        pass

    def _load_tile_data_to_memory(self, tile: Tile, only_points: bool = False, only_grid: bool = False):
        pass

    def save(self, folderpath: str = None, progress_bar: bool = True):
        """
        inherited class can write code here to load the tile data, see backends
        """
        pass

    def load(self, folderpath: str = None):
        """
        inherited class can write code here to load the tile data, see backends
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

    def _update_tiles(self, container_name: str, progress_bar: bool):
        """
        Pick up existing tiles and put them in the correct place in the new grid.  Then add the new points to all of
        the tiles.

        Parameters
        ----------
        container_name
            the folder name of the converted data, equivalent to splitting the output_path variable in the kluster
            dataset
        progress_bar
            if True, display a progress bar
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
            self._add_points_to_tiles(container_name, progress_bar)

    def _add_points_to_tiles(self, container_name: str, progress_bar: bool):
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
        progress_bar
            if True, display a progress bar
        """

        if self.data is not None:
            self._save_grid()
            if not isinstance(self.tile_edges_x, np.ndarray):
                self.tile_edges_x = self.tile_edges_x.compute()
                self.tile_edges_y = self.tile_edges_y.compute()
            binnum = bin2d_with_indices(self.data['x'], self.data['y'], self.tile_edges_x, self.tile_edges_y)
            unique_locs = np.unique(binnum)
            flat_tiles = self.tiles.ravel()
            tilexorigin = self.tile_x_origin.ravel()
            tileyorigin = self.tile_y_origin.ravel()
            if progress_bar:
                print_progress_bar(0, len(unique_locs), 'Adding Points from {}:'.format(container_name))
            for cnt, ul in enumerate(unique_locs):
                if progress_bar:
                    print_progress_bar(cnt + 1, len(unique_locs), 'Adding Points from {}:'.format(container_name))
                point_mask = binnum == ul
                pts = self.data[point_mask]
                if flat_tiles[ul] is None:
                    flat_tiles[ul] = self._build_tile(tilexorigin[ul], tileyorigin[ul])
                    x, y = self._tile_idx_to_origin_point(ul)
                    flat_tiles[ul].name = '{}_{}'.format(x, y)
                if self.sub_type in ['srtile', 'quadtile']:
                    self._load_tile_data_to_memory(flat_tiles[ul])
                flat_tiles[ul].add_points(pts, container_name, progress_bar=False)
                if flat_tiles[ul].is_empty:
                    flat_tiles[ul] = None
                if self.sub_type in ['srtile', 'quadtile']:
                    # just save and reload the points
                    self._save_tile(flat_tiles[ul], ul, only_points=True)
                    self._load_tile(ul, only_points=True)
            self.number_of_tiles = np.count_nonzero(self.tiles != None)

    def add_points(self, data: Union[xr.Dataset, Array, np.ndarray], container_name: str, file_list: list = None,
                   crs: int = None, vertical_reference: str = None, min_time: float = None, max_time: float = None,
                   progress_bar: bool = True):
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
        min_time
            Optional, the minimum time in UTC seconds, only included if you want to track the total min max time of the
            added data
        max_time
            Optional, the maximum time in UTC seconds, only included if you want to track the total min max time of the
            added data
        progress_bar
            if True, display a progress bar
        """

        if isinstance(data, (Array, xr.Dataset)):
            data = data.compute()
        if container_name in self.container:
            raise ValueError('{} is already within this bathygrid instance, remove_points first if you want to replace this data'.format(container_name))
        self.data = data
        self._validate_input_data()
        self._update_metadata(container_name, file_list, crs, vertical_reference, min_time, max_time)
        self._update_base_grid()
        self._update_tiles(container_name, progress_bar)
        self._update_mean_depth()
        self.data = None  # points are in the tiles, clear this attribute to free up memory

    def _remove_tile(self, flat_index: int):
        """
        Removing a tile just involves setting the tile location in self.tiles to None, but we also want to update
        the metadata as well
        """

        tile = self.tiles.flat[flat_index]
        if tile:
            if self.existing_tile_mask is not None and self.existing_tile_mask.any():
                self.existing_tile_mask.flat[flat_index] = False
            self.number_of_tiles -= 1
            self.tiles.flat[flat_index] = None

    def remove_points(self, container_name: str = None, progress_bar: bool = True):
        """
        We go through all the existing tiles and remove the points associated with container_name

        Parameters
        ----------
        container_name
            the folder name of the converted data, equivalent to splitting the output_path variable in the kluster
            dataset
        progress_bar
            if True, display a progress bar
        """

        if container_name in self.container:
            self.container.pop(container_name)
            self.container_timestamp.pop(container_name)
            if not self.is_empty:
                flat_tiles = self.tiles.ravel()
                if progress_bar:
                    print_progress_bar(0, len(flat_tiles), 'Removing Points from {}:'.format(container_name))
                for cnt, tile in enumerate(flat_tiles):
                    if progress_bar:
                        print_progress_bar(cnt + 1, len(flat_tiles), 'Removing Points from {}:'.format(container_name))
                    if tile:
                        tile.remove_points(container_name, progress_bar=False)
                        if tile.is_empty:
                            self._remove_tile(cnt)
                    if self.sub_type in ['srtile', 'quadtile']:
                        # just save and reload the points
                        self._save_tile(flat_tiles[cnt], cnt, only_points=True)
                        self._load_tile(cnt, only_points=True)
            if self.is_empty:
                self.tiles = None
            self._save_grid()

    def get_tiles_by_resolution(self, resolution: float = None, layer: Union[str, list] = 'depth',
                                nodatavalue: float = np.float32(np.nan), z_positive_up: bool = False):
        """
        Tile generator to get the geotransform and data from all tiles

        Yields a generator that contains a tuple of (geotransform used by GDAL, tilecount), the column index, the row index, the number of cells
        in a row and the tile data as a dictionary (ex: {'depth': np.array([[...})

        Parameters
        ----------
        resolution
            resolution of the layer we want to access, if not provided will use the first resolution found, will error if there is
            more than one resolution in the grid
        layer
            string identifier for the layer(s) to access, valid layers include 'depth', 'intensity', 'vertical_uncertainty', 'horizontal_uncertainty', 'total_uncertainty',
            'hypothesis_ratio
        nodatavalue
            nodatavalue to set in the regular grid
        z_positive_up
            if True, will output bands with positive up convention
        """

        if not resolution:
            if len(self.resolutions) > 1:
                raise ValueError(
                    'BathyGrid: you must specify a resolution to return layer data when multiple resolutions are found')
            resolution = self.resolutions[0]
        if self.no_grid:
            raise ValueError('BathyGrid: Grid is empty, gridding has not been run yet.')
        if isinstance(layer, str):
            layer = [layer]
        for cnt, tile in enumerate(self.tiles.flat):
            if tile:
                row, col = self._tile_idx_to_row_col(cnt)
                tile, data, geo, data_col, data_row, tile_cell_count = self.get_tile_data(row, col, resolution, layer, nodatavalue, z_positive_up)
                yield geo, data_col, data_row, tile_cell_count, data

    def get_tile_data(self, row_number: int, column_number: int, resolution: float, layer: Union[str, list] = 'depth',
                      nodatavalue: float = np.float32(np.nan), z_positive_up: bool = False):
        """
        Get the data and relevant information for the tile at the provided row/column number.  If the tile is a subgrid
        (vr grids have grids as tiles), this will return the data for that subgrid.

        You should use this method to access a tile, will also allow you to alter the z sign convention.

        Parameters
        ----------
        row_number
            row number of the desired tile
        column_number
            column number of the desired tile
        resolution
            resolution of the layer we want to access, if not provided will use the first resolution found, will error if there is
            more than one resolution in the grid
        layer
            string identifier for the layer(s) to access, valid layers include 'depth', 'intensity', 'vertical_uncertainty', 'horizontal_uncertainty', 'total_uncertainty',
            'hypothesis_ratio
        nodatavalue
            nodatavalue to set in the regular grid
        z_positive_up
            if True, will output bands with positive up convention

        Returns
        -------
        Union[Tile,BathyGrid]
            Tile or Grid at the given row column index
        dict
            tile layer data for the given resolution
        list
            [x origin, x pixel size, x rotation, y origin, y rotation, -y pixel size] for the given tile
        int
            column index in terms of cell count
        int
            row index in terms of cell count
        int
            width of the tile in number of cells
        """

        if self.positive_up:
            if z_positive_up:  # this is already positive up/height, so switch this off
                z_positive_up = False
            else:  # this is a positive up/height, so switch this on to flip the z convention of the returned data
                z_positive_up = True

        tile = self.tiles[row_number, column_number]
        tile_cell_count = self.tile_size / resolution
        assert tile_cell_count.is_integer()
        tile_cell_count = int(tile_cell_count)
        data_col, data_row = row_number * tile_cell_count, column_number * tile_cell_count
        geo = tile.get_geotransform(resolution)
        data = {}
        for cnt, lyr in enumerate(layer):
            newdata = tile.get_layers_by_name(lyr, resolution, nodatavalue=nodatavalue, z_positive_up=z_positive_up)
            if newdata is not None:
                if isinstance(newdata, list):  # true if 'tile' is actually a subgrid (BathyGrid)
                    newdata = newdata[0]
                data[lyr] = newdata
        return tile, data, geo, data_col, data_row, tile_cell_count

    def get_tile_boundaries(self):
        """
        Return the grid coordinates in such a way that if you were to draw them, you would get a grid.  Start with the
        tile edges and add some more coordinates to draw a box around each tile.  Finish off with a last column that
        gets you back to the row origin for the next row, so that you don't get a draw diagonal line from the end of
        the row to the start of the next row.

        Returns
        -------
        np.array
            1d array of x coordinates for the drawn grid in meters
        np.array
            1d array of y coordinates for the drawn grid in meters
        """

        cellboundaries_x, cellboundaries_y = np.meshgrid(self.tile_edges_x, self.tile_edges_y)
        total_x = np.zeros((cellboundaries_x.shape[0], cellboundaries_x.shape[1] * 5), dtype=np.float32)
        total_y = np.zeros((cellboundaries_x.shape[0], cellboundaries_x.shape[1] * 5), dtype=np.float32)
        total_x[:, ::5] = cellboundaries_x
        total_y[:, ::5] = cellboundaries_y
        total_x[:, 1::5] = cellboundaries_x + self.tile_size
        total_y[:, 1::5] = cellboundaries_y
        total_x[:, 2::5] = cellboundaries_x + self.tile_size
        total_y[:, 2::5] = cellboundaries_y + self.tile_size
        total_x[:, 3::5] = cellboundaries_x
        total_y[:, 3::5] = cellboundaries_y + self.tile_size
        total_x[:, 4::5] = cellboundaries_x
        total_y[:, 4::5] = cellboundaries_y

        # now add one last column to get back to the origin of the next row so you dont get a big diagonal line across your grid
        total_x = np.hstack((total_x, total_x[:, -1].reshape(total_x.shape[0], 1)))
        total_y = np.hstack((total_y, total_y[:, -1].reshape(total_y.shape[0], 1) + self.tile_size))
        # adjust corner point to the end of the grid
        total_y[-1, -1] = self.max_y

        return total_x.ravel(), total_y.ravel()

    def get_tile_neighbors(self, til: Union[Tile, BaseGrid]):
        """
        Return a list of Tile objects for the neighbors to the provided tile.  Neighbors will be in order of [above, right,
        down, left] in the returned list.

        Parameters
        ----------
        til
            Tile that you want to find the neighbors for

        Returns
        -------
        list
            list of neighbor Tiles
        """

        til_row, til_column = np.where(self.tiles == til)
        tils = []
        # up
        try:
            tils.append(self.tiles[til_row - 1, til_column][0])
        except:
            tils.append(None)
        # right
        try:
            tils.append(self.tiles[til_row, til_column + 1][0])
        except:
            tils.append(None)
        # down
        try:
            tils.append(self.tiles[til_row + 1, til_column][0])
        except:
            tils.append(None)
        # left
        try:
            tils.append(self.tiles[til_row, til_column - 1][0])
        except:
            tils.append(None)
        return tils

    def get_tile_neighbor_points(self, til: Tile, buffer_value: float):
        """
        Get the point data for all points that are in neighboring tiles to the provided tile and are within the buffer value
        from the tile border.

        Parameters
        ----------
        til
            Tile that you want to find the neighbor points for
        buffer_value
            offset in meters that you want to use to find the points in the neighboring tiles

        Returns
        -------
        numpy.ndarray
            point data for the neighboring tiles
        """

        data = []
        tils = self.get_tile_neighbors(til)
        # up
        if tils[0]:
            if isinstance(tils[0], Tile):  # single resolution option
                newdata = tils[0].data[tils[0].data['y'] >= (tils[0].max_y - buffer_value)]
                data.append(newdata)
            else:  # for variable resolution, tils will be a list of grids, get all tiles in the grid
                for subtil in tils[0].tiles.flat:
                    newdata = subtil.data[subtil.data['y'] >= (subtil.max_y - buffer_value)]
                    data.append(newdata)
        if tils[1]:
            if isinstance(tils[1], Tile):  # single resolution option
                newdata = tils[1].data[tils[1].data['x'] <= (tils[1].min_x + buffer_value)]
                data.append(newdata)
            else:
                for subtil in tils[1].tiles.flat:
                    newdata = subtil.data[subtil.data['x'] <= (subtil.min_x + buffer_value)]
                    data.append(newdata)
        if tils[2]:
            if isinstance(tils[2], Tile):  # single resolution option
                newdata = tils[2].data[tils[2].data['y'] <= (tils[2].min_y + buffer_value)]
                data.append(newdata)
            else:
                for subtil in tils[2].tiles.flat:
                    newdata = subtil.data[subtil.data['y'] <= (subtil.min_y + buffer_value)]
                    data.append(newdata)
        if tils[3]:
            if isinstance(tils[3], Tile):  # single resolution option
                newdata = tils[3].data[tils[3].data['x'] >= (tils[3].max_x - buffer_value)]
                data.append(newdata)
            else:
                for subtil in tils[3].tiles.flat:
                    newdata = subtil.data[subtil.data['x'] >= (subtil.max_x - buffer_value)]
                    data.append(newdata)
        if data:
            data = np.concatenate(data)
            if not isinstance(data, np.ndarray):
                data = data.compute()
            if data.size == 0:
                data = None
        else:
            data = None
        return data

    def _finalize_chunk(self, column_indices: list, row_indices: list, cells_per_tile: int, layers: list, layerdata: list,
                        nodatavalue: float, for_gdal: bool = True):
        """
        With get_chunks_of_tiles, we build a small grid/geotransform from many tiles until we hit the maximum chunk width.
        Here we take those tiles and build the small grid.            

        Parameters
        ----------
        column_indices
            integer row indices for where the tile fits in the small grid
        row_indices
            integer row indices for where the tile fits in the small grid
        cells_per_tile
            number of cells we would expect along one dimension for each tile (assumes tiles are all square)
        layers
            list of string identifiers for the layers we are interested in
        layerdata
            list of the tile data dicts for each tile
        for_gdal
            if True, performs numpy fliplr to conform to the GDAL specifications

        Returns
        -------
        dict
            dictionary of gridded data, ex: {'depth': np.array([[....})
        """
        finaldata = {}
        # normalize the column/row indices, the data and the geotransform yielded here are just for this chunk
        mincol = min(column_indices)
        curdcol = [col - mincol for col in column_indices]
        minrow = min(row_indices)
        curdrow = [row - minrow for row in row_indices]
        for lyr in layers:
            if lyr in ['depth', 'intensity', 'vertical_uncertainty', 'horizontal_uncertainty', 'total_uncertainty', 'hypothesis_ratio']:
                # ensure nodatavalue is a float32
                nodatavalue = np.float32(nodatavalue)
            elif lyr in ['density', 'hypothesis_count']:
                # density has to have an integer based nodatavalue
                try:
                    nodatavalue = np.int(nodatavalue)
                except ValueError:
                    nodatavalue = 0
            finaldata[lyr] = np.full((max(curdcol) + cells_per_tile, max(curdrow) + cells_per_tile), nodatavalue)
            for cnt, data in enumerate(layerdata):
                if lyr in data:
                    finaldata[lyr][curdcol[cnt]:curdcol[cnt] + cells_per_tile, curdrow[cnt]:curdrow[cnt] + cells_per_tile] = data[lyr]
            if for_gdal:
                finaldata[lyr] = np.fliplr(finaldata[lyr].T)
        return finaldata

    def _split_to_approx_shape(self, arr: np.ndarray, chunk_shape: tuple):
        """
        Take a 2d array and split it up into chunk_shape sized chunks.  We then return the != None values in the array
        to get the list of lists of values.

        Parameters
        ----------
        arr
            2d array
        chunk_shape
            size of the desired chunk, (row_dimension, column_dimension)

        Returns
        -------
        list
            list of lists of the values in each non-None cell
        """

        if arr.ndim != 2 or len(chunk_shape) != 2:
            raise NotImplementedError('_split_to_approx_shape: assumes two dimensions')

        if chunk_shape >= arr.shape:
            valid = arr != None
            if valid.any():
                return [arr[valid].tolist()]
            else:
                return []

        num_sections = int(np.ceil(arr.shape[0] / chunk_shape[0]))
        row_split = np.array_split(arr, num_sections, axis=0)
        num_col_sections = int(np.ceil(row_split[0].shape[1] / chunk_shape[1]))

        chnks = []
        for split_a in row_split:
            col_split = np.array_split(split_a, num_col_sections, axis=1)
            for split_b in col_split:
                valid = split_b != None
                if valid.any():
                    chnks.append(split_b[valid].tolist())
        return chnks

    def _tile_chunk_indices(self, max_chunk_dimension: float = None):
        """
        In order to return chunks of tiles in get_chunks_of_tiles, we need to figure out the tile indices of the tiles
        that are in each chunk.  We take a max chunk dimension, split our grid into chunks of grids, and return a list of
        the row,column indices for each real tile in that grid.

        Parameters
        ----------
        max_chunk_dimension
            size of the chunk, used for both dimensions, i.e. maxchunkdimension=2, chunk shape = (2,2)

        Returns
        -------
        list
            list of lists of row column indices for each real tile in each chunk
        """

        # ensure we get at least one tile, but pick the number of tiles that gets us less than or equal to max_chunk_dimension
        max_length = max(int(np.floor(max_chunk_dimension / self.tile_size)), 1)
        chunk_shape = (max_length, max_length)

        tindex = np.full_like(self.tiles, None)
        tloc = np.where(self.tiles != None)
        tindex[tloc] = np.column_stack(tloc).tolist()

        tile_rows, tile_columns = tloc
        min_tile_row, min_tile_column = min(tile_rows), min(tile_columns)
        max_tile_row, max_tile_column = max(tile_rows), max(tile_columns)
        tindex = tindex[min_tile_row:max_tile_row + 1, min_tile_column:max_tile_column + 1]
        return self._split_to_approx_shape(tindex, chunk_shape)

    def get_chunks_of_tiles(self, resolution: float = None, layer: Union[str, list] = 'depth',
                            nodatavalue: float = np.float32(np.nan), z_positive_up: bool = False,
                            override_maximum_chunk_dimension: float = None, for_gdal: bool = True):
        """
        Grid generator that builds out grids in chunks from the parent grid.  We use it here as building one large grid
        for the whole area sometimes causes memory issues.  This generator will return grids that are less than or equal
        in height and width to the maximum_chunk_dimension.

        Yields the GDAL Geotransform for the smaller grid, the max dimension of the smaller grid, and the dict of the
        smaller gridded dataset as a dictionary (ex: {'depth': np.array([[...})

        Parameters
        ----------
        resolution
            resolution of the layer we want to access, if not provided will use the first resolution found, will error if there is
            more than one resolution in the grid
        layer
            string identifier for the layer(s) to access, valid layers include 'depth', 'intensity', 'vertical_uncertainty', 'horizontal_uncertainty', 'total_uncertainty', 'hypothesis_ratio'
        nodatavalue
            nodatavalue to set in the regular grid
        z_positive_up
            if True, will output bands with positive up convention
        override_maximum_chunk_dimension
            by default, we use the grid_variables.maximum_chunk_dimension, use this optional argument if you want to
            override this value
        for_gdal
            if True, performs numpy fliplr to conform to the GDAL specifications
        """

        if isinstance(layer, str):
            layer = [layer]
        if override_maximum_chunk_dimension:
            max_dimension = override_maximum_chunk_dimension
        else:
            max_dimension = maximum_chunk_dimension
        tile_indices = self._tile_chunk_indices(max_dimension)
        for index_list in tile_indices:
            curgeo = None
            curdata = []
            curdcol = []
            curdrow = []
            curcellcount = []
            curmaxdimension = 0
            for trow, tcol in index_list:
                tile, data, geo, data_col, data_row, tile_cell_count = self.get_tile_data(trow, tcol, resolution, layer, nodatavalue, z_positive_up)
                curdata.append(data)
                curdcol.append(data_col)
                curdrow.append(data_row)
                curcellcount.append(tile_cell_count)
                if curgeo is None:
                    curgeo = geo
                else:
                    # merge the georeference of the two datasets, everything remains the same except the origin, which is now the global origin
                    curgeo = [min([curgeo[0], geo[0]]), curgeo[1], curgeo[2], max([curgeo[3], geo[3]]), curgeo[4], curgeo[5]]
                    # here we take the greatest dimension currently
                    curmaxdimension = max((max(curdcol) - min(curdcol) + tile_cell_count) * resolution,
                                          (max(curdrow) - min(curdrow) + tile_cell_count) * resolution)
            assert all(curcellcount)  # all the cell counts per tile should be the same for the one resolution
            finaldata = self._finalize_chunk(curdcol, curdrow, curcellcount[0], layer, curdata, nodatavalue, for_gdal)
            yield curgeo, curmaxdimension, finaldata

    def get_layers_by_name(self, layer: Union[str, list] = 'depth', resolution: float = None, nodatavalue: float = np.float32(np.nan),
                           z_positive_up: bool = False):
        """
        Return the numpy 2d grid for the provided layer, resolution.  Will check to ensure that you have gridded at this
        resolution already.  Grid returned will have NaN values for empty spaces.

        Parameters
        ----------
        layer
            string identifier for the layer(s) to access, valid layers include 'depth', 'intensity', 'vertical_uncertainty', 'horizontal_uncertainty', 'total_uncertainty', 'hypothesis_ratio'
        resolution
            resolution of the layer we want to access
        nodatavalue
            nodatavalue to set in the regular grid
        z_positive_up
            if True, will output bands with positive up convention

        Returns
        -------
        list
            list of gridded data for each provided layer, resolution across all tiles
        """

        # ensure nodatavalue is a float32
        nodatavalue = np.float32(nodatavalue)
        empty = True
        if isinstance(layer, str):
            layer = [layer]
        if self.no_grid:
            raise ValueError('BathyGrid: Grid is empty, gridding has not been run yet.')
        if not resolution:
            if len(self.resolutions) > 1:
                raise ValueError('BathyGrid: you must specify a resolution to return layer data when multiple resolutions are found')
            resolution = self.resolutions[0]
        data = [self._build_layer_grid(resolution, layername=lyr, nodatavalue=nodatavalue) for lyr in layer]
        for cnt, tile in enumerate(self.tiles.flat):
            if tile:
                row, col = self._tile_idx_to_row_col(cnt)
                tile, newdata, geo, data_col, data_row, tile_cell_count = self.get_tile_data(row, col, resolution, layer, nodatavalue, z_positive_up)
                for cnt, lyr in enumerate(newdata.keys()):
                    data[cnt][data_col:data_col + tile_cell_count, data_row:data_row + tile_cell_count] = newdata[lyr]
                    empty = False
        if empty:
            data = None
        return data

    def get_layers_trimmed(self, layer: Union[str, list] = 'depth', resolution: float = None, nodatavalue: float = np.float32(np.nan),
                           z_positive_up: bool = False):
        """
        Get the layer indicated by the provided layername and trim to the minimum bounding box of real values in the
        layer.

        Parameters
        ----------
        layer
            string identifier for the layer to access, one of 'depth', 'intensity', 'vertical_uncertainty', 'horizontal_uncertainty', 'total_uncertainty', 'hypothesis_ratio'
        resolution
            resolution of the layer we want to access
        nodatavalue
            nodatavalue to set in the regular grid
        z_positive_up
            if True, will output bands with positive up convention

        Returns
        -------
        list
            list of 2dim array of gridded layer trimmed to the minimum bounding box
        list
            new mins to use
        list
            new maxs to use
        """

        data = self.get_layers_by_name(layer, resolution, nodatavalue=nodatavalue, z_positive_up=z_positive_up)
        dat = data[0]  # just use the first layer, the mins/maxs should be the same for all layers

        notnan = ~np.isnan(dat)
        rows = np.any(notnan, axis=1)
        cols = np.any(notnan, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        rmax += 1
        cmax += 1

        for cnt, dat in enumerate(data):
            data[cnt] = dat[rmin:rmax, cmin:cmax]

        return data, [rmin, cmin], [rmax, cmax]

    def _grid_regular(self, algorithm: str, resolution: float, clear_existing: bool, regrid_option: str, auto_resolution: str,
                      progress_bar: bool = True, border_data: np.ndarray = None):
        """
        Run the gridding without Dask, Tile after Tile.

        Parameters
        ----------
        resolution
            resolution of the gridded data in the Tiles
        algorithm
            algorithm to grid by
        clear_existing
            if True, will clear out any existing grids before generating this one
        regrid_option
            controls what parts of the grid will get re-gridded if regrid is True and clear_existing is False, one of 'full', 'update'.  Full mode will
            regrid the entire grid.  Update mode will only update those tiles that have a point_count_changed=True.  If clear_existing is True, will
            automatically run in 'full' mode
        auto_resolution
            if density or depth, allow the tile to auto calculate the appropriate resolution.  If empty string, do not
            support resolution determination
        progress_bar
            if True, display a progress bar
        border_data
            point data that falls on the borders, used in the CUBE algorithm to handle tile edge issues
        """

        if progress_bar:
            print_progress_bar(0, self.tiles.size, 'Gridding {} - {}:'.format(self.name, algorithm))
        grid_border_data = border_data  # this is a vr thing, will include border data from neighboring grids as well.
        for cnt, tile in enumerate(self.tiles.flat):
            if progress_bar:
                print_progress_bar(cnt + 1, self.tiles.size, 'Gridding {} - {}:'.format(self.name, algorithm))
            if tile:
                if regrid_option == 'update' and not clear_existing:
                    # update only those tiles with new points/removed points.  If the point count hasn't changed, we skip
                    if not tile.point_count_changed:
                        for rz in tile.resolutions:
                            if rz not in self.resolutions:
                                self.resolutions.append(rz)
                        continue
                if algorithm == 'cube':
                    if auto_resolution:
                        border_data = self.get_tile_neighbor_points(tile, tile.width / 10)
                    else:
                        border_data = self.get_tile_neighbor_points(tile, min(tile.width / 10, resolution * 3))
                    if grid_border_data is not None:
                        if border_data is not None:
                            border_data = np.concatenate([border_data, grid_border_data])
                        else:
                            border_data = grid_border_data
                else:
                    border_data = None
                if isinstance(tile, BathyGrid) and auto_resolution:  # vrgrid subgrids can calc their own resolution
                    rez = tile.grid(algorithm, None, auto_resolution_mode=auto_resolution, clear_existing=clear_existing, regrid_option=regrid_option, progress_bar=False,
                                    grid_parameters=self.grid_parameters, border_data=border_data)
                elif isinstance(tile, SRTile) and auto_resolution and self.name not in sr_grid_root_names:  # tiles in vrgridtile can be different resolutions
                    rez = tile.grid(algorithm, None, auto_resolution_mode=auto_resolution, clear_existing=clear_existing, regrid_option=regrid_option, progress_bar=False,
                                    grid_parameters=self.grid_parameters, border_data=border_data)
                else:
                    rez = tile.grid(algorithm, resolution, auto_resolution_mode=auto_resolution, clear_existing=clear_existing, regrid_option=regrid_option, progress_bar=False,
                                    grid_parameters=self.grid_parameters, border_data=border_data)
                if isinstance(rez, float) or isinstance(rez, int):
                    rez = [rez]
                for rz in rez:
                    if rz not in self.resolutions:
                        self.resolutions.append(rz)
            if self.sub_type in ['srtile', 'quadtile']:
                # just save and reload the gridded data
                self._save_tile(tile, cnt, only_grid=True)
                self._load_tile(cnt, only_grid=True)
        self.resolutions = np.sort(np.unique(self.resolutions)).tolist()
        self._save_grid()

    def _grid_parallel_worker(self, data_for_workers: list, progress_bar: bool, tile_indices: list):
        futs = []
        data_for_workers = self.client.scatter(data_for_workers)
        futs.append(self.client.map(_gridding_parallel, data_for_workers))
        if progress_bar:
            progress(futs, multi=False)
        wait(futs)
        results = self.client.gather(futs)
        results = [item for sublist in results for item in sublist]
        resolutions = [res[0] for res in results]
        if isinstance(resolutions[0], list):  # this is true for vrgrids
            resolutions = [r for subrez in resolutions for r in subrez]
        tiles = [res[1] for res in results]
        for result_tile, tidx in zip(tiles, tile_indices):
            self.tiles.flat[tidx] = result_tile
            if self.sub_type in ['srtile', 'quadtile']:
                # just save and reload the gridded data
                self._save_tile(result_tile, tidx, only_grid=True)
                self._load_tile(tidx, only_grid=True)
        for rez in resolutions:
            if rez not in self.resolutions:
                self.resolutions.append(rez)
        tiles = None
        resolutions = None
        results = None

    def _grid_parallel(self, algorithm: str, resolution: float, clear_existing: bool, regrid_option: str, auto_resolution: str,
                       progress_bar: bool = True, border_data: np.ndarray = None):
        """
        Use Dask to submit the tiles in parallel to the cluster for processing.  Probably should think up a more
        intelligent way to do this than sending around the whole Tile obejct.  That object has a bunch of other stuff
        that isn't used by the gridding process.  Although maybe with lazy loading of data, that doesnt matter as much.

        Parameters
        ----------
        resolution
            resolution of the gridded data in the Tiles
        algorithm
            algorithm to grid by
        clear_existing
            if True, will clear out any existing grids before generating this one
        regrid_option
            controls what parts of the grid will get re-gridded if regrid is True and clear_existing is False, one of 'full', 'update'.  Full mode will
            regrid the entire grid.  Update mode will only update those tiles that have a point_count_changed=True.  If clear_existing is True, will
            automatically run in 'full' mode
        auto_resolution
            if density or depth, allow the tile to auto calculate the appropriate resolution.  If empty string, do not
            support resolution determination
        progress_bar
            if True, display a progress bar
        border_data
            point data that falls on the borders, used in the CUBE algorithm to handle tile edge issues
        """

        if not self.client:
            self.client = dask_find_or_start_client()

        chunks_at_a_time = len(self.client.ncores())
        total_runs = int(np.ceil(self.number_of_tiles / 8))
        cur_run = 1

        self.resolutions = []
        data_for_workers = []
        chunk_index = 0
        tile_indices = []
        grid_border_data = border_data  # this is a vr thing, will include border data from neighboring grids as well.

        for cnt, tile in enumerate(self.tiles.flat):
            if tile:
                if regrid_option == 'update' and not clear_existing:
                    # update only those tiles with new points/removed points.  If the point count hasn't changed, we skip
                    if not tile.point_count_changed:
                        for rz in tile.resolutions:
                            if rz not in self.resolutions:
                                self.resolutions.append(rz)
                        continue
                if self.sub_type in ['srtile', 'quadtile']:
                    self._load_tile_data_to_memory(tile)
                tile_indices.append(cnt)
                if algorithm == 'cube':
                    if auto_resolution:
                        border_data = self.get_tile_neighbor_points(tile, tile.width / 10)
                    else:
                        border_data = self.get_tile_neighbor_points(tile, min(tile.width / 10, resolution * 3))
                    if grid_border_data is not None:
                        if border_data is not None:
                            border_data = np.concatenate([border_data, grid_border_data])
                        else:
                            border_data = grid_border_data
                else:
                    border_data = None
                data_for_workers.append([tile, algorithm, resolution, clear_existing, regrid_option, auto_resolution, self.name, self.grid_parameters, border_data])
                chunk_index += 1
                if chunk_index == chunks_at_a_time:
                    print('processing surface: group {} out of {}'.format(cur_run, total_runs))
                    cur_run += 1
                    chunk_index = 0
                    self._grid_parallel_worker(data_for_workers, progress_bar, tile_indices)
                    tile_indices = []
                    data_for_workers = []
        if data_for_workers:
            print('processing surface: group {} out of {}'.format(cur_run, total_runs))
            self._grid_parallel_worker(data_for_workers, progress_bar, tile_indices)
        self.resolutions = np.sort(np.unique(self.resolutions)).tolist()
        self._save_grid()

    def grid(self, algorithm: str = 'mean', resolution: float = None, clear_existing: bool = False, auto_resolution_mode: str = 'depth',
             regrid_option: str = 'full', use_dask: bool = False, progress_bar: bool = True, grid_parameters: dict = None,
             border_data: np.ndarray = None):
        """
        Gridding involves calling 'grid' on all child grids/tiles until you eventually call 'grid' on a Tile.  The Tiles
        are the objects that actually contain the points / gridded data

        Parameters
        ----------
        resolution
            resolution of the gridded data in the Tiles
        algorithm
            algorithm to grid by
        clear_existing
            if True, will clear out any existing grids before generating this one.
        auto_resolution_mode
            one of density, depth; chooses the algorithm used to determine the resolution for the grid/tile
        regrid_option
            controls what parts of the grid will get re-gridded if regrid is True and clear_existing is False, one of 'full', 'update'.  Full mode will
            regrid the entire grid.  Update mode will only update those tiles that have a point_count_changed=True.  If clear_existing is True, will
            automatically run in 'full' mode
        use_dask
            if True, will start a dask LocalCluster instance and perform the gridding in parallel
        progress_bar
            if True, display a progress bar
        grid_parameters
            optional dict of settings to pass to the grid algorithm
        border_data
            point data that falls on the borders, used in the CUBE algorithm to handle tile edge issues.  You won't supply it here,
            this argument will be used during VR gridding, with grids passing subgrids the border data automatically.
        """

        if self.grid_algorithm and (self.grid_algorithm != algorithm) and not clear_existing:
            raise ValueError('Bathygrid: gridding with {}, but {} is already used within the grid.  You must clear'.format(algorithm, self.grid_algorithm) +
                             ' existing data first before using a different gridding algorithm')
        if clear_existing and regrid_option == 'update':
            print('Warning: regrid_option=update is ignored when using clear_existing=True.  The entire grid will be re-gridded.')

        self.grid_algorithm = algorithm
        self.grid_resolution = resolution
        self.grid_parameters = grid_parameters
        if resolution is not None:
            resolution = float(resolution)
        if self.is_empty:
            raise ValueError('BathyGrid: Grid is empty, no points have been added')
        auto_resolution = ''
        if resolution is None:
            auto_resolution = auto_resolution_mode.lower()
            self.grid_resolution = 'AUTO_{}'.format(auto_resolution_mode).upper()
            if self.name != 'VRGridTile_Root':
                if auto_resolution == 'depth':
                    resolution = self._calculate_resolution_lookup()
                elif auto_resolution == 'density':
                    resolution = self.resolution_by_density()
        self.resolutions = []

        if use_dask:
            self._grid_parallel(algorithm, resolution, clear_existing, regrid_option, auto_resolution=auto_resolution, progress_bar=progress_bar, border_data=border_data)
        else:
            self._grid_regular(algorithm, resolution, clear_existing, regrid_option, auto_resolution=auto_resolution, progress_bar=progress_bar, border_data=border_data)
        return self.resolutions

    def plot(self, layer: str = 'depth', resolution: float = None):
        """
        Use matplotlib imshow to plot the layer/resolution.

        Parameters
        ----------
        layer
            string identifier for the layer to access, one of 'depth', 'intensity', 'vertical_uncertainty', 'horizontal_uncertainty', 'total_uncertainty', 'hypothesis_ratio'
        resolution
            resolution of the layer we want to access
        """

        if self.no_grid:
            raise ValueError('BathyGrid: Grid is empty, gridding has not been run yet.')
        if not resolution:
            resolution = self.resolutions
        else:
            resolution = [resolution]

        plt.figure()
        for res in resolution:
            x, y, lyrdata, newmins, newmaxs = self.return_surf_xyz(layer, res, True)
            lat2d, lon2d = np.meshgrid(y, x)
            data_m = np.ma.array(lyrdata[0], mask=np.isnan(lyrdata[0]))
            plt.pcolormesh(lon2d, lat2d, data_m.T)
        plt.title('{}'.format(layer))

    def plot_density_histogram(self, number_of_bins: int = 50):
        """
        Build histogram plot of the soundings per cell across all tiles in the grid

        Parameters
        ----------
        number_of_bins
            number of bins to use in the histogram
        """

        density = np.array(self.density_count)
        plt.hist(density, number_of_bins)
        plt.xlabel('Soundings per Cell')
        plt.ylabel('Number of Cells')
        plt.title(f'Density Histogram (bins={number_of_bins})')

    def plot_density_per_square_meter_histogram(self, number_of_bins: int = 50):
        """
        Build histogram plot of the soundings per square meter across all tiles in the grid

        Parameters
        ----------
        number_of_bins
            number of bins to use in the histogram
        """

        density = np.array(self.density_per_square_meter)
        plt.hist(density, number_of_bins)
        plt.xlabel('Soundings per Square Meter')
        plt.ylabel('Number of Cells')
        plt.title(f'Density Histogram (bins={number_of_bins})')

    def plot_z_histogram(self, number_of_bins: int = 50):
        """
        Build histogram plot of the depth or intensity (intensity if is_backscatter) across all tiles in the grid

        Parameters
        ----------
        number_of_bins
            number of bins to use in the histogram
        """

        dkey = self.depth_key
        if dkey in self.layer_names:
            dvals = self.return_layer_values(dkey)
            plt.hist(dvals, number_of_bins)
            plt.ylabel('Number of Cells')
            if self.is_backscatter:
                plt.xlabel('Intensity (dB)')
                plt.title(f'Intensity Histogram (bins={number_of_bins})')
            else:
                plt.xlabel('Depth (meters)')
                plt.title(f'Depth Histogram (bins={number_of_bins})')
        else:
            print(f'{dkey} not found')

    def plot_vertical_uncertainty_histogram(self, number_of_bins: int = 50):
        """
        Build histogram plot of the vertical uncertainty across all tiles in the grid

        Parameters
        ----------
        number_of_bins
            number of bins to use in the histogram
        """

        if 'vertical_uncertainty' in self.layer_names:
            vunc = self.return_layer_values('vertical_uncertainty')
            plt.hist(vunc, number_of_bins)
            plt.xlabel('Vertical Uncertainty (2 sigma, meters)')
            plt.ylabel('Number of Cells')
            plt.title(f'Vertical Uncertainty Histogram (bins={number_of_bins})')
        else:
            print('Vertical Uncertainty not found')

    def plot_horizontal_uncertainty_histogram(self, number_of_bins: int = 50):
        """
        Build histogram plot of the horizontal uncertainty across all tiles in the grid

        Parameters
        ----------
        number_of_bins
            number of bins to use in the histogram
        """

        if 'horizontal_uncertainty' in self.layer_names:
            hunc = self.return_layer_values('horizontal_uncertainty')
            plt.hist(hunc, number_of_bins)
            plt.xlabel('Horizontal Uncertainty (meters)')
            plt.ylabel('Number of Cells')
            plt.title(f'Horizontal Uncertainty Histogram (bins={number_of_bins})')
        else:
            print('Vertical Uncertainty not found')

    def plot_density_vs_depth(self, number_of_bins: int = 50):
        """
        Plot the average density vs depth using the number of bins provided to sum/average the density count
        """

        density, depth = self.density_count_vs_depth
        density, depth = np.array(density), np.array(depth)
        mindepth, maxdepth = min(depth), max(depth)
        bins = np.linspace(mindepth, maxdepth, number_of_bins + 1)
        bin_indices = np.digitize(depth, bins)
        bin_sort = np.argsort(bin_indices)
        unique_indices, uidx, ucounts = np.unique(bin_indices[bin_sort], return_index=True, return_counts=True)
        counts_sum = np.add.reduceat(density[bin_sort], uidx, axis=0)
        counts_mean = counts_sum / ucounts

        plt.plot(bins, counts_mean)
        plt.xlabel('Depth (meters)')
        plt.ylabel('Average Soundings per Cell')
        plt.title(f'Average Density vs Depth (bins={number_of_bins})')

    def plot_density_per_square_meter_vs_depth(self, number_of_bins: int = 50):
        """
        Plot the average density vs depth using the number of bins provided to sum/average the density per square meter
        """

        density, depth = self.density_per_square_meter_vs_depth
        density, depth = np.array(density), np.array(depth)
        mindepth, maxdepth = min(depth), max(depth)
        bins = np.linspace(mindepth, maxdepth, number_of_bins + 1)
        bin_indices = np.digitize(depth, bins)
        bin_sort = np.argsort(bin_indices)
        unique_indices, uidx, ucounts = np.unique(bin_indices[bin_sort], return_index=True, return_counts=True)
        dsm_sum = np.add.reduceat(density[bin_sort], uidx, axis=0)
        dsm_mean = dsm_sum / ucounts

        plt.plot(bins, dsm_mean)
        plt.xlabel('Depth (meters)')
        plt.ylabel('Average Soundings per Square Meter')
        plt.title(f'Average Density vs Depth (bins={number_of_bins})')

    def return_layer_names(self):
        """
        Return a list of layer names based on what layers exist in the BathyGrid instance.

        Returns
        -------
        list
            list of str surface layer names (ex: ['depth', 'density', 'horizontal_uncertainty', 'vertical_uncertainty']
        """

        if self.no_grid:
            return []
        for tile in self.tiles.flat:
            if tile:
                if isinstance(tile, Tile):
                    for resolution in tile.cells:
                        return list(tile.cells[resolution].keys())
                elif tile.number_of_tiles > 0:  # this is a vr grid, with subgrids within the main grid
                    for subtile in tile.tiles.flat:
                        if subtile:
                            for resolution in subtile.cells:
                                return list(subtile.cells[resolution].keys())
        return []

    def return_extents(self):
        """
        Return the 2d extents of the BathyGrid

        Returns
        -------
        list
            [[minx, miny], [maxx, maxy]]
        """

        return [[self.min_x, self.min_y], [self.max_x, self.max_y]]

    def return_surf_xyz(self, layer: Union[str, list] = 'depth', resolution: float = None, cell_boundaries: bool = True,
                        nodatavalue: float = np.float32(np.nan)):
        """
        Return the xyz grid values as well as an index for the valid nodes in the surface.  z is the gridded result that
        matches the provided layername

        Parameters
        ----------
        layer
            string identifier for the layer to access, one of 'depth', 'intensity', 'vertical_uncertainty', 'horizontal_uncertainty', 'total_uncertainty', 'hypothesis_ratio'
        resolution
            resolution of the layer we want to access
        cell_boundaries
            If True, the user wants the cell boundaries, not the node locations.  If False, returns the node locations
            instead.  If you want to export node locations to file, you want this to be false.  If you are building
            a gdal object, you want this to be True.
        nodatavalue
            nodatavalue to set in the regular grid

        Returns
        -------
        np.ndarray
            numpy array, 1d x locations for the grid nodes
        np.ndarray
            numpy array, 1d y locations for the grid nodes
        np.ndarray
            numpy 2d array, 2d grid depth values
        list
            new minimum x,y coordinate for the trimmed layer
        list
            new maximum x,y coordinate for the trimmed layer
        """

        if self.no_grid:
            raise ValueError('BathyGrid: Grid is empty, gridding has not been run yet.')
        if not resolution:
            if len(self.resolutions) > 1:
                raise ValueError('BathyGrid: you must specify a resolution to return layer data when multiple resolutions are found')
            resolution = self.resolutions[0]

        surfs, new_mins, new_maxs = self.get_layers_trimmed(layer, resolution, nodatavalue)

        if not cell_boundaries:  # get the node locations for each cell
            x = (np.arange(self.min_x, self.max_x, resolution) + resolution / 2)[new_mins[1]:new_maxs[1]]
            y = (np.arange(self.min_y, self.max_y, resolution) + resolution / 2)[new_mins[0]:new_maxs[0]]
        else:  # get the cell boundaries for each cell, will be one longer than the node locations option (this is what matplotlib pcolormesh wants)
            x = np.arange(self.min_x, self.max_x, resolution)[new_mins[1]:new_maxs[1] + 1]
            y = np.arange(self.min_y, self.max_y, resolution)[new_mins[0]:new_maxs[0] + 1]
        return x, y, surfs, new_mins, new_maxs

    def _validate_layer_query(self, x_loc: np.array, y_loc: np.array, layer: str = 'depth', resolution: float = None,
                              nodatavalue: float = np.float32(np.nan)):
        """
        validate the inputs to layer_values_at_xy and return the corrected inputs
        """
        asarrays = True
        if isinstance(x_loc, (tuple, list)):
            x_loc = np.array(list(x_loc))
        if isinstance(y_loc, list):
            y_loc = np.array(list(y_loc))
        if isinstance(x_loc, np.ndarray) and isinstance(y_loc, np.ndarray) and x_loc.shape != y_loc.shape:
            raise ValueError('x and y locations must be the same shape, x_loc:{} y_loc:{}'.format(x_loc, y_loc))
        if isinstance(x_loc, (float, int)) or isinstance(y_loc, (float, int)):
            try:
                x_loc = float(x_loc)
                y_loc = float(y_loc)
                asarrays = False
            except:
                raise ValueError(
                    'x_loc and y_loc must either both be arrays or both be integers or floats'.format(x_loc, y_loc))
        if asarrays:
            layer_values = np.full_like(x_loc, np.float32(nodatavalue), dtype=np.float32)
        else:
            layer_values = np.float32(nodatavalue)
        if not resolution:
            query_resolution = sorted(
                self.resolutions)  # should be sorted already, but ensure we start with high rez first
        else:
            query_resolution = [resolution]
        return asarrays, query_resolution, layer_values, x_loc, y_loc

    def layer_values_at_xy(self, x_loc: np.array, y_loc: np.array, layer: str = 'depth', resolution: float = None,
                           nodatavalue: float = np.float32(np.nan)):
        """
        Return the layer values at the given x y locations.

        Parameters
        ----------
        x_loc
            numpy array, 1d x locations for the grid nodes (easting)
        y_loc
            numpy array, 1d y locations for the grid nodes (northing)
        layer
            layer that we want to query
        resolution
            resolution of the layers we want to query, if None will go through all resolutions and return the highest
            resolution layer value at the xy location
        nodatavalue
            nodatavalue to set in the regular grid

        Returns
        -------
        np.array
            1d array of the same size as x_loc/y_loc with the layer values at the locations
        """

        asarrays, query_resolution, layer_values, x_loc, y_loc = self._validate_layer_query(x_loc, y_loc, layer, resolution, nodatavalue)
        for rez in query_resolution:
            if np.isnan(nodatavalue):
                no_values_yet = np.isnan(layer_values)
            else:
                no_values_yet = layer_values == nodatavalue
            if not no_values_yet.any():  # we have an answer for all xy locations
                break
            surf_x, surf_y, surfs, new_mins, new_maxs = self.return_surf_xyz(layer, rez, cell_boundaries=True,
                                                                             nodatavalue=nodatavalue)

            if asarrays:
                query_x_loc = x_loc[no_values_yet]
                query_y_loc = y_loc[no_values_yet]
            else:
                query_x_loc = x_loc
                query_y_loc = y_loc

            # get the cell index for each query value
            digitized_x = np.digitize(query_x_loc, surf_x)
            digitized_y = np.digitize(query_y_loc, surf_y)

            # drop values that have no valid answer
            out_of_bounds = (digitized_x == 0) + (digitized_y == 0) + (digitized_x == len(surf_x)) + (digitized_y == len(surf_y))
            out_of_bounds_idx = np.where(out_of_bounds)[0]
            in_bounds_idx = np.where(~out_of_bounds)[0]
            if asarrays:
                digitized_x = np.delete(digitized_x, out_of_bounds_idx)
                digitized_y = np.delete(digitized_y, out_of_bounds_idx)
            elif out_of_bounds_idx.size:
                continue

            # now align with the cell values
            digitized_x = digitized_x - 1
            digitized_y = digitized_y - 1

            # store layer values for the given valid xy locations
            if digitized_x.size and digitized_y.size:
                # not sure why the np.array(digitized_y) is required, if you just use digitized_y/digitized_x, it seems
                # to return a slice of the array
                if asarrays:
                    valid_index = np.arange(layer_values.size)[no_values_yet][in_bounds_idx]
                    layer_values[valid_index] = surfs[0][np.array(digitized_y), np.array(digitized_x)]
                else:
                    layer_values = surfs[0][np.array(digitized_y), np.array(digitized_x)]
        return layer_values

    def return_layer_values(self, layer: str):
        """
        Return a 1d array of all values in the provided layer name, excluding nodatavalues.

        Parameters
        ----------
        layer

        Returns
        -------
        np.ndarray
            array of all values in the grid across all resolutions, excluding nodatavalues
        """

        layer_values = []
        if self.tiles is not None:
            for tile in self.tiles.flat:
                if tile:
                    layer_values.extend(tile.return_layer_values(layer))
        return np.array(layer_values)

    def return_unique_containers(self):
        """
        Containers are added to the bathygrid in chunks with indexes attached, like 'container_0', 'container_1'.  This
        method will return only unique containers, i.e. ['container']

        Returns
        -------
        list
            list of unique container names
        """

        unique_cont = []
        for cont in self.container:
            groups = cont.split('_')
            if len(groups) > 1:  # want support for kluster container names with count attached, ex: em2040_123_09_07_2010_0, em2040_123_09_07_2010_1
                idx = groups[-1]
                try:
                    test_idx = int(idx)
                    final_cont_name = cont[:-(len(idx) + 1)]
                except:
                    final_cont_name = cont
            else:
                final_cont_name = cont
            if final_cont_name not in unique_cont:
                unique_cont.append(final_cont_name)
        return unique_cont


def _gridding_parallel(data_blob: list):
    """
    Gridding routine suited for running in parallel using the dask cluster.
    """
    tile, algorithm, resolution, clear_existing, regrid_option, auto_resolution, grid_name, grid_parameters, border_data = data_blob

    if isinstance(tile, BathyGrid) and auto_resolution:  # vrgrid subgrids can calc their own resolution
        rez = tile.grid(algorithm, None, auto_resolution_mode=auto_resolution, clear_existing=clear_existing, regrid_option=regrid_option, progress_bar=False,
                        grid_parameters=grid_parameters, border_data=border_data)
    elif isinstance(tile, SRTile) and auto_resolution and grid_name not in sr_grid_root_names:  # tiles in vrgridtile can be different resolutions
        rez = tile.grid(algorithm, None, auto_resolution_mode=auto_resolution, clear_existing=clear_existing, regrid_option=regrid_option, progress_bar=False,
                        grid_parameters=grid_parameters, border_data=border_data)
    else:
        rez = tile.grid(algorithm, resolution, auto_resolution_mode=auto_resolution, clear_existing=clear_existing, regrid_option=regrid_option, progress_bar=False,
                        grid_parameters=grid_parameters, border_data=border_data)
    return rez, tile


def _correct_for_layer_metadata(resfile: str, data: list, nodatavalue: float):
    """
    Gdal bag driver writes the band min/max to include the nodatavalue, we have to write the correct values ourselves,
    should be resolved in GDAL3.3.2, see OSGeo/gdal issue #4057

    Parameters
    ----------
    resfile
        bag for this resolution
    data
        raster layers of the data, as numpy arrays
    nodatavalue
        nodatavalue of the layer
    """

    if os.path.exists(resfile):
        try:
            r5 = h5py.File(resfile, 'r+')
            validdata = data[0] != nodatavalue
            r5['BAG_root']['elevation'].attrs['Maximum Elevation Value'] = np.float32(np.max(data[0][validdata]))
            r5['BAG_root']['elevation'].attrs['Minimum Elevation Value'] = np.float32(np.min(data[0][validdata]))
            if len(data) == 2:
                r5['BAG_root']['uncertainty'].attrs['Maximum Uncertainty Value'] = np.float32(np.max(data[1][validdata]))
                r5['BAG_root']['uncertainty'].attrs['Minimum Uncertainty Value'] = np.float32(np.min(data[1][validdata]))
            r5.close()
        except:
            print('Warning: Unable to adjust minmax for elevation and uncertainty layers, unknown h5py error')


def _set_temporal_extents(resfile: str, start_time: Union[str, int, float, datetime], end_time: Union[str, int, float, datetime]):
    """
    Taken from the HSTB bag.py library.  Sets the min/max time of the BAG by shoveling in the following xml blob:

      <gmd:temporalElement>
        <gmd:EX_TemporalExtent>
          <gmd:extent>
            <gml:TimePeriod gml:id="temporal-extent-1" xsi:type="gml:TimePeriodType">
              <gml:beginPosition>2018-06-29T07:20:48</gml:beginPosition>
              <gml:endPosition>2018-07-06T21:54:43</gml:endPosition>
            </gml:TimePeriod>
          </gmd:extent>
        </gmd:EX_TemporalExtent>
      </gmd:temporalElement>

    Parameters
    ----------
    resfile
        bag for this resolution
    data
        raster layers of the data, as numpy arrays
    nodatavalue
        nodatavalue of the layer
    """

    if os.path.exists(resfile) and start_time and end_time:
        try:
            r5 = h5py.File(resfile, 'r+')
            metadata = r5['BAG_root']['metadata'][:].tobytes().decode().replace("\x00", "")
            xml_root = et.fromstring(metadata)

            if isinstance(start_time, (float, int)):
                start_time = datetime.utcfromtimestamp(start_time).strftime('%Y-%m-%dT%H:%M:%S')
            elif isinstance(start_time, datetime):
                start_time = start_time.strftime('%Y-%m-%dT%H:%M:%S')
            if isinstance(end_time, (float, int)):
                end_time = datetime.utcfromtimestamp(end_time).strftime('%Y-%m-%dT%H:%M:%S')
            elif isinstance(end_time, datetime):
                end_time = end_time.strftime('%Y-%m-%dT%H:%M:%S')
            gmd = '{http://www.isotc211.org/2005/gmd}'
            gml = '{http://www.opengis.net/gml/3.2}'
            bagschema = "{http://www.opennavsurf.org/schema/bag}"
            xsi = '{http://www.w3.org/2001/XMLSchema-instance}'
            et.register_namespace("gmi", "http://www.isotc211.org/2005/gmi")
            et.register_namespace('gmd', "http://www.isotc211.org/2005/gmd")
            et.register_namespace('xsi', "http://www.w3.org/2001/XMLSchema-instance")
            et.register_namespace('gml', "http://www.opengis.net/gml/3.2")
            et.register_namespace('gco', "http://www.isotc211.org/2005/gco")
            et.register_namespace('xlink', "http://www.w3.org/1999/xlink")
            et.register_namespace('bag', "http://www.opennavsurf.org/schema/bag")
            temporal_hierarchy = [gmd + 'identificationInfo', bagschema + 'BAG_DataIdentification', gmd + 'extent', gmd + 'EX_Extent',
                                  gmd + 'temporalElement', gmd + 'EX_TemporalExtent', gmd + 'extent', gml + 'TimePeriod']
            use_gml = gml
            temporal_root = "/".join(temporal_hierarchy)
            begin_elem = xml_root.findall(temporal_root + "/" + use_gml + 'beginPosition')
            end_elem = xml_root.findall(temporal_root + "/" + use_gml + 'endPosition')
            if not begin_elem or not end_elem:
                parent = xml_root
                for elem in temporal_hierarchy:
                    found = parent.findall(elem)
                    if not found:
                        new_elem = et.SubElement(parent, elem)
                        if "TimePeriod" in elem:
                            new_elem.set(use_gml + 'id', "temporal-extent-1")
                            new_elem.set(xsi + 'type', "gml:TimePeriodType")
                        found = [new_elem]
                    parent = found[0]
                if not begin_elem:
                    begin_elem = [et.SubElement(parent, use_gml+"beginPosition")]
                if not end_elem:
                    end_elem = [et.SubElement(parent, use_gml+"endPosition")]
            begin_elem[0].text = start_time
            end_elem[0].text = end_time
            new_metadata = et.tostring(xml_root).decode()
            del r5['BAG_root']['metadata']
            r5['BAG_root'].create_dataset("metadata", maxshape=(None,), data=np.array(list(new_metadata), dtype="S1"))
            r5.close()
        except:
            print('Warning: Unable to add time extents to BAG, unknown h5py error')


def _generate_caris_rxl(resfile: str, wkt_string: str):
    """
    Caris expects the WKT string to be written to a separate file next to the BAG.  We have the wkt string in the bag
    metadata, but we need to write this second file to make Caris happy.  Caris expects the WKT V1 GDAL string format,
    so we ensure that is created and passed in here

    Parameters
    ----------
    resfile
        path to the bag file
    wkt_string
        the WKT v1 GDAL string for the horizontal coordinate system of this surface
    """

    if os.path.exists(resfile) and wkt_string:
        try:
            rxl_path = os.path.splitext(resfile)[0] + '.bag_rxl'
            top = et.Element('caris_registration', version="4.0", generation="USER")
            newtree = et.ElementTree(top)
            coord_elem = et.SubElement(top, 'coordinate_system')
            wktelem = et.SubElement(coord_elem, 'wkt')
            wktelem.text = wkt_string
            xmlstr = minidom.parseString(et.tostring(top)).toprettyxml(indent="  ", encoding='utf-8').decode()
            xmlstr = xmlstr.replace('&quot;', '"').encode('utf-8')
            with open(rxl_path, 'wb') as rxlfile:
                rxlfile.write(xmlstr)
        except:
            print('Warning: Unable to generate Caris RXL file, unknown ElementTree error')


class OperationalGrid(BathyGrid):
    def __init__(self, min_x: float = 0, min_y: float = 0, max_x: float = 0, max_y: float = 0,
                 tile_size: float = 1024.0, set_extents_manually: bool = False, output_folder: str = '', is_backscatter: bool = False):
        super().__init__(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y, tile_size=tile_size, set_extents_manually=set_extents_manually,
                         output_folder=output_folder, is_backscatter=is_backscatter)

    def __repr__(self):
        base_output = super().__repr__()
        output = 'Time of Data (UTC): {} to {}\n'.format(self.min_time, self.max_time)
        try:
            epsg_name = CRS.from_epsg(int(self.epsg)).name
        except:
            epsg_name = 'Unknown'
        output += 'EPSG: {} ({})\n'.format(self.epsg, epsg_name)
        output += 'Path: {}\n'.format(self.output_folder)
        output += base_output
        return output

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
        elif type(self.data) == xr.Dataset:
            if 'x' not in self.data:
                raise ValueError('BathyGrid: xarray Dataset provided for data, but "x" or "y" not found in variable names')
            if self.data['z'].ndim > 1:
                raise ValueError('BathyGrid: xarray Dataset provided for data, but found multiple dimensions, must be one dimensional: {}'.format(self.data.dims))
            self._convert_dataset()  # internally we just convert xarray dataset to numpy for ease of use
        else:
            raise ValueError('QuadTree: numpy structured array or dask array with "x" and "y" as variable must be provided')

    def save(self, folder_path: str = None, progress_bar: bool = True):
        """
        Recursive save for all BathyGrid/Tile objects within this class.

        Parameters
        ----------
        folder_path
            container folder for the grid
        progress_bar
            if True, displays a console progress bar
        """

        if not folder_path:
            if self.output_folder:
                super().save(self.output_folder, progress_bar=progress_bar)
            else:
                raise ValueError('Grid has not been saved before, you must provide a folder path to save.')
        else:
            fpath, fname = os.path.split(folder_path)
            folderpath = create_folder(fpath, fname)
            self.output_folder = folderpath
            super().save(folderpath, progress_bar=progress_bar)

    def load(self, folder_path: str = None):
        """
        Recursive load for all BathyGrid/Tile objects within this class.

        Parameters
        ----------
        folder_path
            container folder for the grid
        """

        if not folder_path:
            if self.output_folder:
                super().load(self.output_folder)
            else:
                raise ValueError('Grid has not been saved before, you must provide a folder path to load.')
        else:
            self.output_folder = folder_path
            super().load(self.output_folder)

    def export(self, output_path: str, export_format: str = 'csv', z_positive_up: bool = True, resolution: float = None,
               **kwargs):
        """
        Export the node data to one of the supported formats

        Parameters
        ----------
        output_path
            filepath for exporting the dataset
        export_format
            format option, one of 'csv', 'geotiff', 'bag'
        z_positive_up
            if True, will output bands with positive up convention
        resolution
            if provided, will only export the given resolution
        """

        fmt = export_format.lower()
        if os.path.exists(output_path):
            tstmp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            foldername, filname = os.path.split(output_path)
            filnm, filext = os.path.splitext(filname)
            output_path = os.path.join(foldername, '{}_{}{}'.format(filnm, tstmp, filext))

        if fmt == 'csv':
            self._export_csv(output_path, z_positive_up=z_positive_up, resolution=resolution)
        elif fmt == 'geotiff':
            self._export_geotiff(output_path, z_positive_up=z_positive_up, resolution=resolution)
        elif fmt == 'bag':
            if self.is_backscatter:
                raise ValueError('Bathygrid: Cannot generate BAG with Backscatter grid')
            self._export_bag(output_path, resolution=resolution, **kwargs)
        else:
            raise ValueError("bathygrid: Unrecognized format {}, must be one of ['csv', 'geotiff', 'bag']".format(fmt))

    def _export_csv(self, output_file: str, z_positive_up: bool = True, resolution: float = None):
        """
        Export the node data to csv

        Parameters
        ----------
        output_file
            output_file to contain the exported data
        z_positive_up
            if True, will output bands with positive up convention
        resolution
            if provided, will only export the given resolution
        """

        basefile, baseext = os.path.splitext(output_file)
        if resolution is not None:
            resolutions = [resolution]
        else:
            resolutions = self.resolutions
        for res in resolutions:
            resfile = basefile + '_{}.csv'.format(res)
            lyrs = self.return_layer_names()
            x, y, lyrdata, newmins, newmaxs = self.return_surf_xyz(lyrs, res, False)
            xx, yy = np.meshgrid(x, y)
            dataset = [xx.ravel(), yy.ravel()]
            dnames = ['x', 'y']
            dfmt = ['%.3f', '%.3f']
            for cnt, lname in enumerate(lyrs):
                if lname == 'depth' and z_positive_up:
                    if self.mean_depth > 0:  # currently positive down
                        lyrdata[cnt] = lyrdata[cnt] * -1
                    lname = 'elevation'
                dataset += [lyrdata[cnt].ravel()]
                dnames += [lname]
                if lname == 'density':
                    dfmt += ['%i']
                else:
                    dfmt += ['%.3f']
            sortidx = np.argsort(dataset[0])
            np.savetxt(resfile, np.stack([d[sortidx] for d in dataset], axis=1),
                       fmt=dfmt, delimiter=' ', comments='',
                       header=' '.join([nm for nm in dnames]))

    def _export_geotiff(self, filepath: str, z_positive_up: bool = True, resolution: float = None):
        """
        Export a GDAL generated geotiff to the provided filepath

        Parameters
        ----------
        filepath
            folder to contain the exported data
        z_positive_up
            if True, will output bands with positive up convention
        resolution
            if provided, will only export the given resolution
        """

        lyrtranslator = {'depth': 'Depth', 'density': 'Density', 'elevation': 'Elevation', 'vertical_uncertainty': 'Vertical Uncertainty',
                         'total_uncertainty': 'Total Uncertainty', 'horizontal_uncertainty': 'Horizontal Uncertainty', 'intensity': 'Intensity'}
        nodatavalue = 1000000.0
        basefile, baseext = os.path.splitext(filepath)
        if resolution is not None:
            resolutions = [resolution]
        else:
            resolutions = self.resolutions
        if self.is_backscatter:
            layernames = ['intensity']
        else:
            layernames = [lname for lname in self.layer_names if lname in ['depth', 'vertical_uncertainty', 'total_uncertainty']]
        finalnames = [lyrtranslator[lname] for lname in layernames]
        if z_positive_up and finalnames.index('Depth') != -1:
            finalnames[finalnames.index('Depth')] = 'Elevation'
        for res in resolutions:
            chunk_count = 1
            for geo_transform, maxdim, data in self.get_chunks_of_tiles(resolution=res, layer=layernames,
                                                                        nodatavalue=nodatavalue, z_positive_up=z_positive_up):
                resfile = basefile + '_{}_{}.tif'.format(res, chunk_count)
                data = list(data.values())
                gdal_raster_create(resfile, data, geo_transform, self.epsg, nodatavalue=nodatavalue, bandnames=finalnames, driver='GTiff')
                chunk_count += 1

    def _export_bag(self, filepath: str, resolution: float = None, individual_name: str = 'unknown',
                    organizational_name: str = 'unknown', position_name: str = 'unknown', attr_date: str = '',
                    vert_crs: str = '', abstract: str = '', process_step_description: str = '', attr_datetime: str = '',
                    restriction_code: str = 'otherRestrictions', other_constraints: str = 'unknown',
                    classification: str = 'unclassified', security_user_note: str = 'none'):
        """
        Export a GDAL generated BAG to the provided filepath

        If attr_date is not provided, will use the current date.  If attr_datetime is not provided, will use the current
        date/time.  If process_step_description is not provided, will use a default 'Generated By GDAL and Kluster'
        message.  If vert_crs is not provided, will use a WKT with value = 'unknown'

        Parameters
        ----------
        filepath
            folder to contain the exported data
        resolution
            if provided, will only export the given resolution
        """

        if not attr_date:
            attr_date = datetime.utcnow().strftime('%Y-%m-%d')
        if not attr_datetime:
            attr_datetime = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
        if not process_step_description:
            process_step_description = 'Generated By GDAL {}'.format(return_gdal_version())
        if not vert_crs:
            vert_crs = 'VERT_CS["unknown", VERT_DATUM["unknown", 2000]]'

        bag_options = ['VAR_INDIVIDUAL_NAME=' + individual_name, 'VAR_ORGANISATION_NAME=' + organizational_name,
                       'VAR_POSITION_NAME=' + position_name, 'VAR_DATE=' + attr_date, 'VAR_VERT_WKT=' + vert_crs,
                       'VAR_ABSTRACT=' + abstract, 'VAR_PROCESS_STEP_DESCRIPTION=' + process_step_description,
                       'VAR_DATETIME=' + attr_datetime, 'VAR_RESTRICTION_CODE=' + restriction_code,
                       'VAR_OTHER_CONSTRAINTS=' + other_constraints, 'VAR_CLASSIFICATION=' + classification,
                       'VAR_SECURITY_USER_NOTE=' + security_user_note]

        lyrtranslator = {'depth': 'Depth', 'density': 'Density', 'elevation': 'Elevation', 'vertical_uncertainty': 'Uncertainty',
                         'total_uncertainty': 'Uncertainty', 'horizontal_uncertainty': 'Horizontal Uncertainty', 'intensity': 'Intensity'}
        nodatavalue = 1000000.0
        z_positive_up = True
        basefile, baseext = os.path.splitext(filepath)
        if resolution is not None:
            resolutions = [resolution]
        else:
            resolutions = self.resolutions
        layernames = [lname for lname in self.layer_names if lname in ['depth', 'vertical_uncertainty', 'total_uncertainty']]
        finalnames = [lyrtranslator[lname] for lname in layernames]
        if z_positive_up and finalnames.index('Depth') != -1:
            finalnames[finalnames.index('Depth')] = 'Elevation'
        for res in resolutions:
            chunk_count = 1
            for geo_transform, maxdim, data in self.get_chunks_of_tiles(resolution=res, layer=layernames,
                                                                        nodatavalue=nodatavalue, z_positive_up=z_positive_up):
                resfile = basefile + '_{}_{}.bag'.format(res, chunk_count)
                data = list(data.values())
                gdal_raster_create(resfile, data, geo_transform, self.epsg, nodatavalue=nodatavalue,
                                   bandnames=finalnames, driver='BAG', creation_options=bag_options)
                _correct_for_layer_metadata(resfile, data, nodatavalue)
                _set_temporal_extents(resfile, self.min_time, self.max_time)
                _generate_caris_rxl(resfile, CRS.from_epsg(self.epsg).to_wkt(version='WKT1_GDAL', pretty=True))
                chunk_count += 1

    def return_attribution(self):
        """
        Used in Kluster, return the important attribution of the class as a dict to display in the gui

        Returns
        -------
        dict
            class attributes in a presentable form
        """

        data = {'grid_folder': self.output_folder, 'name': self.name, 'type': type(self), 'grid_resolution': self.grid_resolution,
                'grid_algorithm': self.grid_algorithm, 'grid_parameters': self.grid_parameters, 'epsg': self.epsg,
                'vertical_reference': self.vertical_reference,
                'height': self.height, 'width': self.width, 'minimum_x': self.min_x, 'maximum_x': self.max_x,
                'minimum_y': self.min_y, 'maximum_y': self.max_y, 'minimum_time_utc': self.min_time,
                'maximum_time_utc': self.max_time, 'tile_size': self.tile_size,
                'subtile_size': self.subtile_size, 'tile_count': self.number_of_tiles, 'resolutions': self.resolutions,
                'storage_type': self.storage_type}
        ucontainers = self.return_unique_containers()
        for cont_name in ucontainers:
            try:  # this works for kluster added containers, that have a suffix with an index
                data['source_{}'.format(cont_name)] = {'time': self.container_timestamp[cont_name + '_0'],
                                                       'multibeam_lines': self.container[cont_name + '_0']}
            except KeyError:
                try:  # this works for all other standard container names
                    data['source_{}'.format(cont_name)] = {'time': self.container_timestamp[cont_name],
                                                           'multibeam_lines': self.container[cont_name]}
                except KeyError:
                    nearest_cont_name = [nm for nm in self.container if nm.find(cont_name) != -1]
                    if nearest_cont_name[0]:
                        data['source_{}'.format(nearest_cont_name[0])] = {'time': self.container_timestamp[nearest_cont_name[0]],
                                                                          'multibeam_lines': self.container[nearest_cont_name[0]]}
                    else:
                        raise ValueError('Unable to find entry for container {}, if you have a suffix with a _ and a number, bathygrid will interpret that as an index starting with 0'.format(cont_name))
        return data
