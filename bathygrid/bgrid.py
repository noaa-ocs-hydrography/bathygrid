import numpy as np
import xarray as xr
from dask.array import Array
from dask.distributed import wait, progress
import matplotlib.pyplot as plt
from typing import Union
from datetime import datetime

from bathygrid.grids import BaseGrid
from bathygrid.tile import SRTile, Tile
from bathygrid.utilities import bin2d_with_indices, dask_find_or_start_client, print_progress_bar, \
    utc_seconds_to_formatted_string, formatted_string_to_utc_seconds
from bathygrid.grid_variables import depth_resolution_lookup, maximum_chunk_dimension


class BathyGrid(BaseGrid):
    """
    Manage a rectangular grid of tiles, each able to operate independently and in parallel.  BathyGrid automates the
    creation and updating of each Tile, which happens under the hood when you add or remove points.

    Used in the VRGridTile as the tiles of the master grid.  Each tile of the VRGridTile is a BathyGrid with tiles within
    that grid.
    """

    def __init__(self, min_x: float = 0, min_y: float = 0, max_x: float = 0, max_y: float = 0, tile_size: float = 1024.0,
                 set_extents_manually: bool = False, output_folder: str = ''):
        super().__init__(min_x=min_x, min_y=min_y, tile_size=tile_size)

        if set_extents_manually:
            self.min_x = min_x
            self.min_y = min_y
            self.max_x = max_x
            self.max_y = max_y

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
        self.sub_type = 'srtile'
        self.storage_type = 'numpy'
        self.client = None

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
    def coverage_area(self):
        """
        Return the coverage area of this grid in the same units as the resolution (generally meters)
        """
        cellcount = self.cell_count
        area = 0
        for rez, cnt in cellcount.items():
            area += cnt * rez
        return area

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

    def _calculate_resolution(self):
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

        return SRTile(tile_x_origin, tile_y_origin, self.tile_size)

    def _build_empty_tile_space(self):
        """
        Build a 2d array of NaN for the size of one of the tiles.
        """

        return np.full((self.tile_size, self.tile_size), np.nan)

    def _build_layer_grid(self, resolution: float, nodatavalue: float = np.float32(np.nan)):
        """
        Build a 2d array of NaN for the size of the whole BathyGrid (given the provided resolution)

        Parameters
        ----------
        resolution
            float resolution that we want to use to build the grid
        """

        # ensure nodatavalue is a float32
        nodatavalue = np.float32(nodatavalue)
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
            binnum = bin2d_with_indices(self.data['x'], self.data['y'], self.tile_edges_x, self.tile_edges_y)
            unique_locs = np.unique(binnum)
            flat_tiles = self.tiles.ravel()
            tilexorigin = self.tile_x_origin.ravel()
            tileyorigin = self.tile_y_origin.ravel()
            if progress_bar:
                print_progress_bar(0, len(unique_locs), 'Adding Points to {}:'.format(self.name))
            for cnt, ul in enumerate(unique_locs):
                if progress_bar:
                    print_progress_bar(cnt + 1, len(unique_locs), 'Adding Points to {}:'.format(self.name))
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
                    print_progress_bar(0, len(flat_tiles), 'Removing Points from {}:'.format(self.name))
                for cnt, tile in enumerate(flat_tiles):
                    if progress_bar:
                        print_progress_bar(cnt + 1, len(flat_tiles), 'Removing Points from {}:'.format(self.name))
                    if tile:
                        tile.remove_points(container_name, progress_bar=False)
                        if tile.is_empty:
                            self._remove_tile(cnt)
                    if self.sub_type in ['srtile', 'quadtile']:
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
            string identifier for the layer(s) to access, valid layers include 'depth', 'horizontal_uncertainty', 'vertical_uncertainty'
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
                col, row = self._tile_idx_to_row_col(cnt)
                tile_cell_count = self.tile_size / resolution
                assert tile_cell_count.is_integer()
                tile_cell_count = int(tile_cell_count)
                data_col, data_row = col * tile_cell_count, row * tile_cell_count
                geo = tile.get_geotransform(resolution)
                data = {}
                for cnt, lyr in enumerate(layer):
                    newdata = tile.get_layers_by_name(lyr, resolution, nodatavalue=nodatavalue, z_positive_up=z_positive_up)
                    if newdata is not None:
                        if isinstance(newdata, list):  # true if 'tile' is actually a subgrid (BathyGrid)
                            newdata = newdata[0]
                        data[lyr] = newdata
                yield geo, data_col, data_row, tile_cell_count, data

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
        # ensure nodatavalue is a float32
        nodatavalue = np.float32(nodatavalue)
        # normalize the column/row indices, the data and the geotransform yielded here are just for this chunk
        mincol = min(column_indices)
        curdcol = [col - mincol for col in column_indices]
        minrow = min(row_indices)
        curdrow = [row - minrow for row in row_indices]
        for lyr in layers:
            finaldata[lyr] = np.full((max(curdcol) + cells_per_tile, max(curdrow) + cells_per_tile), nodatavalue)
            for cnt, data in enumerate(layerdata):
                if lyr in data:
                    finaldata[lyr][curdcol[cnt]:curdcol[cnt] + cells_per_tile, curdrow[cnt]:curdrow[cnt] + cells_per_tile] = data[lyr]
            if for_gdal:
                finaldata[lyr] = np.fliplr(finaldata[lyr].T)
        return finaldata
    
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
            string identifier for the layer(s) to access, valid layers include 'depth', 'horizontal_uncertainty', 'vertical_uncertainty'
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
            max_width = override_maximum_chunk_dimension
        else:
            max_width = maximum_chunk_dimension
        curmaxdimension = 0
        curgeo = None
        curdata = []
        curdcol = []
        curdrow = []
        curcellcount = []
        for geo, data_col, data_row, tile_cell_count, tdata in self.get_tiles_by_resolution(resolution, layer,
                                                                                            nodatavalue=nodatavalue, z_positive_up=z_positive_up):
            if curmaxdimension >= max_width:
                assert all(curcellcount)  # all the tile counts per tile should be the same for the one resolution
                finaldata = self._finalize_chunk(curdcol, curdrow, curcellcount[0], layer, curdata, nodatavalue, for_gdal)
                yield curgeo, curmaxdimension, finaldata
                curgeo = None
                curdata = []
                curdcol = []
                curdrow = []
                curcellcount = []

            curdata.append(tdata)
            curdcol.append(data_col)
            curdrow.append(data_row)
            curcellcount.append(tile_cell_count)
            if curgeo is None:
                curgeo = geo
            else:
                # merge the georeference of the two datasets, everything remains the same except the origin, which is now the global origin
                curgeo = [min([curgeo[0], geo[0]]), curgeo[1], curgeo[2], max([curgeo[3], geo[3]]), curgeo[4], curgeo[5]]
            # here we take the greatest dimension currently to see if we are big enough to stop the chunk
            curmaxdimension = max((max(curdcol) - min(curdcol) + tile_cell_count) * resolution, (max(curdrow) - min(curdrow) + tile_cell_count) * resolution)
        assert all(curcellcount)  # all the cell counts per tile should be the same for the one resolution
        finaldata = self._finalize_chunk(curdcol, curdrow, curcellcount[0], layer, curdata, nodatavalue)
        yield curgeo, curmaxdimension, finaldata

    def get_layers_by_name(self, layer: Union[str, list] = 'depth', resolution: float = None, nodatavalue: float = np.float32(np.nan),
                           z_positive_up: bool = False):
        """
        Return the numpy 2d grid for the provided layer, resolution.  Will check to ensure that you have gridded at this
        resolution already.  Grid returned will have NaN values for empty spaces.

        Parameters
        ----------
        layer
            string identifier for the layer(s) to access, valid layers include 'depth', 'horizontal_uncertainty', 'vertical_uncertainty'
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
        data = [self._build_layer_grid(resolution, nodatavalue=nodatavalue) for lyr in layer]
        for cnt, tile in enumerate(self.tiles.flat):
            if tile:
                col, row = self._tile_idx_to_row_col(cnt)
                tile_cell_count = self.tile_size / resolution
                assert tile_cell_count.is_integer()
                tile_cell_count = int(tile_cell_count)
                data_col, data_row = col * tile_cell_count, row * tile_cell_count
                for cnt, lyr in enumerate(layer):
                    newdata = tile.get_layers_by_name(lyr, resolution, nodatavalue=nodatavalue, z_positive_up=z_positive_up)
                    if newdata is not None:
                        if isinstance(newdata, list):  # true if 'tile' is actually a subgrid (BathyGrid)
                            newdata = newdata[0]
                        data[cnt][data_col:data_col + tile_cell_count, data_row:data_row + tile_cell_count] = newdata
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
            string identifier for the layer to access, one of 'depth', 'horizontal_uncertainty', 'vertical_uncertainty'
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

    def _grid_regular(self, algorithm: str, resolution: float, clear_existing: bool, auto_resolution: bool,
                      progress_bar: bool = True):
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
        auto_resolution
            if True and the tile type supports it, allow the tile to auto calculate the appropriate resolution
        progress_bar
            if True, display a progress bar
        """

        if progress_bar:
            print_progress_bar(0, self.tiles.size, 'Gridding {} - {}:'.format(self.name, algorithm))
        for cnt, tile in enumerate(self.tiles.flat):
            if progress_bar:
                print_progress_bar(cnt + 1, self.tiles.size, 'Gridding {} - {}:'.format(self.name, algorithm))
            if tile:
                if isinstance(tile, BathyGrid) and auto_resolution:  # vrgrid subgrids can calc their own resolution
                    rez = tile.grid(algorithm, None, clear_existing=clear_existing, progress_bar=False)
                elif isinstance(tile, SRTile) and auto_resolution and self.name != 'SRGrid_Root':  # tiles in vrgridtile can be different resolutions
                    rez = tile.grid(algorithm, None, clear_existing=clear_existing, progress_bar=False)
                else:
                    rez = tile.grid(algorithm, resolution, clear_existing=clear_existing, progress_bar=False)
                if isinstance(rez, float) or isinstance(rez, int):
                    rez = [rez]
                for rz in rez:
                    if rz not in self.resolutions:
                        self.resolutions.append(rz)
            if self.sub_type in ['srtile', 'quadtile']:
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
                self._save_tile(result_tile, tidx, only_grid=True)
                self._load_tile(tidx, only_grid=True)
        for rez in resolutions:
            if rez not in self.resolutions:
                self.resolutions.append(rez)
        tiles = None
        resolutions = None
        results = None

    def _grid_parallel(self, algorithm: str, resolution: float, clear_existing: bool, auto_resolution: bool,
                       progress_bar: bool = True):
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
        auto_resolution
            if True and the tile type supports it, allow the tile to auto calculate the appropriate resolution
        progress_bar
            if True, display a progress bar
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
        for cnt, tile in enumerate(self.tiles.flat):
            if tile:
                if self.sub_type in ['srtile', 'quadtile']:
                    self._load_tile_data_to_memory(tile)
                tile_indices.append(cnt)
                data_for_workers.append([tile, algorithm, resolution, clear_existing, auto_resolution, self.name])
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

    def grid(self, algorithm: str = 'mean', resolution: float = None, clear_existing: bool = False, use_dask: bool = False,
             progress_bar: bool = True):
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
            if True, will clear out any existing grids before generating this one.  Otherwise if the resolution exists,
            will only run gridding on tiles that have new points.
        use_dask
            if True, will start a dask LocalCluster instance and perform the gridding in parallel
        progress_bar
            if True, display a progress bar
        """

        if self.grid_algorithm and (self.grid_algorithm != algorithm) and not clear_existing:
            raise ValueError('Bathygrid: gridding with {}, but {} is already used within the grid.  You must clear'.format(algorithm, self.grid_algorithm) +
                             ' existing data first before using a different gridding algorithm')
        self.grid_algorithm = algorithm
        self.grid_resolution = resolution
        if resolution is not None:
            resolution = float(resolution)
        if self.is_empty:
            raise ValueError('BathyGrid: Grid is empty, no points have been added')
        auto_resolution = False
        if resolution is None:
            auto_resolution = True
            self.grid_resolution = 'AUTO'
            resolution = self._calculate_resolution()
        self.resolutions = []

        if use_dask:
            self._grid_parallel(algorithm, resolution, clear_existing, auto_resolution=auto_resolution, progress_bar=progress_bar)
        else:
            self._grid_regular(algorithm, resolution, clear_existing, auto_resolution=auto_resolution, progress_bar=progress_bar)
        return self.resolutions

    def plot(self, layer: str = 'depth', resolution: float = None):
        """
        Use matplotlib imshow to plot the layer/resolution.

        Parameters
        ----------
        layer
            string identifier for the layer to access, one of 'depth', 'horizontal_uncertainty', 'vertical_uncertainty'
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

    def return_layer_names(self):
        """
        Return a list of layer names based on what layers exist in the BathyGrid instance.

        Returns
        -------
        list
            list of str surface layer names (ex: ['depth', 'horizontal_uncertainty', 'vertical_uncertainty']
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
            string identifier for the layer to access, one of 'depth', 'horizontal_uncertainty', 'vertical_uncertainty'
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
    tile, algorithm, resolution, clear_existing, auto_resolution, grid_name = data_blob
    if isinstance(tile, BathyGrid) and auto_resolution:  # vrgrid subgrids can calc their own resolution
        rez = tile.grid(algorithm, None, clear_existing=clear_existing, progress_bar=False)
    elif isinstance(tile, SRTile) and auto_resolution and grid_name != 'SRGrid_Root':  # tiles in vrgridtile can be different resolutions
        rez = tile.grid(algorithm, None, clear_existing=clear_existing, progress_bar=False)
    else:
        rez = tile.grid(algorithm, resolution, clear_existing=clear_existing, progress_bar=False)
    return rez, tile
