import numpy as np
from bathygrid.grids import TileGrid
from bathygrid.utilities import bin2d_with_indices, is_power_of_two
from bathygrid.algorithms import nb_grid_mean, nb_grid_shoalest
from bathygrid.grid_variables import depth_resolution_lookup


class Tile(TileGrid):
    """
    Bathygrid is composed of multiple Tiles.  Each Tile manages its own point data and gridding.  This base tile contains
    some common methods for managing point data.
    """

    def __init__(self, min_x: float, min_y: float, size: float):
        super().__init__(min_x, min_y, size)
        self.algorithm = None

    def clear_grid(self):
        """
        Clear all data associated with grids
        """

        self.cells = {}
        self.cell_edges_x = {}
        self.cell_edges_y = {}

    def clear_points(self):
        """
        Clear all data associated with points
        """

        self.data = None
        self.container = {}  # dict of container name, list of multibeam fil
        self.cell_indices = {}


class SRTile(Tile):
    """
    Single Resolution Tile (SRTile) is a Tile object that can generate only a single resolution gridded data product
    when gridding.  It can contain multiple grids at different resolutions (see resolution key) if you run 'grid' multiple
    times with different resolutions without using clear_existing
    """

    def __init__(self, min_x: float = 0.0, min_y: float = 0.0, size: float = 0.0):
        super().__init__(min_x, min_y, size)

    @property
    def mean_depth(self):
        if self.data is None:
            return 0
        else:
            return float(self.data['z'].mean())

    @property
    def cell_count(self):
        final_count = {}
        for rez in self.cells:
            final_count[rez] = int(np.count_nonzero(~np.isnan(self.cells[rez]['depth'])))
        return final_count

    def _calculate_resolution(self):
        """
        Use the depth resolution lookup to find the appropriate depth resolution band.  The lookup is the max depth and
        the resolution that applies.

        Returns
        -------
        float
            resolution to use at the existing mean_depth
        """

        if self.data is None:
            raise ValueError('SRTile: Unable to calculate resolution when there are no points in the tile')
        dpth_keys = list(depth_resolution_lookup.keys())
        # get next positive value in keys of resolution lookup
        range_index = np.argmax((np.array(dpth_keys) - self.mean_depth) > 0)
        calc_resolution = depth_resolution_lookup[dpth_keys[range_index]]
        # ensure that resolution does not exceed the tile size for obvious reasons
        clipped_rez = min(self.width, calc_resolution)
        return float(clipped_rez)

    def add_points(self, data: np.ndarray, container: str, progress_bar: bool = False):
        """
        Add new points to the Tile object.  Retain the point source (container) so that we can remove them later using
        this tag if necessary

        Parameters
        ----------
        data
            numpy structured array of point data containing 'x', 'y', 'z' (and optionally 'tvu', 'thu')
        container
            name of the source of the point data
        progress_bar
            placeholder, to match the signature of the grid 'add_points' routines.  unused here.
        """

        if self.data is None:
            self.data = data
            self.container = {container: [0, self.data['x'].size]}
            for resolution in self.cell_indices:
                self.cell_indices[resolution] = np.full(self.data.shape, -1)
        else:
            if container in self.container:
                self.remove_points(container)
            self.container[container] = [self.data.size, self.data.size + data.size]
            if not isinstance(self.data, np.ndarray):
                self.data = self.data.compute()
            self.data = np.concatenate([self.data, data])
            for resolution in self.cell_indices:
                self.cell_indices[resolution] = np.append(self.cell_indices[resolution], np.full(data.shape, -1))

    def remove_points(self, container, progress_bar: bool = False):
        """
        Remove existing points from the Tile that match the container tag

        Parameters
        ----------
        container
            name of the source of the point data
        progress_bar
            placeholder, to match the signature of the grid 'add_points' routines.  unused here.
        """

        if container in self.container:
            remove_start, remove_end = self.container[container]
            msk = np.ones(self.data.shape[0], dtype=bool)
            msk[remove_start:remove_end] = False
            chunk_size = msk[remove_start:remove_end].size
            for cont in self.container:
                if self.container[cont][0] >= remove_end:
                    self.container[cont] = [self.container[cont][0] - chunk_size, self.container[cont][1] - chunk_size]
            for resolution in self.cell_indices:
                self.cell_indices[resolution] = self.cell_indices[resolution][msk]
            self.data = self.data[msk]
            self.container.pop(container)

    def new_grid(self, resolution: float, algorithm: str, nodatavalue: float = np.nan):
        """
        Construct a new grid for the given resolution/algorithm and store it in the appropriate class attributes

        Parameters
        ----------
        resolution
            resolution of the grid in x/y units
        algorithm
            algorithm to grid by
        nodatavalue
            value to fill empty space in the grid with
        """

        if not is_power_of_two(resolution):
            raise ValueError(f'Tile: Resolution must be a power of two, got {resolution}')
        resolution = resolution
        grid_x = np.arange(self.min_x, self.max_x, resolution)
        grid_y = np.arange(self.min_y, self.max_y, resolution)
        self.cell_edges_x[resolution] = np.append(grid_x, grid_x[-1] + resolution)
        self.cell_edges_y[resolution] = np.append(grid_y, grid_y[-1] + resolution)
        grid_shape = (grid_x.size, grid_y.size)
        self.cells[resolution] = {}
        if algorithm == 'mean':
            self.cells[resolution]['depth'] = np.full(grid_shape, nodatavalue)
            if self.data is not None and 'tvu' in self.data.dtype.names:
                self.cells[resolution]['vertical_uncertainty'] = np.full(grid_shape, nodatavalue)
            if self.data is not None and 'thu' in self.data.dtype.names:
                self.cells[resolution]['horizontal_uncertainty'] = np.full(grid_shape, nodatavalue)

    def _run_mean_grid(self, resolution: float):
        """
        Run the mean algorithm on the Tile data
        """

        vert_val = None
        horiz_val = None
        vert_grid = None
        horiz_grid = None
        if 'tvu' in self.data.dtype.names:
            vert_val = self.data['tvu']
            vert_grid = self.cells[resolution]['vertical_uncertainty']
        if 'thu' in self.data.dtype.names:
            horiz_val = self.data['thu']
            horiz_grid = self.cells[resolution]['horizontal_uncertainty']
        nb_grid_mean(self.data['z'], self.cell_indices[resolution], self.cells[resolution]['depth'],
                     vert_val, horiz_val, vert_grid, horiz_grid)
        self.cells[resolution]['depth'] = np.round(self.cells[resolution]['depth'], 3)
        if vert_val is not None:
            self.cells[resolution]['vertical_uncertainty'] = np.round(self.cells[resolution]['vertical_uncertainty'], 3)
        if horiz_val is not None:
            self.cells[resolution]['horizontal_uncertainty'] = np.round(self.cells[resolution]['horizontal_uncertainty'], 3)

    def _run_shoalest_grid(self, resolution: float):
        """
        Run the shoalest algorithm on the Tile data
        """

        vert_val = None
        horiz_val = None
        vert_grid = None
        horiz_grid = None
        if 'tvu' in self.data.dtype.names:
            vert_val = self.data['tvu']
            vert_grid = self.cells[resolution]['vertical_uncertainty']
        if 'thu' in self.data.dtype.names:
            horiz_val = self.data['thu']
            horiz_grid = self.cells[resolution]['horizontal_uncertainty']
        nb_grid_shoalest(self.data['z'], self.cell_indices[resolution], self.cells[resolution]['depth'],
                         vert_val, horiz_val, vert_grid, horiz_grid)
        self.cells[resolution]['depth'] = np.round(self.cells[resolution]['depth'], 3)
        if vert_val is not None:
            self.cells[resolution]['vertical_uncertainty'] = np.round(self.cells[resolution]['vertical_uncertainty'], 3)
        if horiz_val is not None:
            self.cells[resolution]['horizontal_uncertainty'] = np.round(self.cells[resolution]['horizontal_uncertainty'], 3)

    def grid(self, algorithm: str, resolution: float = None, clear_existing: bool = False, progress_bar: bool = False):
        """
        Grid the Tile data using the provided algorithm and resolution.  Stores the gridded data in the Tile

        Parameters
        ----------
        resolution
            resolution of the grid in x/y units
        algorithm
            algorithm to grid by
        clear_existing
            If True, clears all the existing gridded data associated with the tile
        progress_bar
            if True, show a progress bar

        Returns
        -------
        float
            resolution of the grid in x/y units
        """

        if resolution is None:
            resolution = self._calculate_resolution()
        if not is_power_of_two(resolution):
            raise ValueError(f'Tile: Resolution must be a power of two, got {resolution}')
        if clear_existing:
            self.clear_grid()
        if not isinstance(self.data, np.ndarray):
            self.data = self.data.compute()

        if resolution not in self.cells or algorithm != self.algorithm:
            self.algorithm = algorithm
            self.new_grid(resolution, algorithm)
        if resolution not in self.cell_indices:
            self.cell_indices[resolution] = bin2d_with_indices(self.data['x'], self.data['y'],
                                                               self.cell_edges_x[resolution],
                                                               self.cell_edges_y[resolution])
        else:
            self.data = np.array(self.data)
            if not isinstance(self.cell_indices[resolution], np.ndarray):
                self.cell_indices[resolution] = self.cell_indices[resolution].compute()
            self.cell_indices[resolution] = np.array(
                self.cell_indices[resolution])  # can't be a memmap object, we need to overwrite data on disk
            if not isinstance(self.cells[resolution]['depth'], np.ndarray):
                self.cells[resolution]['depth'] = self.cells[resolution]['depth'].compute()
                if 'vertical_uncertainty' in self.cells[resolution]:
                    self.cells[resolution]['vertical_uncertainty'] = self.cells[resolution]['vertical_uncertainty'].compute()
                if 'horizontal_uncertainty' in self.cells[resolution]:
                    self.cells[resolution]['horizontal_uncertainty'] = self.cells[resolution]['horizontal_uncertainty'].compute()
            self.cells[resolution]['depth'] = np.array(self.cells[resolution]['depth'])
            if 'vertical_uncertainty' in self.cells[resolution]:
                self.cells[resolution]['vertical_uncertainty'] = np.array(self.cells[resolution]['vertical_uncertainty'])
            if 'horizontal_uncertainty' in self.cells[resolution]:
                self.cells[resolution]['horizontal_uncertainty'] = np.array(self.cells[resolution]['horizontal_uncertainty'])
            new_points = self.cell_indices[resolution] == -1
            if new_points.any():
                self.cell_indices[resolution][new_points] = bin2d_with_indices(self.data['x'][new_points],
                                                                               self.data['y'][new_points],
                                                                               self.cell_edges_x[resolution],
                                                                               self.cell_edges_y[resolution])
            else:  # there are no new points and this resolution already exists in the grid, so skip the gridding
                return resolution
        if algorithm == 'mean':
            self._run_mean_grid(resolution)
        elif algorithm == 'shoalest':
            self._run_shoalest_grid(resolution)
        return resolution

    def get_layers_by_name(self, layer: str = 'depth', resolution: float = None):
        """
        Get the layer at the provided resolution with the provided resolution.

        Parameters
        ----------
        layer
            layer name
        resolution
            resolution of the layer that we want.  If None, pulls the only layer in the Tile, errors if there is more
            than one layer

        Returns
        -------
        Union[da.Array, np.ndarray]
            2d array of the gridded data
        """

        if self.is_empty:
            return None
        if not resolution and len(list(self.cells.keys())) > 1:
            raise ValueError('Tile {}: you must specify a resolution to return layer data when multiple resolutions are found'.format(self.name))
        if resolution:
            if resolution not in list(self.cells.keys()):
                return None
        else:
            resolution = list(self.cells.keys())[0]
        if layer not in self.cells[resolution]:
            raise ValueError('Tile {}: layer {} not found for resolution {}'.format(self.name, layer, resolution))
        return self.cells[resolution][layer]


class VRTile(Tile):
    def __init__(self, min_x: float, min_y: float, size: float):
        super().__init__(min_x, min_y, size)
        self.overlay_cells = {}
