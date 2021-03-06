import numpy as np
from dask.array import Array as darray

from bathygrid.grids import TileGrid
from bathygrid.utilities import bin2d_with_indices, is_power_of_two
from bathygrid.algorithms import np_grid_mean, np_grid_shoalest, calculate_slopes, nb_cube
from bathygrid.grid_variables import depth_resolution_lookup, minimum_points_per_cell, starting_resolution_density, \
    noise_accomodation_factor, revert_to_lookup_threshold


class Tile(TileGrid):
    """
    Bathygrid is composed of multiple Tiles.  Each Tile manages its own point data and gridding.  This base tile contains
    some common methods for managing point data.
    """

    def __init__(self, min_x: float, min_y: float, size: float):
        super().__init__(min_x, min_y, size)
        self.algorithm = None
        self.grid_parameters = None

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

    def __init__(self, min_x: float = 0.0, min_y: float = 0.0, size: float = 0.0, is_backscatter: bool = False):
        super().__init__(min_x, min_y, size)
        self.point_count_changed = False
        self.is_backscatter = is_backscatter

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
    def mean_depth(self):
        """
        Returns the mean z value for this grid, will be mean intensity for backscatter grid, name is a misnomer

        Returns
        -------
        float
            mean value for z data
        """

        if self.data is None:
            return 0.0
        else:
            return float(self.data['z'].mean())

    @property
    def cell_count(self):
        """
        Returns the number of cells for each resolution in the tile

        Returns
        -------
        dict
            dictionary of resolution values as float to number of cells as integer
        """

        final_count = {}
        for rez in self.cells:
            first_key = list(self.cells[rez].keys())[0]
            final_count[rez] = int(np.count_nonzero(~np.isnan(self.cells[rez][first_key])))
        return final_count

    @property
    def density_count(self):
        """
        Returns a list of number of soundings per cell for each populated cell in the tile

        Returns
        -------
        list
            list of sounding counts per cell
        """

        density_count = []
        for rez in self.cells:
            if 'density' not in self.cells[rez]:
                raise ValueError(f'No density layer found for Tile {self.name}')
            density = self.cells[rez]['density']
            if isinstance(density, darray):
                density = density.compute()
            density_count.extend(density[density > 0].tolist())
        return density_count

    @property
    def density_per_square_meter(self):
        """
        Returns a list of number of soundings / m2 per cell for each populated cell in the tile

        Returns
        -------
        list
            list of soundings / m2 per cell
        """

        density_per_meter = []
        for rez in self.cells:
            if 'density' not in self.cells[rez]:
                raise ValueError(f'No density layer found for Tile {self.name}')
            density = self.cells[rez]['density']
            if isinstance(density, darray):
                density = density.compute()
            density_per_meter.extend((density[density > 0] / (rez ** 2)).tolist())
        return density_per_meter

    @property
    def density_count_vs_depth(self):
        """
        Returns a tuple of (density_count, depth in meters)

        Returns
        -------
        tuple
            tuple of (density_count, depth in meters)
        """

        density_per_meter = []
        depth_values = []
        for rez in self.cells:
            if 'density' not in self.cells[rez]:
                raise ValueError(f'No density layer found for Tile {self.name}')
            if 'depth' not in self.cells[rez]:
                raise ValueError(f'No depth layer found for Tile {self.name}')
            density = self.cells[rez]['density']
            depth = self.cells[rez]['depth']
            if isinstance(density, darray):
                density = density.compute()
            if isinstance(depth, darray):
                depth = depth.compute()
            msk = density > 0
            density_per_meter.extend(density[msk].tolist())
            depth_values.extend(depth[msk].tolist())
        return density_per_meter, depth_values

    @property
    def density_per_square_meter_vs_depth(self):
        """
        Returns a tuple of (density_per_square_meter, depth in meters)

        Returns
        -------
        tuple
            tuple of (density_per_square_meter, depth in meters)
        """

        density_per_meter = []
        depth_values = []
        for rez in self.cells:
            if 'density' not in self.cells[rez]:
                raise ValueError(f'No density layer found for Tile {self.name}')
            if 'depth' not in self.cells[rez]:
                raise ValueError(f'No depth layer found for Tile {self.name}')
            density = self.cells[rez]['density']
            depth = self.cells[rez]['depth']
            if isinstance(density, darray):
                density = density.compute()
            if isinstance(depth, darray):
                depth = depth.compute()
            msk = density > 0
            density_per_meter.extend((density[density > 0] / (rez ** 2)).tolist())
            depth_values.extend(depth[msk].tolist())
        return density_per_meter, depth_values

    @property
    def coverage_area_square_meters(self):
        """
        Returns the coverage area of the tile in square meters, ommitting any unpopulated cells

        Returns
        -------
        float
            coverage area in square meters
        """

        area = 0.0
        for rez in self.cells:
            if 'density' not in self.cells[rez]:
                raise ValueError(f'No density layer found for Tile {self.name}')
            density = self.cells[rez]['density']
            if isinstance(density, darray):
                density = density.compute()
            area += round(density[density > 0].size * (rez ** 2), 3)
        return area

    @property
    def coverage_area_square_nm(self):
        """
        Returns the coverage area of the tile in square nautical miles, ommitting any unpopulated cells

        Returns
        -------
        float
            coverage area in square nautical miles
        """

        return self.coverage_area_square_meters / 3434290.012

    def _calculate_resolution_lookup(self):
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
                self._clear_temp_data(resolution)
        self.point_count_changed = True

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
                self._clear_temp_data(resolution)
            self.data = self.data[msk]
            self.container.pop(container)
            self.point_count_changed = True

    def new_grid(self, resolution: float, algorithm: str, nodatavalue: float = np.float32(np.nan)):
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
        if algorithm in ['mean', 'shoalest']:
            self.cells[resolution][self.depth_key] = np.full(grid_shape, nodatavalue, dtype=np.float32)
            self.cells[resolution]['density'] = np.full(grid_shape, 0, dtype=int)
            if self.data is not None and 'tvu' in self.data.dtype.names:
                self.cells[resolution]['vertical_uncertainty'] = np.full(grid_shape, nodatavalue, dtype=np.float32)
            if self.data is not None and 'thu' in self.data.dtype.names:
                self.cells[resolution]['horizontal_uncertainty'] = np.full(grid_shape, nodatavalue, dtype=np.float32)
        elif algorithm == 'cube':
            self.cells[resolution][self.depth_key] = np.full(grid_shape, nodatavalue, dtype=np.float32)
            self.cells[resolution]['density'] = np.full(grid_shape, 0, dtype=int)
            self.cells[resolution]['hypothesis_count'] = np.full(grid_shape, 0, dtype=int)
            self.cells[resolution]['total_uncertainty'] = np.full(grid_shape, nodatavalue, dtype=np.float32)
            self.cells[resolution]['hypothesis_ratio'] = np.full(grid_shape, nodatavalue, dtype=np.float32)

    def _clear_temp_data(self, resolution: float):
        """
        We have some data that is not flushed to disk, we retain it for experiments like the patch test.  But this data
        is invalidated when gridding is re-run or when points are added/removed.
        """
        if resolution in self.cells:
            if 'x_slope' in self.cells[resolution]:
                self.cells[resolution].pop('x_slope')
            if 'y_slope' in self.cells[resolution]:
                self.cells[resolution].pop('y_slope')

    def _grid_algorithm_initialize(self, resolution: float, only_container: str = None):
        vert_val = np.array([])
        horiz_val = np.array([])
        vert_grid = np.array([])
        horiz_grid = np.array([])
        if 'tvu' in self.data.dtype.names and not only_container:
            if not isinstance(self.data, np.ndarray):
                vert_val = self.data['tvu'].compute()
            else:
                vert_val = self.data['tvu']
            vert_grid = self.cells[resolution]['vertical_uncertainty']
        if 'thu' in self.data.dtype.names and not only_container:
            if not isinstance(self.data, np.ndarray):
                horiz_val = self.data['thu'].compute()
            else:
                horiz_val = self.data['thu']
            horiz_grid = self.cells[resolution]['horizontal_uncertainty']
        if only_container:
            depth_val = self.data['z'][self.container[only_container][0]:self.container[only_container][1]]
            self.cells[resolution][only_container] = np.full(self.cells[resolution][self.depth_key].shape, np.float32(np.nan), dtype=np.float32)
            self.cells[resolution][only_container + '_density'] = np.full(self.cells[resolution][self.depth_key].shape, 0, dtype=int)
        else:
            depth_val = self.data['z']
        if not isinstance(self.data, np.ndarray):
            depth_val = depth_val.compute()
        if only_container:
            cindx = self.cell_indices[resolution][self.container[only_container][0]:self.container[only_container][1]]
        else:
            cindx = self.cell_indices[resolution]
        if not isinstance(cindx, np.ndarray):
            cindx = cindx.compute()
        return vert_val, horiz_val, depth_val, vert_grid, horiz_grid, cindx

    def _cube_grid_algorithm_initialize(self, data: np.ndarray, resolution: float, only_container: str = None):
        if not isinstance(data, np.ndarray):
            vert_val = data['tvu'].compute()
            horiz_val = data['thu'].compute()
        else:
            vert_val = data['tvu']
            horiz_val = data['thu']

        if not only_container:
            totalunc_grid = self.cells[resolution]['total_uncertainty']
            hypcnt_grid = self.cells[resolution]['hypothesis_count']
            hypratio_grid = self.cells[resolution]['hypothesis_ratio']
        else:
            totalunc_grid = np.array([])
            hypcnt_grid = np.array([])
            hypratio_grid = np.array([])

        if only_container:
            depth_val = data['z'][self.container[only_container][0]:self.container[only_container][1]]
            self.cells[resolution][only_container] = np.full(self.cells[resolution][self.depth_key].shape, np.float32(np.nan), dtype=np.float32)
            self.cells[resolution][only_container + '_density'] = np.full(self.cells[resolution][self.depth_key].shape, 0, dtype=int)
        else:
            depth_val = data['z']
        if not isinstance(data, np.ndarray):
            depth_val = depth_val.compute()

        if only_container:
            cindx = self.cell_indices[resolution][self.container[only_container][0]:self.container[only_container][1]]
        else:
            cindx = self.cell_indices[resolution]
        if not isinstance(cindx, np.ndarray):
            cindx = cindx.compute()
        return vert_val, horiz_val, depth_val, totalunc_grid, hypcnt_grid, hypratio_grid, cindx

    def _run_mean_grid(self, resolution: float, only_container: str = None):
        """
        Run the mean algorithm on the Tile data
        """

        vert_val, horiz_val, depth_val, vert_grid, horiz_grid, cindx = self._grid_algorithm_initialize(resolution, only_container=only_container)
        if not only_container:
            np_grid_mean(depth_val, cindx, self.cells[resolution][self.depth_key], self.cells[resolution]['density'], vert_val, horiz_val, vert_grid, horiz_grid)
            self.cells[resolution][self.depth_key] = np.round(self.cells[resolution][self.depth_key], 3)
            if vert_val.size > 0:
                self.cells[resolution]['vertical_uncertainty'] = np.round(self.cells[resolution]['vertical_uncertainty'], 3)
            if horiz_val.size > 0:
                self.cells[resolution]['horizontal_uncertainty'] = np.round(self.cells[resolution]['horizontal_uncertainty'], 3)
        else:
            np_grid_mean(depth_val, cindx, self.cells[resolution][only_container], self.cells[resolution][only_container + '_density'], vert_val, horiz_val, vert_grid, horiz_grid)
            self.cells[resolution][only_container] = np.round(self.cells[resolution][only_container], 3)

    def _run_shoalest_grid(self, resolution: float, only_container: str = None):
        """
        Run the shoalest algorithm on the Tile data
        """

        vert_val, horiz_val, depth_val, vert_grid, horiz_grid, cindx = self._grid_algorithm_initialize(resolution, only_container=only_container)
        if not only_container:
            np_grid_shoalest(depth_val, cindx, self.cells[resolution][self.depth_key], self.cells[resolution]['density'], vert_val, horiz_val, vert_grid, horiz_grid)
            self.cells[resolution][self.depth_key] = np.round(self.cells[resolution][self.depth_key], 3)
            if vert_val.size > 0:
                self.cells[resolution]['vertical_uncertainty'] = np.round(self.cells[resolution]['vertical_uncertainty'], 3)
            if horiz_val.size > 0:
                self.cells[resolution]['horizontal_uncertainty'] = np.round(self.cells[resolution]['horizontal_uncertainty'], 3)
        else:
            np_grid_shoalest(depth_val, cindx, self.cells[resolution][only_container], self.cells[resolution][only_container + '_density'], vert_val, horiz_val, vert_grid, horiz_grid)
            self.cells[resolution][only_container] = np.round(self.cells[resolution][only_container], 3)

    def _run_cube_grid(self, resolution: float, grid_parameters: dict = float, only_container: str = None, border_data: np.ndarray = None):
        if border_data is not None:
            data = np.concatenate([self.data, border_data])
        else:
            data = self.data
        vert_val, horiz_val, depth_val, totalunc_grid, hypcnt_grid, hypratio_grid, cindx = self._cube_grid_algorithm_initialize(data, resolution, only_container=only_container)
        if not isinstance(data, np.ndarray):
            x_val = data['x'].compute()
            y_val = data['y'].compute()
        else:
            x_val = data['x']
            y_val = data['y']

        if grid_parameters and 'method' in grid_parameters:
            grid_method = grid_parameters['method']
        else:
            print('WARNING: cube method not found in given parameters, defaulting to "local"')
            grid_method = 'local'

        if grid_parameters and 'iho_order' in grid_parameters:
            iho_order = grid_parameters['iho_order']
        else:
            print('WARNING: cube iho_order not found in given parameters, defaulting to "order1a"')
            iho_order = 'order1a'

        if grid_parameters and 'variance_selection' in grid_parameters:
            grid_variance_selection = grid_parameters['variance_selection']
        else:
            print('WARNING: cube variance_selection not found in given parameters, defaulting to "cube"')
            grid_variance_selection = 'cube'

        if not only_container:
            nb_cube(x_val, y_val, depth_val, cindx, self.cells[resolution][self.depth_key], self.cells[resolution]['density'],
                    vert_val, horiz_val, totalunc_grid, hypcnt_grid, hypratio_grid, self.min_x, self.max_y, iho_order, grid_method,
                    resolution, resolution, variance_selection=grid_variance_selection)
            self.cells[resolution][self.depth_key] = np.round(self.cells[resolution][self.depth_key], 3)
            self.cells[resolution]['total_uncertainty'] = np.round(self.cells[resolution]['total_uncertainty'], 3)
            self.cells[resolution]['hypothesis_ratio'] = np.round(self.cells[resolution]['hypothesis_ratio'], 3)
        else:
            nb_cube(x_val, y_val, depth_val, cindx, self.cells[resolution][only_container], self.cells[resolution][only_container + '_density'],
                    vert_val, horiz_val, totalunc_grid, hypcnt_grid, hypratio_grid, self.min_x, self.max_y, iho_order, grid_method,
                    resolution, resolution, variance_selection=grid_variance_selection)
            self.cells[resolution][only_container] = np.round(self.cells[resolution][only_container], 3)

    def _run_slopes(self, resolution: float):
        if 'x_slope' in self.cells[resolution]:
            # we've already run slope calculations on this object.  These are not saved to disk to conserve space, but
            #   they will remain in the cells buffer
            return
        if not isinstance(self.data, np.ndarray):
            x_val = self.data['x'].compute()
            y_val = self.data['y'].compute()
            depth_val = self.data['z'].compute()
        else:
            x_val = self.data['x']
            y_val = self.data['y']
            depth_val = self.data['z']
        if not isinstance(self.cell_indices[resolution], np.ndarray):
            cindx = self.cell_indices[resolution].compute()
        else:
            cindx = self.cell_indices[resolution]
        if not isinstance(self.cell_edges_x[resolution], np.ndarray):
            cedgex = self.cell_edges_x[resolution].compute()
            cedgey = self.cell_edges_y[resolution].compute()
        else:
            cedgex = self.cell_edges_x[resolution]
            cedgey = self.cell_edges_y[resolution]
        self.cells[resolution]['x_slope'], self.cells[resolution]['y_slope'] = calculate_slopes(x_val, y_val, depth_val, cindx, cedgex, cedgey, visualize=False)

    def _return_cell_counts(self, resolution: float):
        grid_x = np.arange(self.min_x, self.max_x, resolution)
        grid_y = np.arange(self.min_y, self.max_y, resolution)
        cell_edges_x = np.append(grid_x, grid_x[-1] + resolution)
        cell_edges_y = np.append(grid_y, grid_y[-1] + resolution)
        cell_indices = bin2d_with_indices(self.data['x'], self.data['y'], cell_edges_x, cell_edges_y)
        uniqs, counts = np.unique(cell_indices, return_counts=True)
        return uniqs, counts

    def _assess_resolution(self, resolution: float = None):
        """
        Using the points in this tile, assess the given resolution to determine if it is too coarse or too fine.  We use
        the grid_variables minimum_points_per_cell as the ideal points per cell.  If 95% of the cells contain at least
        this many points and not more than 4 times this many points, the resolution is deemed good.

        Parameters
        ----------
        resolution
            resolution of the grid in x/y units

        Returns
        -------
        bool
            if True, resolution is deemed good
        str
            string qualifier, if LOW resolution is too low, if HIGH resolution is too high, if empty string the resolution is good
        """

        uniqs, counts = self._return_cell_counts(resolution)
        # if there are less than minimum_points_per_cell in any cell, the resolution is too fine
        too_fine = (counts < minimum_points_per_cell)
        # if there are greater than minimum_points_per_cell * 4 in all cells, the resolution is too coarse
        # mulitply by 4 as the cell is split into 4 more cells if we use a finer resolution
        too_coarse = (counts > minimum_points_per_cell * 4)
        # check to see if all cells in grid are populated by points
        # fully_populated = len(uniqs) == grid_x.shape[0] * grid_y.shape[0]

        # if any cells have less than minimum points, this resolution is too low
        if too_fine.any():
            return False, 'LOW'
        # all cells have too coarse a resolution
        elif too_coarse.all():
            return False, 'HIGH'
        else:
            return True, ''

    def resolution_by_density_old(self, starting_resolution: float = None):
        """
        DEPRECATED: See resolution_by_density
        A recursive check with the points in this tile to identify the best resolution based on density.  We start with
        the depth resolution lookup resolution, binning the points and determining if grid_variables.check_cells_percentage
        of the cells have the appropriate number of points per cell, see grid_variables.minimum_points_per_cell.  We then
        rerun this check until we settle on the appropriate resolution.

        Parameters
        ----------
        starting_resolution
            the first resolution to evaluate, will go up/down from this resolution in the iterative check

        Returns
        -------
        float
            resolution to use based on the density of the points in the tile
        """

        rez_options = list(depth_resolution_lookup.values())
        if not starting_resolution:
            starting_resolution = starting_resolution_density  # start at a coarse resolution to catch holidays
        else:
            if starting_resolution not in rez_options:
                raise ValueError('Provided resolution {} is not one of the valid resolution options: {}'.format(starting_resolution, rez_options))

        valid_rez = False
        checked_rez = []
        current_rez = starting_resolution
        while not valid_rez:
            valid_rez, rez_adjustment = self._assess_resolution(current_rez)
            checked_rez.append(current_rez)
            curr_rez_index = rez_options.index(current_rez)
            # if you hit the resolution limit in available resolutions, just stop there
            # also, if this is a good resolution, stop here
            if valid_rez or (curr_rez_index == 0 and rez_adjustment == 'HIGH') or (curr_rez_index == len(rez_options) - 1 and rez_adjustment == 'LOW'):
                return current_rez
            # get the next lower resolution
            elif rez_adjustment == 'HIGH':
                current_rez = rez_options[curr_rez_index - 1]
            # get the next higher resolution
            elif rez_adjustment == 'LOW':
                current_rez = rez_options[curr_rez_index + 1]
            # if you are about to check a resolution that has been checked already (i.e. you are going back and forth
            #   between resolutions) go with the greater of the two to be conservative
            if current_rez in checked_rez:
                return max(current_rez, checked_rez[-1])

    def resolution_by_density(self, starting_resolution: float = None, noise_factor: float = None):
        """
        A density based check adapted from the "Computationally efficient variable resolution depth estimation" paper by
        Brian Calder/Glen Rice.  Determine the density for a coarse resolution grid on the tile (starting_resolution)
        and calculate a predicted resolution to fit the cell using the noise_accomodation_factor and
        minimum_points_per_cell parameters.

        Parameters
        ----------
        starting_resolution
            the first resolution to evaluate, will go up/down from this resolution in the iterative check
        noise_factor
            increase this to increase the number of soundings allowed per node, is multiplied against the
            minimum points per cell parameter

        Returns
        -------
        float
            resolution to use based on the density of the points in the tile
        """

        rez_options = sorted(list(depth_resolution_lookup.values()))
        tile_size = self.max_x - self.min_x
        if not starting_resolution:
            starting_resolution = starting_resolution_density  # start at a coarse resolution to catch holidays
        else:
            if starting_resolution not in rez_options:
                raise ValueError('Provided resolution {} is not one of the valid resolution options: {}'.format(starting_resolution, rez_options))
        if not noise_factor:
            noise_factor = noise_accomodation_factor
        starting_resolution = min(tile_size, starting_resolution)
        max_starting_cells = int((tile_size / starting_resolution) ** 2)
        uniqs, counts = self._return_cell_counts(starting_resolution)
        percent_full = len(uniqs) / max_starting_cells
        # try and deal with edge cases, where the tile is not filled in, like along the edge of the survey
        # can't rely on density, that will drive the resolution up, as the tile is nearly empty
        # instead fall back on the old depth lookup method
        if percent_full <= revert_to_lookup_threshold:
            rez_depths = list(depth_resolution_lookup.keys())
            nearest_valid_resolution_index = int(np.searchsorted(rez_depths, self.mean_depth))
            if nearest_valid_resolution_index == len(rez_options):  # greater than any rez option, use the coarsest resolution
                nearest_valid_resolution_index -= 1
            final_rez = depth_resolution_lookup[rez_depths[nearest_valid_resolution_index]]
        else:
            # estimate resolution per coarse grid cell based on density of the points in each cell (from Calder paper)
            cell_density = counts / (starting_resolution ** 2)
            resolution_estimate = np.sqrt(2 * minimum_points_per_cell * (1 + noise_factor) / cell_density)
            # compute weighted average of resolution estimate across all cells to get tile wide resolution estimate
            max_resolution = np.sum(resolution_estimate * counts) / np.sum(counts)
            # get the nearest power of two resolution
            nearest_valid_resolution_index = int(np.searchsorted(rez_options, max_resolution))
            if nearest_valid_resolution_index == len(rez_options):  # greater than any rez option, use the coarsest resolution
                nearest_valid_resolution_index -= 1
            final_rez = rez_options[nearest_valid_resolution_index]
        # resolution cannot be greater than the tile size of course...
        final_rez = min(final_rez, tile_size)
        return final_rez

    def grid(self, algorithm: str, resolution: float = None, clear_existing: bool = False, auto_resolution_mode: str = 'depth',
             regrid_option: str = '', progress_bar: bool = False, grid_parameters: dict = None, border_data: np.ndarray = None):
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
        auto_resolution_mode
            one of density, depth; chooses the algorithm used to determine the resolution for the tile
        regrid_option
            a place holder to match the bgrid grid method
        progress_bar
            a place holder to match the bgrid grid method
        grid_parameters
            optional dict of settings to pass to the grid algorithm
        border_data
            point data that falls on the borders, used in the CUBE algorithm to handle tile edge issues

        Returns
        -------
        float
            resolution of the grid in x/y units
        """

        if resolution is None:
            if auto_resolution_mode == 'depth':
                resolution = self._calculate_resolution_lookup()
            elif auto_resolution_mode == 'density':
                resolution = self.resolution_by_density()
            else:
                raise ValueError('Tile given no resolution and an option of {} which is not supported'.format(auto_resolution_mode))
        if not is_power_of_two(resolution):
            raise ValueError(f'Tile: Resolution must be a power of two, got {resolution}')
        if clear_existing:
            self.clear_grid()
        self._clear_temp_data(resolution)

        if resolution not in self.cells or algorithm != self.algorithm:
            self.algorithm = algorithm
            self.grid_parameters = grid_parameters
            self.new_grid(resolution, algorithm)
        if not isinstance(self.cell_edges_x[resolution], np.ndarray):
            self.cell_edges_x[resolution] = self.cell_edges_x[resolution].compute()
            self.cell_edges_y[resolution] = self.cell_edges_y[resolution].compute()
        if resolution not in self.cell_indices:
            self.cell_indices[resolution] = bin2d_with_indices(self.data['x'], self.data['y'], self.cell_edges_x[resolution], self.cell_edges_y[resolution])
        else:
            if not isinstance(self.cell_indices[resolution], np.ndarray):
                self.cell_indices[resolution] = self.cell_indices[resolution].compute()
            self.cell_indices[resolution] = np.array(self.cell_indices[resolution])  # can't be a memmap object, we need to overwrite data on disk
            if not isinstance(self.cells[resolution][self.depth_key], np.ndarray):
                self.cells[resolution][self.depth_key] = self.cells[resolution][self.depth_key].compute()
                self.cells[resolution]['density'] = self.cells[resolution]['density'].compute()
                for lyr in ['vertical_uncertainty', 'horizontal_uncertainty', 'hypothesis_count', 'total_uncertainty', 'hypothesis_ratio']:
                    if lyr in self.cells[resolution]:
                        self.cells[resolution][lyr] = self.cells[resolution][lyr].compute()
            new_points = self.cell_indices[resolution] == -1
            if new_points.any():
                self.cell_indices[resolution][new_points] = bin2d_with_indices(self.data['x'][new_points], self.data['y'][new_points],
                                                                               self.cell_edges_x[resolution], self.cell_edges_y[resolution])

            self.cells[resolution][self.depth_key] = np.full(self.cells[resolution][self.depth_key].shape, np.nan, dtype=np.float32)
            self.cells[resolution]['density'] = np.full(self.cells[resolution]['density'].shape, 0, dtype=int)
            for lyr in ['vertical_uncertainty', 'horizontal_uncertainty', 'hypothesis_count', 'total_uncertainty', 'hypothesis_ratio']:
                if lyr in self.cells[resolution]:
                    if lyr == 'hypothesis_count':
                        self.cells[resolution][lyr] = np.full(self.cells[resolution][lyr].shape, 0, dtype=int)
                    else:
                        self.cells[resolution][lyr] = np.full(self.cells[resolution][lyr].shape, np.nan, dtype=np.float32)

        if algorithm == 'mean':
            self._run_mean_grid(resolution)
        elif algorithm == 'shoalest':
            self._run_shoalest_grid(resolution)
        elif algorithm == 'cube':
            self._run_cube_grid(resolution, grid_parameters, border_data=border_data)
        self.point_count_changed = False
        return resolution

    def get_layers_by_name(self, layer: str = 'depth', resolution: float = None, nodatavalue: float = np.float32(np.nan),
                           z_positive_up: bool = False):
        """
        Get the layer at the provided resolution with the provided resolution.

        Parameters
        ----------
        layer
            layer name, can either be a grid layer name ('depth') or the name of a container that exists in the grid
            that allows you to return the grid for just that layer, i.e. container query
        resolution
            resolution of the layer that we want.  If None, pulls the only layer in the Tile, errors if there is more
            than one layer
        nodatavalue
            nodatavalue to set in the regular grid
        z_positive_up
            if True, will output bands with positive up convention

        Returns
        -------
        Union[da.Array, np.ndarray]
            2d array of the gridded data
        """

        container_query = False
        if layer in ['depth', 'intensity', 'vertical_uncertainty', 'horizontal_uncertainty', 'total_uncertainty', 'hypothesis_ratio', 'x_slope', 'y_slope']:
            # ensure nodatavalue is a float32
            nodatavalue = np.float32(nodatavalue)
        elif layer in ['density', 'hypothesis_count']:
            # density has to have an integer based nodatavalue
            try:
                nodatavalue = np.int(nodatavalue)
            except ValueError:
                nodatavalue = 0
        else:
            # this must be a container query, layer is the name of a container
            container_query = True
        if self.is_empty:
            return None
        if not resolution and len(list(self.cells.keys())) > 1:
            raise ValueError('Tile {}: you must specify a resolution to return layer data when multiple resolutions are found'.format(self.name))
        if resolution:
            if resolution not in list(self.cells.keys()):
                return None
        else:
            resolution = list(self.cells.keys())[0]
        if layer in ['x_slope', 'y_slope']:
            self._run_slopes(resolution)
        elif container_query:
            if layer in self.container.keys():
                if self.algorithm == 'mean':
                    self._run_mean_grid(resolution, only_container=layer)
                elif self.algorithm == 'shoalest':
                    self._run_shoalest_grid(resolution, only_container=layer)
                elif self.algorithm == 'cube':
                    self._run_cube_grid(resolution, self.grid_parameters, only_container=layer)
            else:
                self.cells[resolution][layer] = np.full_like(self.cells[resolution][self.depth_key], nodatavalue)
        if layer not in self.cells[resolution]:
            raise ValueError('Tile {}: layer {} not found for resolution {}'.format(self.name, layer, resolution))
        try:
            data = np.copy(self.cells[resolution][layer].compute())
        except:
            data = np.copy(self.cells[resolution][layer])
        if np.count_nonzero(np.isnan(data)) == data.size:
            return None
        if layer.lower() == 'depth' and z_positive_up:
            data = data * -1
        if not np.isnan(nodatavalue):  # if nodatavalue is not NaN, we need to replace it with the nodatavalue
            data[np.isnan(data)] = nodatavalue
        return data

    def return_layer_values(self, layer: str):
        """
        Return a 1d array of all values in the provided layer name, excluding nodatavalues.

        Parameters
        ----------
        layer

        Returns
        -------
        list
            list of all values in the grid across all resolutions, excluding nodatavalues
        """
        if layer in ['depth', 'intensity', 'vertical_uncertainty', 'horizontal_uncertainty', 'total_uncertainty', 'hypothesis_ratio']:
            # ensure nodatavalue is a float32
            nodatavalue = np.float32(np.nan)
        elif layer in ['density', 'hypothesis_count']:
            nodatavalue = 0
        else:
            raise ValueError("Bathygrid: return_layer_values - only 'depth', 'density', 'intensity', 'vertical_uncertainty', 'horizontal_uncertainty' currently supported")
        layer_values = []
        for rez in self.cells:
            if layer in self.cells[rez]:
                lvalues = self.cells[rez][layer]
                if isinstance(lvalues, darray):
                    lvalues = lvalues.compute()
                if np.isnan(nodatavalue):
                    layer_values.extend(lvalues[~np.isnan(lvalues)].tolist())
                else:
                    layer_values.extend(lvalues[lvalues != nodatavalue].tolist())
        return layer_values


class VRTile(Tile):
    def __init__(self, min_x: float, min_y: float, size: float):
        super().__init__(min_x, min_y, size)
        self.overlay_cells = {}
