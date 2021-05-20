import numpy as np
from bathygrid.grids import TileGrid
from bathygrid.utilities import bin2d_with_indices, is_power_of_two
from bathygrid.algorithms import nb_grid_mean


class Tile(TileGrid):
    """
    Bathygrid is composed of multiple Tiles.  Each Tile manages its own point data and gridding.
    """
    def __init__(self, min_x: float, min_y: float, size: float):
        super().__init__(min_x, min_y, size)
        self.algorithm = None

    def clear_grid(self):
        self.cells = {}
        self.cell_edges_x = {}
        self.cell_edges_y = {}
        self.min_grid_resolution = None
        self.max_grid_resolution = None

    def clear_points(self):
        self.data = None
        self.container = {}  # dict of container name, list of multibeam fil
        self.cell_indices = {}

    def add_points(self, data: np.ndarray, container: str):
        if self.data is None:
            self.data = data
            self.container = {container: [0, self.data['x'].size]}
            for resolution in self.cell_indices:
                self.cell_indices[resolution] = np.full(self.data.shape, -1)
        else:
            if container in self.container:
                self.remove_points(container)
            self.container[container] = [self.data.size, self.data.size + data.size]
            self.data = np.concatenate([self.data, data])
            for resolution in self.cell_indices:
                self.cell_indices[resolution] = np.append(self.cell_indices[resolution], np.full(data.shape, -1))

    def remove_points(self, container):
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
        if not is_power_of_two(resolution):
            raise ValueError(f'Tile: Resolution must be a power of two, got {resolution}')
        grid_x = np.arange(self.min_x, self.max_x, resolution)
        grid_y = np.arange(self.min_y, self.max_y, resolution)
        self.cell_edges_x[resolution] = np.append(grid_x, grid_x[-1] + resolution)
        self.cell_edges_y[resolution] = np.append(grid_y, grid_y[-1] + resolution)
        grid_shape = (grid_x.size, grid_y.size)
        self.cells[resolution] = {}
        if algorithm == 'mean':
            self.cells[resolution]['depth'] = np.full(grid_shape, nodatavalue)
            self.cells[resolution]['uncertainty'] = np.full(grid_shape, nodatavalue)


class SRTile(Tile):
    def __init__(self, min_x: float, min_y: float, size: float):
        super().__init__(min_x, min_y, size)

    def _run_mean_grid(self, resolution):
        self.cells[resolution]['depth'], self.cells[resolution]['uncertainty'] = nb_grid_mean(self.data['z'], self.data['unc'],
                                                                                              self.cell_indices[resolution],
                                                                                              self.cells[resolution]['depth'],
                                                                                              self.cells[resolution]['uncertainty'])
        self.cells[resolution]['depth'] = np.round(self.cells[resolution]['depth'], 3)
        self.cells[resolution]['uncertainty'] = np.round(self.cells[resolution]['uncertainty'], 3)

    def grid(self, resolution: float, algorithm: str):
        if not is_power_of_two(resolution):
            raise ValueError(f'Tile: Resolution must be a power of two, got {resolution}')

        if resolution not in self.cells or algorithm != self.algorithm:
            self.algorithm = algorithm
            self.new_grid(resolution, algorithm)
        if resolution not in self.cell_indices:
            self.cell_indices[resolution] = bin2d_with_indices(self.data['x'], self.data['y'], self.cell_edges_x[resolution],
                                                               self.cell_edges_y[resolution])
        else:
            new_points = self.cell_indices[resolution] == -1
            self.cell_indices[resolution][new_points] = bin2d_with_indices(self.data['x'][new_points], self.data['y'][new_points],
                                                                           self.cell_edges_x[resolution], self.cell_edges_y[resolution])
        if algorithm == 'mean':
            self._run_mean_grid(resolution)
