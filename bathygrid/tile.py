import numpy as np
from bathygrid.grids import TileGrid
from bathygrid.utilities import bin2d_with_indices, is_power_of_two
from bathygrid.algorithms import grid_mean


class Tile(TileGrid):
    """
    Bathygrid is composed of multiple Tiles.  Each Tile manages its own point data and gridding.
    """
    def __init__(self, min_x: float, min_y: float, size: float):
        super().__init__(min_x, min_y, size)
        self.algorithm = None

    def add_points(self, data: np.ndarray, container: str):
        if self.data is None:
            self.data = data
            self.container = {container: [0, self.data['x'].size]}
        else:
            if container in self.container:
                self.remove_points(container)
            self.container[container] = [self.data.size, self.data.size + data.size]
            self.data = np.concatenate([self.data, data])

    def remove_points(self, container):
        if container in self.container:
            remove_start, remove_end = self.container[container]
            msk = np.ones(self.data.shape[0], dtype=bool)
            msk[remove_start:remove_end] = False
            chunk_size = msk[remove_start:remove_end].size
            for cont in self.container:
                if self.container[cont][0] >= remove_end:
                    self.container[cont] = [self.container[cont][0] - chunk_size, self.container[cont][1] - chunk_size]
            self.container.pop(container)
            self.data = self.data[msk]

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

    def grid_single_resolution(self, resolution: float, algorithm: str):
        if not is_power_of_two(resolution):
            raise ValueError(f'Tile: Resolution must be a power of two, got {resolution}')
        if resolution not in self.cells or algorithm != self.algorithm:
            self.algorithm = algorithm
            self.new_grid(resolution, algorithm)
        cell_indices = bin2d_with_indices(self.data['x'], self.data['y'], self.cell_edges_x[resolution], self.cell_edges_y[resolution])
        if algorithm == 'mean':
            self.cells[resolution]['depth'], self.cells[resolution]['uncertainty'] = grid_mean(self.data['z'], self.data['unc'],
                                                                                               cell_indices, self.cells[resolution]['depth'],
                                                                                               self.cells[resolution]['uncertainty'])
