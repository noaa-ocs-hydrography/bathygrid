import numpy as np


class Tile:
    """
    Bathygrid is composed of multiple Tiles.  Each Tile manages its own point data and gridding.
    """
    def __init__(self, min_x: float, min_y: float, size: float):
        self.data = None
        self.container = None
        self.points_count = None

        self.min_grid_resolution = None
        self.max_grid_resolution = None

        self.min_x = min_x
        self.max_x = min_x + size
        self.min_y = min_y
        self.max_y = min_y + size

        self.width = size
        self.height = size
        self.name = f'{min_x}_{min_y}'

    @property
    def is_empty(self):
        """
        Return True if the Tile has no points
        """

        if not self.points_count:
            return True
        else:
            return False

    def add_points(self, data: np.ndarray, container: str):
        if self.data is None:
            self.data = data
            self.container = {container: [0, self.data['x'].size]}
        else:
            if container in self.container:
                self.remove_container(container)
            self.container[container] = [self.data.size, self.data.size + data.size]
            self.data = np.concatenate([self.data, data])
        self.points_count = self.data['x'].size

    def remove_container(self, container):
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
            self.points_count = self.data['x'].size
