import numpy as np
from bathygrid.utilities import is_power_of_two


class Grid:
    """
    Base grid for the Tile and the BathyGrid.
    """
    def __init__(self):
        self.min_y = None
        self.min_x = None
        self.max_y = None
        self.max_x = None

        self.cell_edges_x = None
        self.cell_edges_y = None

        self.cells = None

        self.width = None
        self.height = None

    @property
    def is_empty(self):
        if self.cells is None:
            return True
        elif not self.cells.any():
            return True
        return False


class BaseGrid(Grid):
    def __init__(self, cell_size):
        super().__init__()
        self.origin_x = 0
        self.origin_y = 0

        self.data = None
        self.container = {}  # dict of container name, list of multibeam files

        self.cell_x_origin = None
        self.cell_y_origin = None

        self.existing_cell_index = None
        self.existing_cell_mask = None

        self.maximum_cells = None
        self.number_of_cells = None

        self.cell_size = cell_size
        if not is_power_of_two(self.cell_size):
            raise ValueError('Grid size must be a power of two.')

    def _build_grid(self):
        if self.min_y is None or self.min_x is None or self.max_y is None or self.max_x is None:
            raise ValueError('UtmGrid not initialized!')

        nearest_lower_x = np.floor((self.min_x - self.origin_x) / self.cell_size) * self.cell_size
        nearest_higher_x = np.ceil((self.max_x - self.origin_x) / self.cell_size) * self.cell_size
        if nearest_higher_x == self.max_x:  # higher value cant be equal, our binning relies on it being less than
            nearest_higher_x += self.cell_size
        nearest_lower_y = np.floor((self.min_y - self.origin_y) / self.cell_size) * self.cell_size
        nearest_higher_y = np.ceil((self.max_y - self.origin_y) / self.cell_size) * self.cell_size
        if nearest_higher_y == self.max_y:
            nearest_higher_y += self.cell_size

        tx_origins = np.arange(nearest_lower_x, nearest_higher_x, self.cell_size)
        ty_origins = np.arange(nearest_lower_y, nearest_higher_y, self.cell_size)
        cell_x_origin, cell_y_origin = np.meshgrid(tx_origins, ty_origins)

        if not self.is_empty:  # we have existing cells, so find where the existing cells are in the new grid
            self.existing_cell_mask = np.full(cell_x_origin.shape, False)
            existing_cell_shape = self.cell_x_origin.shape
            y_corner_coord = np.argwhere(cell_y_origin == self.cell_y_origin[0][0])[0][0]
            x_corner_coord = np.argwhere(cell_x_origin == self.cell_x_origin[0][0])[0][1]
            self.existing_cell_mask[y_corner_coord:y_corner_coord + existing_cell_shape[0], x_corner_coord:x_corner_coord + existing_cell_shape[1]] = True

        self.cell_x_origin, self.cell_y_origin = cell_x_origin, cell_y_origin

        self.cell_edges_x = np.append(tx_origins, tx_origins[-1] + self.cell_size)
        self.cell_edges_y = np.append(ty_origins, ty_origins[-1] + self.cell_size)

        self.width = nearest_higher_x - nearest_lower_x
        self.height = nearest_higher_y, nearest_lower_y
        self.maximum_cells = self.cell_x_origin.size

    def _init_from_extents(self, min_y: float, min_x: float, max_y: float, max_x: float):
        self.min_y = min_y
        self.min_x = min_x
        self.max_y = max_y
        self.max_x = max_x
        self._build_grid()

    def _update_extents(self, min_y: float, min_x: float, max_y: float, max_x: float):
        self.min_y = min(min_y, self.min_y)
        self.min_x = min(min_x, self.min_x)
        self.max_y = max(max_y, self.max_y)
        self.max_x = max(max_x, self.max_x)
        self._build_grid()


class TileGrid(Grid):
    def __init__(self, min_x: float, min_y: float, size: float):
        super().__init__()
        self.data = None

        self.container = {}  # dict of container name, list of multibeam files

        self.min_grid_resolution = None
        self.max_grid_resolution = None

        self.cells = {}
        self.cell_edges_x = {}
        self.cell_edges_y = {}

        self._init_from_size(min_x, min_y, size)

    @property
    def is_empty(self):
        if not self.points_count:
            return True
        return False

    @property
    def points_count(self):
        if self.data is not None:
            return int(self.data.size)
        else:
            return 0

    def _init_from_size(self, min_x: float, min_y: float, size: float):
        self.min_x = min_x
        self.max_x = min_x + size
        self.min_y = min_y
        self.max_y = min_y + size

        self.width = size
        self.height = size
        self.name = f'{min_x}_{min_y}'
