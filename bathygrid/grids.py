import numpy as np
from typing import Union


def is_power_of_two(n: Union[int, float]):
    """
    Return True if number is a power of two, supports n>1 and n<1.

    Parameters
    ----------
    n
        number to check, can be float or int

    Returns
    -------
    bool
        number is power of two
    """

    if n > 1:
        if n != int(n):
            return False
        n = int(n)
        return (n != 0) and (n & (n - 1) == 0)
    elif n == 1:
        return True
    elif n > 0:
        return is_power_of_two(1/n)
    else:
        return False
    

class Grid:
    def __init__(self, tile_size):
        self.origin_x = 0
        self.origin_y = 0
        self.min_y = None
        self.min_x = None
        self.max_y = None
        self.max_x = None

        self.tiles = None

        self.existing_tile_index = None

        self.tile_x_origin = None
        self.tile_y_origin = None
        self.tile_center_x = None
        self.tile_center_y = None
        self.tile_edges_x = None
        self.tile_edges_y = None

        self.existing_tile_mask = None

        self.width = None
        self.height = None
        self.maximum_tiles = None
        self.number_of_tiles = None

        self.tile_size = tile_size
        if not is_power_of_two(self.tile_size):
            raise ValueError('Grid size must be a power of two.')

    @property
    def is_empty(self):
        if self.tiles is None:
            return True
        elif not self.tiles.any():
            return True
        return False

    def _build_grid(self):
        if self.min_y is None or self.min_x is None or self.max_y is None or self.max_x is None:
            raise ValueError('UtmGrid not initialized!')

        nearest_lower_x = np.floor((self.min_x - self.origin_x) / self.tile_size) * self.tile_size
        nearest_higher_x = np.ceil((self.max_x - self.origin_x) / self.tile_size) * self.tile_size
        nearest_lower_y = np.floor((self.min_y - self.origin_y) / self.tile_size) * self.tile_size
        nearest_higher_y = np.ceil((self.max_y - self.origin_y) / self.tile_size) * self.tile_size

        tx_origins = np.arange(nearest_lower_x, nearest_higher_x, self.tile_size)
        ty_origins = np.arange(nearest_lower_y, nearest_higher_y, self.tile_size)
        tile_x_origin, tile_y_origin = np.meshgrid(tx_origins, ty_origins)

        if not self.is_empty:  # we have existing tiles, so find where the existing tiles are in the new grid
            self.existing_tile_mask = np.full(tile_x_origin.shape, False)
            existing_tile_shape = self.tile_x_origin.shape
            y_corner_coord = np.argwhere(tile_y_origin == self.tile_y_origin[0][0])[0][0]
            x_corner_coord = np.argwhere(tile_x_origin == self.tile_x_origin[0][0])[0][1]
            self.existing_tile_mask[y_corner_coord:y_corner_coord + existing_tile_shape[0], x_corner_coord:x_corner_coord + existing_tile_shape[1]] = True

        self.tile_x_origin, self.tile_y_origin = tile_x_origin, tile_y_origin

        self.tile_center_x = self.tile_x_origin + self.tile_size / 2
        self.tile_center_y = self.tile_y_origin + self.tile_size / 2

        self.tile_edges_x = np.append(tx_origins, tx_origins[-1] + self.tile_size)
        self.tile_edges_y = np.append(ty_origins, ty_origins[-1] + self.tile_size)

        self.width = nearest_higher_x - nearest_lower_x
        self.height = nearest_higher_y, nearest_lower_y
        self.maximum_tiles = self.tile_x_origin.size

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

