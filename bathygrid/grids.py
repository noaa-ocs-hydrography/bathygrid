import numpy as np
from bathygrid.utilities import is_power_of_two


class Grid:
    """
    Base grid for the Tile and the BathyGrid.  Contains the coordinate attributes
    """
    def __init__(self):
        self.min_y = None
        self.min_x = None
        self.max_y = None
        self.max_x = None

        self.width = None
        self.height = None


class BaseGrid(Grid):
    """
    Base class for all grid objects.  Contains all the attributes that allow for the generation of a grid of Tiles or
    other Grids within this Grid.
    """
    def __init__(self, min_x: float = 0, min_y: float = 0, tile_size: float = 1024.0):
        super().__init__()
        self.origin_x = min_x
        self.origin_y = min_y

        self.data = None
        self.container = {}  # dict of container name, list of multibeam files
        self.tiles = None

        self.tile_x_origin = None
        self.tile_y_origin = None

        self.tile_edges_x = None
        self.tile_edges_y = None

        self.existing_tile_mask = None

        self.maximum_tiles = None
        self.number_of_tiles = None

        self.can_grow = False
        self.tile_size = tile_size
        if not is_power_of_two(self.tile_size):
            raise ValueError('Grid size must be a power of two.')

    @property
    def is_empty(self):
        """
        Return True if there are any Tiles within this grid
        """
        if self.tiles is None:
            return True
        elif not self.tiles.any():
            return True
        return False

    def _adjust_extents(self):
        """
        If this BathyGrid is allowed to grow, we extend the boundaries by integer increments of tile_size
        """
        if self.can_grow:
            nearest_lower_x = np.floor((self.min_x - self.origin_x) / self.tile_size) * self.tile_size
            nearest_higher_x = np.ceil((self.max_x - self.origin_x) / self.tile_size) * self.tile_size
            nearest_lower_y = np.floor((self.min_y - self.origin_y) / self.tile_size) * self.tile_size
            nearest_higher_y = np.ceil((self.max_y - self.origin_y) / self.tile_size) * self.tile_size
            if nearest_higher_x == self.max_x:  # higher value cant be equal, our binning relies on it being less than
                nearest_higher_x += self.tile_size
            if nearest_higher_y == self.max_y:
                nearest_higher_y += self.tile_size
            self.min_x, self.min_y = nearest_lower_x, nearest_lower_y
            self.max_x, self.max_y = nearest_higher_x, nearest_higher_y

    def _build_grid(self):
        """
        Build a new grid after adjusting or initializing the extents.  Provides the edges and origin coordinates for the
        grid.

        If a grid exists (not is_empty), we pick up all the existing tiles, build a new grid with the new extents, and
        then put down the tiles in the right place in the new grid.  Allows the Grid to grow or contract as points are
        added and removed.
        """

        if self.min_y is None or self.min_x is None or self.max_y is None or self.max_x is None:
            raise ValueError('UtmGrid not initialized!')
        self._adjust_extents()

        tx_origins = np.arange(self.min_x, self.max_x, self.tile_size)
        ty_origins = np.arange(self.min_y, self.max_y, self.tile_size)
        tile_x_origin, tile_y_origin = np.meshgrid(tx_origins, ty_origins)

        if not self.is_empty and self.can_grow:
            # we have existing tiles, so find where the existing tiles are in the new grid.  Only applies if the grid can grow
            self.existing_tile_mask = np.full(tile_x_origin.shape, False)
            existing_tile_shape = self.tile_x_origin.shape
            y_corner_coord = np.argwhere(tile_y_origin == self.tile_y_origin[0][0])[0][0]
            x_corner_coord = np.argwhere(tile_x_origin == self.tile_x_origin[0][0])[0][1]
            self.existing_tile_mask[y_corner_coord:y_corner_coord + existing_tile_shape[0], x_corner_coord:x_corner_coord + existing_tile_shape[1]] = True

        self.tile_x_origin, self.tile_y_origin = tile_x_origin, tile_y_origin

        self.tile_edges_x = np.append(tx_origins, tx_origins[-1] + self.tile_size)
        self.tile_edges_y = np.append(ty_origins, ty_origins[-1] + self.tile_size)

        self.width = self.max_x - self.min_x
        self.height = self.max_y - self.min_y
        self.maximum_tiles = self.tile_x_origin.size

    def _init_from_extents(self, min_y: float, min_x: float, max_y: float, max_x: float):
        """
        Build a new grid by first populating the extents of the grid.

        Parameters
        ----------
        min_y
            minimum y value (latitude, northing) for the grid extents
        min_x
            minimum x value (longitude, easting) for the grid extents
        max_y
            maximum y value (latitude, northing) for the grid extents
        max_x
            maximum x value (longitude, easting) for the grid extents
        """

        self.min_y = min_y
        self.min_x = min_x
        self.max_y = max_y
        self.max_x = max_x
        self._build_grid()

    def _update_extents(self, min_y: float, min_x: float, max_y: float, max_x: float):
        """
        Update the grid with new extents, triggering a new grid build carrying over existing tiles

        Parameters
        ----------
        min_y
            minimum y value (latitude, northing) for the grid extents
        min_x
            minimum x value (longitude, easting) for the grid extents
        max_y
            maximum y value (latitude, northing) for the grid extents
        max_x
            maximum x value (longitude, easting) for the grid extents
        """

        self.min_y = min(min_y, self.min_y)
        self.min_x = min(min_x, self.min_x)
        self.max_y = max(max_y, self.max_y)
        self.max_x = max(max_x, self.max_x)
        self._build_grid()

    def _tile_idx_to_row_col(self, tile_index: int):
        """
        When operating on tiles, we generally flatten the numpy array of tiles and iterate through.  Here we take that
        flattened index and convert it to the row, column of the

        Parameters
        ----------
        tile_index

        Returns
        -------
        int
            row index
        int
            column index
        """

        rows, cols = self.tile_x_origin.shape
        return divmod(tile_index, cols)


class TileGrid(Grid):
    """
    Base grid for all Tiles.  Tiles have cells instead of other Tiles.  Has some routines for checking cell, point count.
    """
    def __init__(self, min_x: float, min_y: float, size: float):
        super().__init__()
        self.data = None

        self.container = {}  # dict of container name, list of multibeam files

        self.cells = {}
        self.cell_edges_x = {}
        self.cell_edges_y = {}
        self.cell_indices = {}

        self._init_from_size(min_x, min_y, size)

    @property
    def is_empty(self):
        """
        Check if the Tile contains any points
        """

        if not self.points_count:
            return True
        return False

    @property
    def points_count(self):
        """
        Return the number of points in this Tile
        """

        if self.data is not None:
            return int(self.data.size)
        else:
            return 0

    def _init_from_size(self, min_x: float, min_y: float, size: float):
        """
        Build the extents and width/height from the provided values

        Parameters
        ----------
        min_x
            minimum x value (longitude, easting) for the grid extents
        min_y
            minimum y value (latitude, northing) for the grid extents
        size
            size of the tile in x/y units, assumes tiles are square
        """

        self.min_x = min_x
        self.max_x = min_x + size
        self.min_y = min_y
        self.max_y = min_y + size

        self.width = size
        self.height = size
        self.name = f'{min_x}_{min_y}'
