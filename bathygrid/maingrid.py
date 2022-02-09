import os

from bathygrid.backends import NumpyGrid, ZarrGrid


class SRGrid(NumpyGrid):
    """
    SRGrid is the basic implementation of the BathyGrid.  This class contains the metadata and other functions required
    to build and maintain the BathyGrid
    """
    def __init__(self, min_x: float = 0, min_y: float = 0, tile_size: float = 1024.0, output_folder: str = '', is_backscatter: bool = False):
        super().__init__(min_x=min_x, min_y=min_y, tile_size=tile_size, output_folder=output_folder, is_backscatter=is_backscatter)
        self.can_grow = True
        self.name = 'SRGrid_Root'
        self.sub_type = 'srtile'
        self.output_folder = output_folder
        if self.output_folder:
            os.makedirs(output_folder, exist_ok=True)
        self.storage_class = NumpyGrid


class VRGridTile(SRGrid):
    """
    VRGridTile is a simple approach to variable resolution gridding.  We build a grid of BathyGrids, where each BathyGrid
    has a certain number of tiles (each tile with size subtile_size).  Each of those tiles can have a different resolution
    depending on depth.
    """

    def __init__(self, min_x: float = 0, min_y: float = 0, tile_size: float = 1024, output_folder: str = '',
                 subtile_size: float = 128, is_backscatter: bool = False):
        super().__init__(min_x=min_x, min_y=min_y, tile_size=tile_size, output_folder=output_folder, is_backscatter=is_backscatter)
        self.can_grow = True
        self.subtile_size = subtile_size
        self.name = 'VRGridTile_Root'
        self.sub_type = 'grid'

    def _build_tile(self, tile_x_origin: float, tile_y_origin: float):
        """
        For the VRGridTile class, the 'Tiles' are in fact BathyGrids, which contain their own tiles.  subtile_size controls
        the size of the Tiles within this BathyGrid.

        Parameters
        ----------
        tile_x_origin
            x origin coordinate for the tile, in the same units as the BathyGrid
        tile_y_origin
            y origin coordinate for the tile, in the same units as the BathyGrid

        Returns
        -------
        BathyGrid
            empty BathyGrid for this origin / tile size
        """

        if self.output_folder:  # necessary if a destination is set so child grids flush to disk
            ofolder = os.path.join(self.output_folder, self.name)
        else:  # grid is in memory only
            ofolder = ''
        return self.storage_class(min_x=tile_x_origin, min_y=tile_y_origin, max_x=tile_x_origin + self.tile_size,
                                  max_y=tile_y_origin + self.tile_size, tile_size=self.subtile_size,
                                  set_extents_manually=True, output_folder=ofolder)


class SRGridZarr(ZarrGrid):
    """
    SRGridZarr is the implementation of the BathyGrid with the Zarr backend.  This class contains the metadata and other
    functions required to build and maintain the BathyGrid.  Zarr should give improvements in compression to saved data.
    """
    def __init__(self, min_x: float = 0, min_y: float = 0, tile_size: float = 1024.0, output_folder: str = '', is_backscatter: bool = False):
        super().__init__(min_x=min_x, min_y=min_y, tile_size=tile_size, output_folder=output_folder, is_backscatter=is_backscatter)
        self.can_grow = True
        self.name = 'SRGridZarr_Root'
        self.sub_type = 'srtile'
        self.output_folder = output_folder
        if self.output_folder:
            os.makedirs(output_folder, exist_ok=True)
        self.storage_class = ZarrGrid


class VRGridTileZarr(SRGridZarr):
    """
    VRGridTileZarr is a simple approach to variable resolution gridding.  We build a grid of BathyGrids, where each BathyGrid
    has a certain number of tiles (each tile with size subtile_size).  Each of those tiles can have a different resolution
    depending on depth.  Uses the Zarr backend for compression advantages.
    """

    def __init__(self, min_x: float = 0, min_y: float = 0, tile_size: float = 1024, output_folder: str = '',
                 subtile_size: float = 128, is_backscatter: bool = False):
        super().__init__(min_x=min_x, min_y=min_y, tile_size=tile_size, output_folder=output_folder, is_backscatter=is_backscatter)
        self.can_grow = True
        self.subtile_size = subtile_size
        self.name = 'VRGridTileZarr_Root'
        self.sub_type = 'grid'

    def _build_tile(self, tile_x_origin: float, tile_y_origin: float):
        """
        For the VRGridTile class, the 'Tiles' are in fact BathyGrids, which contain their own tiles.  subtile_size controls
        the size of the Tiles within this BathyGrid.

        Parameters
        ----------
        tile_x_origin
            x origin coordinate for the tile, in the same units as the BathyGrid
        tile_y_origin
            y origin coordinate for the tile, in the same units as the BathyGrid

        Returns
        -------
        BathyGrid
            empty BathyGrid for this origin / tile size
        """

        if self.output_folder:  # necessary if a destination is set so child grids flush to disk
            ofolder = os.path.join(self.output_folder, self.name)
        else:  # grid is in memory only
            ofolder = ''
        return self.storage_class(min_x=tile_x_origin, min_y=tile_y_origin, max_x=tile_x_origin + self.tile_size,
                                  max_y=tile_y_origin + self.tile_size, tile_size=self.subtile_size,
                                  set_extents_manually=True, output_folder=ofolder)
