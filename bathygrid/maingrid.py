import os
import numpy as np
from dask.array import Array

try:
    from tqdm import tqdm
    tqdm_enabled = True
except:
    tqdm_enabled = False

from bathygrid.backends import NumpyGrid
from bathygrid.utilities import create_folder


class SRGrid(NumpyGrid):
    """
    SRGrid is the basic implementation of the BathyGrid.  This class contains the metadata and other functions required
    to build and maintain the BathyGrid
    """
    def __init__(self, min_x: float = 0, min_y: float = 0, tile_size: float = 1024.0):
        super().__init__(min_x=min_x, min_y=min_y, tile_size=tile_size)
        self.can_grow = True
        self.name = 'SRGrid_Root'
        self.sub_type = 'tile'

    def _convert_dataset(self):
        """
        We currently convert xarray Dataset input into a numpy structured array.  Xarry Datasets appear to be rather
        slow in testing, I believe because they do some stuff under the hood with matching coordinates when you do
        basic operations.  Also, you can't do any fancy indexing with xarray Dataset, at best you can use slice with isel.

        For all these reasons, we just convert to numpy.
        """
        allowed_vars = ['x', 'y', 'z', 'tvu', 'thu']
        dtyp = [(varname, self.data[varname].dtype) for varname in allowed_vars if varname in self.data]
        empty_struct = np.empty(len(self.data['x']), dtype=dtyp)
        for varname, vartype in dtyp:
            empty_struct[varname] = self.data[varname].values
        self.data = empty_struct

    def _update_metadata(self, container_name: str = None, file_list: list = None, epsg: int = None,
                         vertical_reference: str = None):
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
        """

        if file_list:
            self.container[container_name] = file_list
        else:
            self.container[container_name] = ['Unknown']

        if self.epsg and (self.epsg != int(epsg)):
            raise ValueError('BathyGrid: Found existing coordinate system {}, new coordinate system {} must match'.format(self.epsg,
                                                                                                                          epsg))
        if self.vertical_reference and (self.vertical_reference != vertical_reference):
            raise ValueError('BathyGrid: Found existing vertical reference {}, new vertical reference {} must match'.format(self.vertical_reference,
                                                                                                                            vertical_reference))
        self.epsg = int(epsg)
        self.vertical_reference = vertical_reference

    def _validate_input_data(self):
        """
        Ensure you get a structured numpy array as the input dataset.  If dataset is an Xarray Dataset, we convert it to
        Numpy for performance reasons.
        """

        if type(self.data) in [np.ndarray, Array]:
            if not self.data.dtype.names:
                raise ValueError('BathyGrid: numpy array provided for data, but no names were found, array must be a structured array')
            if 'x' not in self.data.dtype.names or 'y' not in self.data.dtype.names:
                raise ValueError('BathyGrid: numpy structured array provided for data, but "x" or "y" not found in variable names')
            self.layernames = [self.rev_layer_lookup[var] for var in self.data.dtype.names if var in ['z', 'tvu']]
        elif type(self.data) == xr.Dataset:
            if 'x' not in self.data:
                raise ValueError('BathyGrid: xarray Dataset provided for data, but "x" or "y" not found in variable names')
            if len(self.data.dims) > 1:
                raise ValueError('BathyGrid: xarray Dataset provided for data, but found multiple dimensions, must be one dimensional: {}'.format(self.data.dims))
            self.layernames = [self.rev_layer_lookup[var] for var in self.data if var in ['z', 'tvu']]
            self._convert_dataset()  # internally we just convert xarray dataset to numpy for ease of use
        else:
            raise ValueError('QuadTree: numpy structured array or dask array with "x" and "y" as variable must be provided')

    def save(self, folder_path: str = None, progress_bar: bool = True):
        """
        Recursive save for all BathyGrid/Tile objects within this class.

        Parameters
        ----------
        folder_path
            container folder for the grid
        progress_bar
            if True, displays a console progress bar
        """

        if not folder_path:
            if self.output_folder:
                super().save(self.output_folder, progress_bar=progress_bar)
            else:
                raise ValueError('Grid has not been saved before, you must provide a folder path to save.')
        else:
            fpath, fname = os.path.split(folder_path)
            folderpath = create_folder(fpath, fname)
            self.output_folder = folderpath
            super().save(folderpath, progress_bar=progress_bar)


class VRGridTile(SRGrid):
    """
    VRGridTile is a simple approach to variable resolution gridding.  We build a grid of BathyGrids, where each BathyGrid
    has a certain number of tiles (each tile with size subtile_size).  Each of those tiles can have a different resolution
    depending on depth.
    """

    def __init__(self, min_x: float = 0, min_y: float = 0, tile_size: float = 1024, subtile_size: float = 128):
        super().__init__(min_x=min_x, min_y=min_y, tile_size=tile_size)
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
        return NumpyGrid(min_x=tile_x_origin, min_y=tile_y_origin, max_x=tile_x_origin + self.tile_size,
                         max_y=tile_y_origin + self.tile_size, tile_size=self.subtile_size,
                         set_extents_manually=True)

