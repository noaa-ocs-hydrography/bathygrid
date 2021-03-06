import os
from bathygrid.bgrid import BathyGrid
from bathygrid.maingrid import SRGrid, VRGridTile, SRGridZarr, VRGridTileZarr
from bathygrid.utilities import is_power_of_two, create_folder
from bathygrid.grid_variables import allowable_grid_root_names


def _validate_load_path(folder_path: str):
    """
    Ensure the folder_path passed to load_grid is a valid path.  Checks for the root folder and some other things.

    Parameters
    ----------
    folder_path
        container folder for the grid

    Returns
    -------
    str
        root folder path, the root folder is the folder within the container folder that holds the grid data
    str
        root folder name
    """

    if not os.path.exists(folder_path):
        raise IOError('Unable to find folder {}'.format(folder_path))
    subfolders = os.listdir(folder_path)
    if len(subfolders) == 0:
        raise IOError('Found no root folders in {}, expected a root folder like "VRGridTile_Root"'.format(folder_path))
    valid_subfolders = [fldr for fldr in subfolders if fldr in allowable_grid_root_names]
    if len(valid_subfolders) > 1:
        raise IOError('Found multiple subfolders in {}, expected one root folder like "VRGridTile_Root"'.format(folder_path))
    elif len(valid_subfolders) == 0:
        raise IOError('Found no root folders in {}, expected a root folder like "VRGridTile_Root"'.format(folder_path))
    return os.path.join(folder_path, valid_subfolders[0]), valid_subfolders[0]


def _validate_create_options(folder_path: str, grid_type: str, grid_backend: str, tile_size: float, subtile_size: float):
    if folder_path:
        fpath, fname = os.path.split(folder_path)
        folderpath = create_folder(fpath, fname)
    else:
        folderpath = ''

    if grid_type not in ['single_resolution', 'variable_resolution_tile']:
        raise ValueError("Grid type {} invalid, must be one of ['single_resolution', 'variable_resolution_tile']".format(grid_type))
    if grid_backend not in ['numpy', 'zarr']:
        raise ValueError("Grid backend {} invalid, must be one of ['numpy', 'zarr']".format(grid_type))
    if not is_power_of_two(tile_size):
        raise ValueError('Tile size {} must be a power of two'.format(tile_size))
    if grid_type == 'variable_resolution_tile' and not is_power_of_two(subtile_size):
        raise ValueError('Sub tile size {} must be a power of two'.format(subtile_size))
    return folderpath


def is_bathygrid(folder_path: str):
    """
    Quick method to determine if the provided folder is a BathyGrid parent folder.  We simply search the subfolders
    for a Bathygrid root folder.

    Returns
    -------
    bool
        True if the given folder is a BathyGrid folder
    """

    if os.path.isdir(folder_path):
        subfolders = os.listdir(folder_path)
        for fldr in subfolders:
            if fldr in allowable_grid_root_names:
                return True
    return False


def load_grid(folder_path: str):
    """
    Load a saved BathyGrid instance from file.  Folder_path is the container that should contain a root folder within,
    ex: 'SRGrid_Root'.  Loading is done lazily, only loading the metadata for each object and a reference to the point
    and grid data, so this should be pretty quick.

    Parameters
    ----------
    folder_path
        container folder for the grid

    Returns
    -------
    BathyGrid
        one of the BathyGrid implementations, ex: SRGrid
    """

    root_folder, root_name = _validate_load_path(folder_path)
    if root_name == 'SRGrid_Root':
        grid_class = SRGrid()
    elif root_name == 'VRGridTile_Root':
        grid_class = VRGridTile()
    elif root_name == 'SRGridZarr_Root':
        grid_class = SRGridZarr()
    elif root_name == 'VRGridTileZarr_Root':
        grid_class = VRGridTileZarr()
    else:
        raise NotImplementedError('{} is not a valid grid type'.format(root_name))
    grid_class.load(folder_path)
    return grid_class


def create_grid(folder_path: str = '', grid_type: str = 'single_resolution', grid_backend: str = 'zarr',
                tile_size: float = 1024.0, subtile_size: float = 128, is_backscatter: bool = False):
    """
    Create and return a new bathygrid instance.  Use add points and grid methods to use the instance.

    Parameters
    ----------
    folder_path
        container folder for the grid, if you want it to immediately flush to disk
    grid_type
        one of 'single_resolution', 'variable_resolution_tile'
    grid_backend
        one of 'zarr', 'numpy', default is zarr to leverage compression benefits
    tile_size
        main tile size, the size in meters of the tiles within the grid, a larger tile size will improve performance,
        but size should be at most 1/2 the length/width of the survey area
    subtile_size
        sub tile size, only used for variable resolution, the size of the subtiles within the tiles, subtiles are the
        smallest unit within the grid that is single resolution
    is_backscatter
        if True, this is a backscatter surface, depth layer will become "intensity" and BAG creation is disabled

    Returns
    -------
    BathyGrid
        one of the BathyGrid implementations, ex: SRGrid
    """

    folderpath = _validate_create_options(folder_path, grid_type, grid_backend, tile_size, subtile_size)
    if grid_type == 'single_resolution':
        if grid_backend == 'numpy':
            grid_class = SRGrid(output_folder=folderpath, tile_size=tile_size, is_backscatter=is_backscatter)
        else:
            grid_class = SRGridZarr(output_folder=folderpath, tile_size=tile_size, is_backscatter=is_backscatter)
    elif grid_type == 'variable_resolution_tile':
        if grid_backend == 'numpy':
            grid_class = VRGridTile(output_folder=folderpath, tile_size=tile_size, subtile_size=subtile_size, is_backscatter=is_backscatter)
        else:
            grid_class = VRGridTileZarr(output_folder=folderpath, tile_size=tile_size, subtile_size=subtile_size, is_backscatter=is_backscatter)
    else:
        raise NotImplementedError('{} is not a valid grid type'.format(grid_type))
    return grid_class
