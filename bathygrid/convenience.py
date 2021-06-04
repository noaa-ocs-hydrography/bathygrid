import os
from bathygrid.bgrid import BathyGrid
from bathygrid.maingrid import SRGrid, VRGridTile
from bathygrid.utilities import is_power_of_two


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
    if len(subfolders) > 1:
        raise IOError('Found multiple folders in {}, expected one root folder.  Found {}'.format(folder_path, subfolders))
    if len(subfolders) == 0:
        raise IOError('Found no root folders in {}, expected a root folder like "VRGridTile_Root"'.format(folder_path))
    if subfolders[0] not in ['VRGridTile_Root', 'SRGrid_Root']:
        raise ValueError('Root folder {} is not one of the valid root folders ["VRGridTile_Root", "SRGrid_Root"]'.format(subfolders[0]))
    return os.path.join(folder_path, subfolders[0]), subfolders[0]


def _validate_create_options(folder_path: str, grid_type: str, tile_size: float, subtile_size: float):
    if folder_path and os.path.exists(folder_path):
        if os.listdir(folder_path):
            raise ValueError('Folder path {} exists and contains folders/files.  Please provide a folder path that does not exist or is empty'.format(folder_path))
    if grid_type not in ['single_resolution', 'variable_resolution_tile']:
        raise ValueError("Grid type {} invalid, must be one of ['single_resolution', 'variable_resolution_tile']".format(grid_type))
    if not is_power_of_two(tile_size):
        raise ValueError('Tile size {} must be a power of two'.format(tile_size))
    if grid_type == 'variable_resolution_tile' and not is_power_of_two(subtile_size):
        raise ValueError('Sub tile size {} must be a power of two'.format(subtile_size))


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
    else:
        raise NotImplementedError('{} is not a valid grid type'.format(root_name))
    grid_class.load(folder_path)
    return grid_class


def create_grid(folder_path: str = '', grid_type: str = 'single_resolution', tile_size: float = 1024.0, subtile_size: float = 128):
    """
    Create a new bathygrid instance

    Parameters
    ----------
    folder_path
        container folder for the grid, if you want it to immediately flush to disk
    grid_type
        one of 'single_resolution', 'variable_resolution_tile'
    tile_size
        main tile size, the size in meters of the tiles within the grid, a larger tile size will improve performance,
        but size should be at most 1/2 the length/width of the survey area
    subtile_size
        sub tile size, only used for variable resolution, the size of the subtiles within the tiles, subtiles are the
        smallest unit within the grid that is single resolution

    Returns
    -------
    BathyGrid
        one of the BathyGrid implementations, ex: SRGrid
    """

    _validate_create_options(folder_path, grid_type, tile_size, subtile_size)
    if grid_type == 'single_resolution':
        grid_class = SRGrid(output_folder=folder_path, tile_size=tile_size)
    elif grid_type == 'variable_resolution_tile':
        grid_class = VRGridTile(output_folder=folder_path, tile_size=tile_size, subtile_size=subtile_size)
    else:
        raise NotImplementedError('{} is not a valid grid type'.format(grid_type))
    return grid_class
