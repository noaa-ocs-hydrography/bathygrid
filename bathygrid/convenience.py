import os
from bathygrid.bgrid import BathyGrid
from bathygrid.maingrid import SRGrid, VRGridTile


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
    grid_class.name = root_name
    grid_class.load(folder_path)
    grid_class.output_folder = folder_path
    return grid_class
