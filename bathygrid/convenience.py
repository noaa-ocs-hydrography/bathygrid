import os
from bathygrid.maingrid import SRGrid, VRGridTile


def _validate_load_path(folder_path: str):
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
    root_folder, root_name = _validate_load_path(folder_path)
    if root_name == 'SRGrid_Root':
        storage_class = SRGrid()
    elif root_name == 'VRGridTile_Root':
        storage_class = VRGridTile()
    else:
        raise NotImplementedError('{} is not a valid grid type'.format(root_name))
    storage_class.name = root_name
    storage_class.load(folder_path)
    storage_class.output_folder = folder_path
    return storage_class
