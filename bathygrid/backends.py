import os
import numpy as np
import json
from shutil import rmtree
import dask.array as da

from bathygrid.bgrid import BathyGrid
from bathygrid.tile import Tile, SRTile
from bathygrid.utilities import print_progress_bar


bathygrid_desired_keys = ['min_y', 'min_x', 'max_y', 'max_x', 'width', 'height', 'origin_x', 'origin_y', 'container',
                          'tile_x_origin', 'tile_y_origin', 'tile_edges_x', 'tile_edges_y', 'existing_tile_mask',
                          'maximum_tiles', 'number_of_tiles', 'can_grow', 'tile_size', 'mean_depth', 'epsg',
                          'vertical_reference', 'resolutions', 'name', 'output_folder', 'sub_type', 'subtile_size',
                          'storage_type']
bathygrid_numpy_to_list = ['tile_x_origin', 'tile_y_origin', 'tile_edges_x', 'tile_edges_y', 'resolutions']
bathygrid_float_to_str = ['min_y', 'min_x', 'max_y', 'max_x', 'width', 'height', 'origin_x', 'origin_y', 'mean_depth']

tile_desired_keys = ['min_y', 'min_x', 'max_y', 'max_x', 'width', 'height', 'container', 'name', 'algorithm']
tile_float_to_str = ['min_y', 'min_x', 'max_y', 'max_x']


class BaseStorage(BathyGrid):
    """
    Base class for handling saving/loading from disk.  Uses json for saving metadata to file.  Inherit this class
    to create a storage backend for the point/grid data.
    """
    def __init__(self, min_x: float = 0, min_y: float = 0, max_x: float = 0, max_y: float = 0,
                 tile_size: float = 1024.0, set_extents_manually: bool = False):
        super().__init__(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y, tile_size=tile_size,
                         set_extents_manually=set_extents_manually)

    def _save_bathygrid_metadata(self, folderpath: str):
        """
        Save the metadata assoicated with the bathygrid object to json

        Parameters
        ----------
        folderpath
            path to folder that will contain the new json file
        """

        if not os.path.exists(folderpath):
            raise EnvironmentError('Unable to save json file to {}, does not exist'.format(folderpath))
        fileout = os.path.join(folderpath, 'metadata.json')
        data = {ky: self.__getattribute__(ky) for ky in bathygrid_desired_keys}
        for ky in bathygrid_numpy_to_list:
            if isinstance(data[ky], np.ndarray):
                data[ky] = data[ky].tolist()
        for ky in bathygrid_float_to_str:
            data[ky] = str(data[ky])
        with open(fileout, 'w') as fout:
            json.dump(data, fout, indent=4)

    def _load_bathygrid_metadata(self, folderpath: str):
        """
        Load the metadata assoicated with the bathygrid object from json

        Parameters
        ----------
        folderpath
            path to folder that contains the json file
        """

        fileout = os.path.join(folderpath, 'metadata.json')
        if not os.path.exists(fileout):
            raise EnvironmentError('Unable to load json file from {}, does not exist'.format(folderpath))
        with open(fileout, 'r') as fout:
            data = json.load(fout)
        for ky in bathygrid_numpy_to_list:
            data[ky] = np.array(data[ky])
        for ky in bathygrid_float_to_str:
            data[ky] = float(data[ky])
        for ky in data:
            self.__setattr__(ky, data[ky])
        self.tiles = np.full(self.tile_x_origin.shape, None, dtype=object)

    def _save_tile_metadata(self, tile: Tile, folderpath: str):
        """
        Save the metadata assoicated with the Tile object to json

        Parameters
        ----------
        tile
            tile object that needs it's metdata saved
        folderpath
            path to folder that will contain the new json file
        """

        if not os.path.exists(folderpath):
            raise EnvironmentError('Unable to save json file to {}, does not exist'.format(folderpath))
        fileout = os.path.join(folderpath, 'metadata.json')
        data = {ky: tile.__getattribute__(ky) for ky in tile_desired_keys}
        for ky in tile_float_to_str:
            data[ky] = str(data[ky])
        with open(fileout, 'w') as fout:
            json.dump(data, fout, indent=4)

    def _load_tile_metadata(self, tile: Tile, folderpath: str):
        """
        Load the metadata assoicated with the Tile object from json

        Parameters
        ----------
        tile
            tile object that needs it's metdata loaded
        folderpath
            path to folder that contains the json file
        """

        fileout = os.path.join(folderpath, 'metadata.json')
        if not os.path.exists(fileout):
            raise EnvironmentError('Unable to load json file from {}, does not exist'.format(folderpath))
        with open(fileout, 'r') as fout:
            data = json.load(fout)
        for ky in tile_float_to_str:
            data[ky] = float(data[ky])
        for ky in data:
            tile.__setattr__(ky, data[ky])

    def _save_tile_data(self, tile: Tile, folderpath: str):
        raise NotImplementedError('_save_tile_data must be implemented in inheriting class')

    def _load_tile_data(self, tile: Tile, folderpath: str):
        raise NotImplementedError('_load_tile_data must be implemented in inheriting class')

    def _get_tile_folder(self, root_folder: str, flat_index: int):
        """
        Build the expected folder path to the tile given its flattened index

        Parameters
        ----------
        root_folder
            folder that contains the tile folder
        flat_index
            1d index of the flattened tiles

        Returns
        -------
        str
            full folder path to the tile folder
        int
            row index of the tile
        int
            column index of the tile
        """

        row, col = self._tile_idx_to_row_col(flat_index)
        tile_name = '{}_{}'.format(row, col)
        tile_folder = os.path.join(root_folder, tile_name)
        return tile_folder, row, col

    def _clear_tile(self, tile_folder: str):
        """
        Delete the given tile_folder and any data it has in it
        """

        if os.path.exists(tile_folder):
            rmtree(tile_folder)

    def save(self, folderpath: str, progress_bar: bool = True):
        """
        Save to a new root folder within the provided folderpath

        Parameters
        ----------
        folderpath
            base folder for the saved bathygrid instance
        progress_bar
            if True, will display a progress bar
        """

        root_folder = os.path.join(folderpath, self.name)
        os.makedirs(root_folder, exist_ok=True)
        self._save_bathygrid_metadata(root_folder)
        if progress_bar:
            print_progress_bar(0, self.tiles.size)
        if self.sub_type == 'tile':
            for cnt, tile in enumerate(self.tiles.flat):
                if progress_bar:
                    print_progress_bar(cnt + 1, self.tiles.size)
                tile_folder, row, col = self._get_tile_folder(root_folder, cnt)
                if tile is None:
                    self._clear_tile(tile_folder)
                else:
                    os.makedirs(tile_folder, exist_ok=True)
                    self._save_tile_metadata(tile, tile_folder)
                    self._save_tile_data(tile, tile_folder)
        else:
            for cnt, subgrid in enumerate(self.tiles.flat):
                if progress_bar:
                    print_progress_bar(cnt + 1, self.tiles.size)
                subgrid_folder, row, col = self._get_tile_folder(root_folder, cnt)
                if subgrid is None:
                    self._clear_tile(subgrid_folder)
                else:
                    subgrid.save(root_folder, progress_bar=False)

    def load(self, folderpath: str):
        """
        Load from a saved bathygrid instance

        Parameters
        ----------
        folderpath
            base folder for the saved bathygrid instance
        """

        root_folder = os.path.join(folderpath, self.name)
        if not os.path.exists(root_folder):
            raise ValueError('Unable to find grid root folder {}'.format(root_folder))
        self._load_bathygrid_metadata(root_folder)
        if self.sub_type == 'tile':
            for idx in range(self.tiles.size):
                tile_folder, row, col = self._get_tile_folder(root_folder, idx)
                if os.path.exists(tile_folder):
                    newtile = SRTile()
                    self._load_tile_metadata(newtile, tile_folder)
                    self._load_tile_data(newtile, tile_folder)
                    self.tiles[row][col] = newtile
                else:
                    self.tiles[row][col] = None
        else:
            if self.storage_type == 'numpy':
                storage_cls = NumpyGrid
            else:
                raise NotImplementedError('{} is not a valid storage type'.format(self.storage_type))
            for idx in range(self.tiles.size):
                subgrid_folder, row, col = self._get_tile_folder(root_folder, idx)
                if os.path.exists(subgrid_folder):
                    newgrid = storage_cls()
                    newgrid.name = os.path.split(subgrid_folder)[1]
                    newgrid.load(root_folder)
                    self.tiles[row][col] = newgrid
                else:
                    self.tiles[row][col] = None


class NumpyGrid(BaseStorage):
    """
    Backend for saving the point data / gridded data to numpy chunked files
    """
    def __init__(self, min_x: float = 0, min_y: float = 0, max_x: float = 0, max_y: float = 0,
                 tile_size: float = 1024.0, set_extents_manually: bool = False):
        super().__init__(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y, tile_size=tile_size,
                         set_extents_manually=set_extents_manually)
        self.storage_type = 'numpy'

    def _save_tile_data(self, tile: Tile, folderpath: str):
        da.to_npy_stack(folderpath + '/data', da.from_array(tile.data))
        for resolution in tile.cells.keys():
            da.to_npy_stack(folderpath + '/cells_{}_depth'.format(resolution), da.from_array(tile.cells[resolution]['depth']))
            da.to_npy_stack(folderpath + '/cells_{}_vertical_uncertainty'.format(resolution), da.from_array(tile.cells[resolution]['vertical_uncertainty']))
            da.to_npy_stack(folderpath + '/cells_{}_horizontal_uncertainty'.format(resolution), da.from_array(tile.cells[resolution]['horizontal_uncertainty']))
            da.to_npy_stack(folderpath + '/cell_edges_x_{}'.format(resolution), da.from_array(tile.cell_edges_x[resolution]))
            da.to_npy_stack(folderpath + '/cell_edges_y_{}'.format(resolution), da.from_array(tile.cell_edges_y[resolution]))

    def _load_tile_data(self, tile: Tile, folderpath: str):
        resolutions = []
        data_folders = os.listdir(folderpath)
        for df in data_folders:
            sections = df.split('_')
            if sections[0] == 'cells':
                resolutions.append(float(sections[1]))
        resolutions.sort()
        tile.data = da.from_npy_stack(folderpath + '/data')
        for resolution in resolutions:
            tile.cells[resolution] = {}
            tile.cells[resolution]['depth'] = da.from_npy_stack(folderpath + '/cells_{}_depth'.format(resolution))
            tile.cells[resolution]['vertical_uncertainty'] = da.from_npy_stack(folderpath + '/cells_{}_vertical_uncertainty'.format(resolution))
            tile.cells[resolution]['horizontal_uncertainty'] = da.from_npy_stack(folderpath + '/cells_{}_horizontal_uncertainty'.format(resolution))
            tile.cell_edges_x[resolution] = da.from_npy_stack(folderpath + '/cell_edges_x_{}'.format(resolution))
            tile.cell_edges_y[resolution] = da.from_npy_stack(folderpath + '/cell_edges_y_{}'.format(resolution))