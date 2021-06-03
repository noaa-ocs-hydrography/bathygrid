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
bathygrid_numpy_to_list = ['tile_x_origin', 'tile_y_origin', 'tile_edges_x', 'tile_edges_y', 'resolutions', 'existing_tile_mask']
bathygrid_float_to_str = ['min_y', 'min_x', 'max_y', 'max_x', 'width', 'height', 'origin_x', 'origin_y', 'mean_depth']

tile_desired_keys = ['min_y', 'min_x', 'max_y', 'max_x', 'width', 'height', 'container', 'name', 'algorithm']
tile_float_to_str = ['min_y', 'min_x', 'max_y', 'max_x']


class BaseStorage(BathyGrid):
    """
    Base class for handling saving/loading from disk.  Uses json for saving metadata to file.  Inherit this class
    to create a storage backend for the point/grid data.
    """
    def __init__(self, min_x: float = 0, min_y: float = 0, max_x: float = 0, max_y: float = 0,
                 tile_size: float = 1024.0, set_extents_manually: bool = False, output_folder: str = ''):
        super().__init__(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y, tile_size=tile_size,
                         set_extents_manually=set_extents_manually, output_folder=output_folder)

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

        os.makedirs(folderpath, exist_ok=True)
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

    def _save_tile_data(self, tile: Tile, folderpath: str, only_points: bool = False, only_grid: bool = False):
        raise NotImplementedError('_save_tile_data must be implemented in inheriting class')

    def _load_tile_data(self, tile: Tile, folderpath: str, only_points: bool = False, only_grid: bool = False):
        raise NotImplementedError('_load_tile_data must be implemented in inheriting class')

    def _load_tile_data_to_memory(self, tile: Tile, only_points: bool = False, only_grid: bool = False):
        raise NotImplementedError('_load_tile_data_to_memory must be implemented in inheriting class')

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
        tile_name = self._get_new_tile_name(flat_index)
        row, col = self._tile_idx_to_row_col(flat_index)
        tile_folder = os.path.join(root_folder, tile_name)
        return tile_folder, row, col

    def _get_new_tile_name(self, flat_index: int):
        """
        Build a new tile_name given the flattened tile index

        Parameters
        ----------
        flat_index
            1d index of the flattened tiles

        Returns
        -------
        str
            new tile name
        """

        x, y = self._tile_idx_to_origin_point(flat_index)
        tile_name = '{}_{}'.format(x, y)
        return tile_name

    def _clear_tile(self, tile_folder: str):
        """
        Delete the given tile_folder and any data it has in it

        Parameters
        ----------
        tile_folder
            path to folder that contains the tile
        """

        if os.path.exists(tile_folder):
            rmtree(tile_folder)

    def _clear_tile_contents(self, tile_folder: str):
        """
        Clear a tile from disk by removing the folder/data, and then recreating the folder

        Parameters
        ----------
        tile_folder
            path to folder that contains the tile
        """

        self._clear_tile(tile_folder)
        os.makedirs(tile_folder)

    def _save_grid(self):
        """
        Save the grid data to disk, grids do not contain points or gridded data (that is in the tiles) so the only
        thing saved to disk is the grid metadata
        """

        if self.output_folder:
            root_folder = os.path.join(self.output_folder, self.name)
            os.makedirs(root_folder, exist_ok=True)
            self._save_bathygrid_metadata(root_folder)

    def _save_tile(self, tile: Tile, flat_index: int, only_points: bool = False, only_grid: bool = False):
        """
        Save the tile data to disk, if there is no tile, clear the existing data.  Saving a tile will recreate whatever
        is currently on disk.

        Parameters
        ----------
        tile
            tile object that needs it's metdata loaded
        flat_index
            1d index of the flattened tiles
        """

        if self.output_folder:
            root_folder = os.path.join(self.output_folder, self.name)
            tile_folder, row, col = self._get_tile_folder(root_folder, flat_index)
            if tile is None:
                self._clear_tile(tile_folder)
            else:
                self._load_tile_data_to_memory(tile, only_points=only_points, only_grid=only_grid)
                self._save_tile_metadata(tile, tile_folder)
                self._save_tile_data(tile, tile_folder, only_points=only_points, only_grid=only_grid)

    def save(self, folderpath: str = None, progress_bar: bool = True):
        """
        Save to a new root folder within the provided folderpath

        Parameters
        ----------
        folderpath
            base folder for the saved bathygrid instance
        progress_bar
            if True, will display a progress bar
        """
        if folderpath:
            self.output_folder = folderpath
            self._save_grid()
            if progress_bar:
                print_progress_bar(0, self.tiles.size, 'Saving to {}:'.format(os.path.split(folderpath)[1]))
            if self.sub_type in ['srtile', 'quadtile']:
                for cnt, tile in enumerate(self.tiles.flat):
                    if progress_bar:
                        print_progress_bar(cnt + 1, self.tiles.size, 'Saving to {}:'.format(os.path.split(folderpath)[1]))
                    self._save_tile(tile, cnt)
            else:  # this grid has grids underneath it, we need to save those grids via recursion
                for cnt, subgrid in enumerate(self.tiles.flat):
                    if progress_bar:
                        print_progress_bar(cnt + 1, self.tiles.size, 'Saving to {}:'.format(os.path.split(folderpath)[1]))
                    subgrid_folder, row, col = self._get_tile_folder(self.output_folder, cnt)
                    if subgrid is None:
                        self._clear_tile(subgrid_folder)
                    else:
                        subgrid.save(os.path.join(self.output_folder, self.name), progress_bar=False)

    def _load_grid(self):
        """
        Load the grid data from disk, grids do not contain points or gridded data (that is in the tiles) so the only
        thing saved to disk is the grid metadata
        """

        if self.output_folder:
            root_folder = os.path.join(self.output_folder, self.name)
            if not os.path.exists(root_folder):
                raise ValueError('Unable to find grid root folder {}'.format(root_folder))
            self._load_bathygrid_metadata(root_folder)

    def _load_tile(self, flat_index: int, only_points: bool = False, only_grid: bool = False):
        """
        Save the tile data to disk, if there is no tile, clear the existing data.  Saving a tile will recreate whatever
        is currently on disk.

        Parameters
        ----------
        flat_index
            1d index of the flattened tiles
        """

        if self.output_folder:
            root_folder = os.path.join(self.output_folder, self.name)
            tile_folder, row, col = self._get_tile_folder(root_folder, flat_index)
            if os.path.exists(tile_folder):
                if self.tiles.flat[flat_index] is None:
                    if self.sub_type == 'srtile':
                        newtile = SRTile()
                    else:
                        raise NotImplementedError('Only srtile is currently supported, {} do not exist yet'.format(self.sub_type))
                else:
                    newtile = self.tiles.flat[flat_index]
                self._load_tile_metadata(newtile, tile_folder)
                self._load_tile_data(newtile, tile_folder, only_points=only_points, only_grid=only_grid)
                self.tiles.flat[flat_index] = newtile
            else:
                self.tiles.flat[flat_index] = None

    def load(self, folderpath: str = None):
        """
        Load from a saved bathygrid instance

        Parameters
        ----------
        folderpath
            base folder for the saved bathygrid instance
        """
        if folderpath:
            self.output_folder = folderpath
            self._load_grid()
            if self.sub_type in ['srtile', 'quadtile']:
                for idx in range(self.tiles.size):
                    self._load_tile(idx)
            else:  # this grid has grids underneath it, we need to load those grids via recursion
                root_folder = os.path.join(self.output_folder, self.name)
                if self.storage_type == 'numpy':
                    storage_cls = NumpyGrid
                else:
                    raise NotImplementedError('{} is not a valid storage type'.format(self.storage_type))
                for idx in range(self.tiles.size):
                    subgrid_folder, row, col = self._get_tile_folder(self.output_folder, idx)
                    if os.path.exists(subgrid_folder):
                        newgrid = storage_cls()
                        newgrid.name = os.path.split(subgrid_folder)[1]
                        newgrid.load(root_folder)
                        self.tiles[row][col] = newgrid
                    else:
                        self.tiles[row][col] = None


class NumpyGrid(BaseStorage):
    """
    Backend for saving the point data / gridded data to numpy chunked files, I think this is going to be less efficient
    and create too many files/folders.  But there isn't really an option to lazy load in numpy unless I want to carry
    around a loaded npz object and pull out the arrays on demand.
    """

    def __init__(self, min_x: float = 0, min_y: float = 0, max_x: float = 0, max_y: float = 0,
                 tile_size: float = 1024.0, set_extents_manually: bool = False, output_folder: str = ''):
        super().__init__(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y, tile_size=tile_size,
                         set_extents_manually=set_extents_manually, output_folder=output_folder)
        self.storage_type = 'numpy'

    def _numpygrid_todask(self, arr):
        if not isinstance(arr, da.Array):
            return da.from_array(arr)
        return arr

    def _save_tile_data(self, tile: Tile, folderpath: str, only_points: bool = False, only_grid: bool = False):
        """
        Convert the arrays to dask and save them as stacked numpy arrays
        """

        if not only_grid:
            rmtree(folderpath + '/data', ignore_errors=True)  # ignore errors so that this passes if this folder does not exist
            da.to_npy_stack(folderpath + '/data', self._numpygrid_todask(tile.data))
        if not only_points:
            for resolution in tile.cells.keys():
                rmtree(folderpath + '/cells_{}_depth'.format(resolution), ignore_errors=True)
                da.to_npy_stack(folderpath + '/cells_{}_depth'.format(resolution), self._numpygrid_todask(tile.cells[resolution]['depth']))
                rmtree(folderpath + '/cells_{}_vertical_uncertainty'.format(resolution), ignore_errors=True)
                da.to_npy_stack(folderpath + '/cells_{}_vertical_uncertainty'.format(resolution), self._numpygrid_todask(tile.cells[resolution]['vertical_uncertainty']))
                rmtree(folderpath + '/cells_{}_horizontal_uncertainty'.format(resolution), ignore_errors=True)
                da.to_npy_stack(folderpath + '/cells_{}_horizontal_uncertainty'.format(resolution), self._numpygrid_todask(tile.cells[resolution]['horizontal_uncertainty']))
                rmtree(folderpath + '/cell_edges_x_{}'.format(resolution), ignore_errors=True)
                da.to_npy_stack(folderpath + '/cell_edges_x_{}'.format(resolution), self._numpygrid_todask(tile.cell_edges_x[resolution]))
                rmtree(folderpath + '/cell_edges_y_{}'.format(resolution), ignore_errors=True)
                da.to_npy_stack(folderpath + '/cell_edges_y_{}'.format(resolution), self._numpygrid_todask(tile.cell_edges_y[resolution]))
        # both require saving the indices, which are updated on adding/removing points and when gridding
        for resolution in tile.cell_indices.keys():
            rmtree(folderpath + '/cell_{}_indices'.format(resolution), ignore_errors=True)
            da.to_npy_stack(folderpath + '/cell_{}_indices'.format(resolution), self._numpygrid_todask(tile.cell_indices[resolution]))

    def _load_tile_data(self, tile: Tile, folderpath: str, only_points: bool = False, only_grid: bool = False):
        """
        lazy load from the saved tile arrays into dask arrays and populate the tile attributes.
        """

        resolutions = []
        data_folders = os.listdir(folderpath)
        for df in data_folders:
            sections = df.split('_')
            if sections[0] == 'cells':
                resolutions.append(float(sections[1]))
        resolutions.sort()
        if not only_grid:
            tile.data = da.from_npy_stack(folderpath + '/data')
        if not only_points:
            for resolution in resolutions:
                tile.cells[resolution] = {}
                tile.cells[resolution]['depth'] = da.from_npy_stack(folderpath + '/cells_{}_depth'.format(resolution))
                tile.cells[resolution]['vertical_uncertainty'] = da.from_npy_stack(folderpath + '/cells_{}_vertical_uncertainty'.format(resolution))
                tile.cells[resolution]['horizontal_uncertainty'] = da.from_npy_stack(folderpath + '/cells_{}_horizontal_uncertainty'.format(resolution))
                tile.cell_edges_x[resolution] = da.from_npy_stack(folderpath + '/cell_edges_x_{}'.format(resolution))
                tile.cell_edges_y[resolution] = da.from_npy_stack(folderpath + '/cell_edges_y_{}'.format(resolution))
        for resolution in resolutions:
            tile.cell_indices[resolution] = da.from_npy_stack(folderpath + '/cell_{}_indices'.format(resolution))

    def _load_tile_data_to_memory(self, tile: Tile, only_points: bool = False, only_grid: bool = False):
        """
        Expects you to have run _load_tile_data already.  This is the next step that loads it into memory.

        Pull the tile data into memory, allowing the data to be worked on efficiently, and to break the link between data
        and file on disk.  This allows you to delete the data on disk and retain the data in memory.

        With Numpy memmap, this is a bit weird.  You have to call del to break the link and then replace the reference.
        See below.
        """

        if not only_grid:
            tmpdata = np.array(tile.data)
            del tile.data
            tile.data = tmpdata
        if not only_points:
            cell_resolutions = list(tile.cells.keys())
            for resolution in cell_resolutions:
                for lyrname in tile.cells[resolution]:
                    tmpdata = np.array(tile.cells[resolution][lyrname])
                    del tile.cells[resolution][lyrname]
                    tile.cells[resolution][lyrname] = tmpdata
                tmpdata = np.array(tile.cell_edges_x[resolution])
                del tile.cell_edges_x[resolution]
                tile.cell_edges_x[resolution] = tmpdata
                tmpdata = np.array(tile.cell_edges_y[resolution])
                del tile.cell_edges_y[resolution]
                tile.cell_edges_y[resolution] = tmpdata
        cellidx_resolutions = list(tile.cell_indices.keys())
        for resolution in cellidx_resolutions:
            tmpdata = np.array(tile.cell_indices[resolution])
            del tile.cell_indices[resolution]
            tile.cell_indices[resolution] = tmpdata
