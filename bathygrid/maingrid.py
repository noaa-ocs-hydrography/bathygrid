import os
import numpy as np
from dask.array import Array
import xarray as xr
from datetime import datetime

from bathygrid.backends import NumpyGrid
from bathygrid.utilities import create_folder, gdal_raster_create, return_gdal_version


class SRGrid(NumpyGrid):
    """
    SRGrid is the basic implementation of the BathyGrid.  This class contains the metadata and other functions required
    to build and maintain the BathyGrid
    """
    def __init__(self, min_x: float = 0, min_y: float = 0, tile_size: float = 1024.0, output_folder: str = ''):
        super().__init__(min_x=min_x, min_y=min_y, tile_size=tile_size, output_folder=output_folder)
        self.can_grow = True
        self.name = 'SRGrid_Root'
        self.sub_type = 'srtile'
        self.output_folder = output_folder
        if self.output_folder:
            os.makedirs(output_folder, exist_ok=True)

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
        elif type(self.data) == xr.Dataset:
            if 'x' not in self.data:
                raise ValueError('BathyGrid: xarray Dataset provided for data, but "x" or "y" not found in variable names')
            if len(self.data.dims) > 1:
                raise ValueError('BathyGrid: xarray Dataset provided for data, but found multiple dimensions, must be one dimensional: {}'.format(self.data.dims))
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

    def load(self, folder_path: str = None):
        """
        Recursive load for all BathyGrid/Tile objects within this class.

        Parameters
        ----------
        folder_path
            container folder for the grid
        """

        if not folder_path:
            if self.output_folder:
                super().load(self.output_folder)
            else:
                raise ValueError('Grid has not been saved before, you must provide a folder path to load.')
        else:
            self.output_folder = folder_path
            super().load(self.output_folder)

    def export(self, output_path: str, export_format: str = 'csv', z_positive_up: bool = True, resolution: float = None,
               **kwargs):
        """
        Export the node data to one of the supported formats

        Parameters
        ----------
        output_path
            filepath for exporting the dataset
        export_format
            format option, one of 'csv', 'geotiff', 'bag'
        z_positive_up
            if True, will output bands with positive up convention
        resolution
            if provided, will only export the given resolution
        """

        fmt = export_format.lower()
        if os.path.exists(output_path):
            tstmp = datetime.now().strftime('%Y%m%d_%H%M%S')
            foldername, filname = os.path.split(output_path)
            filnm, filext = os.path.splitext(filname)
            output_path = os.path.join(foldername, '{}_{}{}'.format(filnm, tstmp, filext))

        if fmt == 'csv':
            self._export_csv(output_path, z_positive_up=z_positive_up, resolution=resolution)
        elif fmt == 'geotiff':
            self._export_geotiff(output_path, z_positive_up=z_positive_up, resolution=resolution)
        elif fmt == 'bag':
            self._export_bag(output_path, z_positive_up=z_positive_up, resolution=resolution, **kwargs)
        else:
            raise ValueError("bathygrid: Unrecognized format {}, must be one of ['csv', 'geotiff', 'bag']".format(fmt))

    def _export_csv(self, output_file: str, z_positive_up: bool = True, resolution: float = None):
        """
        Export the node data to csv

        Parameters
        ----------
        output_file
            output_file to contain the exported data
        z_positive_up
            if True, will output bands with positive up convention
        resolution
            if provided, will only export the given resolution
        """

        basefile, baseext = os.path.splitext(output_file)
        if resolution is not None:
            resolutions = [resolution]
        else:
            resolutions = self.resolutions
        for res in resolutions:
            resfile = basefile + '_{}.csv'.format(res)
            lyrs = self.return_layer_names()
            x, y, lyrdata, newmins, newmaxs = self.return_surf_xyz(lyrs, res, False)
            xx, yy = np.meshgrid(x, y)
            dataset = [xx.ravel(), yy.ravel()]
            dnames = ['x', 'y']
            for cnt, lname in enumerate(lyrs):
                if lname == 'depth' and z_positive_up:
                    lyrdata[cnt] = lyrdata[cnt] * -1
                    lname = 'elevation'
                dataset += [lyrdata[cnt].ravel()]
                dnames += [lname]

            sortidx = np.argsort(dataset[0])
            np.savetxt(resfile, np.stack([d[sortidx] for d in dataset], axis=1),
                       fmt=['%.3f' for d in dataset], delimiter=' ', comments='',
                       header=' '.join([nm for nm in dnames]))

    def _gdal_preprocessing(self, resolution: float, nodatavalue: float = 1000000.0, z_positive_up: bool = True,
                            layer_names: list = ['depth', 'vertical_uncertainty']):
        """
        Build the regular grid of depth and vertical uncertainty that raster outputs require.  Additionally, return
        the origin/pixel size (geotransform) and the bandnames to display in the raster.

        If vertical uncertainty is not found, will only return a list of [depth grid]

        Set all NaN in the dataset given to the provided nodatavalue (can't seem to get NaN nodatavalue to display in
        Caris)

        Parameters
        ----------
        resolution
            resolution that you want to return the data for
        nodatavalue
            nodatavalue to set in the regular grid
        z_positive_up
            if True, will output bands with positive up convention
        layer_names
            the layer names that you want to return the data for

        Returns
        -------
        list
            list of either [2d array of depth] or [2d array of depth, 2d array of vert uncertainty]
        list
            [x origin, x pixel size, x rotation, y origin, y rotation, -y pixel size]
        list
            list of band names, ex: ['Depth', 'Vertical Uncertainty']
        """

        finalnames = []
        lyrtranslator = {'depth': 'Depth', 'elevation': 'Elevation', 'vertical_uncertainty': 'Vertical Uncertainty',
                         'horizontal_uncertainty': 'Horizontal Uncertainty'}

        x, y, lyrdata, newmins, newmaxs = self.return_surf_xyz(layer_names, resolution, True)
        geo_transform = [np.float32(x[0]), resolution, 0, np.float32(y[0]), 0, -resolution]
        layer_names = [lyrtranslator[ln] for ln in layer_names]
        for cnt, lname in enumerate(layer_names):
            if lname == 'Depth' and z_positive_up:
                lyrdata[cnt] = lyrdata[cnt] * -1
                lname = 'Elevation'
            lyrdata[cnt] = lyrdata[cnt][:, ::-1]
            lyrdata[cnt][np.isnan(lyrdata[cnt])] = nodatavalue
            finalnames.append(lname)
        return lyrdata, geo_transform, layer_names

    def _export_geotiff(self, filepath: str, z_positive_up: bool = True, resolution: float = None):
        """
        Export a GDAL generated geotiff to the provided filepath

        Parameters
        ----------
        filepath
            folder to contain the exported data
        z_positive_up
            if True, will output bands with positive up convention
        resolution
            if provided, will only export the given resolution
        """

        nodatavalue = 1000000.0
        basefile, baseext = os.path.splitext(filepath)
        if resolution is not None:
            resolutions = [resolution]
        else:
            resolutions = self.resolutions
        for res in resolutions:
            resfile = basefile + '_{}.tiff'.format(res)
            data, geo_transform, bandnames = self._gdal_preprocessing(resolution=res, nodatavalue=nodatavalue,
                                                                      z_positive_up=z_positive_up)
            gdal_raster_create(resfile, data, geo_transform, self.epsg, nodatavalue=nodatavalue, bandnames=bandnames,
                               driver='GTiff')

    def _export_bag(self, filepath: str, z_positive_up: bool = True, resolution: float = None, individual_name: str = 'unknown',
                    organizational_name: str = 'unknown', position_name: str = 'unknown', attr_date: str = '',
                    vert_crs: str = '', abstract: str = '', process_step_description: str = '', attr_datetime: str = '',
                    restriction_code: str = 'otherRestrictions', other_constraints: str = 'unknown',
                    classification: str = 'unclassified', security_user_note: str = 'none'):
        """
        Export a GDAL generated BAG to the provided filepath

        If attr_date is not provided, will use the current date.  If attr_datetime is not provided, will use the current
        date/time.  If process_step_description is not provided, will use a default 'Generated By GDAL and Kluster'
        message.  If vert_crs is not provided, will use a WKT with value = 'unknown'

        Parameters
        ----------
        filepath
            folder to contain the exported data
        z_positive_up
            if True, will output bands with positive up convention
        resolution
            if provided, will only export the given resolution
        """

        if not attr_date:
            attr_date = datetime.now().strftime('%Y-%m-%d')
        if not attr_datetime:
            attr_datetime = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        if not process_step_description:
            process_step_description = 'Generated By GDAL {}'.format(return_gdal_version())
        if not vert_crs:
            vert_crs = 'VERT_CS["unknown", VERT_DATUM["unknown", 2000]]'

        bag_options = ['VAR_INDIVIDUAL_NAME=' + individual_name, 'VAR_ORGANISATION_NAME=' + organizational_name,
                       'VAR_POSITION_NAME=' + position_name, 'VAR_DATE=' + attr_date, 'VAR_VERT_WKT=' + vert_crs,
                       'VAR_ABSTRACT=' + abstract, 'VAR_PROCESS_STEP_DESCRIPTION=' + process_step_description,
                       'VAR_DATETIME=' + attr_datetime, 'VAR_RESTRICTION_CODE=' + restriction_code,
                       'VAR_OTHER_CONSTRAINTS=' + other_constraints, 'VAR_CLASSIFICATION=' + classification,
                       'VAR_SECURITY_USER_NOTE=' + security_user_note]

        nodatavalue = 1000000.0
        basefile, baseext = os.path.splitext(filepath)
        if resolution is not None:
            resolutions = [resolution]
        else:
            resolutions = self.resolutions
        for res in resolutions:
            resfile = basefile + '_{}.bag'.format(res)
            data, geo_transform, bandnames = self._gdal_preprocessing(resolution=res, nodatavalue=nodatavalue,
                                                                      z_positive_up=z_positive_up)
            gdal_raster_create(resfile, data, geo_transform, self.epsg, nodatavalue=nodatavalue, bandnames=bandnames,
                               driver='BAG', creation_options=bag_options)


class VRGridTile(SRGrid):
    """
    VRGridTile is a simple approach to variable resolution gridding.  We build a grid of BathyGrids, where each BathyGrid
    has a certain number of tiles (each tile with size subtile_size).  Each of those tiles can have a different resolution
    depending on depth.
    """

    def __init__(self, min_x: float = 0, min_y: float = 0, tile_size: float = 1024, output_folder: str = '', subtile_size: float = 128):
        super().__init__(min_x=min_x, min_y=min_y, tile_size=tile_size, output_folder=output_folder)
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
        return NumpyGrid(min_x=tile_x_origin, min_y=tile_y_origin, max_x=tile_x_origin + self.tile_size,
                         max_y=tile_y_origin + self.tile_size, tile_size=self.subtile_size,
                         set_extents_manually=True, output_folder=ofolder)

