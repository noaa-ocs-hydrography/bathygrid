import os
from xml.etree import ElementTree as et
from xml.dom import minidom
import numpy as np
from dask.array import Array
import xarray as xr
from datetime import datetime
import h5py
from pyproj import CRS
from typing import Union

from bathygrid.backends import NumpyGrid
from bathygrid.utilities import create_folder, gdal_raster_create, return_gdal_version


def _correct_for_layer_metadata(resfile: str, data: list, nodatavalue: float):
    """
    Gdal bag driver writes the band min/max to include the nodatavalue, we have to write the correct values ourselves,
    should be resolved in GDAL3.3.2, see OSGeo/gdal issue #4057

    Parameters
    ----------
    resfile
        bag for this resolution
    data
        raster layers of the data, as numpy arrays
    nodatavalue
        nodatavalue of the layer
    """

    if os.path.exists(resfile):
        r5 = h5py.File(resfile, 'r+')
        validdata = data[0] != nodatavalue
        r5['BAG_root']['elevation'].attrs['Maximum Elevation Value'] = np.float32(np.max(data[0][validdata]))
        r5['BAG_root']['elevation'].attrs['Minimum Elevation Value'] = np.float32(np.min(data[0][validdata]))
        if len(data) == 2:
            r5['BAG_root']['uncertainty'].attrs['Maximum Uncertainty Value'] = np.float32(np.max(data[1][validdata]))
            r5['BAG_root']['uncertainty'].attrs['Minimum Uncertainty Value'] = np.float32(np.min(data[1][validdata]))
        r5.close()


def _set_temporal_extents(resfile: str, start_time: Union[str, int, float, datetime], end_time: Union[str, int, float, datetime]):
    """
    Taken from the HSTB bag.py library.  Sets the min/max time of the BAG by shoveling in the following xml blob:

      <gmd:temporalElement>
        <gmd:EX_TemporalExtent>
          <gmd:extent>
            <gml:TimePeriod gml:id="temporal-extent-1" xsi:type="gml:TimePeriodType">
              <gml:beginPosition>2018-06-29T07:20:48</gml:beginPosition>
              <gml:endPosition>2018-07-06T21:54:43</gml:endPosition>
            </gml:TimePeriod>
          </gmd:extent>
        </gmd:EX_TemporalExtent>
      </gmd:temporalElement>

    Parameters
    ----------
    resfile
        bag for this resolution
    data
        raster layers of the data, as numpy arrays
    nodatavalue
        nodatavalue of the layer
    """

    if os.path.exists(resfile) and start_time and end_time:
        r5 = h5py.File(resfile, 'r+')
        metadata = r5['BAG_root']['metadata'][:].tobytes().decode().replace("\x00", "")
        xml_root = et.fromstring(metadata)

        if isinstance(start_time, (float, int)):
            start_time = datetime.utcfromtimestamp(start_time).strftime('%Y-%m-%dT%H:%M:%S')
        elif isinstance(start_time, datetime):
            start_time = start_time.strftime('%Y-%m-%dT%H:%M:%S')
        if isinstance(end_time, (float, int)):
            end_time = datetime.utcfromtimestamp(end_time).strftime('%Y-%m-%dT%H:%M:%S')
        elif isinstance(end_time, datetime):
            end_time = end_time.strftime('%Y-%m-%dT%H:%M:%S')
        gmd = '{http://www.isotc211.org/2005/gmd}'
        gml = '{http://www.opengis.net/gml/3.2}'
        bagschema = "{http://www.opennavsurf.org/schema/bag}"
        xsi = '{http://www.w3.org/2001/XMLSchema-instance}'
        et.register_namespace("gmi", "http://www.isotc211.org/2005/gmi")
        et.register_namespace('gmd', "http://www.isotc211.org/2005/gmd")
        et.register_namespace('xsi', "http://www.w3.org/2001/XMLSchema-instance")
        et.register_namespace('gml', "http://www.opengis.net/gml/3.2")
        et.register_namespace('gco', "http://www.isotc211.org/2005/gco")
        et.register_namespace('xlink', "http://www.w3.org/1999/xlink")
        et.register_namespace('bag', "http://www.opennavsurf.org/schema/bag")
        temporal_hierarchy = [gmd + 'identificationInfo', bagschema + 'BAG_DataIdentification', gmd + 'extent', gmd + 'EX_Extent',
                              gmd + 'temporalElement', gmd + 'EX_TemporalExtent', gmd + 'extent', gml + 'TimePeriod']
        use_gml = gml
        temporal_root = "/".join(temporal_hierarchy)
        begin_elem = xml_root.findall(temporal_root + "/" + use_gml + 'beginPosition')
        end_elem = xml_root.findall(temporal_root + "/" + use_gml + 'endPosition')
        if not begin_elem or not end_elem:
            parent = xml_root
            for elem in temporal_hierarchy:
                found = parent.findall(elem)
                if not found:
                    new_elem = et.SubElement(parent, elem)
                    if "TimePeriod" in elem:
                        new_elem.set(use_gml + 'id', "temporal-extent-1")
                        new_elem.set(xsi + 'type', "gml:TimePeriodType")
                    found = [new_elem]
                parent = found[0]
            if not begin_elem:
                begin_elem = [et.SubElement(parent, use_gml+"beginPosition")]
            if not end_elem:
                end_elem = [et.SubElement(parent, use_gml+"endPosition")]
        begin_elem[0].text = start_time
        end_elem[0].text = end_time
        new_metadata = et.tostring(xml_root).decode()
        del r5['BAG_root']['metadata']
        r5['BAG_root'].create_dataset("metadata", maxshape=(None,), data=np.array(list(new_metadata), dtype="S1"))
        r5.close()


def _generate_caris_rxl(resfile: str, wkt_string: str):
    """
    Caris expects the WKT string to be written to a separate file next to the BAG.  We have the wkt string in the bag
    metadata, but we need to write this second file to make Caris happy.  Caris expects the WKT V1 GDAL string format,
    so we ensure that is created and passed in here

    Parameters
    ----------
    resfile
        path to the bag file
    wkt_string
        the WKT v1 GDAL string for the horizontal coordinate system of this surface
    """

    if os.path.exists(resfile) and wkt_string:
        rxl_path = os.path.splitext(resfile)[0] + '.bag_rxl'
        top = et.Element('caris_registration', version="4.0", generation="USER")
        newtree = et.ElementTree(top)
        coord_elem = et.SubElement(top, 'coordinate_system')
        wktelem = et.SubElement(coord_elem, 'wkt')
        wktelem.text = wkt_string
        xmlstr = minidom.parseString(et.tostring(top)).toprettyxml(indent="  ", encoding='utf-8').decode()
        xmlstr = xmlstr.replace('&quot;', '"').encode('utf-8')
        with open(rxl_path, 'wb') as rxlfile:
            rxlfile.write(xmlstr)


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
            if self.data['z'].ndim > 1:
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
            self._export_bag(output_path, resolution=resolution, **kwargs)
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

        lyrtranslator = {'depth': 'Depth', 'elevation': 'Elevation', 'vertical_uncertainty': 'Vertical Uncertainty',
                         'horizontal_uncertainty': 'Horizontal Uncertainty'}
        nodatavalue = 1000000.0
        basefile, baseext = os.path.splitext(filepath)
        if resolution is not None:
            resolutions = [resolution]
        else:
            resolutions = self.resolutions
        layernames = [lname for lname in self.layer_names if lname in ['depth', 'vertical_uncertainty']]
        finalnames = [lyrtranslator[lname] for lname in layernames]
        if z_positive_up and finalnames.index('Depth') != -1:
            finalnames[finalnames.index('Depth')] = 'Elevation'
        for res in resolutions:
            chunk_count = 1
            for geo_transform, maxdim, data in self.get_chunks_of_tiles(resolution=res, layer=layernames,
                                                                        nodatavalue=nodatavalue, z_positive_up=z_positive_up):
                resfile = basefile + '_{}_{}.tif'.format(res, chunk_count)
                data = list(data.values())
                gdal_raster_create(resfile, data, geo_transform, self.epsg, nodatavalue=nodatavalue, bandnames=finalnames,
                                   driver='GTiff')
                chunk_count += 1

    def _export_bag(self, filepath: str, resolution: float = None, individual_name: str = 'unknown',
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

        lyrtranslator = {'depth': 'Depth', 'elevation': 'Elevation', 'vertical_uncertainty': 'Vertical Uncertainty',
                         'horizontal_uncertainty': 'Horizontal Uncertainty'}
        nodatavalue = 1000000.0
        z_positive_up = True
        basefile, baseext = os.path.splitext(filepath)
        if resolution is not None:
            resolutions = [resolution]
        else:
            resolutions = self.resolutions
        layernames = [lname for lname in self.layer_names if lname in ['depth', 'vertical_uncertainty']]
        finalnames = [lyrtranslator[lname] for lname in layernames]
        if z_positive_up and finalnames.index('Depth') != -1:
            finalnames[finalnames.index('Depth')] = 'Elevation'
        for res in resolutions:
            chunk_count = 1
            for geo_transform, maxdim, data in self.get_chunks_of_tiles(resolution=res, layer=layernames,
                                                                        nodatavalue=nodatavalue, z_positive_up=z_positive_up):
                resfile = basefile + '_{}_{}.bag'.format(res, chunk_count)
                data = list(data.values())
                gdal_raster_create(resfile, data, geo_transform, self.epsg, nodatavalue=nodatavalue,
                                   bandnames=finalnames, driver='BAG', creation_options=bag_options)
                _correct_for_layer_metadata(resfile, data, nodatavalue)
                _set_temporal_extents(resfile, self.min_time, self.max_time)
                _generate_caris_rxl(resfile, CRS.from_epsg(self.epsg).to_wkt(version='WKT1_GDAL', pretty=True))
                chunk_count += 1

    def return_attribution(self):
        """
        Used in Kluster, return the important attribution of the class as a dict to display in the gui

        Returns
        -------
        dict
            class attributes in a presentable form
        """

        data = {'grid_folder': self.output_folder, 'name': self.name, 'type': type(self), 'grid_resolution': self.grid_resolution,
                'grid_algorithm': self.grid_algorithm, 'epsg': self.epsg, 'vertical_reference': self.vertical_reference,
                'height': self.height, 'width': self.width, 'minimum_x': self.min_x, 'maximum_x': self.max_x,
                'minimum_y': self.min_y, 'maximum_y': self.max_y, 'minimum_time_utc': self.min_time,
                'maximum_time_utc': self.max_time, 'tile_size': self.tile_size,
                'subtile_size': self.subtile_size, 'tile_count': self.number_of_tiles, 'resolutions': self.resolutions,
                'storage_type': self.storage_type}
        ucontainers = self.return_unique_containers()
        for cont_name in ucontainers:
            try:  # this works for kluster added containers, that have a suffix with an index
                data['source_{}'.format(cont_name)] = {'time': self.container_timestamp[cont_name + '_0'],
                                                       'multibeam_lines': self.container[cont_name + '_0']}
            except KeyError:
                try:  # this works for all other standard container names
                    data['source_{}'.format(cont_name)] = {'time': self.container_timestamp[cont_name],
                                                           'multibeam_lines': self.container[cont_name]}
                except KeyError:
                    nearest_cont_name = [nm for nm in self.container if nm.find(cont_name) != -1]
                    if nearest_cont_name[0]:
                        data['source_{}'.format(nearest_cont_name[0])] = {'time': self.container_timestamp[nearest_cont_name[0]],
                                                                          'multibeam_lines': self.container[nearest_cont_name[0]]}
                    else:
                        raise ValueError('Unable to find entry for container {}, if you have a suffix with a _ and a number, bathygrid will interpret that as an index starting with 0'.format(cont_name))
        return data


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
