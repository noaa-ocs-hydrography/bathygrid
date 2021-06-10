import os, sys

# BGRID and TILE
# used to determine resolutions in grid tiles, depth versus resolution
#  ex: 0 to 20 meters is 0.5 m resolution, 20 to 40 meters is 1.0 meters, 40 to 80 meters is 4.0 meters resolution, etc.
depth_resolution_lookup = {20: 0.5, 40: 1.0, 80: 4.0, 160: 8.0, 320: 16.0, 640: 32.0, 1280: 64.0, 2560: 128.0,
                           5120: 256.0, 10240: 512.0, 20480: 1024.0}

# BACKEND
# these are the attributes that are written to disk for grids, see backend
bathygrid_desired_keys = ['min_y', 'min_x', 'max_y', 'max_x', 'width', 'height', 'origin_x', 'origin_y', 'container',
                          'tile_x_origin', 'tile_y_origin', 'tile_edges_x', 'tile_edges_y', 'existing_tile_mask',
                          'maximum_tiles', 'number_of_tiles', 'can_grow', 'tile_size', 'mean_depth', 'epsg',
                          'vertical_reference', 'resolutions', 'name', 'output_folder', 'sub_type', 'subtile_size',
                          'storage_type', 'grid_algorithm', 'container_timestamp', 'grid_resolution']
# these attributes are written to disk but need translation between numpy and list for JSON to work
bathygrid_numpy_to_list = ['tile_x_origin', 'tile_y_origin', 'tile_edges_x', 'tile_edges_y', 'existing_tile_mask']
# these attributes are written to disk but need translation between float and string for JSON to work
bathygrid_float_to_str = ['min_y', 'min_x', 'max_y', 'max_x', 'width', 'height', 'origin_x', 'origin_y', 'mean_depth']
# these are the attributes that are written to disk for tiles, see backend
tile_desired_keys = ['min_y', 'min_x', 'max_y', 'max_x', 'width', 'height', 'container', 'name', 'algorithm']
# these are the tile attributes that need translation between float and string for JSON to work
tile_float_to_str = ['min_y', 'min_x', 'max_y', 'max_x']
