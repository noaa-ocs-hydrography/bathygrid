import os, sys

# BGRID and TILE
# used to determine resolutions in grid tiles, depth versus resolution
#  ex: 0 to 20 meters is 0.5 m resolution, 20 to 40 meters is 1.0 meters, 40 to 80 meters is 4.0 meters resolution, etc.
depth_resolution_lookup = {10: 0.25, 20: 0.5, 40: 1.0, 60: 2.0, 80: 4.0, 160: 8.0, 320: 16.0, 640: 32.0, 1280: 64.0, 2560: 128.0,
                           5120: 256.0, 10240: 512.0, 20480: 1024.0}
# maximum grid dimension before we start building separate tiffs for export and visualization, we check both width and height
# of the sub grid and use the greater value.  see bgrid get_chunks_of_tiles
maximum_chunk_dimension = 32768.0
# density based resolution finder will use as fine a resolution as possible, as long as there are at least this many
# points per cell
minimum_points_per_cell = 5
# used in density based resolution estimate, see tile.resolution_by_densityv2
starting_resolution_density = 16.0
noise_accomodation_factor = 0.75
revert_to_lookup_threshold = 0.75

# BACKEND
# allowable grid root names, we use this to check for grid type on reload and a number of other things
allowable_grid_root_names = ['SRGrid_Root', 'VRGridTile_Root', 'SRGridZarr_Root', 'VRGridTileZarr_Root']
sr_grid_root_names = ['SRGrid_Root', 'SRGridZarr_Root']
vr_grid_root_names = ['VRGridTile_Root', 'VRGridTileZarr_Root']
# these are the attributes that are written to disk for grids, see backend
bathygrid_desired_keys = ['min_y', 'min_x', 'max_y', 'max_x', 'min_time', 'max_time', 'width', 'height', 'origin_x',
                          'origin_y', 'container', 'tile_x_origin', 'tile_y_origin', 'tile_edges_x', 'tile_edges_y',
                          'existing_tile_mask', 'maximum_tiles', 'number_of_tiles', 'can_grow', 'tile_size', 'mean_depth', 'epsg',
                          'vertical_reference', 'resolutions', 'name', 'output_folder', 'sub_type', 'subtile_size',
                          'storage_type', 'grid_algorithm', 'container_timestamp', 'grid_resolution', 'version']
# these attributes are written to disk but need translation between numpy and list for JSON to work
bathygrid_numpy_to_list = ['tile_x_origin', 'tile_y_origin', 'tile_edges_x', 'tile_edges_y', 'existing_tile_mask']
# these attributes are written to disk but need translation between float and string for JSON to work
bathygrid_float_to_str = ['min_y', 'min_x', 'max_y', 'max_x', 'width', 'height', 'origin_x', 'origin_y', 'mean_depth']
# these are the attributes that are written to disk for tiles, see backend
tile_desired_keys = ['min_y', 'min_x', 'max_y', 'max_x', 'width', 'height', 'container', 'name', 'algorithm', 'point_count_changed']
# these are the tile attributes that need translation between float and string for JSON to work
tile_float_to_str = ['min_y', 'min_x', 'max_y', 'max_x']
