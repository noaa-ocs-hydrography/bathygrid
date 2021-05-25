import numpy as np
from bathygrid.grids import TileGrid
from bathygrid.utilities import bin2d_with_indices, is_power_of_two
from bathygrid.algorithms import nb_grid_mean


class Tile(TileGrid):
    """
    Bathygrid is composed of multiple Tiles.  Each Tile manages its own point data and gridding.
    """
    def __init__(self, min_x: float, min_y: float, size: float):
        super().__init__(min_x, min_y, size)
        self.algorithm = None

    def clear_grid(self):
        self.cells = {}
        self.cell_edges_x = {}
        self.cell_edges_y = {}

    def clear_points(self):
        self.data = None
        self.container = {}  # dict of container name, list of multibeam fil
        self.cell_indices = {}


class SRTile(Tile):
    def __init__(self, min_x: float, min_y: float, size: float):
        super().__init__(min_x, min_y, size)

    def add_points(self, data: np.ndarray, container: str):
        if self.data is None:
            self.data = data
            self.container = {container: [0, self.data['x'].size]}
            for resolution in self.cell_indices:
                self.cell_indices[resolution] = np.full(self.data.shape, -1)
        else:
            if container in self.container:
                self.remove_points(container)
            self.container[container] = [self.data.size, self.data.size + data.size]
            self.data = np.concatenate([self.data, data])
            for resolution in self.cell_indices:
                self.cell_indices[resolution] = np.append(self.cell_indices[resolution], np.full(data.shape, -1))

    def remove_points(self, container):
        if container in self.container:
            remove_start, remove_end = self.container[container]
            msk = np.ones(self.data.shape[0], dtype=bool)
            msk[remove_start:remove_end] = False
            chunk_size = msk[remove_start:remove_end].size
            for cont in self.container:
                if self.container[cont][0] >= remove_end:
                    self.container[cont] = [self.container[cont][0] - chunk_size, self.container[cont][1] - chunk_size]
            for resolution in self.cell_indices:
                self.cell_indices[resolution] = self.cell_indices[resolution][msk]
            self.data = self.data[msk]
            self.container.pop(container)

    def new_grid(self, resolution: float, algorithm: str, nodatavalue: float = np.nan):
        if not is_power_of_two(resolution):
            raise ValueError(f'Tile: Resolution must be a power of two, got {resolution}')
        resolution = resolution
        grid_x = np.arange(self.min_x, self.max_x, resolution)
        grid_y = np.arange(self.min_y, self.max_y, resolution)
        self.cell_edges_x[resolution] = np.append(grid_x, grid_x[-1] + resolution)
        self.cell_edges_y[resolution] = np.append(grid_y, grid_y[-1] + resolution)
        grid_shape = (grid_x.size, grid_y.size)
        self.cells[resolution] = {}
        if algorithm == 'mean':
            self.cells[resolution]['depth'] = np.full(grid_shape, nodatavalue)
            self.cells[resolution]['vertical_uncertainty'] = np.full(grid_shape, nodatavalue)
            self.cells[resolution]['horizontal_uncertainty'] = np.full(grid_shape, nodatavalue)

    def _run_mean_grid(self, resolution):
        nb_grid_mean(self.data['z'], self.data['tvu'], self.data['thu'], self.cell_indices[resolution],
                     self.cells[resolution]['depth'], self.cells[resolution]['vertical_uncertainty'],
                     self.cells[resolution]['horizontal_uncertainty'])
        self.cells[resolution]['depth'] = np.round(self.cells[resolution]['depth'], 3)
        self.cells[resolution]['vertical_uncertainty'] = np.round(self.cells[resolution]['vertical_uncertainty'], 3)
        self.cells[resolution]['horizontal_uncertainty'] = np.round(self.cells[resolution]['horizontal_uncertainty'], 3)

    def grid(self, algorithm: str, resolution: float, clear_existing: bool = False):
        if not is_power_of_two(resolution):
            raise ValueError(f'Tile: Resolution must be a power of two, got {resolution}')
        if clear_existing:
            self.clear_grid()

        if resolution not in self.cells or algorithm != self.algorithm:
            self.algorithm = algorithm
            self.new_grid(resolution, algorithm)
        if resolution not in self.cell_indices:
            self.cell_indices[resolution] = bin2d_with_indices(self.data['x'], self.data['y'], self.cell_edges_x[resolution],
                                                               self.cell_edges_y[resolution])
        else:
            new_points = self.cell_indices[resolution] == -1
            if new_points.any():
                self.cell_indices[resolution][new_points] = bin2d_with_indices(self.data['x'][new_points], self.data['y'][new_points],
                                                                               self.cell_edges_x[resolution], self.cell_edges_y[resolution])
        if algorithm == 'mean':
            self._run_mean_grid(resolution)
        return resolution

    def get_layer_by_name(self, layer: str = 'depth', resolution: float = None):
        if self.is_empty:
            raise ValueError('Tile: Grid is empty, no layer "{}" found'.format(layer))
        if not resolution and len(list(self.cells.keys())) > 1:
            raise ValueError('Tile: you must specify a resolution to return layer data when multiple resolutions are found')
        if resolution:
            if resolution not in list(self.cells.keys()):
                raise ValueError('Tile: resolution {} not found in Tile'.format(resolution))
        else:
            resolution = list(self.cells.keys())[0]
        return self.cells[resolution][layer]


class VRTile(Tile):
    def __init__(self, min_x: float, min_y: float, size: float):
        super().__init__(min_x, min_y, size)
        self.overlay_cells = {}
#
#
# class QuadTree:
#     """
#     Adapted from https://github.com/GliderToolsCommunity/GliderTools/blob/master/glidertools/mapping.py
#
#     Recursively splits data into quadrants
#
#     Object oriented quadtree can access children recursively
#
#     Ultimately, we want to:
#
#     - save huge datasets in a way that allows for lazy loading by location
#     - save indices that allow you to update the grid when the soundings change
#     - allow for constraining ultimate grid sizes to powers of two
#     - allow for utm and geographic coordinate systems
#     - implement mean/CUBE algorithms for cell depth
#
#     - write locks on quads, file lock?  one master lock for now?
#     """
#
#     def __init__(self, manager, mins=None, maxs=None, max_points_per_quad=5,
#                  max_grid_size=128, min_grid_size=1, location=[], index=[], parent=None):
#         self.parent = parent  # parent quad instance, None if this is the top quad
#         self.location = location
#         self.tree_depth = len(location)
#         self.manager = manager
#         if manager is None:
#             manager = self.root.manager
#
#         # can't save boolean to json/zarr attribute, need to encode as a diff type, this kind of sucks but leaving it for now
#         self.is_leaf = False   # is the end of a quad split, contains the data has no children
#         self.quad_index = -1
#
#         if not index and not parent:  # first run through with data gets here
#             self._validate_inputs(min_grid_size, max_grid_size, max_points_per_quad)
#             data = manager.data
#             self.index = np.arange(data['x'].shape[0]).tolist()
#             self.settings = {'max_points_per_quad': max_points_per_quad, 'max_grid_size': max_grid_size,
#                              'min_grid_size': min_grid_size, 'min_depth': 0, 'max_depth': 0,
#                              'min_tvu': 0, 'max_tvu': 0, 'min_thu': 0, 'max_thu': 0,
#                              'number_of_points': len(self.index)}
#         elif not index:  # get here when you intialize empty quad to then load()
#             data = None
#             self.index = []
#         else:  # initialize child with index
#             data = manager.data[index]
#             self.index = index
#
#         if self.index:
#             xval = data['x']
#             yval = data['y']
#         else:
#             xval = None
#             yval = None
#
#         if mins is None and maxs is None:
#             if self.index:  # first run through of data gets here
#                 self.mins = [np.min(xval).astype(xval.dtype), (np.min(yval).astype(yval.dtype))]
#                 self.maxs = [np.max(xval).astype(xval.dtype), np.max(yval).astype(yval.dtype)]
#                 self._align_toplevel_grid()
#                 self.mins = [self.mins[0].astype(xval.dtype), self.mins[1].astype(yval.dtype)]
#                 self.maxs = [self.maxs[0].astype(xval.dtype), self.maxs[1].astype(yval.dtype)]
#                 manager._build_node_data_matrix(self.mins, self.maxs)
#             else:  # get here when you intialize empty quad to then load()
#                 self.mins = [0, 0]
#                 self.maxs = [0, 0]
#         else:
#             self.mins = mins
#             self.maxs = maxs
#
#         self.children = []
#
#         should_divide = False
#         if self.index:
#             top_left_idx, top_right_idx, bottom_left_idx, bottom_right_idx, xmin, xmax, ymin, ymax, xmid, ymid = self._build_quadrant_indices(xval, yval)
#             should_divide = self._build_split_check(len(top_left_idx), len(top_right_idx), len(bottom_left_idx), len(bottom_right_idx),
#                                                     max_grid_size, min_grid_size, max_points_per_quad)
#             if should_divide:
#                 props = dict(max_points_per_quad=max_points_per_quad, min_grid_size=min_grid_size, max_grid_size=max_grid_size, parent=self)
#                 self.children.append(QuadTree(None, [xmin, ymid], [xmid, ymax], index=top_left_idx, location=location + [0], **props))
#                 self.children.append(QuadTree(None, [xmid, ymid], [xmax, ymax], index=top_right_idx, location=location + [1], **props))
#                 self.children.append(QuadTree(None, [xmin, ymin], [xmid, ymid], index=bottom_left_idx, location=location + [2], **props))
#                 self.children.append(QuadTree(None, [xmid, ymin], [xmax, ymid], index=bottom_right_idx, location=location + [3], **props))
#                 self.index = []
#
#         if not should_divide:
#             self.is_leaf = True
#             if self.index and 'z' in data.dtype.names:
#                 quad_depth = manager.data['z'][self.index].mean().astype(manager.data['z'].dtype)
#                 quad_tvu = None
#                 if 'tvu' in data.dtype.names:
#                     quad_tvu = manager.data['tvu'][self.index].mean().astype(manager.data['tvu'].dtype)
#                 # quad_node_x = (self.mins[0] + (self.maxs[0] - self.mins[0]) / 2).astype(self.mins[0].dtype)
#                 # quad_node_y = (self.mins[1] + (self.maxs[1] - self.mins[1]) / 2).astype(self.mins[1].dtype)
#
#                 self.quad_index = self._return_root_quad_index(manager.mins, self.mins, min_grid_size,
#                                                                manager.node_data, manager.is_vr)
#                 if isinstance(self.quad_index, int):  # current vr implementation just gets you a flattened list of node values
#                     manager.node_data.append([quad_depth, quad_tvu])
#                 else:  # SR builds a MxN matrix of node values
#                     manager.node_data['z'][self.quad_index[0], self.quad_index[1]] = quad_depth
#                     if 'tvu' in data.dtype.names:
#                         manager.node_data['tvu'][self.quad_index[0], self.quad_index[1]] = quad_tvu
#
#     def __getitem__(self, args, silent=False):
#         """
#         Go through the quadtree and locate the quadtree at the provided index, see self.loc
#         """
#
#         args = np.array(args, ndmin=1)
#         if any(args > 3):
#             raise UserWarning("A quadtree only has 4 possible children, provided locations: {}".format(args))
#         quadtree = self
#         passed = []
#         for depth in args:
#             if (len(quadtree.children) > 0) | (not silent):
#                 quadtree = quadtree.children[depth]
#                 passed += [depth]
#             else:
#                 return None
#
#         return quadtree
#
#     def __repr__(self):
#         return "<{} : {}>".format(str(self.__class__)[1:-1], str(self.location))
#
#     def __str__(self):
#         location = str(self.location)[1:-1]
#         location = location if location != "" else "[] - base QuadTree has no location"
#
#         # boundaries and spacing to make it pretty
#         left, top = self.mins
#         right, bot = self.maxs
#         wspace = " " * len("{:.2f}".format(top))
#         strtspace = " " * (15 - max(0, (len("{:.2f}".format(top)) - 6)))
#
#         # text output (what youll see when you print the object)
#         about_tree = "\n".join(
#             [
#                 "",
#                 "QuadTree object",
#                 "===============",
#                 "  location:         {}".format(location),
#                 "  tree depth:       {}".format(len(self.location)),
#                 "  n_points:         {}".format(len(self.index)),
#                 "  boundaries:       {:.2f}".format(top),
#                 "{}{:.2f}{}{:.2f}".format(strtspace, left, wspace, right),
#                 "                    {:.2f}".format(bot),
#                 "  children_points:  {}".format(str([len(c.index) for c in self.children])),
#             ]
#         )
#         return about_tree
#
#     def _return_root_quad_index(self, root_mins, mins, min_grid_size, current_node_data, is_vr):
#         if is_vr:
#             return np.int32(len(current_node_data) - 1)
#         else:
#             return [int((mins[0] - root_mins[0]) / min_grid_size),
#                     int((mins[1] - root_mins[1]) / min_grid_size)]
#
#     def _validate_inputs(self, min_grid_size, max_grid_size, max_points_per_quad):
#         if not is_power_of_two(min_grid_size):
#             raise ValueError('QuadTree: Only supports min_grid_size that is power of two, received {}'.format(min_grid_size))
#         if not is_power_of_two(max_grid_size):
#             raise ValueError('QuadTree: Only supports max_grid_size that is power of two, received {}'.format(max_grid_size))
#         if (not isinstance(max_points_per_quad, int)) or (max_points_per_quad <= 0):
#             raise ValueError('QuadTree: max points per quad must be a positive integer, received {}'.format(max_points_per_quad))
#
#     def _align_toplevel_grid(self):
#         """
#         So that our grids will line up nicely with each other, we set the origin to the nearest multiple of 128 and
#         adjust the width/height of the quadtree to the nearest power of two.  This way when we use powers of two
#         resolution, everything will work out nicely.
#         """
#
#         # align origin with nearest multple of 128
#         self.mins[0] -= self.mins[0] % 128
#         self.mins[1] -= self.mins[1] % 128
#
#         width = self.maxs[0] - self.mins[0]
#         height = self.maxs[1] - self.mins[1]
#         greatest_dim = max(width, height)
#         nearest_pow_two = int(2 ** np.ceil(np.log2(greatest_dim)))
#         width_adjustment = (nearest_pow_two - width)
#         height_adjustment = (nearest_pow_two - height)
#
#         self.maxs[0] += width_adjustment
#         self.maxs[1] += height_adjustment
#
#     def _build_quadrant_indices(self, xval: Union[np.ndarray, da.Array], yval: Union[np.ndarray, da.Array]):
#         """
#         Determine the data indices that split the data into four quadrants
#         Parameters
#         ----------
#         xval
#             x coordinate for all points
#         yval
#             y coordinate for all points
#
#         Returns
#         -------
#         np.array
#             data indices that correspond to points in the top left quadrant
#         np.array
#             data indices that correspond to points in the top right quadrant
#         np.array
#             data indices that correspond to points in the bottom left quadrant
#         np.array
#             data indices that correspond to points in the bottom right quadrant
#         float
#             minimum x value of the input points
#         float
#             maximum x value of the input points
#         float
#             minimum y value of the input points
#         float
#             maximum y value of the input points
#         float
#             x midpoint value of the input points
#         float
#             y midpoint value of the input points
#         """
#
#         xmin, ymin = self.mins
#         xmax, ymax = self.maxs
#         xmid = (0.5 * (xmin + xmax)).astype(xmin.dtype)
#         ymid = (0.5 * (ymin + ymax)).astype(ymin.dtype)
#
#         # split the data into four quadrants
#         xval_lessthan = xval <= xmid
#         xval_greaterthan = xval >= xmid
#         yval_lessthan = yval <= ymid
#         yval_greaterthan = yval >= ymid
#
#         idx = np.array(self.index)
#         index_q0 = idx[xval_lessthan & yval_greaterthan].tolist()  # top left
#         index_q1 = idx[xval_greaterthan & yval_greaterthan].tolist()  # top left
#         index_q2 = idx[xval_lessthan & yval_lessthan].tolist()  # top left
#         index_q3 = idx[xval_greaterthan & yval_lessthan].tolist()  # top left
#
#         return index_q0, index_q1, index_q2, index_q3, xmin, xmax, ymin, ymax, xmid, ymid
#
#     def _build_split_check(self, q0_size: int, q1_size: int, q2_size: int, q3_size: int, max_grid_size, min_grid_size, max_points_per_quad):
#         """
#         Builds a check to determine whether or not this quadrant should be divided.  Uses:
#
#         point_check - points in the quad must not exceed the provided maximum allowable points
#         max_size_check - quad size must not exceed the provided maximum allowable grid size
#         min_size_check - quad size (after splitting) must not end up less than minimum allowable grid size
#         too_few_points_check - if you know that splitting will lead to less than allowable max points, dont split
#         empty_quad_check - if there are three quadrants that are empty, split so that you don't end up with big
#             quads that are mostly empty
#
#         Parameters
#         ----------
#         q0_size
#             size of points that belong to the top left quadrant
#         q1_size
#             size of points that belong to the top right quadrant
#         q2_size
#             size of points that belong to the bottom left quadrant
#         q3_size
#             size of points that belong to the bottom right quadrant
#
#         Returns
#         -------
#         bool
#             if True, split this quad into 4 quadrants
#         """
#         n_points = len(self.index)
#         sizes = [self.maxs[0] - self.mins[0], self.maxs[1] - self.mins[1]]
#
#         point_check = n_points > max_points_per_quad
#         max_size_check = sizes[0] > max_grid_size
#         min_size_check = sizes[0] / 2 >= min_grid_size
#
#         too_few_points_check = True
#         empty_quad_check = False
#         if n_points <= max_points_per_quad * 4:  # only do these checks if there are just a few points, they are costly
#             too_few_points_quads = [q0_size >= max_points_per_quad or q0_size == 0,
#                                     q1_size >= max_points_per_quad or q1_size == 0,
#                                     q2_size >= max_points_per_quad or q2_size == 0,
#                                     q3_size >= max_points_per_quad or q3_size == 0]
#             too_few_points_check = np.count_nonzero(too_few_points_quads) == 4
#             if n_points <= max_points_per_quad:
#                 empty_quads = [q0_size == 0, q1_size == 0, q2_size == 0, q3_size == 0]
#                 empty_quad_check = np.count_nonzero(empty_quads) == 3
#                 too_few_points_check = True  # hotwire this, we always split when there are three empty quadrants and we are greater than min resolution
#
#         if (point_check or max_size_check or empty_quad_check) and min_size_check:
#             return True
#         return False
#
#     def _traverse_tree(self):
#         """
#         iterate through the quadtree
#         """
#         if not self.children:
#             yield self
#         for child in self.children:
#             yield from child._traverse_tree()
#
#     def loc(self, *args: list, silent=False):
#         """
#         Get a child quad by index
#
#         self.loc(0,1,2) returns the bottom left (2) of the top right (1) of the top left (0) child quad of self
#
#         Parameters
#         ----------
#         args
#             list of the quad indices to use to locate a quad
#         silent
#             if True, will return None if the index does not exist.  Otherwise raises IndexError
#
#         Returns
#         -------
#         QuadTree
#             QuadTree instance at that location
#         """
#
#         return self.__getitem__(args, silent=silent)
#
#     def query_xy(self, x: float, y: float):
#         """
#         Given the provided x/y value, find the leaf that contains the point.  The point does not have to be an actual
#         point in the quadtree, will find the leaf that theoretically would contain it.
#
#         Returns None if point is out of bounds
#
#         search_qtree = qtree.query_xy(538999, 5292700)
#         search_qtree.is_leaf
#         Out[10]: True
#
#         Parameters
#         ----------
#         x
#             x value of point to search for
#         y
#             y value of point to search for
#
#         Returns
#         -------
#         QuadTree
#             leaf node that contains the given point
#         """
#         xmid = 0.5 * (self.mins[0] + self.maxs[0])
#         ymid = 0.5 * (self.mins[1] + self.maxs[1])
#         idx = np.where([(x < xmid) & (y > ymid), (x >= xmid) & (y > ymid), (x < xmid) & (y <= ymid), (x >= xmid) & (y <= ymid)])[0].tolist()
#         self = self.loc(*idx, silent=True)
#         while not self.is_leaf:
#             self = self.query_xy(x, y)
#
#         return self
#
#     def get_leaves(self):
#         """
#         Get a list of all leaves from the quadtree
#
#         Returns
#         -------
#         list
#             list of QuadTrees that are leaves
#         """
#         return list(set(list(self._traverse_tree())))
#
#     def get_leaves_attr(self, attr: str):
#         """
#         Get the attribute corresponding to attr from all leaves
#
#         Parameters
#         ----------
#         attr
#             str name of an attribute
#
#         Returns
#         -------
#         list
#             list of attribute values
#         """
#
#         return [getattr(q, attr) for q in self.leaves]
#
#     def draw_tree(self, ax: plt.Axes = None, tree_depth: int = None, exclude_empty: bool = False,
#                   line_width: int = 1, edge_color='red', plot_nodes: bool = False, plot_points: bool = False):
#         """
#         Recursively plot an x/y box drawing of the qtree.
#
#         Parameters
#         ----------
#         ax
#             matplotlib subplot, provide if you want to draw on an existing plot
#         tree_depth
#             optional, if provided will only plot the tree from this level down
#         exclude_empty
#             optional, if provided will only plot the leaves that contain points
#         line_width
#             line width in the outline of the Rect object plotted for each quad
#         edge_color
#             color of the outline of the Rect object plotted for each quad
#         plot_nodes
#             if True, will plot the node of each quad
#         plot_points
#             if True, will plot all the points given to the QuadTree
#         """
#
#         manager = self.root.manager
#         manager._finalize_data()
#
#         root_quad = self.root
#         norm = matplotlib.colors.Normalize(vmin=root_quad.settings['min_depth'], vmax=root_quad.settings['max_depth'])
#         cmap = matplotlib.cm.rainbow
#
#         if ax is None:
#             ax = plt.subplots(figsize=[11, 7], dpi=150)[1]
#
#         if tree_depth is None or tree_depth == 0:
#             if exclude_empty and not self.index:
#                 pass
#             else:
#                 sizes = [self.maxs[0] - self.mins[0], self.maxs[1] - self.mins[1]]
#                 if self.quad_index != -1:
#                     try:
#                         idx = self.quad_index[0], self.quad_index[1]
#                     except:
#                         idx = self.quad_index
#                     quad_z = manager.node_data['z'][idx].compute()
#                     rect = matplotlib.patches.Rectangle(self.mins, *sizes, zorder=2, alpha=0.5, lw=line_width, ec=edge_color, fc=cmap(norm(quad_z)))
#                     if plot_nodes:
#                         quad_x = manager.node_data['x'][idx].compute()
#                         quad_y = manager.node_data['y'][idx].compute()
#                         ax.scatter(quad_x, quad_y, s=5)
#                     if plot_points:
#                         ax.scatter(manager.data['x'][self.index].compute(),
#                                    manager.data['y'][self.index].compute(), s=2)
#                 else:  # no depth for the quad
#                     rect = matplotlib.patches.Rectangle(self.mins, *sizes, zorder=2, alpha=1, lw=line_width, ec=edge_color, fc='None')
#                 ax.add_patch(rect)
#
#         if tree_depth is None:
#             for child in self.children:
#                 child.draw_tree(ax, tree_depth=None, exclude_empty=exclude_empty, line_width=line_width, edge_color=edge_color, plot_points=plot_points, plot_nodes=plot_nodes)
#         elif tree_depth > 0:
#             for child in self.children:
#                 child.draw_tree(ax, tree_depth=tree_depth - 1, exclude_empty=exclude_empty, line_width=line_width, edge_color=edge_color, plot_points=plot_points, plot_nodes=plot_nodes)
#
#         if (self.tree_depth == 0) or (tree_depth is None and self.tree_depth == 0):
#             xsize = self.maxs[0] - self.mins[0]
#             ysize = self.maxs[1] - self.mins[1]
#             ax.set_ylim(self.mins[1] - ysize / 10, self.maxs[1] + ysize / 10)
#             ax.set_xlim(self.mins[0] - xsize / 10, self.maxs[0] + xsize / 10)
#             divider = make_axes_locatable(ax)
#             cax = divider.append_axes('right', size='5%', pad=0.05)
#             plt.gcf().colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='vertical', label='Depth (+down, meters)')
#
#         return ax
#
#     @property
#     def root(self):
#         """
#         Return the root of the tree
#
#         Returns
#         -------
#         QuadTree
#             root quadtree for the tree
#         """
#
#         parent = self
#         for _ in self.location:
#             parent = parent.parent
#         return parent
#
#     @property
#     def siblings(self):
#         """
#         Return a list of siblings for this QuadTree, returns None if this QuadTree has no parent (top level)
#
#         Returns
#         -------
#         list
#             list of QuadTree instances for the siblings of this QuadTree
#         """
#
#         if self.parent is None:
#             return None
#
#         siblings = self.parent.children.copy()
#         siblings.remove(self)
#         return siblings
#
#     def _get_border_children(self, quad, location):
#         """Returns all T/L/R/B boundaries as defined by bound_location"""
#         bounds = [[2, 3], [0, 2], [0, 1], [1, 3]]
#         bound_location = bounds[location]
#         if not quad.is_leaf:
#             for i in bound_location:
#                 yield from self._get_border_children(quad.children[i], location)
#         else:
#             yield quad
#
#     @property
#     def neighbours(self):
#         """
#         Return a list of all neighbors for this QuadTree (orthogonal)
#
#         Returns
#         -------
#         list
#             list of QuadTrees that are orthogonally adjacent to this QuadTree
#         """
#
#         neighbours = []
#         root = self.root
#         if self == root:
#             return neighbours
#
#         ########################
#         # IMMEDIATELY ADJACENT #
#         sizes = [self.maxs[0] - self.mins[0], self.maxs[1] - self.mins[1]]
#         coords = [(self.mins[0] + sizes[0] / 2, self.maxs[1] + sizes[1] / 2,),
#                   (self.maxs[0] + sizes[0] / 2, self.mins[1] + sizes[1] / 2,),
#                   (self.mins[0] + sizes[0] / 2, self.mins[1] - sizes[1] / 2,),
#                   (self.maxs[0] - sizes[0] / 2, self.mins[1] + sizes[1] / 2,),]
#         # loop through top, right, bottom, left
#         for i in range(4):
#             x, y = coords[i]
#             query_quad = root.query_xy(x, y)
#             if query_quad is not None:
#                 same_size_idx = query_quad.location[: self.tree_depth]
#                 same_size_quad = root[same_size_idx]
#                 neighbours += list(self._get_border_children(same_size_quad, i))
#
#         #############
#         # DIAGONALS #
#         root_sizes = [root.maxs[0] - root.mins[0], root.maxs[1] - root.mins[1]]
#         xs, ys = (root_sizes / 2 ** root.max_tree_depth) / 2
#         neighbours += [
#             root.query_xy(self.mins[0] - xs, self.mins[1] - ys),  # TL
#             root.query_xy(self.maxs[0] + xs, self.mins[1] - ys),  # TR
#             root.query_xy(self.mins[0] - xs, self.maxs[1] + ys),  # BL
#             root.query_xy(self.maxs[0] + xs, self.maxs[1] + ys),  # BR
#         ]
#
#         unique_neighbours = list(set(neighbours))
#         try:
#             unique_neighbours.remove(self)
#         except ValueError:
#             pass
#
#         return unique_neighbours
#
#     @property
#     def max_tree_depth(self):
#         """
#         Return the max depth of this QuadTree tree
#
#         Returns
#         -------
#         int
#             max depth of tree (a 7 means there are 7 levels to the tree)
#         """
#
#         depths = np.array([leaf.tree_depth for leaf in self.leaves])
#
#         return depths.max()
#
#     @property
#     def leaves(self):
#         """
#         Return a list of all leaves for the tree, leaves being QuadTrees with no children
#
#         Returns
#         -------
#         list
#             list of QuadTrees
#         """
#
#         return self.get_leaves()
#
#     @property
#     def leaves_with_data(self):
#         """
#         Return a list of all leaves for the tree that have data
#
#         Returns
#         -------
#         list
#             list of QuadTrees
#         """
#
#         return [lv for lv in self.get_leaves() if lv.index]
#
