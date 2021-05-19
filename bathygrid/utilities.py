import os
import numpy as np
from datetime import datetime
from typing import Union


def create_folder(output_directory: str, fldrname: str):
    """
    Generate a new folder with folder name fldrname in output_directory.  Will create output_directory if it does
    not exist.  If fldrname exists, will generate a folder with a time tag next to it instead.  Will always
    create a folder this way.

    Parameters
    ----------
    output_directory
        path to containing folder
    fldrname
        name of new folder to create

    Returns
    -------
    str
        path to the created folder
    """

    os.makedirs(output_directory, exist_ok=True)
    tstmp = datetime.now().strftime('%Y%m%d_%H%M%S')
    try:
        fldr_path = os.path.join(output_directory, fldrname)
        os.mkdir(fldr_path)
    except FileExistsError:
        fldr_path = os.path.join(output_directory, fldrname + '_{}'.format(tstmp))
        os.mkdir(fldr_path)
    return fldr_path


def is_power_of_two(n: Union[int, float]):
    """
    Return True if number is a power of two, supports n>1 and n<1.

    Parameters
    ----------
    n
        number to check, can be float or int

    Returns
    -------
    bool
        number is power of two
    """

    if n > 1:
        if n != int(n):
            return False
        n = int(n)
        return (n != 0) and (n & (n - 1) == 0)
    elif n == 1:
        return True
    elif n > 0:
        return is_power_of_two(1 / n)
    else:
        return False


def bin2d_with_indices(x: np.array, y: np.array, x_edges: np.array, y_edges: np.array):
    """
    Started out using scipy binned_statistic_2d, but I found that it would append bins regardless of the number of bins
    you ask for (even when all points are inside the explicit bin edges) and the binnumber would be difficult to
    translate.  Since our bin edges are always sorted, a 2d binning isn't really that hard, so we do it using
    searchsorted for speed.

    Parameters
    ----------
    x
        x coordinate of the points, should be same shape as y (one dimensional)
    y
        y coordinate of the points, should be same shape as x (one dimensional)
    x_edges
        the bounds for the bins in the x dimension, should be one larger than the total expected bins in this dimension
    y_edges
        the bounds for the bins in the y dimension, should be one larger than the total expected bins in this dimension

    Returns
    -------
    np.array
        one dimensional integer index array that indicates which bin each point falls within.  Applies to the flattened
        bins (is a one dimensional index)
    """

    xshape = x_edges.shape[0] - 1  # edges will be one longer than the number of tiles in this dimension
    yshape = y_edges.shape[0] - 1
    base_indices = np.arange(xshape * yshape).reshape(yshape, xshape)
    x_idx = np.searchsorted(x_edges, x, side='right') - 1
    y_idx = np.searchsorted(y_edges, y, side='right') - 1
    return base_indices[y_idx, x_idx]
