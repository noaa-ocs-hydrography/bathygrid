import os
import numpy as np
from datetime import datetime
from typing import Union
from dask.distributed import get_client, Client
import psutil


def dask_find_or_start_client(address: str = None, silent: bool = False):
    """
    Either start or return Dask client in local/networked cluster mode

    Parameters
    ----------
    address
        ip address for existing or desired new dask server instance
    silent
        whether or not to print messages

    Returns
    -------
    dask.distributed.client.Client
        Client instance representing Local Cluster/Networked Cluster operations
    """

    client = None
    try:
        if address is None:
            client = get_client()
            if not silent:
                print('Using existing local cluster client...')
        else:
            client = get_client(address=address)
            if not silent:
                print('Using existing client on address {}...'.format(address))
    except ValueError:  # no global client found and no address provided
        logical_core_count = psutil.cpu_count(True)
        mem_total_gb = psutil.virtual_memory().total / 1000000000
        # currently trying to support >8 workers is a mem hog.  Limit to 8, maybe expose this in the gui somewhere

        if mem_total_gb > 24:  # basic test to see if we have enough memory, using an approx necessary amount of memory
            num_workers = min(logical_core_count, 8)
        else:  # if you have less, just limit to 4 workers
            num_workers = min(logical_core_count, 4)

        if address is None:
            if not silent:
                print('Starting local cluster client...')
            client = Client(n_workers=num_workers)
        else:
            if not silent:
                print('Starting client on address {}...'.format(address))
            client = Client(address=address, n_workers=num_workers)
    if client is not None:
        print(client)
    return client


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

    if x_edges.size == 2 and y_edges.size == 2:
        return np.zeros(x.shape, dtype=int)
    xshape = x_edges.shape[0] - 1  # edges will be one longer than the number of tiles in this dimension
    yshape = y_edges.shape[0] - 1
    base_indices = np.arange(xshape * yshape).reshape(yshape, xshape)
    x_idx = np.searchsorted(x_edges, x, side='right') - 1
    y_idx = np.searchsorted(y_edges, y, side='right') - 1
    return base_indices[y_idx, x_idx]


def print_progress_bar(iteration, total, prefix='Progress:', suffix='Complete', decimals=1, length=70, fill='█', print_end="\r"):
    """
    Call in a loop to generate a text progress bar, ex:

    # A List of Items
    items = list(range(0, 57))
    l = len(items)

    # Initial call to print 0% progress
    printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for i, item in enumerate(items):
        # Do stuff...
        time.sleep(0.1)
        # Update Progress Bar
        printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

    Parameters
    ----------
    iteration
        current iteration out of the total iterations
    total
        total iterations
    prefix
        prefix string for the display
    suffix
        suffix string for the display
    decimals
        number of decimals in the progress percentage
    length
        character length of the bar
    fill
        bar fill character
    print_end
        end character
    """

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()
