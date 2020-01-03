import math
import os
import logging
import itertools
from collections import Iterable
from timeit import default_timer

import numpy as np

log = logging.getLogger(__name__)


class measure_time(object):
    r"""Simple context-manager to simplify time measurement.

    Example usage:

        with measure_time() as fit_elapsed:
            r = d.fit()
        # now do something with `fit_elapsed()`.
    """

    def __init__(self):
        self.start = None
        self.elapsed = None

    def __enter__(self):
        self.start = default_timer()
        return self

    def __exit__(self, *args):
        self.elapsed = default_timer() - self.start

    def __call__(self, *args, **kwargs):
        if self.elapsed:
            return self.elapsed
        return default_timer() - self.start


def read_large_txt(filename, delimiter=None, dtype=None):
    r"""Basic reimplementation of ``np.genfromtxt`` that consumes less memory.

    :param filename: File to load
    :param delimiter: Delimiter character, such as `,` for CSV
    :param dtype: The numpy element type for all tokens in the file
    :return: np.ndarray containing the file's content
    """
    with open(filename) as f:
        nrows = sum(1 for _ in f)
        f.seek(0)
        ncols = len(next(f).split(delimiter))
        f.seek(0)

        log.debug('%s: creating array: %s x %s -> %s bytes', filename, nrows, ncols, nrows * ncols * 8)

        out = np.empty((nrows, ncols), dtype=dtype)
        for i, line in enumerate(f):
            out[i] = line.split(delimiter)
    return out


class LoadedDataset:
    r"""Simple holder for a (x, y) dataset in memory."""

    def __init__(self, filename: str, x: np.ndarray, y: np.ndarray):
        self.filename = filename
        self.x = x
        self.y = y


def load_dataset(filename, shuffle=False) -> LoadedDataset:
    r"""Load a dataset CSV file into memory.

    This function loads the file's entire content into an np.ndarray.
    Each line represents a row in the multidimensional array, each comma-delimited token a column.
    :param filename: Path of the file to load
    :param shuffle:
    """
    # np.genfromtxt uses too much memory!
    data = read_large_txt(filename, delimiter=',')
    if shuffle:
        np.random.shuffle(data)
    y = data[:, -1]  # extract the last column from all rows
    x = data[:, :-1]  # extract the first m-1 columns from all rows
    return LoadedDataset(filename, x, y)


def filename_to_dataset_name(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def split_range_into_two_chunks(count):
    half = count / 2
    return math.ceil(half), math.floor(half)


def make_curved_topology(input_dim, latent_dim, num_layers, factor=25):
    r"""Create a 1D array of neurons per layer, that first increases in size and then decreases again.

    :param input_dim: The dimensionality of the input data (1st layer size)
    :param latent_dim: The reduced dimensionality (last layer size)
    :param num_layers: Number of layers
    :param factor: Factor describing the maximum increase
    :return: list of neuron counts
    """
    assert num_layers > 1
    increasing_count, decreasing_count = split_range_into_two_chunks(num_layers)
    increasing = [int(x) for x in np.linspace(input_dim, input_dim * factor, increasing_count)]
    # skip first element (same as last in |increasing|)
    decreasing = [int(x) for x in np.linspace(input_dim * factor, latent_dim, 1 + decreasing_count)[1:]]
    return increasing + decreasing


def assign(lst: list, idx: int, value, fill=None):
    r"""Assign :paramref:`value` to the list :paramref:`lst` at position :paramref:`idx`.

    If needed, the list is padded with :paramref:`fill`.

    :param lst: The target list
    :param idx: The index that shall be valid
    :param value: The new value :paramref:`lst` shall have at pos :paramref:`idx`
    :param fill: Padding value if the list is smaller than :paramref:`idx`
    """
    diff = len(lst) - idx
    if diff >= 0:
        lst[idx] = value
    else:
        lst.extend(itertools.repeat(fill, -diff))
        lst.append(value)


def get_iterable(x) -> Iterable:
    if isinstance(x, Iterable):
        return x
    else:
        return x,
