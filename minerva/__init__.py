import copy
import os
import logging
from typing import Optional

import numpy as np
import pandas as pd
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import History

from minerva.autoencoder import build_and_train_autoencoder
from minerva.datasets import DatasetRegistry
from minerva.metafeatures.metafeature import DatasetMetafeatures
from minerva.metafeatures.metafeatures import calculate_metafeatures

log = logging.getLogger(__name__)

_default_ds_registry: Optional[DatasetRegistry] = None


def _get_default_registry():
    global _default_ds_registry
    if _default_ds_registry is None:
        storage_dir = os.path.join(os.path.dirname(os.path.abspath(__path__)), '_meta')
        _default_ds_registry = setup_dataset_registry('file:///' + storage_dir)
    return _default_ds_registry


class NoMatchFound(RuntimeError):
    r"""Exception thrown in case there are no candidate training datasets."""
    pass


class AutoencoderModel:
    r"""Autoencoder model and associated data.

    Objects of this class contain the built autoencoder, the encoder part model, the model's training history,
    and information about the closest training dataset.
    """

    def __init__(self, training_distance: float, model: Model, encoder: Model, history: History,
                 metafeatures: DatasetMetafeatures):
        self.training_distance = training_distance
        self.model = model
        self.encoder = encoder
        self.history = history
        self.metafeatures = metafeatures

    def encode(self, x: np.ndarray):
        return self.encoder.predict(x)


def autoencoder_for_dataset(x: np.ndarray, y: Optional[np.ndarray] = None,
                            name: Optional[str] = None,
                            type: Optional[str] = None,
                            ds_registry: Optional[DatasetRegistry] = None,
                            **kwargs) -> AutoencoderModel:
    """Build an autoencoder for the given dataset.

    :param x: The input data.
    :param y: Optional golden labels for the input data.
     These are usually unavailable for real-world data, but can be useful when testing different metafeature combinations.
    :param name: Optional name of the dataset for debugging purposes.
    :param type: Optional type of machine learning this dataset will be used for.
     This restricts the training space to datasets that have been used for the same type.
    :param ds_registry: Registry containing the training datasets + their results.
     If this is `None`, a global default registry will be used, populated with the training datasets distributed as part
     of this library.
    :param kwargs: Used to override configuration properties of the resulting autoencoder.
    """
    if y is None:
        y = np.zeros((x.shape[0],))
    metafeatures = calculate_metafeatures(name, x, y)
    metafeatures_x = pd.Series(name=name, data=metafeatures.to_dict())

    if ds_registry is None:
        ds_registry = _get_default_registry()
    nearest_training_datasets = ds_registry.find_nearest_datasets(metafeatures_x, type)
    if not nearest_training_datasets:
        raise NoMatchFound('no matches found - registry subset empty')

    (metadata, distance) = nearest_training_datasets[0]

    log.debug('Creating model based on training dataset %s - dist %s', metadata['name'], distance)
    cfg = metadata['autoencoder']['trajectory'][-1]['incumbent']
    new_cfg = copy.copy(cfg)
    new_cfg.update(kwargs)
    return AutoencoderModel(distance, *build_and_train_autoencoder(x, new_cfg), metafeatures)


def setup_dataset_registry(uri: str) -> DatasetRegistry:
    r"""Setup a DatasetRegistry from the given data source.

    Currently two data sources are available:

    * PostgreSQL database. The URI format follows the one described `here`_.
    * JSON exports of dataset metadata. This can either be a single file or the path of a directory
      and must start with `file://`.

    .. _here: https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls
    :param uri: URI of a data source containing a set of training datasets.
    :return: A :class:`DatasetRegistry` with all entries of the given data source.
    """
    ds_registry = DatasetRegistry()

    if uri.startswith('postgresql://') or uri.startswith('postgresql+'):
        from minerva.metadata.db import MetadataConnection
        loader = MetadataConnection(uri)
        objects = loader.load_all()
    else:
        from minerva.metadata.json import load_all
        objects = load_all(uri)

    for metadata in objects:
        ds_registry.add(metadata)

    return ds_registry
