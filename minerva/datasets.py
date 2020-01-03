import typing

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from minerva.metafeatures.metafeature import MetaFeatureValue


class DatasetRegistry:
    r"""Registry for training datasets.

    Provides convenience methods for finding training datasets with metafeatures similar to a given input dataset.
    """

    def __init__(self):
        self.datasets = {}  # type: typing.Dict[str, dict]
        self.metafeatures = {}  # type: typing.Dict[str, typing.Dict[str, MetaFeatureValue]]
        self.knn = NearestNeighbors()

    def add(self, metadata: dict):
        r"""Add a training dataset's metadata to this registry.

        :param metadata: Metadata dictionary of the dataset to add.
        """
        metafeatures = [MetaFeatureValue(*v) for v in metadata['metafeatures']['data']]
        metafeatures = {mf.name: float(mf.value) for mf in metafeatures if mf.is_metafeature()}

        self.datasets[metadata['name']] = metadata
        self.metafeatures[metadata['name']] = metafeatures

    def find_nearest_datasets(self, new_metafeatures: pd.Series, type: str):
        r"""For an input dataset's metafeatures, find the datasets with the closest metafeature vector.

        :param new_metafeatures: Metafeatures of the "test dataset".
        :param type: Optional type of the dataset. If set, candidate datasets will be filtered by type.
         This effectively allows the user to create per-algorithm training sets that could be optimized for
         the respective algorithm.
        :return: A list of datasets and their distance to the test dataset's metafeatures.
        """
        all_metafeatures = {}
        for (name, metafeatures) in self.metafeatures.items():
            # kmeans typed datasets have a 'kmeans' key etc.
            if not type or type in self.datasets[name]:
                all_metafeatures[name] = metafeatures
        all_metafeatures = pd.DataFrame(all_metafeatures).transpose()

        x_train, x_test = self._scale(all_metafeatures, new_metafeatures)
        self.knn.fit(x_train)
        x_test = x_test.values.reshape((1, -1))
        distances, neighbor_indices = self.knn.kneighbors(x_test, return_distance=True)
        return [(self.datasets[all_metafeatures.index[i]], d) for (i, d) in zip(neighbor_indices[0], distances[0])]

    def _scale(self, x_train, x_test):
        assert isinstance(x_test, pd.Series), type(x_test)
        assert x_test.values.dtype == np.float64, x_test.values.dtype

        scaled_x_train = x_train.copy(deep=True)
        x_test = x_test.copy(deep=True)
        mins = pd.DataFrame(data=[scaled_x_train.min(), x_test]).min()
        maxs = pd.DataFrame(data=[scaled_x_train.max(), x_test]).max()
        divisor = (maxs - mins)
        divisor[divisor == 0] = 1
        scaled_x_train = (scaled_x_train - mins) / divisor
        x_test = (x_test - mins) / divisor
        return scaled_x_train, x_test
