import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, jaccard_score  # clustering
from sklearn.metrics import accuracy_score, f1_score  # classification


class ClusteringMetrics:
    r"""Class storing the different clustering metrics computed as part of this experiment.

    This can be serialized to a dict by using :func:`.to_dict`.
    """

    def __init__(self, y: np.ndarray, y_pred: np.ndarray):
        self.acc = jaccard_score(y, y_pred, average='macro')
        self.ami = adjusted_mutual_info_score(y, y_pred, average_method='max')
        self.ari = adjusted_rand_score(y, y_pred)

    def to_dict(self):
        return self.__dict__


class ClassificationMetrics:
    r"""Class storing the different classification metrics computed as part of this experiment.

    This can be serialized to a dict by using :func:`.to_dict`.
    """

    def __init__(self, y: np.ndarray, y_pred: np.ndarray):
        self.acc = accuracy_score(y, y_pred)
        self.f1 = f1_score(y, y_pred, average='micro')

    def to_dict(self):
        return self.__dict__
