from abc import ABC
from typing import Union

import numpy as np
from numpy.typing import NDArray

from lkmeans.clustering.utils import assign_to_cluster, select_optimizer, set_type
from lkmeans.distance import DistanceCalculator


class Clustering(ABC):
    def __init__(self, n_clusters: int, *, p: Union[float, int] = 2,
                 max_iter: int = 100, max_iter_with_no_progress: int = 15) -> None:
        self._n_clusters = n_clusters
        self._max_iter = max_iter
        self._p = p
        self._max_iter_with_no_progress = max_iter_with_no_progress

        self._distance_calculator = DistanceCalculator(self._p)
        self._optimizer = select_optimizer(self._p)

        self._inertia = 0.
        self._cluster_centers = np.array([])

    @property
    def inertia_(self) -> float:
        return self._inertia

    @property
    def cluster_centers_(self) -> NDArray:
        return self._cluster_centers

    @staticmethod
    def _validate_data(data: NDArray, n_clusters: int) -> None:
        if data.shape[0] < n_clusters:
            raise ValueError(f'Clustering of {data.shape[0]} samples with {n_clusters} centers is not possible')

    def predict(self, X: NDArray) -> list[int]:
        X = set_type(X)
        _, labels = assign_to_cluster(X, self._cluster_centers, self._n_clusters, self._distance_calculator)
        return labels
