from abc import ABC
from typing import Union

import numpy as np
from numpy.typing import NDArray

from lkmeans.clustering.utils import assign_to_cluster, set_type
from lkmeans.distance import DistanceCalculator
from lkmeans.optimizers import get_optimizer


class Clustering(ABC):
    def __init__(
        self,
        n_clusters: int,
        *,
        p: Union[float, int] = 2,
        max_iter: int = 100,
        max_iter_with_no_progress: int = 15,
    ) -> None:
        self._n_clusters = n_clusters
        self._max_iter = max_iter
        self._p = p
        self._max_iter_with_no_progress = max_iter_with_no_progress

        self._distance_calculator = DistanceCalculator(self._p)
        self._optimizer = get_optimizer(self._p)

        self._inertia = 0.0
        self._cluster_centers = np.array([])

    def _optimize_centroid(self, cluster: NDArray) -> NDArray:
        data_dimension = cluster.shape[1]
        new_centroid = np.array([])

        for coordinate_id in range(data_dimension):
            dimension_slice = cluster[:, coordinate_id]
            value = self._optimizer(dimension_slice)
            new_centroid = np.append(new_centroid, value)
        new_centroid = np.array(new_centroid)
        return new_centroid

    @property
    def inertia_(self) -> float:
        return self._inertia

    @property
    def cluster_centers_(self) -> NDArray:
        return self._cluster_centers

    @staticmethod
    def _validate_data(data: NDArray, n_clusters: int) -> None:
        if data.shape[0] < n_clusters:
            raise ValueError(f"Clustering of {data.shape[0]} samples with {n_clusters} centers is not possible")

    def predict(self, X: NDArray) -> list[int]:
        X = set_type(X)
        _, labels = assign_to_cluster(X, self._cluster_centers, self._n_clusters, self._distance_calculator)
        return labels

    def _get_repr_params(self) -> str:
        return f"n_clusters={self._n_clusters}, p={self._p}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({self._get_repr_params()})"
