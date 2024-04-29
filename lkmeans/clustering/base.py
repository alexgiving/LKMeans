from abc import ABC
from functools import partial
from typing import Callable, Union

import numpy as np
from numpy.typing import NDArray

from lkmeans.clustering.utils import set_type
from lkmeans.distance import DistanceCalculator

from lkmeans.optimizers import bound_optimizer, mean_optimizer, median_optimizer, slsqp_optimizer


def assign_to_cluster(
        X: NDArray,
        centroids: NDArray,
        n_clusters: int,
        distance_calculator: DistanceCalculator,
        ) -> tuple[list[list[float]], list[int]]:
    clusters = [[] for _ in range(n_clusters)]
    labels = []

    for point in X:
        distances_to_each_centroid = distance_calculator.get_pairwise_distance(point, centroids)
        closest_centroid = int(np.argmin(distances_to_each_centroid))
        clusters[closest_centroid].append(point)
        labels.append(closest_centroid)
    return clusters, labels


def _select_optimizer(p: float) -> Callable:
    if p == 2:
        return mean_optimizer
    if p == 1:
        return median_optimizer
    elif 0 < p < 1:
        return partial(bound_optimizer, p=p)
    elif p > 1:
        return partial(slsqp_optimizer, p=p)
    raise ValueError('Parameter p must be greater than 0!')


class Clustering(ABC):
    def __init__(self, n_clusters: int, *, p: Union[float, int] = 2,
                 max_iter: int = 100, max_iter_with_no_progress: int = 15) -> None:
        self._n_clusters = n_clusters
        self._max_iter = max_iter
        self._p = p
        self._max_iter_with_no_progress = max_iter_with_no_progress

        self._distance_calculator = DistanceCalculator(self._p)
        self._optimizer = _select_optimizer(self._p)

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