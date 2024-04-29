from abc import abstractmethod
from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

from lkmeans.clustering.base import Clustering
from lkmeans.clustering.unsupervised import assign_to_cluster, init_centroids
from lkmeans.clustering.utils import calculate_inertia, set_type
from lkmeans.distance import DistanceCalculator


class SupervisedClustering(Clustering):

    @staticmethod
    @abstractmethod
    def _assign_to_cluster(
            X: NDArray,
            centroids: NDArray,
            n_clusters: int,
            distance_calculator: DistanceCalculator,
            ) -> tuple[list[list[float]], list[int]]:
        ...

    @abstractmethod
    def _init_centroids(data: NDArray, n_clusters: int, targets: NDArray) -> NDArray:
        ...

    @abstractmethod
    def _fit(self, X: NDArray, targets: NDArray) -> None:
        ...

    def fit(self, X: NDArray, targets: NDArray) -> None:
        X = set_type(X)
        self._fit(X, targets)

    def fit_predict(self, X: NDArray, targets: NDArray) -> list[int]:
        X = set_type(X)
        self._fit(X, targets)
        labels = self.predict(X)
        return labels


# pylint: disable= too-few-public-methods, too-many-arguments
class SoftSSLKMeans(SupervisedClustering):

    def _optimize_centroid(self, cluster: NDArray) -> NDArray:
        data_dimension = cluster.shape[1]
        new_centroid = np.array([])

        for coordinate_id in range(data_dimension):
            dimension_slice = cluster[:, coordinate_id]
            value = self._optimizer(dimension_slice)
            new_centroid = np.append(new_centroid, value)
        new_centroid = np.array(new_centroid)
        return new_centroid

    def _init_centroids(self, data: NDArray, n_clusters: int, targets: NDArray) -> NDArray:
        if n_clusters != len(set(targets)):
            raise ValueError('Standard is used')
            return init_centroids(data, n_clusters)

        centroids = np.array([])
        for target_id in range(n_clusters):
            supervised_data = data[targets == target_id]
            centroid = self._optimize_centroid(supervised_data)
            centroids = np.append(centroids, centroid)
        return centroids

    def _fit(self, X: NDArray, targets: NDArray) -> None:
        self._validate_data(X, self._n_clusters)

        centroids = self._init_centroids(X, self._n_clusters, targets)

        iter_with_no_progress = 0
        for _ in range(self._max_iter):
            if iter_with_no_progress >= self._max_iter_with_no_progress:
                break

            bias_centroids = deepcopy(centroids)
            clusters, _ = assign_to_cluster(X, centroids, self._n_clusters, self._distance_calculator)

            # update centroids using the specified optimizer
            for cluster_id, cluster in enumerate(clusters):
                cluster = np.array(cluster, copy=True)
                centroids[cluster_id] = deepcopy(self._optimize_centroid(cluster))

            if np.array_equal(bias_centroids, centroids):
                iter_with_no_progress += 1
            else:
                iter_with_no_progress = 0

        self._inertia = calculate_inertia(X, centroids)
        self._cluster_centers = deepcopy(centroids)
