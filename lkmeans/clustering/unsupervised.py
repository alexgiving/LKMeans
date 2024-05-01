from abc import abstractmethod
from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

from lkmeans.clustering.base import Clustering
from lkmeans.clustering.utils import assign_to_cluster, calculate_inertia, set_type


def init_centroids(data: NDArray, n_clusters: int) -> NDArray:
    indices = np.random.choice(data.shape[0], n_clusters, replace=False)
    centroids = data[indices]
    return centroids


class UnsupervisedClustering(Clustering):

    @abstractmethod
    def _fit(self, X: NDArray) -> None:
        ...

    def fit(self, X: NDArray) -> None:
        X = set_type(X)
        self._fit(X)

    def fit_predict(self, X: NDArray) -> list[int]:
        X = set_type(X)
        self._fit(X)
        labels = self.predict(X)
        return labels


# pylint: disable= too-few-public-methods, too-many-arguments
class LKMeans(UnsupervisedClustering):

    def _optimize_centroid(self, cluster: NDArray) -> NDArray:
        data_dimension = cluster.shape[1]
        new_centroid = np.array([])

        for coordinate_id in range(data_dimension):
            dimension_slice = cluster[:, coordinate_id]
            value = self._optimizer(dimension_slice)
            new_centroid = np.append(new_centroid, value)
        new_centroid = np.array(new_centroid)
        return new_centroid

    def _fit(self, X: NDArray) -> None:
        self._validate_data(X, self._n_clusters)

        centroids = init_centroids(X, self._n_clusters)

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
