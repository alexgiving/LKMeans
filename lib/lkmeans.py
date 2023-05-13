from copy import deepcopy

import numpy as np

from lib.minkowski import pairwise_minkowski_distance
from lib.optimizers import (bound_optimizer, mean_optimizer, median_optimizer,
                            slsqp_optimizer)


def assign_to_cluster(
        X: np.ndarray,
        centroids: np.ndarray,
        n_clusters: int,
        p: float | int
    ) -> tuple[list[list[float]], list[int]]:
    clusters = [[] for _ in range(n_clusters)]
    labels = []

    for point in X:
        distances_to_each_cebtroid = pairwise_minkowski_distance(
            point, centroids, p)
        closest_centroid = int(np.argmin(distances_to_each_cebtroid))
        clusters[closest_centroid].append(point)
        labels.append(closest_centroid)
    return clusters, labels


# pylint: disable= too-few-public-methods, too-many-arguments
class LKMeans:
    def __init__(self,
                 n_clusters: int,
                 p: float | int = 2,
                 n_init: int = 5,
                 max_iter: int = 100,
                 max_iter_with_no_progress: int = 15) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.p = p
        self.n_init = n_init
        self.max_iter_with_no_progress = max_iter_with_no_progress

        self.inertia_ = 0
        self.cluster_centers_ = np.array([])

    @staticmethod
    def _init_centroids(data: np.ndarray, n_clusters: int) -> np.ndarray:
        indices = np.random.choice(
            data.shape[0], n_clusters, replace=False)
        centroids = data[indices]
        return centroids

    @staticmethod
    def _optimize_centroid(cluster: np.ndarray, p: float | int) -> np.ndarray:
        data_dimension = cluster.shape[1]
        new_centroid = np.array([])

        for coordinate_id in range(data_dimension):
            dimension_slice = cluster[:, coordinate_id]
            value = 0
            if p == 2:
                value = mean_optimizer(dimension_slice)
            if p == 1:
                value = median_optimizer(dimension_slice)
            elif 0 < p < 1:
                value = bound_optimizer(dimension_slice, p)
            elif p > 1:
                value = slsqp_optimizer(dimension_slice, p)
            else:
                raise ValueError('Parameter p must be greater than 0!')
            new_centroid = np.append(new_centroid, value)
        new_centroid = np.array(new_centroid)
        return new_centroid

    @staticmethod
    def _inertia(X: np.ndarray, centroids: np.ndarray) -> float:
        n_clusters = centroids.shape[0]
        distances = np.empty((X.shape[0], n_clusters))
        for i in range(n_clusters):
            distances[:, i] = np.sum((X - centroids[i, :])**2, axis=1)
        return np.sum(np.min(distances, axis=1))

    def fit(self, X: np.ndarray) -> None:
        centroids = self._init_centroids(X, self.n_clusters)

        iter_with_no_progress = 0
        for _ in range(self.max_iter):
            if iter_with_no_progress >= self.max_iter_with_no_progress:
                break

            bias_centroids = deepcopy(centroids)
            clusters, _ = assign_to_cluster(
                X, centroids, self.n_clusters, self.p)

            # update centroids using the specified optimizer
            for cluster_id, cluster in enumerate(clusters):
                cluster = np.array(cluster, copy=True)
                centroids[cluster_id] = deepcopy(
                    self._optimize_centroid(cluster, self.p)
                )

            if np.array_equal(bias_centroids, centroids):
                iter_with_no_progress += 1
            else:
                iter_with_no_progress = 0

        self.inertia_ = self._inertia(X, centroids)
        self.cluster_centers_ = deepcopy(centroids)

    def predict(self, X: np.ndarray) -> list[int]:
        _, labels = assign_to_cluster(
            X, self.cluster_centers_, self.n_clusters, self.p)
        return labels

    def fit_predict(self, X: np.ndarray) -> list[int]:
        self.fit(X)
        labels = self.predict(X)
        return labels
