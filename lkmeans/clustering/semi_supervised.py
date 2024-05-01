from abc import abstractmethod
from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

from lkmeans.clustering.base import Clustering
from lkmeans.clustering.unsupervised import assign_to_cluster, init_centroids
from lkmeans.clustering.utils import calculate_inertia, set_type
from lkmeans.distance import DistanceCalculator


def select_supervisor_targets(targets: NDArray, selection_ratio: float) -> NDArray:
    targets = targets.astype(np.float16)
    num_not_selected_targets = len(targets) - int(len(targets) * selection_ratio)
    not_selected_indices = np.random.choice(len(targets), num_not_selected_targets, replace=False)
    output_targets = deepcopy(targets)
    output_targets[not_selected_indices] = np.nan
    return output_targets


def assign_to_cluster_with_supervision(
        X: NDArray,
        centroids: NDArray,
        n_clusters: int,
        distance_calculator: DistanceCalculator,
        targets: NDArray,
        ) -> tuple[list[list[float]], list[int]]:
    clusters = [[] for _ in range(n_clusters)]
    labels = []

    for point, real_target in zip(X, targets):
        if not np.isnan(real_target):
            centroid = int(real_target)
        else:
            distances_to_each_centroid = distance_calculator.get_pairwise_distance(point, centroids)
            centroid = int(np.argmin(distances_to_each_centroid))
        clusters[centroid].append(point)
        labels.append(centroid)
    return clusters, labels


class SupervisedClustering(Clustering):

    def _optimize_centroid(self, cluster: NDArray) -> NDArray:
        data_dimension = cluster.shape[1]
        new_centroid = np.array([])

        for coordinate_id in range(data_dimension):
            dimension_slice = cluster[:, coordinate_id]
            value = self._optimizer(dimension_slice)
            new_centroid = np.append(new_centroid, value)
        new_centroid = np.array(new_centroid)
        return new_centroid

    def _init_supervised_centroids(self, data: NDArray, n_clusters: int, targets: NDArray) -> NDArray:
        unique_targets = set(targets[~np.isnan(targets)])

        centroids = []
        for target_id in unique_targets:
            supervised_data = data[targets == target_id]
            centroid = self._optimize_centroid(supervised_data)
            centroids.append(np.expand_dims(centroid, axis=0))
        output_centroids = np.concatenate(centroids, axis=0)

        if len(unique_targets) < n_clusters:
            no_target_data = data[np.isnan(targets)]
            remain_centroids = n_clusters - len(unique_targets)
            padding_centroids = init_centroids(no_target_data, remain_centroids)
            output_centroids = np.concatenate([output_centroids, padding_centroids], axis=0)
        return output_centroids

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

    def _fit(self, X: NDArray, targets: NDArray) -> None:
        self._validate_data(X, self._n_clusters)

        centroids = self._init_supervised_centroids(X, self._n_clusters, targets)

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


# pylint: disable= too-few-public-methods, too-many-arguments
class HardSSLKMeans(SupervisedClustering):

    def _fit(self, X: NDArray, targets: NDArray) -> None:
        self._validate_data(X, self._n_clusters)

        centroids = self._init_supervised_centroids(X, self._n_clusters, targets)

        iter_with_no_progress = 0
        for _ in range(self._max_iter):
            if iter_with_no_progress >= self._max_iter_with_no_progress:
                break

            bias_centroids = deepcopy(centroids)
            clusters, _ = assign_to_cluster_with_supervision(X, centroids, self._n_clusters,
                                                             self._distance_calculator, targets)

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
