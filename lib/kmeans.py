import numpy as np

from lib.minkowski import pairwise_minkowski_distance
from lib.optimizers import (mean_optimizer, median_optimizer,
                            segment_SLSQP_optimizer)
from lib.types import p_type


def assign_to_cluster(X: np.ndarray, centroids: np.ndarray, n_clusters: int, p: p_type):
    clusters = [[] for _ in range(n_clusters)]
    labels = []

    for point in X:
        distances_to_each_cebtroid = pairwise_minkowski_distance(
            point, centroids, p)

        closest_centroid = np.argmin(distances_to_each_cebtroid)
        clusters[closest_centroid].append(point)
        labels.append(closest_centroid)
    return clusters, labels


class KMeans:
    def __init__(self, n_clusters: int, max_iter: int = 100, p: p_type = 2):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.p = p
        self.max_iter_with_no_progress = 15

    def fit(self, X: np.ndarray):
        # initialize centroids randomly
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[indices]

        iter_with_no_progress = 0

        for _ in range(self.max_iter):
            if iter_with_no_progress >= self.max_iter_with_no_progress:
                break
            bias_centroids = self.centroids.copy()
            clusters, _ = assign_to_cluster(
                X, self.centroids, self.n_clusters, self.p)

            # update centroids using the specified optimizer
            for cluster_id, cluster in enumerate(clusters):
                cluster = np.array(cluster, copy=True)
                data_dimension = cluster.shape[1]
                new_centroid = np.array([])

                for coordinate_id in range(data_dimension):
                    dimension_slice = cluster[:, coordinate_id]

                    if self.p == 2:
                        value = mean_optimizer(dimension_slice)
                    elif self.p == 1:
                        value = median_optimizer(dimension_slice)
                    elif 0 < self.p < 1:
                        value = segment_SLSQP_optimizer(
                            dimension_slice, self.p)
                    else:
                        raise ValueError(f'Unsupported value of p: {self.p}')
                    new_centroid = np.append(new_centroid, value)
                self.centroids[cluster_id] = new_centroid.copy()

            if np.array_equal(bias_centroids, self.centroids):
                iter_with_no_progress += 1
            else:
                iter_with_no_progress = 0

        _, labels = assign_to_cluster(
            X, self.centroids, self.n_clusters, self.p)

        return self.centroids, labels
