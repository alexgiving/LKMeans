import numpy as np

from lib.minkowski import pairwise_minkowski_distance
from lib.optimizers import (mean_optimizer, median_optimizer,
                            segment_SLSQP_optimizer)
from lib.types import p_type


class KMeans:
    def __init__(self, n_clusters: int, max_iter: int = 10, p: p_type = 2):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.p = p

    @staticmethod
    def assign_to_cluster(X: np.ndarray, centroids: np.ndarray, n_clusters: int, p: p_type):
        clusters = [[] for _ in range(n_clusters)]
        labels = []

        # assign each data point to the closest centroid
        for point in X:
            distances_to_each_cebtroid = pairwise_minkowski_distance(
                point, centroids, p)

            closest_centroid = np.argmin(distances_to_each_cebtroid)
            clusters[closest_centroid].append(point)
            labels.append(closest_centroid)
        return clusters, labels

    def fit(self, X: np.ndarray):
        # initialize centroids randomly
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[indices]

        for _ in range(self.max_iter):
            clusters, _ = self.assign_to_cluster(
                X, self.centroids, self.n_clusters, self.p)

            # update centroids using the specified optimizer
            for cluster_id, cluster in enumerate(clusters):
                cluster = np.array(cluster)
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
                    # elif self.optimizer in ('Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'trust-constr'):
                    #     self.centroids[j] = minimize(lambda x: minkowski_distance(x, cluster, self.p), self.centroids[j], method=self.optimizer).x.copy()
                self.centroids[cluster_id] = new_centroid

        _, labels = self.assign_to_cluster(
            X, self.centroids, self.n_clusters, self.p)

        return self.centroids, labels
