from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

from lkmeans.clustering.unsupervised.unsupervised_clustering import (
    UnsupervisedClustering,
)
from lkmeans.clustering.utils import (
    assign_to_cluster,
    calculate_inertia,
    init_centroids,
)


class LKMeans(UnsupervisedClustering):

    def _fit(self, X: NDArray) -> None:
        self._validate_data(X, self._n_clusters)

        centroids = init_centroids(X, self._n_clusters)

        iter_with_no_progress = 0
        for _ in range(self._max_iter):
            if iter_with_no_progress >= self._max_iter_with_no_progress:
                break

            bias_centroids = deepcopy(centroids)
            clusters, _ = assign_to_cluster(
                X, centroids, self._n_clusters, self._distance_calculator
            )

            for cluster_id, cluster in enumerate(clusters):
                cluster = np.array(cluster, copy=True)
                centroids[cluster_id] = deepcopy(self._optimize_centroid(cluster))

            if np.array_equal(bias_centroids, centroids):
                iter_with_no_progress += 1
            else:
                iter_with_no_progress = 0

        self._inertia = calculate_inertia(X, centroids)
        self._cluster_centers = deepcopy(centroids)
