from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

from lkmeans.clustering.semi_supervised.supervised_clustering import SemiSupervisedClustering
from lkmeans.clustering.semi_supervised.utils import assign_to_cluster_with_supervision
from lkmeans.clustering.utils import calculate_inertia


class HardSemiSupervisedLKMeans(SemiSupervisedClustering):

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
