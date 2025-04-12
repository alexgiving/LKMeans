from abc import abstractmethod
from typing import Set

import numpy as np
from numpy.typing import NDArray

from lkmeans.clustering.base import Clustering
from lkmeans.clustering.unsupervised.lkmeans import init_centroids
from lkmeans.clustering.utils import set_type


class SemiSupervisedClustering(Clustering):

    def _init_supervised_centroids(
        self, data: NDArray, n_clusters: int, targets: NDArray
    ) -> NDArray:
        unique_targets: Set[int] = set(targets[~np.isnan(targets)])

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
        return np.array(output_centroids)

    @abstractmethod
    def _fit(self, X: NDArray, targets: NDArray) -> None: ...

    def fit(self, X: NDArray, targets: NDArray) -> None:
        X = set_type(X)
        self._fit(X, targets)

    def fit_predict(self, X: NDArray, targets: NDArray) -> list[int]:
        X = set_type(X)
        self._fit(X, targets)
        labels = self.predict(X)
        return labels
