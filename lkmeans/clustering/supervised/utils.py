from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

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
