from functools import partial
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from lkmeans.distance import DistanceCalculator
from lkmeans.optimizers import bound_optimizer, mean_optimizer, median_optimizer, slsqp_optimizer


def set_type(data: Any) -> NDArray:
    if not isinstance(data, np.ndarray):
        return np.array(data)
    return data


def calculate_inertia(X: NDArray, centroids: NDArray) -> float:
    n_clusters = centroids.shape[0]
    distances = np.empty((X.shape[0], n_clusters))
    for i in range(n_clusters):
        distances[:, i] = np.sum((X - centroids[i, :])**2, axis=1)
    return np.sum(np.min(distances, axis=1))


def assign_to_cluster(
        X: NDArray,
        centroids: NDArray,
        n_clusters: int,
        distance_calculator: DistanceCalculator,
        ) -> tuple[list[list[float]], list[int]]:
    clusters = [[] for _ in range(n_clusters)]
    labels = []

    for point in X:
        distances_to_each_centroid = distance_calculator.get_pairwise_distance(point, centroids)
        closest_centroid = int(np.argmin(distances_to_each_centroid))
        clusters[closest_centroid].append(point)
        labels.append(closest_centroid)
    return clusters, labels


def select_optimizer(p: float) -> Callable:
    if p == 2:
        return mean_optimizer
    if p == 1:
        return median_optimizer
    elif 0 < p < 1:
        return partial(bound_optimizer, p=p)
    elif p > 1:
        return partial(slsqp_optimizer, p=p)
    raise ValueError('Parameter p must be greater than 0!')