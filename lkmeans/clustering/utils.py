from typing import Any

import numpy as np
from numpy.typing import NDArray

from lkmeans.distance import DistanceCalculator


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


def init_centroids(data: NDArray, n_clusters: int) -> NDArray:
    indices = np.random.choice(data.shape[0], n_clusters, replace=False)
    centroids = data[indices]
    return centroids


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
