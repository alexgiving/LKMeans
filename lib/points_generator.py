from typing import List, Tuple

import numpy as np


def generate_gaussian_clusters(
    centroid_locations: List[List],
    sigma: float,
    dimension: int,
    n_points_per_cluster: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:

    clusters = []
    labels = []
    randomizer = np.random.default_rng()

    for cluster_id, centroid_location in enumerate(centroid_locations):
        cluster_size = (n_points_per_cluster, dimension)
        labels_size = (n_points_per_cluster, )

        cluster_points = randomizer.normal(
            loc=centroid_location,
            scale=sigma,
            size=cluster_size
        )

        cluster_labels = np.full(shape=labels_size, fill_value=cluster_id)

        clusters.append(cluster_points)
        labels.append(cluster_labels)

    return np.concatenate(clusters), np.concatenate(labels)


def generate_cluster_centroids(
    dimension: int,
    n_clusters: int,
    distance_factor: float
    ) -> np.ndarray:

    centroid_locations = np.stack([np.random.random(dimension) * distance_factor for _ in range(n_clusters)])
    return centroid_locations
