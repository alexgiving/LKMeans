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


def move_towards_mean(mu: np.ndarray, mu_mean: np.ndarray, t: float) -> np.ndarray:
    '''
    Returns a new point that is moved towards the mean according to the rule:
    point(t) = point + t * (mean - point), 0 ≤ t ≤ 1
    '''
    new_mu = mu + t * (mu_mean - mu)
    return new_mu



def generate_2_mix_distribution(
        probability: float,
        mu_1: np.ndarray,
        mu_2: np.ndarray,
        cov_matrix_1: np.ndarray,
        cov_matrix_2: np.ndarray,
        n_samples: int,
        t: float
        ) -> Tuple[np.ndarray, np.ndarray]:


    mu_mean = (mu_1 + mu_2) / 2

    n_1 = int(probability * n_samples)
    mu_2 = move_towards_mean(mu_2, mu_mean, t)

    n_2 = n_samples - n_1
    mu_1 = move_towards_mean(mu_1, mu_mean, t)

    distribution_1 = np.random.multivariate_normal(np.squeeze(mu_1, axis=0), cov_matrix_1, n_1)
    distribution_2 = np.random.multivariate_normal(np.squeeze(mu_2, axis=0), cov_matrix_2, n_2)

    samples = np.concatenate((distribution_1, distribution_2), axis=0)
    labels = np.concatenate((np.zeros(n_1), np.ones(n_2)))


    # Shuffle the samples and labels
    permutation = np.random.permutation(n_samples)
    samples = samples[permutation]
    labels = labels[permutation]

    return samples, labels
