from typing import Any

import numpy as np
from numpy.typing import NDArray


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
