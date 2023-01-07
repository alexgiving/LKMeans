"""KMeans class."""
from copy import deepcopy

import numpy as np


class KMeans:
    """KMeans class."""
    def __init__(self, k=3, max_iter=1_000_000):
        """Initialize KMeans."""
        self.k = k
        self.max_iter = max_iter


    def generate_random_centers(self, data):
        """Generate random centers."""
        mean = np.mean(data, axis = 0)
        std  = np.std(data,  axis = 0)
        centers = np.random.randn(self.k, data.shape[1])*std + mean
        return centers

    def calculate_error(self, centers_new, centers_old):
        """Calculate error."""
        return np.linalg.norm(centers_new - centers_old)


    def calculate_distance(self, data, centers):
        """Calculate distance to every center."""
        distances = np.zeros((data.shape[0], self.k))
        for i in range(self.k):
            distances[:,i] = np.linalg.norm(data - centers[i], axis=1)
        return distances


    def fit(self, data):
        """Fit data."""
        centers = self.generate_random_centers(data)

        centers_old = np.zeros(centers.shape)
        centers_new = deepcopy(centers)

        clusters = np.zeros(data.shape[0])
        error = self.calculate_error(centers_new, centers_old)

        iteration = 0
        while error != 0 and iteration < self.max_iter:
            distances = self.calculate_distance(data, centers)
            # Assign all training data to closest center
            clusters = np.argmin(distances, axis = 1)
            centers_old = deepcopy(centers_new)
            # Calculate mean for every cluster and update the center
            for i in range(self.k):
                centers_new[i] = np.mean(data[clusters == i], axis=0)
            error = self.calculate_error(centers_new, centers_old)
            iteration += 1
        return centers_new
