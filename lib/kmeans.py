"""KMeans class."""
from copy import deepcopy

import numpy as np

from lib.errors import InvalidDistanceMetricException, InvalidAxesException


class KMeans:
    """
    Base class for KMeans algotithm.
    """
    def __init__(self, k=3, max_iter=1_000_000, parameter=0.1):
        """Initialize KMeans."""
        self.k = k
        self.max_iter = max_iter
        self.parameter = parameter

        if parameter <= 0:
            raise InvalidDistanceMetricException("parameter cannot be less or equal 0")


    def _generate_random_centers(self, data):
        """Generate random centers."""
        mean = np.mean(data, axis = 0)
        std = np.std(data,  axis = 0)
        centers = np.random.randn(self.k, data.shape[1])*std + mean
        return centers


    def get_metric(self, axis=0):
        """Get metric."""
        if axis == 0:
            return lambda u, v: np.sum(np.abs(u-v)**self.parameter)**(1/self.parameter)
        if axis == 1:
            return lambda u, v: np.sum(np.abs(u-v)**self.parameter, axis=1)**(1/self.parameter)
        raise InvalidAxesException("axis parameter can be only 0 or 1")


    def calculate_error(self, centers_new, centers_old):
        """Calculate error."""
        func = self.get_metric()
        res = func(centers_new, centers_old)
        return res


    def calculate_distance(self, data, centers):
        """Calculate distance to every center."""
        distances = np.zeros((data.shape[0], self.k))
        func = self.get_metric(axis=1)
        for i in range(self.k):
            distances[:,i] = func(data, centers[i])
        return distances


    def fit(self, data):
        """Fit data."""
        centers = self._generate_random_centers(data)

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

    def transform(self, centers, data):
        '''Align input data for their clusters'''
        distances = self.calculate_distance(data, centers)
        clusters = np.argmin(distances, axis = 1)
        return clusters

    def fit_transform(self, data):
        '''Fit data and align input data for their clusters'''
        centers = self.fit(data)
        clusters = self.transform(centers, data)
        return clusters
