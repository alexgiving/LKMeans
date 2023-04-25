from typing import Union

import numpy as np
from scipy.optimize import minimize, root

from lib.minkowski import minkowski_distance


def sgd_optimizer(
        cluster: np.ndarray,
        p: Union[float, int],
        learning_rate: float = 0.01,
        grad_descent: float = 0.1,
        n_iters: int = 50) -> np.ndarray:
    '''
    SGD optimizer
    Amorim, Renato. (2012). Feature Weighting for Clustering: Using K-Means and the Minkowski Metric. 
    '''

    def minkowski_loss(cluster: np.ndarray, centroid: np.ndarray, p: float) -> np.ndarray:
        '''
        SGD Minkowski Loss function.
        Return the coordinate sum of the Minkowski differences
        Formula: [∑_j (xji - ci)^p]
        '''
        loss = []
        for point in cluster:
            absolute_difference = np.abs(point - centroid)
            power_in_sum = np.power(absolute_difference, p)
            loss.append(power_in_sum)
        loss = np.array(loss)
        dim_loss = np.sum(loss, axis=0)
        return dim_loss

    learning_rate = 0.01
    grad_descent = 0.1
    n_iters = 50
    centroid = np.mean(cluster, axis=0)

    for sgd_iteration in range(n_iters):
        if sgd_iteration == n_iters / 2:
            learning_rate *= grad_descent
        elif sgd_iteration == n_iters / 4:
            learning_rate *= grad_descent
        loss = minkowski_loss(cluster, centroid, p)
        grad = np.gradient(loss, axis=0)
        centroid -= learning_rate * grad
    return centroid


def extremum_optimizer(cluster: np.ndarray, p: Union[float, int]) -> np.ndarray:
    '''
    Find extremum of minkowski function by root of 1 derivative.
    '''
    def minkowski_1_derivative(parameter_to_solve: np.ndarray, cluster: np.ndarray, p: float):
        return np.sum([np.abs(point-parameter_to_solve)**(p-1) for point in cluster], axis=0)

    sol = root(minkowski_1_derivative, np.mean(
        cluster, axis=0), args=(cluster, p), method='hybr')
    return sol.x


def segment_solver_optimizer(cluster: np.ndarray, p: Union[float, int]) -> np.ndarray:
    data_dimension = cluster.shape[0]
    new_centroid = []

    for coordinate_id in range(data_dimension):
        coordinate_sliced_points = cluster[coordinate_id, :]
        unique_coordinate_sliced_points = sorted(set(coordinate_sliced_points))

        research_minima_value = np.inf
        research_minimal_coordinate = np.mean(coordinate_sliced_points)

        for bound_id in range(len(unique_coordinate_sliced_points) - 1):

            bounds = [(unique_coordinate_sliced_points[bound_id],
                       unique_coordinate_sliced_points[bound_id+1])]

            start_value = np.mean(bounds)

            minima_point = minimize(
                fun=lambda x: minkowski_distance(
                    x, coordinate_sliced_points, p),
                x0=start_value,
                method='SLSQP',
                bounds=bounds
            ).x
            minimal_point_value = minkowski_distance(
                minima_point, coordinate_sliced_points, p)

            if minimal_point_value < research_minima_value:
                research_minima_value = minimal_point_value
                research_minimal_coordinate = minima_point
        new_centroid.append(np.mean(coordinate_sliced_points))
    assert len(new_centroid) == data_dimension
    return np.array(new_centroid)


class KMeans:
    def __init__(self, n_clusters, max_iter=100, p=2):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.p = p
        self.centroids = []
        self.labels = []

    def fit(self, X):
        # initialize centroids randomly
        self.centroids = [X[i] for i in np.random.choice(
            X.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            clusters = [[] for _ in range(self.n_clusters)]
            self.labels = []

            # assign each data point to the closest centroid
            for x in X:
                distances_to_each_cebtroid = [minkowski_distance(
                    x, centroid, self.p) for centroid in self.centroids]
                closest_centroid = np.argmin(distances_to_each_cebtroid)
                clusters[closest_centroid].append(x)
                self.labels.append(closest_centroid)

            # update centroids using the specified optimizer
            for cluster_id, cluster in enumerate(clusters):
                cluster = np.array(cluster)
                if len(cluster) == 0:
                    continue
                if self.p == 2:
                    self.centroids[cluster_id] = np.mean(cluster, axis=0)
                elif self.p == 1:
                    self.centroids[cluster_id] = np.median(cluster, axis=0)
                elif self.p > 1:
                    self.centroids[cluster_id] = sgd_optimizer(cluster, self.p)
                elif 0 < self.p < 1:
                    self.centroids[cluster_id] = segment_solver_optimizer(
                        cluster, self.p)
                else:
                    raise ValueError(f'Unsupported value of p: {self.p}')
                    # if self.optimizer == 'SLSQP':
                    #     bounds = [(None, None)] * self.centroids[j].shape[0]
                    #     self.centroids[j] = minimize(
                    #         lambda x: minkowski_distance(x, cluster, self.p),
                    #         self.centroids[j].flatten(),
                    #         method=self.optimizer,
                    #         bounds=bounds
                    #     ).x.copy()
                    # elif self.optimizer in ('Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'trust-constr'):
                    #     self.centroids[j] = minimize(lambda x: minkowski_distance(x, cluster, self.p), self.centroids[j], method=self.optimizer).x.copy()

        return self.centroids, self.labels
