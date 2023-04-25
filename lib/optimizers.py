import warnings

import numpy as np
from scipy.optimize import minimize

from lib.minkowski import minkowski_distance
from lib.types import p_type


def median_optimizer(dimension_slice: np.ndarray):
    value = np.median(dimension_slice)
    return value


def mean_optimizer(dimension_slice: np.ndarray):
    value = np.mean(dimension_slice)
    return value


def segment_SLSQP_optimizer(dimension_slice: np.ndarray, p: p_type):
    dimension_slice = np.unique(dimension_slice)

    research_minima_value = np.inf
    value = np.median(dimension_slice)

    for bound_id in range(len(dimension_slice) - 1):

        bounds = [(dimension_slice[bound_id],
                   dimension_slice[bound_id + 1])]

        x0 = np.median(bounds)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(
                fun=lambda x: minkowski_distance(x, dimension_slice, p),
                x0=x0,
                method='SLSQP',
                bounds=bounds
            )

        if res.success:
            minima_point = res.x[0]
            minimal_point_value = res.fun
            if minimal_point_value < research_minima_value:
                research_minima_value = minimal_point_value
                value = minima_point

    return value


# def sgd_optimizer(
#         cluster: np.ndarray,
#         p: p_type,
#         learning_rate: float = 0.01,
#         grad_descent: float = 0.1,
#         n_iters: int = 50) -> np.ndarray:
#     '''
#     SGD optimizer
#     Amorim, Renato. (2012). Feature Weighting for Clustering: Using K-Means and the Minkowski Metric.
#     '''

#     def minkowski_loss(cluster: np.ndarray, centroid: np.ndarray, p: p_type) -> np.ndarray:
#         '''
#         SGD Minkowski Loss function.
#         Return the coordinate sum of the Minkowski differences
#         Formula: [∑_j (xji - ci)^p]
#         '''
#         loss = []
#         for point in cluster:
#             absolute_difference = np.abs(point - centroid)
#             power_in_sum = np.power(absolute_difference, p)
#             loss.append(power_in_sum)
#         loss = np.array(loss)
#         dim_loss = np.sum(loss, axis=0)
#         return dim_loss

#     learning_rate = 0.01
#     grad_descent = 0.1
#     n_iters = 50
#     centroid = np.mean(cluster, axis=0)

#     for sgd_iteration in range(n_iters):
#         if sgd_iteration == n_iters / 2:
#             learning_rate *= grad_descent
#         elif sgd_iteration == n_iters / 4:
#             learning_rate *= grad_descent
#         loss = minkowski_loss(cluster, centroid, p)
#         grad = np.gradient(loss, axis=0)
#         centroid -= learning_rate * grad
#     return centroid


# def extremum_optimizer(cluster: np.ndarray, p: p_type) -> np.ndarray:
#     '''
#     Find extremum of minkowski function by root of 1 derivative.
#     '''
#     def minkowski_1_derivative(parameter_to_solve: np.ndarray, cluster: np.ndarray, p: float):
#         return np.sum([np.abs(point-parameter_to_solve)**(p-1) for point in cluster], axis=0)

#     sol = root(minkowski_1_derivative, np.mean(
#         cluster, axis=0), args=(cluster, p), method='hybr')
#     return sol.x