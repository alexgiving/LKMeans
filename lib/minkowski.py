import numpy as np


def minkowski_distance(point_a: np.ndarray, point_b: np.ndarray, p: float) -> np.ndarray:
    '''
    Minkowski distance function.
    '''
    absolute_difference = np.abs(point_a - point_b)
    power_in_sum = np.power(absolute_difference, p)
    summa = np.sum(power_in_sum)
    return np.power(summa, 1/p)


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


def minkowski_1_derivative(parameter_to_solve: np.ndarray, cluster: np.ndarray, p: float):
    return np.sum([np.abs(point-parameter_to_solve)**(p-1) for point in cluster], axis=0)
