import numpy as np


def minkowski_distance(point_a: np.ndarray, point_b: np.ndarray, p: float) -> np.ndarray:
    '''
    Minkowski distance function.
    '''
    absolute_difference = np.abs(point_a - point_b)
    power_in_sum = np.power(absolute_difference, p)
    summa = np.sum(power_in_sum)
    return np.power(summa, 1/p)

