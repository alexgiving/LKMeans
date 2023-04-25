from typing import List, Union

import numpy as np


def minkowski_distance(point_a: np.ndarray, point_b: np.ndarray, p: float) -> np.ndarray:
    '''
    Minkowski distance function.
    '''

    difference = point_a - point_b
    absolute_difference = np.abs(difference)
    power_in_sum = np.power(absolute_difference, p)
    summa = np.sum(power_in_sum)
    return np.power(summa, 1/p)


def pairwise_minkowski_distance(point_a: np.ndarray, points: Union[np.ndarray, List], p: float) -> np.ndarray:
    '''
    Pairwise Minkowski distance function.
    '''

    result = np.array([])
    for point in points:
        dist = minkowski_distance(point_a, point, p)
        result = np.append(result, dist)
    return result
