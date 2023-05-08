import numpy as np


def minkowski_distance(point_a: np.ndarray, point_b: np.ndarray, p: float) -> np.ndarray:
    '''
    Minkowski distance function.
    '''
    return np.power(np.sum(np.power(np.abs(point_a - point_b), p)), 1/p)


def pairwise_minkowski_distance(point_a: np.ndarray,
                                points: np.ndarray | list,
                                p: float
                                ) -> np.ndarray:
    '''
    Pairwise Minkowski distance function.
    '''

    result = np.array(
        [minkowski_distance(point_a, point, p) for point in points]
    )
    return result
