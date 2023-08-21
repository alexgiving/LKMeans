import numpy as np
from numpy.typing import NDArray


def minkowski_distance(point_a: NDArray, point_b: NDArray, p: float) -> NDArray:
    '''
    Minkowski distance function.
    '''
    return np.power(np.sum(np.power(np.abs(point_a - point_b), p)), 1/p)


def pairwise_minkowski_distance(point_a: NDArray,
                                points: NDArray | list,
                                p: float
                                ) -> NDArray:
    '''
    Pairwise Minkowski distance function.
    '''

    result = np.array(
        [minkowski_distance(point_a, point, p) for point in points]
    )
    return result
