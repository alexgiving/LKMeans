from typing import List, Union

import numpy as np
from numpy.typing import NDArray


def _minkowski_distance(point_a: NDArray, point_b: NDArray, p: float) -> NDArray:
    """
    Minkowski distance function.
    """
    return np.array(np.power(np.sum(np.power(np.abs(point_a - point_b), p)), 1 / p))


def _pairwise_minkowski_distance(point_a: NDArray, points: NDArray | list, p: float) -> NDArray:
    """
    Pairwise Minkowski distance function.
    """

    result = np.array([_minkowski_distance(point_a, point, p) for point in points])
    return result


class DistanceCalculator:

    def __init__(self, p: Union[float, int]) -> None:
        self._p = p

    def get_distance(self, point_a: NDArray, point_b: NDArray) -> NDArray:
        return _minkowski_distance(point_a, point_b, self._p)

    def get_pairwise_distance(self, point_a: NDArray, points: Union[NDArray, List]) -> NDArray:
        return _pairwise_minkowski_distance(point_a, points, self._p)
