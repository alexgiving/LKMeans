from typing import Union

import numpy as np
from numpy.typing import NDArray

from lkmeans.distance import DistanceCalculator
from lkmeans.optimizers.optimizer import Optimizer


class BoundOptimizer(Optimizer):
    '''
    Special LKMeans optimizer.
    Based on idea that for 0 < p < 1 the minkowski function is a concave function.
    Returns the optimal point of input array with Minkowski metric parameter `p`
    '''

    def __init__(self, p: Union[float, int]) -> None:
        super().__init__()
        self._p = p
        self._distance_calculator = DistanceCalculator(self._p)

    def _optimize(self, data: NDArray) -> float:
        points = np.unique(data)

        optimal_point = points[0]
        smallest_distant = self._distance_calculator.get_distance(optimal_point, data)
        for applicant in points:
            distance_of_applicant = self._distance_calculator.get_distance(applicant, data)
            if distance_of_applicant < smallest_distant:
                optimal_point = applicant
                smallest_distant = distance_of_applicant
        return float(optimal_point)
