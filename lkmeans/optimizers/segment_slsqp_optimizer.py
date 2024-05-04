import warnings
from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from lkmeans.distance import DistanceCalculator
from lkmeans.optimizers.optimizer import Optimizer


# pylint: disable=too-few-public-methods
class SegmentSLSQPOptimizer(Optimizer):

    def __init__(self, p: Union[float, int], tol: float = 1e-1000) -> None:
        super().__init__()
        self._p = p
        self._distance_calculator = DistanceCalculator(self._p)
        self._tol = tol

    def _optimize(self, data: NDArray) -> float:
        data = np.unique(data)

        median = np.median(data)
        fun_median = self._distance_calculator.get_distance(np.array(median), data)

        minimized_fun_median = fun_median
        for bound_id in range(len(data) - 1):

            bounds = [(data[bound_id], data[bound_id + 1])]

            x0 = np.mean(bounds)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = minimize(
                    fun=lambda centre: self._distance_calculator.get_distance(
                        centre, data),
                    x0=x0,
                    method='SLSQP',
                    bounds=bounds,
                    tol=self._tol
                )

            if res.success:
                minima_point = res.x[0]
                minimal_point_value = res.fun
                if minimal_point_value < minimized_fun_median:
                    minimized_fun_median = minimal_point_value
                    median = minima_point
        return float(median)
