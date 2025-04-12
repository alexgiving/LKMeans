import warnings
from typing import Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from lkmeans.distance import DistanceCalculator
from lkmeans.optimizers.optimizer import Optimizer


# pylint: disable=too-few-public-methods
class SLSQPOptimizer(Optimizer):

    def __init__(self, p: Union[float, int], tol: float = 1e-1000) -> None:
        super().__init__()
        self._p = p
        self._distance_calculator = DistanceCalculator(self._p)
        self._tol = tol

    def _optimize(self, data: NDArray) -> float:
        x0 = np.mean(data)
        bounds = [(min(data), max(data))]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = minimize(
                fun=lambda centre: self._distance_calculator.get_distance(centre, data),
                x0=x0,
                method="SLSQP",
                bounds=bounds,
                tol=self._tol,
            ).x[0]
        return float(res)
