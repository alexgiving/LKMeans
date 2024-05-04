import numpy as np
from numpy.typing import NDArray

from lkmeans.optimizers.optimizer import Optimizer


# pylint: disable=too-few-public-methods
class MeanOptimizer(Optimizer):
    '''
    Standard K-Mean optimizer.
    Returns the optimal `mean` point of input array
    '''

    def _optimize(self, data: NDArray) -> float:
        return float(np.mean(data))
