import numpy as np
from numpy.typing import NDArray

from lkmeans.optimizers.optimizer import Optimizer


class MedianOptimizer(Optimizer):
    '''
    Standard K-Medoids optimizer.
    Returns the optimal `median` point of input array
    '''

    def _optimize(self, data: NDArray) -> float:
        return float(np.median(data))
