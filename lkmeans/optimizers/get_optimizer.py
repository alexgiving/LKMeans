from typing import Union

from lkmeans.optimizers.bound_optimizer import BoundOptimizer
from lkmeans.optimizers.mean_optimizer import MeanOptimizer
from lkmeans.optimizers.median_optimizer import MedianOptimizer
from lkmeans.optimizers.optimizer import Optimizer
from lkmeans.optimizers.slsqp_optimizer import SLSQPOptimizer


def get_optimizer(p: Union[float, int]) -> Optimizer:
    if p == 2:
        return MeanOptimizer()
    if p == 1:
        return MedianOptimizer()
    if 0 < p < 1:
        return BoundOptimizer(p)
    if p > 1:
        return SLSQPOptimizer(p)
    raise ValueError('Parameter p must be greater than 0!')
