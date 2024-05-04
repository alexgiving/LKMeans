from abc import ABC, abstractmethod

from numpy.typing import NDArray


# pylint: disable=too-few-public-methods
class Optimizer(ABC):

    @abstractmethod
    def _optimize(self, data: NDArray) -> float:
        ...

    def __call__(self, data: NDArray) -> float:
        return self._optimize(data)
