from abc import ABC, abstractmethod

from numpy.typing import NDArray


class Optimizer(ABC):

    @abstractmethod
    def _optimize(self, data: NDArray) -> float:
        ...

    def __call__(self, data: NDArray) -> float:
        return self._optimize(data)
