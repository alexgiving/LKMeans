from abc import abstractmethod

from numpy.typing import NDArray

from lkmeans.clustering.base import Clustering
from lkmeans.clustering.utils import set_type


class UnsupervisedClustering(Clustering):

    @abstractmethod
    def _fit(self, X: NDArray) -> None:
        ...

    def fit(self, X: NDArray) -> None:
        X = set_type(X)
        self._fit(X)

    def fit_predict(self, X: NDArray) -> list[int]:
        X = set_type(X)
        self._fit(X)
        labels = self.predict(X)
        return labels