import numpy as np
from numpy.typing import NDArray

from lkmeans.clustering.self_supervised.get_preprocessor import get_preprocessor
from lkmeans.clustering.self_supervised.preprocessor_parameters import (
    PreprocessorParameters,
)
from lkmeans.clustering.self_supervised.preprocessor_type import PreprocessorType


class SelfSupervisedPreprocessor:
    def __init__(self, preprocessor_type: PreprocessorType, parameters: PreprocessorParameters) -> None:
        self._preprocessor_type = preprocessor_type
        self._preprocessor = get_preprocessor(self._preprocessor_type, parameters)

    def preprocess(self, X: NDArray) -> NDArray:
        result = self._preprocessor.fit_transform(X)
        return np.array(result)
