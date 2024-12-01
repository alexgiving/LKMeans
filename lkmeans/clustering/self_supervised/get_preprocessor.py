from typing import Union

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from lkmeans.clustering.self_supervised.preprocessor_parameters import PreprocessorParameters
from lkmeans.clustering.self_supervised.preprocessor_type import PreprocessorType

AnyPreprocessor = Union[PCA, TSNE]

def get_preprocessor(preprocessor_type: PreprocessorType, parameters: PreprocessorParameters) -> AnyPreprocessor:
    preprocessor_map = {
        PreprocessorType.PCA: PCA,
        PreprocessorType.TSNE: TSNE
    }
    return preprocessor_map[preprocessor_type](n_components=parameters.n_components)
