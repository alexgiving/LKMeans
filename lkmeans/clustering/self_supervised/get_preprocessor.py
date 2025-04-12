from typing import Union

from sklearn.decomposition import PCA
from sklearn.manifold import (
    MDS,
    TSNE,
    Isomap,
    LocallyLinearEmbedding,
    SpectralEmbedding,
)
from umap import UMAP

from lkmeans.clustering.self_supervised.preprocessor_parameters import (
    PreprocessorParameters,
)
from lkmeans.clustering.self_supervised.preprocessor_type import PreprocessorType

AnyPreprocessor = Union[PCA, TSNE, SpectralEmbedding, LocallyLinearEmbedding, MDS, Isomap, UMAP]


def get_preprocessor(
    preprocessor_type: PreprocessorType, parameters: PreprocessorParameters
) -> AnyPreprocessor:
    preprocessor_map = {
        PreprocessorType.PCA: PCA,
        PreprocessorType.TSNE: TSNE,
        PreprocessorType.SPECTRAL_EMBEDDINGS: SpectralEmbedding,
        PreprocessorType.LOCALLY_LINEAR_EMBEDDINGS: LocallyLinearEmbedding,
        PreprocessorType.MDS: MDS,
        PreprocessorType.ISOMAP: Isomap,
        PreprocessorType.UMAP: UMAP,
    }
    return preprocessor_map[preprocessor_type](n_components=parameters.n_components)
