from typing import Union

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, SpectralEmbedding, LocallyLinearEmbedding, MDS, Isomap

from lkmeans.clustering.self_supervised.preprocessor_parameters import PreprocessorParameters
from lkmeans.clustering.self_supervised.preprocessor_type import PreprocessorType
from umap import UMAP

AnyPreprocessor = Union[PCA, TSNE, SpectralEmbedding, LocallyLinearEmbedding, MDS, Isomap, UMAP]

def get_preprocessor(preprocessor_type: PreprocessorType, parameters: PreprocessorParameters) -> AnyPreprocessor:
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
