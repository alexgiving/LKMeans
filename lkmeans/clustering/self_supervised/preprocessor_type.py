from enum import Enum


class PreprocessorType(Enum):
    PCA = "pca"

    TSNE = "tsne"
    SPECTRAL_EMBEDDINGS = "spectral_embeddings"
    LOCALLY_LINEAR_EMBEDDINGS = "locally_linear_embeddings"
    MDS = "mds"
    ISOMAP = "isomap"

    UMAP = "umap"
