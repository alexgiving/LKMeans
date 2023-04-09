from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def fit_to_2d_tsne(data):
    """Fit data to 2d."""
    tsne = TSNE(n_components=2, random_state=0, perplexity=30, n_iter=5000)
    return tsne.fit_transform(data)


def fit_to_2d_PCA(data):
    """Fit data to 2d."""
    pca = PCA(n_components=2)
    pca.fit(data)
    return pca.transform(data)
