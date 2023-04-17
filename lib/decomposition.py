from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def get_tsne_clusters(clusters: np.ndarray,
                      labels: np.ndarray,
                      centroids: Optional[np.ndarray] = None) -> plt.Figure:

    fig = plt.figure()
    axe = fig.add_subplot(1,1,1)
    color_map = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'purple'}

    colors = [color_map[label] for label in labels]
    tsne = TSNE(n_components=2, random_state=42)

    if not isinstance(centroids, type(None)):
        concuted = np.concatenate((clusters, np.array(centroids)), axis=0)
        clusters_tsne = tsne.fit_transform(concuted)

        axe.scatter(clusters_tsne[:-len(centroids),0], clusters_tsne[:-len(centroids),1], c=colors)
        axe.scatter(clusters_tsne[-len(centroids):,0], clusters_tsne[-len(centroids):,1], marker='*', s=100, c='black')
    else:
        clusters_tsne = tsne.fit_transform(clusters)
        axe.scatter(clusters_tsne[:,0], clusters_tsne[:,1], c=colors)
    return fig
