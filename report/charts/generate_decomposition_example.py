from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from lkmeans.clustering.self_supervised.get_preprocessor import get_preprocessor
from lkmeans.clustering.self_supervised.preprocessor_parameters import PreprocessorParameters
from lkmeans.clustering.self_supervised.preprocessor_type import PreprocessorType


import matplotlib.pyplot as plt
from lkmeans.examples.main import DataType, generate_data
import numpy as np

from tap import Tap


class ArgumentParser(Tap):
    t_parameter: Optional[float] = None
    n_points: int = 1000
    num_clusters: int = 2
    dimension: int = 20
    dataset: DataType = DataType.GENERATED

    path: Path = Path("data/decomposer")
    """Path to save results"""


def main():
    args = ArgumentParser(underscores_to_dashes=True).parse_args()
    args.path.mkdir(exist_ok=True)

    np.random.seed(5)
    color_map = {
        0: 'blue',
        1: 'green',
        2: 'darkgoldenrod',
        3: 'red',
        4: 'purple',
        5: 'orange',
        6: 'cyan',
        7: 'magenta',
        8: 'lime',
        9: 'pink'
    }

    if args.dataset is DataType.GENERATED:
        filename = args.path / f'{args.num_clusters}_cluster_decomposer_analysis_t_{args.t_parameter}.png'
    else:
        filename = args.path / f'{args.dataset.value}_decomposer_analysis.png'

    clusters, labels = generate_data(args)

    fig, axes = plt.subplots(2, 3, figsize=(int(5.2*3), 10))
    colors = [color_map[label] for label in labels]

    preprocessors = [
        ("PCA", PreprocessorType.PCA),
        ("Spectral Embeddings", PreprocessorType.SPECTRAL_EMBEDDINGS),
        ("Locally Linear Embeddings", PreprocessorType.LOCALLY_LINEAR_EMBEDDINGS),
        ("MDS", PreprocessorType.MDS),
        ("ISOMAP", PreprocessorType.ISOMAP),
        ("UMAP", PreprocessorType.UMAP),
    ]

    parameters = PreprocessorParameters(n_components=2)
    for item, row_ax in enumerate(axes):
        for ax in row_ax:
            name, preprocessor_type = preprocessors[item]
            method = get_preprocessor(preprocessor_type, parameters)

            X_transformed = method.fit_transform(clusters)
            ax.scatter(X_transformed[:, 0], X_transformed[:, 1], c=colors)
            ax.set_title(name)
            item += 1

    fig.savefig(filename, dpi=800, bbox_inches='tight')

if __name__ == '__main__':
    main()
