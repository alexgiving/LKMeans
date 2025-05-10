from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt

from lkmeans.clustering.self_supervised.get_preprocessor import get_preprocessor
from lkmeans.clustering.self_supervised.preprocessor_parameters import PreprocessorParameters
from lkmeans.clustering.self_supervised.preprocessor_type import PreprocessorType
from lkmeans.examples.data.experiment_data import get_experiment_data
from lkmeans.examples.data.points_generator import generate_mix_distribution


import matplotlib.pyplot as plt
import numpy as np


parser = ArgumentParser()

parser.add_argument(
    '--path',
    type=Path,
    default=Path('data/decomposer'),
    help='Path to save results'
)


def main():
    args = parser.parse_args()
    args.path.mkdir(exist_ok=True)

    dimension = 20
    n_points = 1000
    np.random.seed(5)

    n_clusters, prob, mu_list, cov_matrices = get_experiment_data(num_clusters=2, dimension=dimension)
    color_map = {3: 'red', 1: 'green', 0: 'blue', 3: 'yellow', 4: 'purple'}

    for t in [0.4, 0.8]:
        filename = args.path / f'{n_clusters}_cluster_decomposer_analysis_t_{t}.png'
        clusters, labels, _ = generate_mix_distribution(
            probability=prob,
            mu_list=mu_list,
            cov_matrices=cov_matrices,
            n_samples=n_points,
            t=t
        )

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
