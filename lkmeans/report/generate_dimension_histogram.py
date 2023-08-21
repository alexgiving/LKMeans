from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt

from lkmeans.data_generation.points_generator import generate_mix_distribution
from lkmeans.examples.experiment_data import get_experiment_data

parser = ArgumentParser()

parser.add_argument(
    '--path',
    type=Path,
    default=Path('images'),
    help='Path to save results'
)


def main():
    args = parser.parse_args()
    args.path.mkdir(exist_ok=True)

    dimension = 20
    n_points = 100

    n_clusters, prob, mu_list, cov_matrices = get_experiment_data(experiment_id=1, dimension=dimension)

    for t in [0.2, 0.4, 0.9]:
        filename = args.path / f'{n_clusters}_cluster_hist_t_{t}.png'
        clusters, _, _ = generate_mix_distribution(
            probability=prob,
            mu_list=mu_list,
            cov_matrices=cov_matrices,
            n_samples=n_points,
            t=t
        )

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(clusters[:, 0], bins=15)
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
        fig.savefig(str(filename), dpi=300, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    main()
