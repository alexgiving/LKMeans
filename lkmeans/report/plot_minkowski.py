from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lkmeans.clustering.utils import assign_to_cluster
from lkmeans.data_generation.points_generator import generate_mix_distribution
from lkmeans.distance import DistanceCalculator
from lkmeans.examples.experiment_data import get_experiment_data

parser = ArgumentParser()

parser.add_argument(
    '--path',
    type=Path,
    default=Path('images'),
    help='Path to save results'
)

parser.add_argument(
    '--p',
    type=float,
    default=2,
    help='Minkowski parameter'
)

parser.add_argument(
    '--t',
    type=float,
    default=0.,
    help='T parameter of distribution'
)


# pylint: disable=too-many-locals
def main():
    args = parser.parse_args()
    args.path.mkdir(exist_ok=True)
    p = int(args.p) if (args.p).is_integer() else args.p

    dimension = 20
    n_points = 10
    n_observation = 10000

    distance_calculator = DistanceCalculator(p)

    n_clusters, prob, mu_list, cov_matrices = get_experiment_data(num_clusters=2, dimension=dimension)

    filename = args.path / f'plot_minkowski_function_with_p_{p}.png'
    samples, _, centroids = generate_mix_distribution(
        probability=prob,
        mu_list=mu_list,
        cov_matrices=cov_matrices,
        n_samples=n_points,
        t=0.1
    )

    dim = 0

    clusters, _ = assign_to_cluster(samples, centroids, n_clusters, p)
    cluster = np.array(clusters[0])
    dimension_data = cluster[:,dim]

    points = np.linspace(min(dimension_data), max(dimension_data), n_observation)
    minkowski_values = distance_calculator.get_pairwise_distance(
        point_a = dimension_data,
        points=points,
    )

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(points, minkowski_values)
    ax.axis('off')
    fig.savefig(str(filename), dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
