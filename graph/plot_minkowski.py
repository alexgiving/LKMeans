from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lib.data import get_experiment_data
from lib.lkmeans import assign_to_cluster
from lib.minkowski import pairwise_minkowski_distance
from lib.points_generator import generate_mix_distribution

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

    n_clusters, prob, mu_list, cov_matrices = get_experiment_data(experiment_id=1, dimension=dimension)

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
    minkowski_values = pairwise_minkowski_distance(
        point_a = dimension_data,
        points=points,
        p=p
    )

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(points, minkowski_values)
    # from lib.minkowski import minkowski_distance
    # ax.scatter(centroids[0][dim], minkowski_distance(centroids[0][dim], dimension_data, p))
    fig.savefig(str(filename), dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
