from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from tap import Tap

from lib.data import get_experiment_data
from lib.kmeans import assign_to_cluster
from lib.minkowski import minkowski_distance, pairwise_minkowski_distance
from lib.points_generator import generate_mix_distribution


class ArgumentParser(Tap):
    path: Path = Path('images/')
    '''Path to save image'''

    p: float | int = 2
    '''Minkowski parameter'''

    t: float = 0.
    '''Parameter of data distribution'''

    @staticmethod
    def to_number(string: str) -> Union[float, int]:
        return float(string) if '.' in string else int(string)

    def configure(self):
        self.add_argument('-p', type=self.to_number)


# pylint: disable=too-many-locals
def main():
    args = ArgumentParser().parse_args()
    args.path.mkdir(exist_ok=True)
    p = args.p

    dimension = 20
    n_points = 10

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

    points = np.linspace(min(dimension_data), max(dimension_data), 100)
    minkowski_values = pairwise_minkowski_distance(
        point_a = dimension_data,
        points=points,
        p=p
    )

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.scatter(points, minkowski_values)
    ax.scatter(centroids[0][dim], minkowski_distance(centroids[0][dim], dimension_data, p))
    fig.savefig(str(filename), dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
