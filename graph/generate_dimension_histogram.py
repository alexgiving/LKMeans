from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tap import Tap

from lib.experiment import get_covariance_matrix
from lib.points_generator import generate_mix_distribution


class ArgumentParser(Tap):
    path: Path = Path('images/')


def main():
    args = ArgumentParser().parse_args()
    path = args.path

    dimension = 20
    n_points = 100

    sigma_list = [1, 1]
    prob = 0.5
    mu_list = [np.array([x + [0] * (dimension-2)])
               for x in [[-4, 0], [4, 0]]]


    cov_matrix_list = [get_covariance_matrix(
        sigma, dimension) for sigma in sigma_list]

    for t in [0, 0.3, 0.7, 0.9]:
        filename = path / f'2cluster_hist_t_{t}.png'
        clusters, _, _ = generate_mix_distribution(
            probability=prob,
            mu_list=mu_list,
            cov_matrix_list=cov_matrix_list,
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
