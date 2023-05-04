from pathlib import Path

import numpy as np

from lib import run_experiment


def main():
    experiments_path = Path('experiments')

    dimension = 20
    n_clusters = 3
    T_parameter = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    minkowski_parameter = [0.2, 0.5, 1, 2]
    repeats = 20
    n_points = [100, 500]

    sigma_list = [1, 1, 1]
    prob = 1/3
    mu_list = [np.array([x + [0] * (dimension-3)])
                for x in [[4, 0, 0], [0, 4, 0], [0, 0, 4]]]

    minkowski_parameter = [0.2, 0.5, 0.7, 0.9, 1, 2, 5, 10]

    for points in n_points:
        experiment_name = f'Experiment 2, N_points:{points}'
        output_path = experiments_path / f'experiment_2_{points}'

        run_experiment(
            dimension=dimension,
            n_clusters=n_clusters,
            distance_parameters=T_parameter,
            minkowski_parameters=minkowski_parameter,
            repeats=repeats,
            n_points=points,
            sigma_list=sigma_list,
            prob=prob,
            mu_list=mu_list,
            experiment_name=experiment_name,
            output_path=output_path
        )


if __name__ == '__main__':
    main()
