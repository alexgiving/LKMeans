from pathlib import Path

import numpy as np
from tap import Tap

from lib.experiment import get_covariance_matrix, run_experiment


class ArgumentParser(Tap):
    path: Path = Path('experiments')
    '''Path to save results'''

    experiment_id: int = 1
    '''
    Experiment id:
        1) 2 normal distributions

        2) 3 normal distributions
    '''


def _get_experiment_data(experiment_id: int, dimension: int) -> tuple[int, float, list[np.ndarray], list[np.ndarray]]:

    n_clusters: int = 0
    prob: float = 0.
    mu_prefix: list[list[float | int]] = [[]]
    sigma_list: list[float | int] = []

    if experiment_id == 1:
        print('Experiment with 2 clusters')
        n_clusters = 2
        sigma_list = [1, 1]
        prob = 0.5
        mu_prefix = [[-4, 0], [4, 0]]

    elif experiment_id == 2:
        print('Experiment with 3 clusters')
        n_clusters = 3
        sigma_list = [1, 1, 1]
        prob = 1/3
        mu_prefix = [[4, 0, 0], [0, 4, 0], [0, 0, 4]]

    else:
        KeyError(f'Not supported experiment type: {experiment_id}')

    mu_list = [np.array([x + [0] * (dimension - len(x))]) for x in mu_prefix]
    cov_matrix = [get_covariance_matrix(sigma, dimension) for sigma in sigma_list]
    return n_clusters, prob, mu_list, cov_matrix


def main():
    args = ArgumentParser().parse_args()
    experiments_path = args.path

    minkowski_parameter = [0.6, 2]
    T_parameter = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    repeats = 10
    n_points = [100, 500]

    dimension = 20
    T_parameter = [0.4, 1]
    n_clusters, prob, mu_list, cov_matrices = _get_experiment_data(
        experiment_id=args.experiment_id, dimension=dimension)

    for points in n_points:
        experiment_name = f'Clusters:{n_clusters}, points:{points}'
        output_path = experiments_path / f'exp_1_{points}'

        run_experiment(
            n_clusters=n_clusters,
            distance_parameters=T_parameter,
            minkowski_parameters=minkowski_parameter,
            repeats=repeats,
            n_points=points,
            cov_matrices=cov_matrices,
            prob=prob,
            mu_list=mu_list,
            experiment_name=experiment_name,
            output_path=output_path
        )


if __name__ == '__main__':
    main()
