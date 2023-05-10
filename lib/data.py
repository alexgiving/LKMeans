import numpy as np

from lib.experiment import get_covariance_matrix


def get_experiment_data(experiment_id: int, dimension: int) -> tuple[int, float, list[np.ndarray], list[np.ndarray]]:

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
        raise KeyError(f'Not supported experiment type: {experiment_id}')

    mu_list = [np.array([x + [0] * (dimension - len(x))]) for x in mu_prefix]
    cov_matrix = [get_covariance_matrix(sigma, dimension) for sigma in sigma_list]
    return n_clusters, prob, mu_list, cov_matrix
