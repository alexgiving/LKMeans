import numpy as np
from numpy.typing import NDArray


def get_covariance_matrix(sigma: float, dimension: int) -> NDArray:
    return np.eye(dimension) * sigma


def get_experiment_data(num_clusters: int, dimension: int) -> tuple[int, float, list[NDArray], list[NDArray]]:
    '''
    Function for generation the synthetic data for experiments
    by dimension and number of clusters
    '''
    prob: float = 0.
    mu_prefix: list[list[float | int]] = [[]]
    sigma_list: list[float | int] = []

    if num_clusters == 2:
        sigma_list = [1, 1]
        prob = 0.5
        mu_prefix = [[-4, 0], [4, 0]]

    elif num_clusters == 3:
        sigma_list = [1, 1, 1]
        prob = 1/3
        mu_prefix = [[4, 0, 0], [0, 4, 0], [0, 0, 4]]

    else:
        raise KeyError(f'Not supported experiment type: {num_clusters}')

    mu_list = [np.array([x + [0] * (dimension - len(x))]) for x in mu_prefix]
    cov_matrix = [get_covariance_matrix(sigma, dimension) for sigma in sigma_list]
    return num_clusters, prob, mu_list, cov_matrix
