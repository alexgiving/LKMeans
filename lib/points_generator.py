import numpy as np


def move_towards_mean(mu: np.ndarray, mu_mean: np.ndarray, t: float) -> np.ndarray:
    '''
    Returns a new point that is moved towards the mean according to the rule:
    point(t) = point + t * (mean - point), 0 ≤ t ≤ 1
    '''
    new_mu = mu + t * (mu_mean - mu)
    return new_mu


# pylint: disable= too-many-locals
def generate_mix_distribution(
    probability: float,
    mu_list: list[np.ndarray],
    cov_matrix_list: list[np.ndarray],
    n_samples: int,
    t: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Returns points from gaussian distributions.
    Generated by distribution probability, covariation matrix and means vector.
    '''

    mu_mean = np.mean(mu_list, axis=0)

    modified_mu_list = []
    for mu in mu_list:
        modified_mu_list.append(move_towards_mean(mu, mu_mean, t))

    # need be dynamic
    if len(mu_list) == 2:
        n_1 = int(probability * n_samples)
        n_list = [n_1, n_samples - n_1]
    elif len(mu_list) == 3:
        n_1 = int(probability * n_samples)
        n_2 = int(probability * n_samples)
        n_3 = n_samples - n_1 - n_2
        n_list = [n_1, n_2, n_3]

    distributions = []
    for n, mu, cov_matrix in zip(n_list, modified_mu_list, cov_matrix_list):
        distribution = np.random.multivariate_normal(
            np.squeeze(mu, axis=0), cov_matrix, n)
        distributions.append(distribution)

    samples = np.concatenate(distributions, axis=0)

    labels = np.array([])
    for lable_id, n in enumerate(n_list):
        samples_labels = np.full(n, lable_id)
        labels = np.concatenate((labels, samples_labels))

    # Shuffle the samples and labels
    permutation = np.random.permutation(n_samples)
    samples = samples[permutation]
    labels = labels[permutation]

    return samples, labels, np.mean(modified_mu_list, axis=1)
