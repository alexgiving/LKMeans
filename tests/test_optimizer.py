import numpy as np
import pytest

from lib.optimizers import (bound_optimizer, mean_optimizer, median_optimizer,
                            segment_SLSQP_optimizer)


def get_test_data(size: int, center: float) -> np.ndarray:
    data = np.random.random(size)
    reverted_data = data * -1
    samples = np.concatenate([data, reverted_data])
    samples = samples + center
    return samples


testing_params = [-356.189, -2.56, 0, 1.111, 34.954]

params = [
    (np.array([1, 2, 3]), 2),
    (np.array([1, 2, 3, 4]), 2.5),
    (np.array([0]), 0),
    (np.array([0, 0]), 0),
    (np.array([1, 3, 5, 7]), 4),
]

p_values = [0.01, 0.2, 0.5]


@pytest.mark.optimizers
@pytest.mark.parametrize("median", testing_params)
def test_median_calculation(median) -> None:
    n_samples = 50
    samples = get_test_data(n_samples, median)
    np.testing.assert_almost_equal(median, median_optimizer(samples))


@pytest.mark.optimizers
@pytest.mark.parametrize("test_input, expected_output", params)
def test_median_optimizer_inputs(test_input, expected_output):
    np.testing.assert_almost_equal(mean_optimizer(test_input), expected_output)


@pytest.mark.optimizers
@pytest.mark.parametrize("median", testing_params)
def test_mean_optimizer(median) -> None:
    n_samples = 50
    samples = get_test_data(n_samples, median)
    np.testing.assert_almost_equal(median, mean_optimizer(samples))


@pytest.mark.optimizers
@pytest.mark.parametrize("test_input, expected_output", params)
def test_mean_optimizer_inputs(test_input, expected_output):
    np.testing.assert_almost_equal(mean_optimizer(test_input), expected_output)


@pytest.mark.optimizers
@pytest.mark.parametrize("median", testing_params)
@pytest.mark.parametrize("p", p_values)
def test_segment_SLSQP_calculation(median, p) -> None:
    n_samples = 50
    samples = get_test_data(n_samples, median)
    np.testing.assert_almost_equal(median, segment_SLSQP_optimizer(samples, p))


@pytest.mark.optimizers
@pytest.mark.parametrize("test_input, expected_output", params)
def test_segment_SLSQP_optimizer_inputs(test_input, expected_output):
    np.testing.assert_almost_equal(mean_optimizer(test_input), expected_output)


@pytest.mark.optimizers
@pytest.mark.parametrize("median", testing_params)
@pytest.mark.parametrize("p", p_values)
def test_bound_calculation(median, p) -> None:
    n_samples = 50
    samples = get_test_data(n_samples, median)
    np.testing.assert_almost_equal(median, bound_optimizer(samples, p), decimal=0)
