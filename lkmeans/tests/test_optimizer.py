import numpy as np
import pytest
from numpy.typing import NDArray

from lkmeans.optimizers import BoundOptimizer, MeanOptimizer, MedianOptimizer, SegmentSLSQPOptimizer, SLSQPOptimizer


def get_test_data(size: int, center: float) -> NDArray:
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
    optimizer = MedianOptimizer()
    np.testing.assert_almost_equal(median, optimizer(samples))


@pytest.mark.optimizers
@pytest.mark.parametrize("test_input, expected_output", params)
def test_median_optimizer_inputs(test_input, expected_output):
    optimizer = MedianOptimizer()
    np.testing.assert_almost_equal(optimizer(test_input), expected_output)


@pytest.mark.optimizers
@pytest.mark.parametrize("median", testing_params)
def test_mean_optimizer(median) -> None:
    n_samples = 50
    samples = get_test_data(n_samples, median)
    optimizer = MeanOptimizer()
    np.testing.assert_almost_equal(median, optimizer(samples))


@pytest.mark.optimizers
@pytest.mark.parametrize("test_input, expected_output", params)
def test_mean_optimizer_inputs(test_input, expected_output):
    optimizer = MeanOptimizer()
    np.testing.assert_almost_equal(optimizer(test_input), expected_output)


@pytest.mark.optimizers
@pytest.mark.parametrize("median", testing_params)
@pytest.mark.parametrize("p", p_values)
def test_slsqp_calculation(median, p) -> None:
    n_samples = 50
    samples = get_test_data(n_samples, median)
    optimizer = SLSQPOptimizer(p)
    np.testing.assert_almost_equal(median, optimizer(samples), decimal=0)


@pytest.mark.optimizers
@pytest.mark.parametrize("median", testing_params)
@pytest.mark.parametrize("p", p_values)
def test_segment_slsqp_calculation(median, p) -> None:
    n_samples = 50
    samples = get_test_data(n_samples, median)
    optimizer = SegmentSLSQPOptimizer(p)
    np.testing.assert_almost_equal(median, optimizer(samples))


@pytest.mark.optimizers
@pytest.mark.parametrize("median", testing_params)
@pytest.mark.parametrize("p", p_values)
def test_bound_calculation(median, p) -> None:
    n_samples = 50
    samples = get_test_data(n_samples, median)
    optimizer = BoundOptimizer(p)
    np.testing.assert_almost_equal(median, optimizer(samples), decimal=0)
