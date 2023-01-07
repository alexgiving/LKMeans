"""Test for distance.py"""
import time

import numpy as np
import pytest
import scipy

from lib.distance import get_lambda_minkowski, lk_norm_matrix


testdata = [
    (np.array([[-2.743351, 8.78014917], [ 6.21909165, 2.74060441]]), 0.001),
    (np.array([[-8.743351, 0.4917], [ -4.9165, 15.]]), 0.01),
    ]


@pytest.mark.parametrize("input_array, parameter", testdata)
def test_quality_lk_norm_matrix(input_array, parameter):
    """Test for lk_norm."""
    res_minkowski = lk_norm_matrix(input_array, input_array, parameter)

    metric = get_lambda_minkowski(parameter)
    res_custom = scipy.spatial.distance.cdist(input_array, input_array, metric)

    np.testing.assert_array_equal(res_minkowski, res_custom)


@pytest.mark.parametrize("input_array, parameter", testdata)
def test_time_lk_norm_matrix(input_array, parameter):
    """Test for lk_norm."""
    iterations = 500

    original_times = []
    for _ in range(iterations):
        start_time = time.time()
        _ = lk_norm_matrix(input_array, input_array, parameter)
        end_time = time.time()
        original_times.append(end_time - start_time)

    custom_times = []
    for _ in range(iterations):
        start_time = time.time()
        metric = get_lambda_minkowski(parameter)
        _ = scipy.spatial.distance.cdist(input_array, input_array, metric)
        end_time = time.time()
        custom_times.append(end_time - start_time)

    assert np.mean(custom_times) > np.mean(original_times)
