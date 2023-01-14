"""Test for distance.py"""
import time

import numpy as np
import pytest
import scipy

from lib.distance import get_lambda_minkowski, lk_norm_matrix


test_inputs = [
    np.array([[-2.743351, 8.78014917], [ 6.21909165, 2.74060441]]),
    np.array([[-8.743351, 0.4917], [ -4.9165, 15.]]),
    ]
test_parameters = [0.1, 0.01, 2]


@pytest.mark.parametrize("input_array", test_inputs)
@pytest.mark.parametrize("parameter", test_parameters)
def test_quality_lk_norm_matrix(input_array, parameter):
    '''
    Test that provides equals of calculations of lk norm
    via lambda function and scipy builtin minkowski
    Run pytest -s to see boopst of builtin calculations.
    '''
    start_time = time.time()
    res_minkowski = lk_norm_matrix(input_array, input_array, parameter)
    end_time = time.time()
    builtin_time = end_time - start_time

    start_time = time.time()
    metric = get_lambda_minkowski(parameter)
    res_custom = scipy.spatial.distance.cdist(input_array, input_array, metric)
    end_time = time.time()
    custom_time = end_time - start_time

    print(f"Lambda: {parameter}\nOriginal time: {builtin_time}\n",
            f"Lambda time: {custom_time}\nOriginal faster in {custom_time/builtin_time}x times\n")
    np.testing.assert_array_equal(res_minkowski, res_custom)


@pytest.mark.parametrize("input_array", test_inputs)
@pytest.mark.parametrize("parameter", test_parameters)
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
