"""Test for distance.py"""
import numpy as np
import pytest
import scipy

from lib.distance import get_lambda_minkowski, lk_norm_matrix


@pytest.mark.parametrize("input_array, parameter", [
    (np.array([[-2.743351, 8.78014917], [ 6.21909165, 2.74060441]]), 0.001),
    (np.array([[-8.743351, 0.4917], [ -4.9165, 15.]]), 0.01),
    ])
def test_lk_norm_matrix(input_array, parameter):
    """Test for lk_norm."""
    res_minkowski = lk_norm_matrix(input_array, input_array, parameter)

    metric = get_lambda_minkowski(parameter)
    res_custom = scipy.spatial.distance.cdist(input_array, input_array, metric)

    np.testing.assert_array_equal(res_minkowski, res_custom)
